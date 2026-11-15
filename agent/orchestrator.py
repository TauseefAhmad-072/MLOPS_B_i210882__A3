"""Conversational orchestrator.

Two planning modes share the same tool layer and safety executor:
  * "llm"            - Claude tool-calling loop (needs ANTHROPIC_API_KEY, or an injected `llm` callable)
  * "offline-rules"  - deterministic keyword planner, clearly labelled in every response
Every operational statement in an answer is built from ToolResult data; failed tools produce
"unknown/unavailable" statements, never invented values.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Callable

import httpx

import agent.tools  # noqa: F401  (registers tools)
from agent.config import get_settings
from agent.schemas import ChatResponse, ToolResult
from agent.tools.base import REGISTRY, execute_tool

SYSTEM_PROMPT = """You are an MLOps assistant operating real infrastructure through tools (DVC, MLflow, Airflow, Kubernetes).
Rules:
1. Every operational claim (metrics, versions, pod state, DAG state) MUST come from a tool result in this conversation. Never answer from memory.
2. If a tool fails, times out or is unavailable, say exactly that and mark the fact as UNKNOWN. Do not guess.
3. Do not call state-changing tools unless the user explicitly asked for that action. Never trigger retraining just because a model looks old/weak: only recommend it, citing evidence (freshness policy, failed quality gate, supplied degradation results).
4. The highest metric alone does not make a model production-ready; mention caveats returned by tools.
5. Structure multi-tool answers as: Known facts / Uncertainties / Recommended actions."""

TRIGGER_RE = re.compile(r"\b(trigger|start|kick ?off|launch|run|execute)\b.*\b(pipeline|dag|training|workflow)\b", re.I)


def anthropic_llm(system: str, messages: list, tools: list) -> dict:
    s = get_settings()
    r = httpx.post("https://api.anthropic.com/v1/messages", timeout=60,
                   headers={"x-api-key": s.anthropic_api_key, "anthropic-version": "2023-06-01",
                            "content-type": "application/json"},
                   json={"model": s.llm_model, "max_tokens": 1500, "system": system, "messages": messages, "tools": tools})
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ rendering helpers
def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render(res: ToolResult) -> list[str]:
    """Turn one ToolResult into factual sentences. Failed tools yield an explicit 'unknown' line."""
    if not res.ok:
        return [f"[{res.tool}] {res.status.upper()}: {res.error}"]
    d, t = res.data or {}, res.tool
    if t == "mlflow_best_run":
        b = d["best"]
        m = ", ".join(f"{k}={_f(v)}" for k, v in b["metrics"].items())
        return [f"Best run by {d['metric']} among {d['considered_runs']} recent runs: {b['run_id']} ({b['name']}), {m}.",
                f"Quality gate tag: {b['tags'].get('quality_gate', 'absent')}; run age {d['age_days']} days."] + \
               [f"Caveat: {c}" for c in d["caveats"]]
    if t == "mlflow_compare_runs":
        return [f"Ranking by {d['metric']}: " + "; ".join(f"{i+1}. {r['name']} ({r['run_id'][:8]}) {_f(r[d['metric']])} gate={r['quality_gate']}"
                                                         for i, r in enumerate(d["ranking"])),
                "Caveat: ranking uses one metric on one hold-out split; a higher score alone does not make a run production-ready."]
    if t == "mlflow_list_runs":
        return [f"{r['run_id'][:8]} {r['name']} f1={r['metrics'].get('f1')} gate={r['tags'].get('quality_gate')}" for r in d["runs"]] or ["No runs found."]
    if t == "mlflow_list_experiments":
        return [f"Experiment {e['experiment_id']}: {e['name']}" for e in d["experiments"]]
    if t == "mlflow_model_registry":
        return [json.dumps(d, default=str)]
    if t == "dvc_run_provenance":
        return [f"Run {d['run_id'][:8]} used dataset {d['dataset_path']} version {d['dataset_version']} (md5 {d['dataset_md5']}), "
                f"Git commit {d['git_commit'][:8]}. Data matched recorded hash at training time: {d['data_matched_recorded_hash']}."]
    if t == "dvc_get_dataset_version":
        return [f"Current dataset {d['path']}: version {d['dataset_version']}, md5 {d['md5']}, recorded at commit {d['dataset_commit'][:8]} "
                f"(HEAD {d['git_head']}); workspace matches recorded hash: {d['workspace_matches_recorded']}."]
    if t == "dvc_list_datasets":
        return [f"Tracked: {x['dvc_file']} (md5 {x['md5']}, {x['size']} bytes)" for x in d["datasets"]] + [f"Remotes: {d['remotes']}"]
    if t == "dvc_check_changes":
        return [("Tracked data HAS changed: " + ", ".join(map(str, d["modified_or_deleted"]))) if d["changed"] else "Tracked data is unchanged vs. the recorded version."]
    if t == "dvc_compare_versions":
        df = d["diff"]
        return [f"Diff {d['rev_a']} -> {d['rev_b']}: modified={[x['path'] for x in df.get('modified', [])]} "
                f"added={[x['path'] for x in df.get('added', [])]} deleted={[x['path'] for x in df.get('deleted', [])]}"]
    if t == "dvc_explain_reproduce":
        return [f"To reproduce {d['rev']} (commit {d['commit'][:8]}):"] + [f"  $ {s}" for s in d["steps"]]
    if t == "airflow_get_dag_runs":
        return [f"{r['dag_run_id']}: {r['state']} (start {r['start_date']}, end {r['end_date']})" for r in d["runs"]] or [f"No runs for DAG {d['dag_id']}."]
    if t == "airflow_get_failed_tasks":
        return [f"Run {d['run_id']}: " + ", ".join(f"{x['task_id']}={x['state']}" for x in d["tasks"]),
                f"Failed tasks: {[x['task_id'] for x in d['failed']] or 'none'}"]
    if t == "airflow_trigger_training_dag":
        return [f"Triggered DAG {d['dag_id']}: run {d['dag_run_id']} state={d['state']}."]
    if t == "airflow_list_dags":
        return [f"DAG {x['dag_id']} (paused={x['paused']})" for x in d["dags"]]
    if t == "airflow_get_task_logs":
        return [f"Log tail for {d['task_id']} (try {d['try_number']}):", d["log_tail"][-600:]]
    if t == "k8s_diagnose_deployment":
        dep = d["deployment"]
        out = [f"Deployment {dep['name']}: desired={dep['desired']} ready={dep['ready']} available={dep['available']}; verdict={d['verdict']}."]
        out += [f"Finding: {x['cause']} | evidence: {x['evidence']} | fix: {x['suggested_fix']}" for x in d["findings"]]
        return out
    if t == "k8s_deployed_version":
        return [f"Deployment {d['deployment']} runs image {d['image']} (tag {d['image_tag']}), version label {d['version_label']}."]
    if t == "k8s_list_workloads":
        return [f"Deployment {x['name']}: {x['ready']}/{x['desired']} ready, images {x['images']}" for x in d["deployments"]] + \
               [f"Pod {p['name']}: {p['phase']} ready={p['ready']} restarts={p['restarts']}" for p in d["pods"]]
    if t == "k8s_unhealthy_pods":
        return [f"Unhealthy: {u['pod']} - {u['problems']}" for u in d["unhealthy"]] or ["No unhealthy pods."]
    if t == "k8s_services":
        return [f"Service {s['name']} ({s['type']}) ports={s['ports']} ready endpoints={s['ready_pods']}/{s['matching_pods']}" for s in d["services"]]
    if t == "k8s_pod_logs":
        return [f"Logs of {d['pod']}:", d["logs"][-800:]]
    if t == "k8s_pod_status":
        return [f"Pod {p['name']}: phase={p['phase']} ready={p['ready']} containers={[(c['name'], c['restart_count'], c['waiting_reason']) for c in p['containers']]}" for p in d["pods"]]
    return [json.dumps(d, default=str)[:600]]


class Orchestrator:
    def __init__(self, llm: Callable | None = None):
        self.llm = llm

    # ------------------------------------------------------------ public
    def handle(self, message: str, confirm: bool = False, request_id: str | None = None) -> ChatResponse:
        rid = request_id or uuid.uuid4().hex[:12]
        use_llm = self.llm is not None or bool(get_settings().anthropic_api_key)
        if use_llm:
            try:
                return self._llm_mode(message, confirm, rid)
            except Exception as e:  # noqa: BLE001
                resp = self._rules_mode(message, confirm, rid)
                resp.uncertainties.append(f"LLM planner failed ({type(e).__name__}); fell back to offline rule-based planner.")
                return resp
        return self._rules_mode(message, confirm, rid)

    # ------------------------------------------------------------ tool use with guards
    def _exec(self, name: str, args: dict, rid: str, message: str, confirm: bool, calls: list) -> ToolResult:
        t = REGISTRY.get(name)
        if t and t.mutating:
            args = {**args, "confirm": confirm}  # confirmation comes ONLY from the user's request, never from the planner
            if name == "airflow_trigger_training_dag" and not TRIGGER_RE.search(message):
                res = ToolResult(tool=name, status="denied", request_id=rid,
                                 error="the user did not explicitly ask to trigger a pipeline; retraining is never started implicitly")
                calls.append(res)
                return res
        res = execute_tool(name, args, request_id=rid)
        calls.append(res)
        return res

    # ------------------------------------------------------------ LLM mode
    def _llm_mode(self, message: str, confirm: bool, rid: str) -> ChatResponse:
        llm = self.llm or anthropic_llm
        tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in REGISTRY.values()]
        msgs: list = [{"role": "user", "content": message}]
        calls: list[ToolResult] = []
        text = ""
        for _ in range(8):
            resp = llm(SYSTEM_PROMPT, msgs, tools)
            blocks = resp["content"]
            msgs.append({"role": "assistant", "content": blocks})
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if not uses:
                break
            results = []
            for u in uses:
                r = self._exec(u["name"], dict(u.get("input") or {}), rid, message, confirm, calls)
                results.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": not r.ok,
                                "content": json.dumps(r.model_dump(), default=str)[:6000]})
            msgs.append({"role": "user", "content": results})
        else:
            text += "\n(stopped: tool-call limit reached)"
        return ChatResponse(request_id=rid, mode="llm", answer=text.strip(), tool_calls=calls)

    # ------------------------------------------------------------ offline rule-based mode
    def _rules_mode(self, message: str, confirm: bool, rid: str) -> ChatResponse:
        m = message.lower()
        calls: list[ToolResult] = []
        run = lambda n, a=None: self._exec(n, a or {}, rid, message, confirm, calls)  # noqa: E731
        s = get_settings()
        metric = next((x for x in ("f1", "accuracy", "precision", "recall", "roc_auc") if x in m), "f1")
        has = lambda *w: any(x in m for x in w)  # noqa: E731

        if (has("best", "recent model", "latest model") and sum([has("dataset", "data version"), has("deploy", "kubernetes", "k8s"),
                                                                 has("retrain", "pipeline", "airflow")]) >= 2) or has("retraining necessary", "need retraining", "should we retrain"):
            return self._assess(message, confirm, rid, calls, metric)

        if re.search(r"\brestart\b", m):
            dep = re.search(r"restart\s+(?:the\s+)?(?:deployment\s+)?([a-z0-9-]{3,})", m)
            name = dep.group(1) if dep and dep.group(1) not in ("deployment", "the", "my") else s.k8s_deployment
            if name in ("api", "service", "app"):
                name = s.k8s_deployment
            return self._finish(rid, "offline-rules", calls, [run("k8s_restart_deployment", {"deployment": name})],
                                note=None if confirm else "Confirmation required: resend with confirm=true to execute.")
        sc = re.search(r"scale\s+(?:\S+\s+)*?(?:to\s+)?(\d+)\s*(?:replicas)?", m)
        if sc and has("scale"):
            return self._finish(rid, "offline-rules", calls, [run("k8s_scale_deployment", {"deployment": s.k8s_deployment, "replicas": int(sc.group(1))})],
                                note=None if confirm else "Confirmation required: resend with confirm=true to execute.")

        if TRIGGER_RE.search(m):
            r = run("airflow_trigger_training_dag")
            return self._finish(rid, "offline-rules", calls, [r], note=None if confirm else
                                "Confirmation required: resend with confirm=true to start the DAG run.")

        if has("dataset", "dvc", "data version", "data changed", "data drift"):
            ref = re.search(r"experiment\s*#?\s*(\w+)", m) or re.search(r"\b([0-9a-f]{32})\b", m)
            revs = re.findall(r"\bdata-v\d+\b|\b[0-9a-f]{7,40}\b", m)
            if ref and has("version", "used", "which"):
                rs = [run("dvc_run_provenance", {"run_ref": ref.group(1)})]
            elif has("changed", "modified", "drift", "change"):
                rs = [run("dvc_check_changes")]
            elif has("compare", "diff") and len(revs) >= 2:
                rs = [run("dvc_compare_versions", {"rev_a": revs[0], "rev_b": revs[1]})]
            elif has("reproduce", "restore", "checkout") and revs:
                rs = [run("dvc_checkout", {"rev": revs[0]})] if has("checkout", "restore") and confirm else [run("dvc_explain_reproduce", {"rev": revs[0]})]
            else:
                rs = [run("dvc_list_datasets"), run("dvc_get_dataset_version")]
            return self._finish(rid, "offline-rules", calls, rs)

        if has("kubernetes", "k8s", "pod", "deployment", "unavailable", "prediction api", "replica", "service", "inference"):
            if has("why", "unavailable", "diagnose", "down", "not working", "fail", "log", "health", "broken", "error"):
                rs = [run("k8s_diagnose_deployment")]
            elif has("version"):
                rs = [run("k8s_deployed_version")]
            elif has("service"):
                rs = [run("k8s_services")]
            elif has("unhealthy", "restart", "crash"):
                rs = [run("k8s_unhealthy_pods")]
            else:
                rs = [run("k8s_list_workloads")]
            return self._finish(rid, "offline-rules", calls, rs)

        if has("airflow", "dag", "pipeline", "workflow"):
            if has("list") and has("dag"):
                rs = [run("airflow_list_dags")]
            elif has("fail", "error", "why"):
                f = run("airflow_get_failed_tasks")
                rs = [f]
                if f.ok and f.data["failed"]:
                    first = next((t for t in f.data["failed"] if t["state"] == "failed"), f.data["failed"][0])
                    rs.append(run("airflow_get_task_logs", {"task_id": first["task_id"], "run_id": f.data["run_id"],
                                                             "try_number": max(first.get("try_number") or 1, 1)}))
            else:
                rs = [run("airflow_get_dag_runs")]
            return self._finish(rid, "offline-rules", calls, rs)

        if has("experiment", "run", "model", "metric", "f1", "accuracy", "compare", "registry", "mlflow"):
            if has("registry", "registered", "version") and has("model"):
                rs = [run("mlflow_model_registry")]
            elif has("compare"):
                n = re.search(r"last\s+(\w+)\s+(?:training\s+)?runs", m)
                words = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
                last_n = int(n.group(1)) if n and n.group(1).isdigit() else words.get(n.group(1), 5) if n else 5
                rs = [run("mlflow_compare_runs", {"last_n": last_n, "metric": metric})]
            elif has("best", "top", "recommend"):
                rs = [run("mlflow_best_run", {"metric": metric})]
            else:
                rs = [run("mlflow_list_runs", {"max_results": 5})]
            return self._finish(rid, "offline-rules", calls, rs)

        return ChatResponse(request_id=rid, mode="offline-rules", tool_calls=[], answer=(
            "I can help with: dataset provenance (DVC), experiment comparison (MLflow), pipeline status/trigger (Airflow), "
            "and deployment diagnosis (Kubernetes). Try: 'Which dataset version was used for experiment 1?'"))

    def _finish(self, rid, mode, calls, results, note=None) -> ChatResponse:
        lines: list[str] = []
        for r in results:
            lines += render(r)
        if note:
            lines.append(note)
        facts = [x for r in results if r.ok for x in render(r)]
        unc = [x for r in results if not r.ok for x in render(r)]
        return ChatResponse(request_id=rid, mode=mode, answer="\n".join(lines), tool_calls=calls, facts=facts, uncertainties=unc)

    # ------------------------------------------------------------ mandatory multi-tool scenario
    def _assess(self, message: str, confirm: bool, rid: str, calls: list, metric: str) -> ChatResponse:
        s = get_settings()
        run = lambda n, a=None: self._exec(n, a or {}, rid, message, confirm, calls)  # noqa: E731
        facts: list[str] = []
        unc: list[str] = []
        recs: list[str] = []
        retrain_evidence: list[str] = []

        def take(res: ToolResult, label: str) -> list[str]:
            lines = render(res)
            (facts if res.ok else unc).extend(lines if res.ok else [f"{label} UNKNOWN - {lines[0]}"])
            return lines

        # 1. MLflow
        best = run("mlflow_best_run", {"metric": metric})
        take(best, "Best model")
        run_id = best.data["best"]["run_id"] if best.ok else None
        if best.ok:
            b = best.data["best"]
            if b["tags"].get("quality_gate") == "failed":
                retrain_evidence.append("the selected run failed its quality gate")
            if b["metrics"].get(metric, 1) < s.f1_gate and metric == "f1":
                retrain_evidence.append(f"best f1 {b['metrics'][metric]} is below policy gate {s.f1_gate}")
            if best.data["age_days"] is not None and best.data["age_days"] > s.model_max_age_days:
                retrain_evidence.append(f"model age {best.data['age_days']}d exceeds freshness policy of {s.model_max_age_days}d")
        # 2. DVC
        if run_id:
            take(run("dvc_run_provenance", {"run_ref": run_id}), "Training dataset version")
        else:
            unc.append("Training dataset version UNKNOWN - no model could be selected.")
        # 3. Kubernetes
        dg = run("k8s_diagnose_deployment")
        take(dg, "Deployment health")
        take(run("k8s_deployed_version"), "Deployed version")
        if dg.ok and dg.data["verdict"] == "unhealthy":
            recs.append("Investigate the deployment findings above before considering any retraining: this is an infrastructure issue, not a model-quality issue.")
        # 4. Airflow
        runs = run("airflow_get_dag_runs", {"limit": 5})
        take(runs, "Recent training workflows")
        if runs.ok and runs.data["runs"] and runs.data["runs"][0]["state"] == "failed":
            ft = run("airflow_get_failed_tasks")
            take(ft, "Failed task details")
            if ft.ok and any(t["task_id"] == "quality_gate" for t in ft.data["failed"]):
                retrain_evidence.append("the latest Airflow run failed at the quality_gate task")
            elif ft.ok:
                recs.append("Latest training run failed before the quality gate: fix " + ", ".join(t["task_id"] for t in ft.data["failed"]) + " (a pipeline problem, not evidence the deployed model is degraded).")
        # 5. Correlate
        unc.append("The deployment does not record which MLflow run its model came from, so 'deployed model == selected best run' cannot be verified from tool data.")
        unc.append("No production-degradation results were supplied; live model quality is unknown.")
        if retrain_evidence:
            recs.append("Retraining is justified by: " + "; ".join(retrain_evidence) + ". It was NOT started automatically; ask explicitly to trigger the pipeline (with confirm=true).")
        else:
            missing = [n for n, r in (("Kubernetes", dg), ("Airflow", runs), ("MLflow", best)) if not r.ok]
            recs.append("Retraining is NOT recommended on the evidence collected (no failed quality gate, no degradation data, within freshness "
                        f"policy of {s.model_max_age_days} days)."
                        + (f" This conclusion is PARTIAL because {', '.join(missing)} evidence was unavailable." if missing else ""))
        recs.append("No retraining was triggered by this request.")
        ans = "KNOWN FACTS\n" + "\n".join(f"- {x}" for x in facts) + "\n\nUNCERTAINTIES\n" + "\n".join(f"- {x}" for x in unc) + \
              "\n\nRECOMMENDED ACTIONS\n" + "\n".join(f"- {x}" for x in recs)
        return ChatResponse(request_id=rid, mode="offline-rules", answer=ans, tool_calls=calls, facts=facts,
                            uncertainties=unc, recommendations=recs)
