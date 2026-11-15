"""Regenerates REPORT.md from docs/evidence/*. Re-run after `scripts/capture_evidence.py` on live infrastructure."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.config import get_settings  # noqa: E402
from agent.tools import REGISTRY  # noqa: E402

E = Path("docs/evidence")
S = get_settings()
rd = lambda n: (E / n).read_text().strip() if (E / n).exists() else "(not captured yet)"  # noqa: E731
rows = "\n".join(f"| `{t.name}` | {t.module} | {'**yes** (confirm=true)' if t.mutating else 'no'} | {t.description} |" for t in REGISTRY.values())
ev = {p.stem: json.loads(p.read_text()) for p in sorted(E.glob("[0-9]_*.json"))}
nulls = re.search(r"(\d+) null", rd("failure_corrupt-dataset.txt"))
gate = re.search(r"f1=([\d.]+)", rd("failure_low-quality-model.txt"))


def conv(k, title, note):
    e = ev[k]
    calls = "\n".join(f"  - `{c['tool']}` -> **{c['status']}** ({c['duration_ms']} ms, req `{c['request_id']}`)" for c in e["tool_calls"])
    live = any(c["status"] == "ok" for c in e["tool_calls"] if c["tool"].startswith(("airflow", "k8s")))
    return (f"### {title}\n*{note if not live else 'LIVE infrastructure evidence'}*\n\n**User:** {e['question']}\n\n"
            f"**Tool calls:**\n{calls}\n\n**Agent answer (verbatim):**\n```\n{e['answer']}\n```\n")


Path("REPORT.md").write_text(f"""# Technical Report - MLOps AI Agent (Assignment 03)

**Use case:** customer-churn prediction (synthetic, offline, no personal data). Model: RandomForest. Quality gate: F1 >= {S.f1_gate}.

## 1. Architecture
See `architecture.md`. FastAPI -> Orchestrator (Claude tool-calling or deterministic offline planner, always labelled in `mode`) -> safety executor -> four tool modules (DVC, MLflow, Airflow 3, Kubernetes) -> evidence-based answer. Training is an Airflow DAG (validate -> preprocess -> train -> evaluate/log -> quality gate -> promote) writing MLflow runs tagged with DVC provenance; serving is a FastAPI container on Kubernetes (2 replicas, ClusterIP, probes, requests/limits, ConfigMap).

## 2. Tool definitions and permissions ({len(REGISTRY)} tools)
| Tool | Module | State-changing | Purpose |
|---|---|---|---|
{rows}

Controls: argument schema validation; {S.tool_timeout_s:.0f}s default timeout; JSON logs (`request_id`, `tool`, `status`, `duration_ms`) with secret redaction; confirmation comes only from the user's request, never from the model; Kubernetes namespace allow-list; Airflow DAG allow-list; DVC/git sub-command allow-list (no shell); retraining is never auto-triggered.

## 3. Evidence status
Evidence files in `docs/evidence/`. Sections marked LIVE below were produced against running Airflow/Kubernetes; otherwise the agent honestly reports the service as unavailable.

DVC checkout round-trip (real):
```
{rd('dvc_checkout_demo.txt')}
```
Automated tests: `docs/evidence/pytest_results.txt` ({rd('pytest_results.txt').splitlines()[-1] if (E / 'pytest_results.txt').exists() else ''}).

## 4. Five tested agent conversations
{conv('1_dataset_provenance', '4.1 Dataset provenance', 'DVC/MLflow executed against real local stores')}
{conv('2_model_selection', '4.2 Model selection', 'MLflow executed; note the failed-gate run is shown with its gate status')}
{conv('3_pipeline_execution', '4.3 Pipeline execution', 'Without confirm=true the agent refuses; re-run with confirm=true against live Airflow')}
{conv('4_deployment_diagnosis', '4.4 Deployment diagnosis', 'No cluster reachable when captured: agent reports UNAVAILABLE and invents nothing')}
{conv('5_multi_tool', '4.5 Multi-tool reasoning', 'Unavailable systems are listed as UNKNOWN and the conclusion is marked PARTIAL')}

## 5. Two failure investigations
**F1 - corrupted dataset** (`scripts/inject_failures.py corrupt-dataset`). Evidence: `{rd('failure_corrupt-dataset.txt')}`. Root cause: nulls injected into a copy of the dataset ({nulls.group(1) if nulls else '?'} cells). `validate_dataset` raises `AirflowFailException` (no retry); every downstream task is skipped, so nothing is trained or promoted. Resolution: `dvc checkout` the tracked version and re-run.

**F2 - low-quality model** (`low-quality-model`). Evidence: `{rd('failure_low-quality-model.txt')}`. MLflow records the run with `quality_gate=failed` (f1={gate.group(1)[:6] if gate else '?'}); `promote_model` refuses unless the tag is `passed` (`test_failed_quality_gate_blocks_promotion`). Root cause: 45% label noise with a 1-tree depth-1 forest. Resolution: restore hyper-parameters; the agent keeps ranking this run last with `gate=failed`.

**F3 - Kubernetes bad ConfigMap** (`bad-model-path`, needs a cluster). Expected: pods Running but not Ready; kubelet `Unhealthy` events (503 on `/ready`); logs `model file not found ...`; Service with 0 ready endpoints. Diagnosis logic is covered by `test_failure_demo_k8s_bad_model_path_is_diagnosed`.

## 6. Screenshots
Place in `screenshots/`: `dvc_status.png`, `mlflow_runs.png`, `airflow_dag_graph.png`, `kubectl_pods.png`.

## 7. Known limitations and improvements
* "Experiment N" means MLflow experiment id N (latest run).
* The deployment does not record the MLflow run id of its model, so "deployed model == best model" is reported as an uncertainty. Improvement: put `MLFLOW_RUN_ID` in the ConfigMap.
* Single hold-out split, single metric; add cross-validation and drift monitoring.
* Offline planner is keyword-based; LLM mode is the general planner.
* Agent API has only an optional shared key; production needs real authN/Z.
* DVC remote is local; use S3/GCS/Azure for teams.
""")
print("REPORT.md written")
