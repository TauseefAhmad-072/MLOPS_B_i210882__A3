"""MLflow experiment-intelligence tools (MlflowClient only - never scrapes the web UI)."""
from __future__ import annotations

import time

from agent.config import get_settings
from agent.tools.base import ServiceUnavailable, ToolError, tool

SHOWN_TAGS = ("quality_gate", "dvc.dataset_version", "dvc.dataset_md5", "dvc.git_commit", "dvc.dataset_path",
              "dvc.data_matches_recorded", "airflow.dag_run_id", "mlflow.runName")


def get_client():
    from mlflow import MlflowClient
    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if any(x in msg for x in ("Connection", "connect", "Max retries", "Timeout", "refused")):
            raise ServiceUnavailable(f"MLflow tracking server unreachable: {msg[:200]}") from e
        raise ToolError(f"MLflow error: {msg[:300]}") from e


def _exp_ids(client, experiment: str | None) -> list[str]:
    if experiment is None:
        exp = _call(client.get_experiment_by_name, get_settings().mlflow_experiment)
        if exp:
            return [exp.experiment_id]
        return [e.experiment_id for e in _call(client.search_experiments)]
    if experiment.isdigit():
        return [experiment]
    exp = _call(client.get_experiment_by_name, experiment)
    if not exp:
        raise ToolError(f"experiment '{experiment}' not found")
    return [exp.experiment_id]


def run_summary(run) -> dict:
    tags = {k: v for k, v in run.data.tags.items() if k in SHOWN_TAGS}
    return {"run_id": run.info.run_id, "name": run.info.run_name, "experiment_id": run.info.experiment_id,
            "status": run.info.status, "start_time": run.info.start_time,
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
            "params": dict(run.data.params), "tags": tags}


@tool("mlflow_list_experiments", "List MLflow experiments.", "mlflow",
      {"max_results": {"type": "integer"}})
def list_experiments(max_results: int = 20) -> dict:
    exps = _call(get_client().search_experiments, max_results=max_results)
    return {"experiments": [{"experiment_id": e.experiment_id, "name": e.name} for e in exps]}


@tool("mlflow_list_runs", "List the most recent training runs (newest first) with metrics, params and tags.",
      "mlflow", {"experiment": {"type": "string"}, "max_results": {"type": "integer"}})
def list_runs(experiment: str | None = None, max_results: int = 10) -> dict:
    c = get_client()
    runs = _call(c.search_runs, _exp_ids(c, experiment), order_by=["attributes.start_time DESC"],
                 max_results=max_results)
    return {"runs": [run_summary(r) for r in runs]}


@tool("mlflow_get_run", "Get metrics, parameters, tags and artifacts of one run.", "mlflow",
      {"run_id": {"type": "string"}}, ("run_id",))
def get_run(run_id: str) -> dict:
    c = get_client()
    run = _call(c.get_run, run_id)
    arts = _call(c.list_artifacts, run_id)
    return {**run_summary(run), "artifacts": [a.path for a in arts]}


def _caveats(run: dict, metric: str, higher_is_better: bool) -> list[str]:
    s, cav = get_settings(), []
    if run["status"] != "FINISHED":
        cav.append(f"run status is {run['status']}, not FINISHED")
    gate = run["tags"].get("quality_gate")
    if gate != "passed":
        cav.append(f"quality gate tag is '{gate or 'absent'}' (not 'passed')")
    if "dvc.dataset_md5" not in run["tags"] or run["tags"].get("dvc.dataset_md5") == "unknown":
        cav.append("no dataset provenance recorded")
    if run["tags"].get("dvc.data_matches_recorded") == "false":
        cav.append("training data did not match the DVC-recorded hash (uncommitted data change)")
    n_test = run["params"].get("n_test")
    if n_test and int(n_test) < 100:
        cav.append(f"small evaluation set (n_test={n_test})")
    if metric == "f1" and run["metrics"].get("f1", 0) < s.f1_gate:
        cav.append(f"f1 below policy gate {s.f1_gate}")
    cav.append("selected on a single metric on one hold-out split; validate on production-like data before promotion")
    return cav


@tool("mlflow_best_run", "Find the best run by a metric among the most recent runs, with readiness caveats.",
      "mlflow", {"metric": {"type": "string"}, "experiment": {"type": "string"},
                 "last_n": {"type": "integer"}, "higher_is_better": {"type": "boolean"}})
def best_run(metric: str = "f1", experiment: str | None = None, last_n: int = 20,
             higher_is_better: bool = True) -> dict:
    runs = list_runs(experiment, last_n)["runs"]
    scored = [r for r in runs if metric in r["metrics"]]
    if not scored:
        raise ToolError(f"no runs with metric '{metric}' found among {len(runs)} recent runs")
    best = (max if higher_is_better else min)(scored, key=lambda r: r["metrics"][metric])
    age_days = round((time.time() * 1000 - best["start_time"]) / 86_400_000, 2) if best["start_time"] else None
    return {"metric": metric, "considered_runs": len(runs), "best": best, "age_days": age_days,
            "caveats": _caveats(best, metric, higher_is_better),
            "note": "best on this metric does not by itself make a model production-ready"}


@tool("mlflow_compare_runs", "Compare runs side by side on all metrics; give run_ids or last_n recent runs.",
      "mlflow", {"run_ids": {"type": "array"}, "experiment": {"type": "string"},
                 "last_n": {"type": "integer"}, "metric": {"type": "string"}})
def compare_runs(run_ids: list | None = None, experiment: str | None = None, last_n: int = 5,
                 metric: str = "f1") -> dict:
    c = get_client()
    runs = [run_summary(_call(c.get_run, r)) for r in run_ids] if run_ids else list_runs(experiment, last_n)["runs"]
    if len(runs) < 2:
        raise ToolError("need at least two runs to compare")
    ranked = sorted((r for r in runs if metric in r["metrics"]), key=lambda r: -r["metrics"][metric])
    return {"metric": metric, "ranking": [{"run_id": r["run_id"], "name": r["name"], metric: r["metrics"][metric],
                                           "quality_gate": r["tags"].get("quality_gate")} for r in ranked],
            "runs": runs}


@tool("mlflow_model_registry", "Registered models and their versions (if a registry is available).", "mlflow",
      {"name": {"type": "string"}})
def model_registry(name: str | None = None) -> dict:
    c = get_client()
    if name:
        vs = _call(c.search_model_versions, f"name='{name}'")
        return {"versions": [{"name": v.name, "version": v.version, "run_id": v.run_id, "status": v.status} for v in vs]}
    models = _call(c.search_registered_models)
    return {"models": [{"name": m.name, "latest": [{"version": v.version, "run_id": v.run_id}
                                                    for v in (m.latest_versions or [])]} for m in models]}
