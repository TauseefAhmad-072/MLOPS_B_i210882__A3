"""Airflow 3 REST API (/api/v2) tools using JWT auth. Only approved DAGs can be triggered."""
from __future__ import annotations

import httpx

from agent.config import get_settings
from agent.logging_utils import redact_text
from agent.tools.base import PermissionDenied, ServiceUnavailable, ToolError, tool

_transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport here
_token_cache: dict[str, str] = {}


def _client() -> httpx.Client:
    return httpx.Client(base_url=get_settings().airflow_base_url, timeout=10, transport=_transport)


def _token(c: httpx.Client) -> str:
    s = get_settings()
    if s.airflow_token:
        return s.airflow_token
    if "t" in _token_cache:
        return _token_cache["t"]
    if not s.airflow_password:
        raise ToolError("AIRFLOW_PASSWORD (or AIRFLOW_TOKEN) is not configured")
    r = c.post("/auth/token", json={"username": s.airflow_username, "password": s.airflow_password})
    if r.status_code >= 400:
        raise ToolError(f"Airflow authentication failed (HTTP {r.status_code})")
    _token_cache["t"] = r.json()["access_token"]
    return _token_cache["t"]


def _api(method: str, path: str, **kw):
    try:
        with _client() as c:
            h = {"Authorization": f"Bearer {_token(c)}", "Accept": "application/json"}
            r = c.request(method, f"/api/v2{path}", headers=h, **kw)
    except httpx.TransportError as e:
        raise ServiceUnavailable(f"Airflow API unreachable at {get_settings().airflow_base_url}: {type(e).__name__}") from e
    if r.status_code == 401:
        _token_cache.clear()
        raise ToolError("Airflow rejected the credentials (HTTP 401)")
    if r.status_code == 404:
        raise ToolError(f"not found: {path}")
    if r.status_code >= 400:
        raise ToolError(f"Airflow API error HTTP {r.status_code}: {redact_text(r.text[:300])}")
    return r.json()


def _run_view(r: dict) -> dict:
    return {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date")}


@tool("airflow_list_dags", "List available Airflow DAGs with paused/active state.", "airflow")
def list_dags() -> dict:
    d = _api("GET", "/dags", params={"limit": 50})
    return {"dags": [{"dag_id": x["dag_id"], "paused": x.get("is_paused"), "stale": x.get("is_stale")}
                     for x in d.get("dags", [])]}


@tool("airflow_get_dag_runs", "Recent DAG runs and their state (newest first).", "airflow",
      {"dag_id": {"type": "string"}, "limit": {"type": "integer"}})
def get_dag_runs(dag_id: str | None = None, limit: int = 5) -> dict:
    dag_id = dag_id or get_settings().training_dag_id
    d = _api("GET", f"/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-start_date"})
    return {"dag_id": dag_id, "runs": [_run_view(r) for r in d.get("dag_runs", [])]}


@tool("airflow_trigger_training_dag", "Trigger the approved training DAG. Starts a real pipeline run.", "airflow",
      {"dag_id": {"type": "string"}, "conf": {"type": "object"}, "confirm": {"type": "boolean"}},
      mutating=True)
def trigger_dag(dag_id: str | None = None, conf: dict | None = None, confirm: bool = False) -> dict:
    s = get_settings()
    dag_id = dag_id or s.training_dag_id
    if dag_id not in s.approved_dags:
        raise PermissionDenied(f"DAG '{dag_id}' is not on the approved list {list(s.approved_dags)}")
    r = _api("POST", f"/dags/{dag_id}/dagRuns", json={"logical_date": None, "conf": conf or {}})
    return {"dag_id": dag_id, **_run_view(r)}


def _latest_run_id(dag_id: str) -> str:
    runs = get_dag_runs(dag_id, 1)["runs"]
    if not runs:
        raise ToolError(f"no runs found for DAG '{dag_id}'")
    return runs[0]["dag_run_id"]


@tool("airflow_get_failed_tasks", "Identify failed/upstream-failed tasks of a DAG run (default: latest run).",
      "airflow", {"dag_id": {"type": "string"}, "run_id": {"type": "string"}})
def failed_tasks(dag_id: str | None = None, run_id: str | None = None) -> dict:
    dag_id = dag_id or get_settings().training_dag_id
    run_id = run_id or _latest_run_id(dag_id)
    d = _api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
    tis = [{"task_id": t["task_id"], "state": t["state"], "try_number": t.get("try_number"),
            "max_tries": t.get("max_tries"), "start_date": t.get("start_date"), "end_date": t.get("end_date")}
           for t in d.get("task_instances", [])]
    return {"dag_id": dag_id, "run_id": run_id, "tasks": tis,
            "failed": [t for t in tis if t["state"] in ("failed", "upstream_failed")]}


@tool("airflow_get_task_logs", "Fetch log tail of one task instance for failure diagnostics.", "airflow",
      {"task_id": {"type": "string"}, "dag_id": {"type": "string"}, "run_id": {"type": "string"},
       "try_number": {"type": "integer"}}, ("task_id",))
def task_logs(task_id: str, dag_id: str | None = None, run_id: str | None = None, try_number: int = 1) -> dict:
    dag_id = dag_id or get_settings().training_dag_id
    run_id = run_id or _latest_run_id(dag_id)
    d = _api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}")
    content = d.get("content", d) if isinstance(d, dict) else d
    lines = [x.get("event", str(x)) if isinstance(x, dict) else str(x) for x in content] \
        if isinstance(content, list) else [str(content)]
    text = redact_text("\n".join(lines))
    return {"dag_id": dag_id, "run_id": run_id, "task_id": task_id, "try_number": try_number, "log_tail": text[-3000:]}
