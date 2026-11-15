import ast
from pathlib import Path

import httpx

from agent.tools import airflow_tools, execute_tool
from tests.conftest import airflow_transport

RUNS = {"dag_runs": [{"dag_run_id": "r1", "state": "failed", "run_type": "manual", "logical_date": None,
                      "start_date": "2026-11-10T10:00:00Z", "end_date": "2026-11-10T10:02:00Z"}]}
TIS = {"task_instances": [
    {"task_id": "validate_dataset", "state": "success", "try_number": 1, "max_tries": 2},
    {"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 2},
    {"task_id": "promote_model", "state": "upstream_failed", "try_number": 0, "max_tries": 2}]}


def test_dag_runs_use_jwt_bearer_auth(monkeypatch):
    calls = []
    monkeypatch.setattr(airflow_tools, "_transport", airflow_transport({("GET", "/api/v2/dags/training_pipeline/dagRuns"): RUNS}, calls))
    r = execute_tool("airflow_get_dag_runs")
    assert r.ok and r.data["runs"][0]["state"] == "failed"
    assert calls[-1][2] == "Bearer jwt-test" and calls[0][1] == "/auth/token"


def test_failed_tasks_and_logs_identified(monkeypatch):
    routes = {("GET", "/api/v2/dags/training_pipeline/dagRuns"): RUNS,
              ("GET", "/api/v2/dags/training_pipeline/dagRuns/r1/taskInstances"): TIS,
              ("GET", "/api/v2/dags/training_pipeline/dagRuns/r1/taskInstances/quality_gate/logs/1"):
                  {"content": [{"event": "quality gate FAILED: f1=0.17 < 0.70"}]}}
    monkeypatch.setattr(airflow_tools, "_transport", airflow_transport(routes))
    f = execute_tool("airflow_get_failed_tasks")
    assert [t["task_id"] for t in f.data["failed"]] == ["quality_gate", "promote_model"]
    assert "f1=0.17" in execute_tool("airflow_get_task_logs", {"task_id": "quality_gate"}).data["log_tail"]


def test_trigger_requires_confirmation_and_approved_dag(monkeypatch):
    calls = []
    monkeypatch.setattr(airflow_tools, "_transport", airflow_transport(
        {("POST", "/api/v2/dags/training_pipeline/dagRuns"): {"dag_run_id": "new", "state": "queued"}}, calls))
    assert execute_tool("airflow_trigger_training_dag", {}).status == "denied"            # no confirm
    assert execute_tool("airflow_trigger_training_dag", {"dag_id": "other_dag", "confirm": True}).status == "denied"  # not approved
    assert not any(c[0] == "POST" for c in calls)
    ok = execute_tool("airflow_trigger_training_dag", {"confirm": True})
    assert ok.ok and ok.data["dag_run_id"] == "new"


def test_airflow_down_reports_unavailable(monkeypatch):
    def refuse(req):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(airflow_tools, "_transport", httpx.MockTransport(refuse))
    assert execute_tool("airflow_list_dags").status == "unavailable"


def test_training_dag_has_required_stages_order_retries_and_gate():
    src = Path("dags/training_pipeline.py").read_text()
    tree = ast.parse(src)
    fns = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for t in ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate", "promote_model"):
        assert t in fns
    order = ["v = validate_dataset()", "s = preprocess_data(v)", "m = train_model(s)",
             "e = evaluate_and_log(m, s, v)", "g = quality_gate(e)", "promote_model(g)"]
    idx = [src.index(x) for x in order]
    assert idx == sorted(idx)                       # data-flow chain enforces task dependencies
    assert '"retries": 2' in src and "AirflowFailException" in src
