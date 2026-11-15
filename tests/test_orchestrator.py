import pytest
from fastapi.testclient import TestClient

import agent.orchestrator as orch
from agent.main import app
from agent.schemas import ToolResult
from agent.tools import REGISTRY, airflow_tools
from tests.conftest import FakeApps, FakeCore, airflow_transport, make_pod

O = orch.Orchestrator


def fake_exec(table, seen):
    def run(name, args=None, request_id=None, timeout=None):
        seen.append((name, args or {}))
        r = table.get(name)
        if r is None:
            return ToolResult(tool=name, status="unavailable", error=f"{name} backend down", request_id=request_id or "")
        return ToolResult(tool=name, status="ok", data=r, request_id=request_id or "")
    return run


def test_tool_selection_for_dataset_question(monkeypatch):
    seen = []
    monkeypatch.setattr(orch, "execute_tool", fake_exec({"dvc_run_provenance": {
        "run_id": "r" * 32, "dataset_path": "d", "dataset_md5": "m", "dataset_version": "v2.1", "git_commit": "a4c92e1xxxx",
        "data_matched_recorded_hash": "true"}}, seen))
    r = O().handle("Which dataset version was used for experiment 25?")
    assert seen == [("dvc_run_provenance", {"run_ref": "25"})]
    assert "v2.1" in r.answer and "a4c92e1" in r.answer


@pytest.mark.parametrize("q,tool", [
    ("Compare the last five training runs and recommend the best candidate based on F1 score", "mlflow_compare_runs"),
    ("Why is my prediction API unavailable?", "k8s_diagnose_deployment"),
    ("what is the status of the airflow pipeline runs", "airflow_get_dag_runs"),
    ("has the tracked dataset changed?", "dvc_check_changes"),
])
def test_tool_selection_matrix(monkeypatch, q, tool):
    seen = []
    monkeypatch.setattr(orch, "execute_tool", fake_exec({}, seen))
    O().handle(q)
    assert seen[0][0] == tool


def test_agent_does_not_invent_results_when_all_tools_unavailable(monkeypatch):
    monkeypatch.setattr(orch, "execute_tool", fake_exec({}, []))
    r = O().handle("Find the best recent model, identify the dataset version used to train it, check the deployment status, "
                   "and explain whether retraining is necessary.")
    assert "UNKNOWN" in r.answer and "UNAVAILABLE" in r.answer
    assert r.facts == []                                    # nothing may be stated as fact
    assert "f1=" not in r.answer and "version data-v" not in r.answer
    assert "PARTIAL" in r.answer                            # conclusion flagged as partial
    assert "No retraining was triggered" in r.answer


def _good_table(gate="passed", f1=0.97, age=1.0, dag_state="success"):
    return {
        "mlflow_best_run": {"metric": "f1", "considered_runs": 3, "age_days": age, "caveats": ["small split"],
                            "best": {"run_id": "b" * 32, "name": "rf", "metrics": {"f1": f1}, "tags": {"quality_gate": gate}}},
        "dvc_run_provenance": {"run_id": "b" * 32, "dataset_path": "d", "dataset_md5": "m", "dataset_version": "data-v2",
                               "git_commit": "6dac4c4cdead", "data_matched_recorded_hash": "true"},
        "k8s_diagnose_deployment": {"deployment": {"name": "ml", "desired": 2, "ready": 2, "available": 2}, "findings": [], "verdict": "healthy"},
        "k8s_deployed_version": {"deployment": "ml", "image": "img:1", "image_tag": "1", "version_label": "1.0.0"},
        "airflow_get_dag_runs": {"dag_id": "t", "runs": [{"dag_run_id": "r1", "state": dag_state, "start_date": "s", "end_date": "e"}]},
        "airflow_get_failed_tasks": {"dag_id": "t", "run_id": "r1", "tasks": [{"task_id": "quality_gate", "state": "failed"}],
                                     "failed": [{"task_id": "quality_gate", "state": "failed"}]}}


MULTI = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."


def test_multi_tool_scenario_touches_all_four_systems_and_never_triggers(monkeypatch):
    seen = []
    monkeypatch.setattr(orch, "execute_tool", fake_exec(_good_table(), seen))
    r = O().handle(MULTI)
    names = [n for n, _ in seen]
    assert {"mlflow_best_run", "dvc_run_provenance", "k8s_diagnose_deployment", "airflow_get_dag_runs"} <= set(names)
    assert "airflow_trigger_training_dag" not in names
    assert "NOT recommended" in r.answer and r.facts and r.uncertainties and r.recommendations


def test_retraining_recommended_only_with_evidence(monkeypatch):
    monkeypatch.setattr(orch, "execute_tool", fake_exec(_good_table(dag_state="failed"), []))
    r = O().handle(MULTI)
    assert "justified by" in r.answer and "quality_gate" in r.answer and "NOT started automatically" in r.answer
    monkeypatch.setattr(orch, "execute_tool", fake_exec(_good_table(age=90), []))
    assert "freshness policy" in O().handle(MULTI).answer


def test_state_changing_chat_requests_need_confirmation(k8s):
    _, apps = k8s(FakeCore([]), FakeApps())
    r = O().handle("please restart the deployment")
    assert r.tool_calls[0].status == "denied" and apps.patched == []
    r = O().handle("please restart the deployment", confirm=True)
    assert r.tool_calls[0].status == "ok" and apps.patched


def test_llm_cannot_trigger_retraining_unless_user_asked(monkeypatch):
    monkeypatch.setattr(airflow_tools, "_transport", airflow_transport({}))
    script = iter([{"content": [{"type": "tool_use", "id": "t1", "name": "airflow_trigger_training_dag", "input": {"confirm": True}}]},
                   {"content": [{"type": "text", "text": "done"}]}])
    r = O(llm=lambda s, m, t: next(script)).handle("Is the model stale? Assess it.", confirm=True)
    assert r.mode == "llm" and r.tool_calls[0].status == "denied" and "did not explicitly ask" in r.tool_calls[0].error


def test_llm_tool_loop_executes_tools_and_returns_final_text(mlflow_runs):
    script = iter([{"content": [{"type": "tool_use", "id": "t1", "name": "mlflow_best_run", "input": {"metric": "f1"}}]},
                   {"content": [{"type": "text", "text": "Best is run-b but its gate failed."}]}])
    seen_tools = []

    def llm(system, msgs, tools):
        seen_tools.append(len(tools))
        return next(script)
    r = O(llm=llm).handle("which run is best?")
    assert r.answer.startswith("Best is run-b") and r.tool_calls[0].ok and seen_tools[0] == len(REGISTRY)


def test_llm_failure_falls_back_to_labelled_offline_mode(monkeypatch):
    monkeypatch.setattr(orch, "execute_tool", fake_exec({}, []))

    def broken(*a):
        raise RuntimeError("api down")
    r = O(llm=broken).handle("list dags in airflow")
    assert r.mode == "offline-rules" and any("fell back" in u for u in r.uncertainties)


def test_chat_endpoint_and_api_key(monkeypatch):
    monkeypatch.setattr(orch, "execute_tool", fake_exec({}, []))
    c = TestClient(app)
    assert c.get("/health").json()["tools"] >= 12
    assert c.post("/chat", json={"message": ""}).status_code == 422
    monkeypatch.setenv("AGENT_API_KEY", "k")
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert c.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "k"}).status_code == 200


# ---------------- failure-injection diagnosis demos (mocked evidence, real analysis code)
def test_failure_demo_k8s_bad_model_path_is_diagnosed(k8s):
    logs = {"a": '{"level":"ERROR","msg":"model file not found at models/does-not-exist.joblib"}'}
    from tests.conftest import ev
    k8s(FakeCore([make_pod("a", ready=False), make_pod("b", ready=False)],
                 [ev("a", "Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")], logs), FakeApps(ready=0))
    r = O().handle("Why is my prediction API unavailable?")
    assert "unhealthy" in r.answer and "model file not found" in r.answer and "Readiness probe failing" in r.answer


def test_failure_demo_airflow_quality_gate_failure_is_diagnosed(monkeypatch):
    routes = {("GET", "/api/v2/dags/training_pipeline/dagRuns"): {"dag_runs": [{"dag_run_id": "r1", "state": "failed"}]},
              ("GET", "/api/v2/dags/training_pipeline/dagRuns/r1/taskInstances"): {"task_instances": [
                  {"task_id": "validate_dataset", "state": "success", "try_number": 1},
                  {"task_id": "quality_gate", "state": "failed", "try_number": 1},
                  {"task_id": "promote_model", "state": "upstream_failed", "try_number": 0}]},
              ("GET", "/api/v2/dags/training_pipeline/dagRuns/r1/taskInstances/quality_gate/logs/1"):
                  {"content": [{"event": "quality gate FAILED: f1=0.17 < 0.70"}]}}
    monkeypatch.setattr(airflow_tools, "_transport", airflow_transport(routes))
    r = O().handle("Why did the training pipeline fail?")
    assert "quality_gate" in r.answer and "f1=0.17" in r.answer and "promote_model=upstream_failed" in r.answer
