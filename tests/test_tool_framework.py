import logging
import time

from agent.tools import REGISTRY, execute_tool
from agent.tools.base import Tool, ToolError


def test_each_module_exposes_at_least_three_tools():
    for module in ("dvc", "mlflow", "airflow", "kubernetes"):
        assert sum(1 for t in REGISTRY.values() if t.module == module) >= 3, module


def test_unknown_tool_and_invalid_arguments_are_rejected():
    assert execute_tool("nope").status == "invalid"
    assert "missing required" in execute_tool("mlflow_get_run", {}).error
    assert "unknown argument" in execute_tool("mlflow_list_runs", {"bogus": 1}).error
    assert "must be integer" in execute_tool("mlflow_list_runs", {"max_results": "ten"}).error


def test_tool_timeout_is_reported(monkeypatch):
    monkeypatch.setitem(REGISTRY, "slow", Tool("slow", "x", "test", {}, (), lambda: time.sleep(2) or {}))
    r = execute_tool("slow", timeout=0.1)
    assert r.status == "timeout" and "exceeded" in r.error


def test_structured_logs_have_required_fields_and_no_secrets(monkeypatch, caplog):
    def boom():
        raise ToolError("auth failed with token=SUPERSECRET123 and Bearer abc.def.ghi")
    monkeypatch.setitem(REGISTRY, "boom", Tool("boom", "x", "test", {}, (), boom))
    with caplog.at_level(logging.INFO, logger="mlops-agent"):
        r = execute_tool("boom", request_id="req-42")
    rec = next(x for x in caplog.records if getattr(x, "fields", {}).get("tool") == "boom")
    f = rec.fields
    assert f["request_id"] == "req-42" and f["status"] == "error" and "duration_ms" in f
    assert "SUPERSECRET123" not in str(f) and "abc.def.ghi" not in str(f)
    assert "SUPERSECRET123" not in (r.error or "")
