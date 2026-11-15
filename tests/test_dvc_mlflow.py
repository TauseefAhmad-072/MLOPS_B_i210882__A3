import pytest

from agent.tools import dvc_tools, execute_tool
from agent.tools.base import PermissionDenied


def test_dvc_command_allowlist_blocks_unapproved_operations():
    for binary, args in [("dvc", ["push"]), ("dvc", ["remote", "add", "x", "y"]), ("git", ["push"]), ("rm", ["-rf", "/"])]:
        with pytest.raises(PermissionDenied):
            dvc_tools._run(binary, args)


def test_dvc_rejects_option_injection_in_revisions():
    r = execute_tool("dvc_explain_reproduce", {"rev": "--output=/etc/passwd"})
    assert r.status == "denied"


def test_dvc_check_changes_parses_status(monkeypatch):
    monkeypatch.setattr(dvc_tools, "_run", lambda b, a, timeout=60: '{"uncommitted": {"modified": ["data/x.csv"]}}')
    r = execute_tool("dvc_check_changes")
    assert r.ok and r.data["changed"] is True and r.data["modified_or_deleted"] == ["data/x.csv"]


def test_dataset_provenance_comes_from_recorded_mlflow_tags(mlflow_runs):
    r = execute_tool("dvc_run_provenance", {"run_ref": mlflow_runs["ids"][0]})
    assert r.ok and r.data["dataset_version"] == "data-v2" and r.data["git_commit"] == "a4c92e1ffff"


def test_missing_provenance_is_an_error_not_a_guess(mlflow_runs):
    r = execute_tool("dvc_run_provenance", {"run_ref": mlflow_runs["ids"][2]})  # run-c has no dvc tags
    assert r.status == "error" and "no recorded DVC provenance" in r.error


def test_best_run_selects_max_metric_and_flags_caveats(mlflow_runs):
    r = execute_tool("mlflow_best_run", {"metric": "f1"})
    best = r.data["best"]
    assert best["name"] == "run-b" and best["metrics"]["f1"] == 0.97
    assert any("quality gate" in c for c in r.data["caveats"])  # highest F1 but gate failed -> flagged
    assert "production-ready" in r.data["note"]


def test_compare_runs_ranks_by_metric(mlflow_runs):
    r = execute_tool("mlflow_compare_runs", {"last_n": 3, "metric": "f1"})
    assert [x["name"] for x in r.data["ranking"]] == ["run-b", "run-c", "run-a"]


def test_best_run_errors_for_unknown_metric(mlflow_runs):
    assert execute_tool("mlflow_best_run", {"metric": "nonexistent"}).status == "error"


def test_mlflow_server_unreachable_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    assert execute_tool("mlflow_list_runs").status in ("unavailable", "error")
