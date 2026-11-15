from agent.tools import execute_tool
from tests.conftest import FakeApps, FakeCore, ev, make_pod


def test_unauthorized_modifications_are_rejected(k8s):
    core, apps = k8s(FakeCore([make_pod("p1")]), FakeApps(ready=2))
    assert execute_tool("k8s_restart_deployment", {"deployment": "ml-inference-api"}).status == "denied"            # no confirm
    assert execute_tool("k8s_restart_deployment", {"deployment": "x", "namespace": "kube-system", "confirm": True}).status == "denied"  # wrong ns
    assert execute_tool("k8s_scale_deployment", {"deployment": "x", "replicas": 50, "confirm": True}).status == "denied"  # out of range
    assert execute_tool("k8s_list_workloads", {"namespace": "default"}).status == "denied"                         # reads restricted too
    assert apps.patched == []


def test_confirmed_restart_in_approved_namespace_succeeds(k8s):
    _, apps = k8s(FakeCore([]), FakeApps())
    r = execute_tool("k8s_restart_deployment", {"deployment": "ml-inference-api", "confirm": True})
    assert r.ok and apps.patched[0][0] == "restart" and apps.patched[0][2] == "mlops-agent"


def test_healthy_deployment_verdict(k8s):
    k8s(FakeCore([make_pod("a"), make_pod("b")]), FakeApps(ready=2))
    r = execute_tool("k8s_diagnose_deployment")
    assert r.data["verdict"] == "healthy" and r.data["findings"] == []
    assert execute_tool("k8s_deployed_version").data["image_tag"] == "1.0.0"


def test_diagnoses_image_pull_failure(k8s):
    k8s(FakeCore([make_pod("a", "Pending", False, waiting="ImagePullBackOff"), make_pod("b", "Pending", False, waiting="ErrImagePull")],
                 [ev("a", "Failed", "pull access denied")]), FakeApps(ready=0, image="mlops-inference:9.9.9"))
    r = execute_tool("k8s_diagnose_deployment")
    causes = [f["cause"] for f in r.data["findings"]]
    assert "Image cannot be pulled" in causes and r.data["verdict"] == "unhealthy"


def test_diagnoses_readiness_failure_from_logs_and_events(k8s):
    logs = {"a": '{"level":"ERROR","msg":"model file not found at models/does-not-exist.joblib"}'}
    k8s(FakeCore([make_pod("a", ready=False), make_pod("b", ready=False)],
                 [ev("a", "Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")], logs), FakeApps(ready=0))
    r = execute_tool("k8s_diagnose_deployment")
    text = str(r.data["findings"])
    assert "Readiness probe failing" in text and "model file not found" in text
    assert any("no ready endpoints" in f["cause"] for f in r.data["findings"])


def test_crashloop_and_oom_detected(k8s):
    k8s(FakeCore([make_pod("a", ready=False, waiting="CrashLoopBackOff", restarts=7, last_reason="OOMKilled", exit_code=137)]), FakeApps(ready=0))
    r = execute_tool("k8s_unhealthy_pods")
    assert r.data["unhealthy"][0]["pod"] == "a"
    causes = [f["cause"] for f in execute_tool("k8s_diagnose_deployment").data["findings"]]
    assert "Container crash-looping" in causes and "Container OOMKilled" in causes


def test_cluster_unreachable_is_unavailable_not_fabricated():
    r = execute_tool("k8s_list_workloads")   # no kubeconfig in test env / CI
    assert r.status == "unavailable" and r.data is None
