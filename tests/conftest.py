import json
import time

import httpx
import pytest

from agent.config import get_settings  # noqa: F401
from agent.tools import airflow_tools, kubernetes_tools


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://airflow.test")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "pw-for-tests")
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("K8S_NAMESPACE", "mlops-agent")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # force offline planner unless a test injects an llm
    monkeypatch.setattr(airflow_tools, "_transport", None)
    airflow_tools._token_cache.clear()
    monkeypatch.setattr(kubernetes_tools, "_cached", None)


@pytest.fixture
def mlflow_runs(env):
    """Three real runs in a throw-away MLflow store."""
    from mlflow import MlflowClient
    c = MlflowClient()
    exp = c.create_experiment("churn-classifier")
    ids = []
    for name, f1, gate, prov in [("run-a", 0.95, "passed", True), ("run-b", 0.97, "failed", True), ("run-c", 0.96, "passed", False)]:
        r = c.create_run(exp, run_name=name)
        c.log_metric(r.info.run_id, "f1", f1)
        c.log_metric(r.info.run_id, "accuracy", f1 - 0.01)
        c.log_param(r.info.run_id, "n_test", 143)
        c.set_tag(r.info.run_id, "quality_gate", gate)
        if prov:
            for k, v in {"dvc.dataset_md5": "abc123", "dvc.dataset_version": "data-v2", "dvc.git_commit": "a4c92e1ffff",
                         "dvc.dataset_path": "data/customer_churn.csv", "dvc.data_matches_recorded": "true"}.items():
                c.set_tag(r.info.run_id, k, v)
        c.set_terminated(r.info.run_id, "FINISHED")
        ids.append(r.info.run_id)
        time.sleep(0.01)
    return {"exp": exp, "ids": ids}


# ---------------- Airflow mock transport
def airflow_transport(routes: dict, calls: list | None = None):
    def handler(req: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append((req.method, req.url.path, req.headers.get("authorization"), req.content))
        if req.url.path == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt-test"})
        for (m, path), body in routes.items():
            if req.method == m and req.url.path == path:
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={"detail": "nf"})
    return httpx.MockTransport(handler)


# ---------------- Kubernetes fakes (camelCase JSON dicts == real API shape)
def make_pod(name, phase="Running", ready=True, waiting=None, restarts=0, last_reason=None, exit_code=None,
             labels=None, unschedulable=None):
    cs = {"name": "api", "ready": ready, "restartCount": restarts, "image": "mlops-inference:1.0.0",
          "state": {"waiting": {"reason": waiting, "message": f"{waiting} detail"}} if waiting else {"running": {}},
          "lastState": {"terminated": {"reason": last_reason, "exitCode": exit_code}} if last_reason else {}}
    conds = [{"type": "Ready", "status": "True" if ready else "False"}]
    if unschedulable:
        conds.append({"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": unschedulable})
    return {"metadata": {"name": name, "labels": labels or {"app": "ml-inference-api"}},
            "spec": {"nodeName": "node1"}, "status": {"phase": phase, "conditions": conds, "containerStatuses": [cs]}}


class FakeCore:
    def __init__(self, pods, events=(), logs=None, services=None):
        self.pods, self.events, self.logs = pods, list(events), logs or {}
        self.services = services if services is not None else [{
            "metadata": {"name": "ml-inference-api"},
            "spec": {"type": "ClusterIP", "clusterIP": "10.0.0.1", "selector": {"app": "ml-inference-api"},
                     "ports": [{"port": 80, "targetPort": 8000}]}}]

    def list_namespaced_pod(self, ns, **kw): return {"items": self.pods}
    def list_namespaced_event(self, ns, **kw): return {"items": self.events}
    def list_namespaced_service(self, ns, **kw): return {"items": self.services}

    def read_namespaced_pod_log(self, pod, ns, **kw):
        return self.logs.get(pod, "")


class FakeApps:
    def __init__(self, replicas=2, ready=0, image="mlops-inference:1.0.0"):
        self.patched = []
        self.dep = {"metadata": {"name": "ml-inference-api", "generation": 3},
                    "spec": {"replicas": replicas, "selector": {"matchLabels": {"app": "ml-inference-api"}},
                             "template": {"metadata": {"labels": {"app": "ml-inference-api", "app.kubernetes.io/version": "1.0.0"}},
                                          "spec": {"containers": [{"name": "api", "image": image}]}}},
                    "status": {"readyReplicas": ready, "availableReplicas": ready, "updatedReplicas": replicas, "observedGeneration": 3}}

    def read_namespaced_deployment(self, name, ns): return self.dep
    def list_namespaced_deployment(self, ns): return {"items": [self.dep]}
    def patch_namespaced_deployment(self, name, ns, body):
        self.patched.append(("restart", name, ns, body))
        return {}

    def patch_namespaced_deployment_scale(self, name, ns, body):
        self.patched.append(("scale", name, ns, body))
        return {}


@pytest.fixture
def k8s(monkeypatch):
    def install(core, apps):
        monkeypatch.setattr(kubernetes_tools, "_clients", lambda: (core, apps))
        return core, apps
    return install


def ev(obj, reason, msg, typ="Warning"):
    return {"involvedObject": {"name": obj}, "type": typ, "reason": reason, "message": msg, "count": 3}


def dumps(x):
    return json.dumps(x)
