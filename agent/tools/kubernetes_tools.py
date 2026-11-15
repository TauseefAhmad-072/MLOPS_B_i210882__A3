"""Kubernetes operations tools. Reads are namespace-restricted; mutations need confirm=True + approved namespace."""
from __future__ import annotations

from agent.config import get_settings
from agent.tools.base import PermissionDenied, ServiceUnavailable, ToolError, tool

_cached: tuple | None = None


def _clients():
    """Return (CoreV1Api, AppsV1Api). Tests monkeypatch this function."""
    global _cached
    if _cached:
        return _cached
    try:
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except Exception:  # noqa: BLE001
            config.load_kube_config()
        _cached = (client.CoreV1Api(), client.AppsV1Api())
        return _cached
    except Exception as e:  # noqa: BLE001
        raise ServiceUnavailable(f"Kubernetes cluster not reachable/configured: {type(e).__name__}: {str(e)[:150]}") from e


def _d(obj) -> dict:
    """Normalise API objects to plain camelCase dicts (same shape as the K8s JSON)."""
    if isinstance(obj, dict):
        return obj
    from kubernetes import client
    return client.ApiClient().sanitize_for_serialization(obj)


def _ns(namespace: str | None) -> str:
    s = get_settings()
    ns = namespace or s.k8s_namespace
    if ns not in s.allowed_namespaces:
        raise PermissionDenied(f"namespace '{ns}' is not approved (allowed: {list(s.allowed_namespaces)})")
    return ns


def _call(fn, *a, **kw):
    try:
        return _d(fn(*a, **kw))
    except (PermissionDenied, ToolError):
        raise
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status", None)
        if status == 404:
            raise ToolError("resource not found") from e
        if status in (401, 403):
            raise ToolError(f"Kubernetes denied the request (HTTP {status}); check RBAC") from e
        raise ServiceUnavailable(f"Kubernetes API call failed: {type(e).__name__}: {str(e)[:150]}") from e


def pod_view(p: dict) -> dict:
    st, meta = p.get("status", {}), p.get("metadata", {})
    cs = []
    for c in st.get("containerStatuses", []) or []:
        state = c.get("state", {}) or {}
        waiting = state.get("waiting") or {}
        last = (c.get("lastState", {}) or {}).get("terminated") or {}
        cs.append({"name": c.get("name"), "ready": c.get("ready"), "restart_count": c.get("restartCount", 0),
                   "image": c.get("image"), "waiting_reason": waiting.get("reason"),
                   "waiting_message": (waiting.get("message") or "")[:200],
                   "last_terminated_reason": last.get("reason"), "last_exit_code": last.get("exitCode")})
    ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in st.get("conditions", []) or [])
    return {"name": meta.get("name"), "phase": st.get("phase"), "ready": ready, "node": p.get("spec", {}).get("nodeName"),
            "containers": cs, "labels": meta.get("labels", {}),
            "unschedulable_reason": next((c.get("message") for c in st.get("conditions", []) or []
                                          if c.get("reason") == "Unschedulable"), None)}


def is_unhealthy(p: dict) -> list[str]:
    reasons = []
    if p["phase"] not in ("Running", "Succeeded"):
        reasons.append(f"phase={p['phase']}")
    if p["phase"] == "Running" and not p["ready"]:
        reasons.append("not Ready")
    for c in p["containers"]:
        if c["waiting_reason"]:
            reasons.append(f"{c['name']} waiting: {c['waiting_reason']}")
        if c["restart_count"]:
            reasons.append(f"{c['name']} restarted {c['restart_count']}x (last: {c['last_terminated_reason']})")
    return reasons


def _pods(ns: str, selector: str | None = None) -> list[dict]:
    core, _ = _clients()
    kw = {"label_selector": selector} if selector else {}
    return [pod_view(p) for p in _call(core.list_namespaced_pod, ns, **kw).get("items", [])]


def _deployment(ns: str, name: str) -> dict:
    _, apps = _clients()
    return _call(apps.read_namespaced_deployment, name, ns)


def _dep_view(d: dict) -> dict:
    sp, st = d.get("spec", {}), d.get("status", {})
    ctrs = sp.get("template", {}).get("spec", {}).get("containers", [])
    return {"name": d["metadata"]["name"], "desired": sp.get("replicas"), "ready": st.get("readyReplicas", 0),
            "available": st.get("availableReplicas", 0), "updated": st.get("updatedReplicas", 0),
            "images": [c.get("image") for c in ctrs], "selector": sp.get("selector", {}).get("matchLabels", {}),
            "conditions": [{"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason")}
                           for c in st.get("conditions", []) or []]}


@tool("k8s_list_workloads", "List Deployments and Pods in the approved namespace.", "kubernetes",
      {"namespace": {"type": "string"}})
def list_workloads(namespace: str | None = None) -> dict:
    ns = _ns(namespace)
    _, apps = _clients()
    deps = [_dep_view(d) for d in _call(apps.list_namespaced_deployment, ns).get("items", [])]
    pods = _pods(ns)
    return {"namespace": ns, "deployments": deps,
            "pods": [{"name": p["name"], "phase": p["phase"], "ready": p["ready"],
                      "restarts": sum(c["restart_count"] for c in p["containers"])} for p in pods]}


@tool("k8s_pod_status", "Detailed status of Pods (container states, restarts, probes outcome).", "kubernetes",
      {"namespace": {"type": "string"}, "pod": {"type": "string"}})
def pod_status(namespace: str | None = None, pod: str | None = None) -> dict:
    pods = _pods(_ns(namespace))
    if pod:
        pods = [p for p in pods if p["name"] == pod]
        if not pods:
            raise ToolError(f"pod '{pod}' not found")
    return {"pods": pods}


@tool("k8s_pod_logs", "Tail logs of a Pod.", "kubernetes",
      {"pod": {"type": "string"}, "namespace": {"type": "string"}, "tail_lines": {"type": "integer"},
       "previous": {"type": "boolean"}}, ("pod",))
def pod_logs(pod: str, namespace: str | None = None, tail_lines: int = 50, previous: bool = False) -> dict:
    from agent.logging_utils import redact_text
    ns = _ns(namespace)
    core, _ = _clients()
    try:
        txt = core.read_namespaced_pod_log(pod, ns, tail_lines=min(tail_lines, 500), previous=previous)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"could not read logs of {pod}: {str(e)[:200]}") from e
    return {"pod": pod, "previous": previous, "logs": redact_text(txt if isinstance(txt, str) else str(txt))[-4000:]}


@tool("k8s_services", "Inspect Services (type, ports, selector) and whether they have ready endpoints.", "kubernetes",
      {"namespace": {"type": "string"}})
def services(namespace: str | None = None) -> dict:
    ns = _ns(namespace)
    core, _ = _clients()
    pods = _pods(ns)
    out = []
    for s in _call(core.list_namespaced_service, ns).get("items", []):
        sel = s.get("spec", {}).get("selector", {}) or {}
        match = [p for p in pods if all(p["labels"].get(k) == v for k, v in sel.items())]
        out.append({"name": s["metadata"]["name"], "type": s["spec"].get("type"), "cluster_ip": s["spec"].get("clusterIP"),
                    "ports": [{"port": p.get("port"), "target": p.get("targetPort")} for p in s["spec"].get("ports", [])],
                    "selector": sel, "matching_pods": len(match), "ready_pods": sum(1 for p in match if p["ready"])})
    return {"namespace": ns, "services": out}


def _events(ns: str) -> list[dict]:
    core, _ = _clients()
    ev = _call(core.list_namespaced_event, ns).get("items", [])
    ev.sort(key=lambda e: str(e.get("lastTimestamp") or e.get("eventTime") or ""), reverse=True)
    return [{"object": e.get("involvedObject", {}).get("name"), "type": e.get("type"), "reason": e.get("reason"),
             "message": (e.get("message") or "")[:250], "count": e.get("count", 1)} for e in ev[:40]]


@tool("k8s_unhealthy_pods", "Find unhealthy Pods, restarts and recent Warning events.", "kubernetes",
      {"namespace": {"type": "string"}})
def unhealthy_pods(namespace: str | None = None) -> dict:
    ns = _ns(namespace)
    bad = [{"pod": p["name"], "problems": is_unhealthy(p)} for p in _pods(ns) if is_unhealthy(p)]
    return {"namespace": ns, "unhealthy": bad, "warning_events": [e for e in _events(ns) if e["type"] == "Warning"][:15]}


@tool("k8s_deployed_version", "Report the deployed application version (image tag, version label, ConfigMap).",
      "kubernetes", {"namespace": {"type": "string"}, "deployment": {"type": "string"}})
def deployed_version(namespace: str | None = None, deployment: str | None = None) -> dict:
    ns = _ns(namespace)
    name = deployment or get_settings().k8s_deployment
    d = _deployment(ns, name)
    ctr = d["spec"]["template"]["spec"]["containers"][0]
    labels = d["spec"]["template"]["metadata"].get("labels", {})
    return {"deployment": name, "image": ctr["image"], "image_tag": ctr["image"].rsplit(":", 1)[-1],
            "version_label": labels.get("app.kubernetes.io/version"),
            "generation": d["metadata"].get("generation"), "observed_generation": d.get("status", {}).get("observedGeneration")}


# ---------- evidence-based diagnosis ----------
def analyze_evidence(dep: dict, pods: list[dict], events: list[dict], svc: list[dict], logs: dict[str, str]) -> list[dict]:
    """Pure function: turns collected evidence into findings (testable without a cluster)."""
    f: list[dict] = []

    def add(cause, evidence, fix):
        f.append({"cause": cause, "evidence": evidence, "suggested_fix": fix})

    if dep["desired"] and not dep["ready"]:
        add("Deployment has no ready replicas", f"desired={dep['desired']} ready={dep['ready']}", "see findings below")
    for p in pods:
        for c in p["containers"]:
            r = c["waiting_reason"]
            if r in ("ImagePullBackOff", "ErrImagePull"):
                add("Image cannot be pulled", f"{p['name']}: {r} {c['waiting_message']} (image {c['image']})",
                    "verify image name/tag exists and the cluster can access the registry (kind: `kind load docker-image`)")
            elif r == "CrashLoopBackOff":
                add("Container crash-looping", f"{p['name']}: exit={c['last_exit_code']} reason={c['last_terminated_reason']}; "
                    f"log tail: {(logs.get(p['name']) or '')[-200:]!r}", "fix the startup error shown in logs, then roll out")
            elif r == "CreateContainerConfigError":
                add("Bad container configuration (e.g. missing ConfigMap/Secret key)", f"{p['name']}: {c['waiting_message']}",
                    "check envFrom/configMapKeyRef names")
            if c["last_terminated_reason"] == "OOMKilled":
                add("Container OOMKilled", f"{p['name']} restarts={c['restart_count']}", "raise memory limit or reduce usage")
        if p["phase"] == "Pending" and p.get("unschedulable_reason"):
            add("Pod cannot be scheduled", f"{p['name']}: {p['unschedulable_reason']}", "reduce requests or add node capacity")
        if p["phase"] == "Running" and not p["ready"] and not any(c["waiting_reason"] for c in p["containers"]):
            tail = (logs.get(p["name"]) or "")[-250:]
            add("Readiness probe failing (container running but not Ready)", f"{p['name']} log tail: {tail!r}",
                "inspect the /ready endpoint cause (config/model path) and the ConfigMap")
    for e in events:
        if e["reason"] == "Unhealthy":
            add("Probe failures reported by kubelet", f"{e['object']}: {e['message']}", "correlate with application logs")
            break
    for s in svc:
        if s["ready_pods"] == 0:
            add("Service has no ready endpoints", f"service {s['name']}: {s['matching_pods']} matching pods, 0 ready",
                "restore Pod readiness; verify selector labels")
    return f


@tool("k8s_diagnose_deployment", "Why is my prediction API unavailable? Collects pods, events, services and logs, "
      "then reports evidence-based findings.", "kubernetes", {"namespace": {"type": "string"}, "deployment": {"type": "string"}})
def diagnose(namespace: str | None = None, deployment: str | None = None) -> dict:
    ns = _ns(namespace)
    name = deployment or get_settings().k8s_deployment
    dep = _dep_view(_deployment(ns, name))
    pods = [p for p in _pods(ns) if all(p["labels"].get(k) == v for k, v in dep["selector"].items())]
    events = _events(ns)
    svc = services(ns)["services"]
    logs: dict[str, str] = {}
    for p in pods:
        if not p["ready"]:
            for prev in (False, True):
                try:
                    logs[p["name"]] = pod_logs(p["name"], ns, 40, prev)["logs"]
                    if logs[p["name"]].strip():
                        break
                except ToolError:
                    continue
    findings = analyze_evidence(dep, pods, events, svc, logs)
    return {"deployment": dep, "pods": [{"name": p["name"], "phase": p["phase"], "ready": p["ready"]} for p in pods],
            "warning_events": [e for e in events if e["type"] == "Warning"][:8], "services": svc,
            "log_samples": {k: v[-500:] for k, v in logs.items()}, "findings": findings,
            "verdict": "healthy" if not findings and dep["ready"] else "unhealthy" if findings else "unknown"}


@tool("k8s_restart_deployment", "Rolling-restart a Deployment. CHANGES CLUSTER STATE.", "kubernetes",
      {"deployment": {"type": "string"}, "namespace": {"type": "string"}, "confirm": {"type": "boolean"}},
      ("deployment",), mutating=True)
def restart_deployment(deployment: str, namespace: str | None = None, confirm: bool = False) -> dict:
    import datetime
    ns = _ns(namespace)
    _, apps = _clients()
    body = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()}}}}}
    _call(apps.patch_namespaced_deployment, deployment, ns, body)
    return {"deployment": deployment, "namespace": ns, "action": "rolling restart requested"}


@tool("k8s_scale_deployment", "Scale a Deployment (1-5 replicas). CHANGES CLUSTER STATE.", "kubernetes",
      {"deployment": {"type": "string"}, "replicas": {"type": "integer"}, "namespace": {"type": "string"},
       "confirm": {"type": "boolean"}}, ("deployment", "replicas"), mutating=True)
def scale_deployment(deployment: str, replicas: int, namespace: str | None = None, confirm: bool = False) -> dict:
    ns = _ns(namespace)
    if not 1 <= replicas <= 5:
        raise PermissionDenied("replicas must be between 1 and 5")
    _, apps = _clients()
    _call(apps.patch_namespaced_deployment_scale, deployment, ns, {"spec": {"replicas": replicas}})
    return {"deployment": deployment, "namespace": ns, "replicas": replicas}
