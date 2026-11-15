# Architecture
```
Engineer ─► FastAPI agent (agent/main.py)  POST /chat {message, confirm}
              └► Orchestrator (agent/orchestrator.py)
                   ├─ llm mode: Claude tool-calling loop (max 8 steps)      ┐ same tools,
                   └─ offline-rules mode: deterministic planner + playbooks ┘ same safety layer
                        └► execute_tool (agent/tools/base.py): validate → permission/confirm → timeout → JSON log
                             ├ dvc_tools        subprocess allow-list (dvc/git), MLflow provenance tags
                             ├ mlflow_tools     MlflowClient (search_runs, registry)
                             ├ airflow_tools    Airflow 3 REST /api/v2, JWT from /auth/token
                             └ kubernetes_tools official python client, namespace allow-list
Training:  data/*.dvc (DVC, local remote) → Airflow DAG → MLflow run (tags: dvc.dataset_md5/version/git_commit)
Serving:   api/main.py in Docker → K8s Deployment(2) + ClusterIP + probes + limits + ConfigMap
```
**Provenance chain.** Every training run records `dvc.dataset_md5`, `dvc.git_commit`, `dvc.dataset_version` (git tag `data-v*` on the commit that last changed the `.dvc` pointer) and whether the data on disk matched the recorded hash. `dvc_run_provenance` reads those tags — it never infers.
**Multi-tool playbook** (`Orchestrator._assess`): MLflow best run → DVC provenance → K8s diagnose + version → Airflow runs (+failed tasks) → correlate → *Known facts / Uncertainties / Recommended actions*. Missing tools yield UNKNOWN entries and a "PARTIAL" conclusion.
**Quality gate.** `F1_GATE` (default 0.70) applied in the DAG; `promote_model` refuses any run whose MLflow tag `quality_gate != passed`.
