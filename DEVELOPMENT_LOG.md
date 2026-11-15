# DEVELOPMENT LOG
- Inspected sandbox: Linux, Python 3.12, no docker/kubectl/cluster; pip available → built real adapters + mocked tests for Airflow/K8s.
- MLflow 3.18 disables the file store by default → switched default backend to `sqlite:///mlflow.db` (needs sqlalchemy+alembic; added to deps).
- `.gitignore` initially ignored the CSV, which stopped `dvc add` creating `data/.gitignore`; removed the ignore rule.
- Phase P0/P1: config, logging, schemas, tool registry/executor; training pipeline; DVC bootstrap (v1/v2); MLflow runs. Smoke-tested all 27 tools.
- Phase P2/P3: Airflow client (JWT), DAG, K8s tools + `analyze_evidence`; manifests, Dockerfile, compose.
- Phase P4: orchestrator (LLM + rules), multi-tool playbook, retrain-evidence policy, trigger guard.
- Phase P5: 43 tests; fixed a brittle DAG-order test; migrated inference API to lifespan; ruff clean; CI; failure injection run locally (validation fail: 70 nulls; gate fail: f1=0.17).
- Honesty fix: multi-tool answer marks conclusion PARTIAL when Kubernetes/Airflow evidence is missing.
