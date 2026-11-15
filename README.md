# Assignment 03 — Production-Ready MLOps AI Agent

A conversational agent that **calls real tools** against DVC, MLflow, Apache Airflow 3 and Kubernetes, and answers only
from tool evidence. Dataset: synthetic customer-churn data (generated offline by `scripts/make_dataset.py`, no personal data). Model: RandomForest.

```
User -> FastAPI (/chat) -> Orchestrator (LLM tool-calling | offline rules) -> Safety executor
        -> dvc_tools | mlflow_tools | airflow_tools | kubernetes_tools -> evidence-based answer
```
See `architecture.md` and `REPORT.md`.

> **Fresh clone?** The DVC *local remote* (`.dvc-storage/`, ~0.3 MB) is committed so `dvc pull` works right after cloning.
> MLflow history (`mlflow.db`) is not committed: run `./scripts/bootstrap_dvc.sh` once to recreate dataset versions and training runs.

## 1. Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # then: export $(grep -v '^#' .env | xargs)
```
## 2. DVC + MLflow (local, no Docker needed)
```bash
./scripts/bootstrap_dvc.sh          # git+dvc init, local DVC remote, dataset v1 & v2 (tags data-v1/data-v2), one MLflow run each
# MLflow default store: sqlite:///mlflow.db (set MLFLOW_TRACKING_URI=http://localhost:5000 to use the compose server)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional UI (needs `pip install mlflow`)
```
Reproduce an old dataset: `git checkout data-v1 -- data/customer_churn.csv.dvc && dvc checkout` (the agent's `dvc_checkout` tool does this after confirmation).
## 3. Airflow 3 + MLflow server (Docker Compose)
```bash
docker compose up -d          # Airflow http://localhost:8080 (admin/admin, LOCAL DEV ONLY), MLflow http://localhost:5000
export MLFLOW_TRACKING_URI=http://localhost:5000 AIRFLOW_BASE_URL=http://localhost:8080 AIRFLOW_PASSWORD=admin
```
DAG `training_pipeline`: `validate_dataset → preprocess_data → train_model → evaluate_and_log → quality_gate → promote_model`
(2 retries with backoff for transient errors; validation / gate failures use `AirflowFailException`, so nothing downstream runs).
## 4. Kubernetes (kind / minikube / Docker Desktop)
```bash
./scripts/deploy_k8s.sh       # exports model, builds image, loads it, applies namespace/configmap/deployment(2 replicas)/ClusterIP service
```
Optional least-privilege identity for in-cluster use: `kubernetes/rbac.yaml`.
## 5. Run the agent
```bash
uvicorn agent.main:app --port 8000
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Which dataset version was used for experiment 1?"}' | python -m json.tool
# state-changing request needs explicit confirmation:
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message":"restart the deployment","confirm":true}'
```
**Planner modes** (reported in every response as `mode`): with `ANTHROPIC_API_KEY` set the agent uses Claude tool-calling;
without it, a deterministic keyword planner ("offline-rules") selects tools. Both use the same tools, permissions and logging.
## 6. Tests & lint
```bash
pytest          # 43 tests, no cluster/Airflow/LLM needed (mocked)
ruff check .
```
## 7. Failure demonstrations
```bash
python scripts/inject_failures.py corrupt-dataset   [--local]   # F1: validation fails, downstream skipped
python scripts/inject_failures.py low-quality-model [--local]   # F2: quality gate fails, promote never runs
python scripts/inject_failures.py bad-model-path                # F3: pods Running but never Ready (K8s)
python scripts/inject_failures.py restore-k8s
```
Then ask: *"Why did the training pipeline fail?"* / *"Why is my prediction API unavailable?"*
`python scripts/capture_evidence.py` stores the five demo conversations in `docs/evidence/`.

## Safety model
* Tools run through one executor: schema validation, timeout, structured JSON logs (`request_id, tool, status, duration_ms`), secret redaction.
* State-changing tools (`k8s_restart/scale`, `airflow_trigger`, `dvc_checkout`) need `confirm=true` **from the user request** (the LLM cannot supply it).
* Kubernetes: approved namespace allow-list (`ALLOWED_NAMESPACES`), replicas 1–5. Airflow: approved DAG allow-list. DVC/git: sub-command allow-list, no shell, revision regex.
* Retraining is never started implicitly; it is only *recommended* with evidence (failed gate, freshness policy `MODEL_MAX_AGE_DAYS`, supplied degradation results).
* No secrets in the repo; `airflow/simple_auth_passwords.json` is a local-dev default only.

## Updating the repo later (typical loop)
```bash
git pull
# ...change code / add live evidence...
pytest && ruff check .
python scripts/capture_evidence.py && python scripts/build_report.py   # refresh REPORT.md from live evidence
git add -A && git commit -m "describe change" && git push
```
Data change: edit/regenerate the dataset -> `dvc add data/customer_churn.csv` -> `git add data/customer_churn.csv.dvc` -> commit -> `git tag data-v3` -> `dvc push` -> commit `.dvc-storage`.
