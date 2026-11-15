# Technical Report - MLOps AI Agent (Assignment 03)

**Use case:** customer-churn prediction (synthetic, offline, no personal data). Model: RandomForest. Quality gate: F1 >= 0.7.

## 1. Architecture
See `architecture.md`. FastAPI -> Orchestrator (Claude tool-calling or deterministic offline planner, always labelled in `mode`) -> safety executor -> four tool modules (DVC, MLflow, Airflow 3, Kubernetes) -> evidence-based answer. Training is an Airflow DAG (validate -> preprocess -> train -> evaluate/log -> quality gate -> promote) writing MLflow runs tagged with DVC provenance; serving is a FastAPI container on Kubernetes (2 replicas, ClusterIP, probes, requests/limits, ConfigMap).

## 2. Tool definitions and permissions (27 tools)
| Tool | Module | State-changing | Purpose |
|---|---|---|---|
| `airflow_list_dags` | airflow | no | List available Airflow DAGs with paused/active state. |
| `airflow_get_dag_runs` | airflow | no | Recent DAG runs and their state (newest first). |
| `airflow_trigger_training_dag` | airflow | **yes** (confirm=true) | Trigger the approved training DAG. Starts a real pipeline run. |
| `airflow_get_failed_tasks` | airflow | no | Identify failed/upstream-failed tasks of a DAG run (default: latest run). |
| `airflow_get_task_logs` | airflow | no | Fetch log tail of one task instance for failure diagnostics. |
| `dvc_list_datasets` | dvc | no | List datasets tracked by DVC (path, md5, size). |
| `dvc_get_dataset_version` | dvc | no | Current recorded dataset version (md5), Git commit and version tag. |
| `dvc_check_changes` | dvc | no | Detect whether DVC-tracked data differs from what is recorded in Git. |
| `dvc_compare_versions` | dvc | no | Compare tracked data between two Git revisions (tags/commits). |
| `dvc_explain_reproduce` | dvc | no | Explain (without executing) how to reproduce a previous dataset version. |
| `dvc_checkout` | dvc | **yes** (confirm=true) | Restore the DVC-tracked dataset to the version recorded at a Git revision. CHANGES THE WORKSPACE. |
| `dvc_run_provenance` | dvc | no | Dataset version recorded for an MLflow run or experiment id ('experiment 25'). |
| `k8s_list_workloads` | kubernetes | no | List Deployments and Pods in the approved namespace. |
| `k8s_pod_status` | kubernetes | no | Detailed status of Pods (container states, restarts, probes outcome). |
| `k8s_pod_logs` | kubernetes | no | Tail logs of a Pod. |
| `k8s_services` | kubernetes | no | Inspect Services (type, ports, selector) and whether they have ready endpoints. |
| `k8s_unhealthy_pods` | kubernetes | no | Find unhealthy Pods, restarts and recent Warning events. |
| `k8s_deployed_version` | kubernetes | no | Report the deployed application version (image tag, version label, ConfigMap). |
| `k8s_diagnose_deployment` | kubernetes | no | Why is my prediction API unavailable? Collects pods, events, services and logs, then reports evidence-based findings. |
| `k8s_restart_deployment` | kubernetes | **yes** (confirm=true) | Rolling-restart a Deployment. CHANGES CLUSTER STATE. |
| `k8s_scale_deployment` | kubernetes | **yes** (confirm=true) | Scale a Deployment (1-5 replicas). CHANGES CLUSTER STATE. |
| `mlflow_list_experiments` | mlflow | no | List MLflow experiments. |
| `mlflow_list_runs` | mlflow | no | List the most recent training runs (newest first) with metrics, params and tags. |
| `mlflow_get_run` | mlflow | no | Get metrics, parameters, tags and artifacts of one run. |
| `mlflow_best_run` | mlflow | no | Find the best run by a metric among the most recent runs, with readiness caveats. |
| `mlflow_compare_runs` | mlflow | no | Compare runs side by side on all metrics; give run_ids or last_n recent runs. |
| `mlflow_model_registry` | mlflow | no | Registered models and their versions (if a registry is available). |

Controls: argument schema validation; 20s default timeout; JSON logs (`request_id`, `tool`, `status`, `duration_ms`) with secret redaction; confirmation comes only from the user's request, never from the model; Kubernetes namespace allow-list; Airflow DAG allow-list; DVC/git sub-command allow-list (no shell); retraining is never auto-triggered.

## 3. Evidence status
Evidence files in `docs/evidence/`. Sections marked LIVE below were produced against running Airflow/Kubernetes; otherwise the agent honestly reports the service as unavailable.

DVC checkout round-trip (real):
```
md5 before (data-v2): b3a5a71f5a021cf320f242f424e2f839
dvc_checkout data-v1: ok | md5 now: 83599631444ad972d85c2aff18f54af0
workspace matches recorded (v1 pointer): True
restored to HEAD pointer, md5: b3a5a71f5a021cf320f242f424e2f839
```
Automated tests: `docs/evidence/pytest_results.txt` (======================= 43 passed, 3 warnings in 11.46s ========================).

## 4. Five tested agent conversations
### 4.1 Dataset provenance
*DVC/MLflow executed against real local stores*

**User:** Which dataset version was used for experiment 1?

**Tool calls:**
  - `dvc_run_provenance` -> **ok** (1872.7 ms, req `76a0663b3ffd`)

**Agent answer (verbatim):**
```
Run 7d073de9 used dataset data/customer_churn.csv version data-v2 (md5 b3a5a71f5a021cf320f242f424e2f839), Git commit 11079740. Data matched recorded hash at training time: true.
```

### 4.2 Model selection
*MLflow executed; note the failed-gate run is shown with its gate status*

**User:** Compare the last five training runs and recommend the best candidate based on F1 score

**Tool calls:**
  - `mlflow_compare_runs` -> **ok** (14.5 ms, req `c896d51c3925`)

**Agent answer (verbatim):**
```
Ranking by f1: 1. rf-1794762780 (e90d7961) 0.8144 gate=passed; 2. rf-1794762773 (c9ba7039) 0.7449 gate=passed; 3. rf-1794762806 (7d073de9) 0.4634 gate=failed
Caveat: ranking uses one metric on one hold-out split; a higher score alone does not make a run production-ready.
```

### 4.3 Pipeline execution
*Without confirm=true the agent refuses; re-run with confirm=true against live Airflow*

**User:** Trigger the model training pipeline and report whether execution succeeded.

**Tool calls:**
  - `airflow_trigger_training_dag` -> **denied** (0.0 ms, req `a9e32d031545`)

**Agent answer (verbatim):**
```
[airflow_trigger_training_dag] DENIED: state-changing operation requires explicit confirmation (confirm=true)
Confirmation required: resend with confirm=true to start the DAG run.
```

### 4.4 Deployment diagnosis
*No cluster reachable when captured: agent reports UNAVAILABLE and invents nothing*

**User:** Why is my prediction API unavailable?

**Tool calls:**
  - `k8s_diagnose_deployment` -> **unavailable** (94.4 ms, req `e4071cf3fc09`)

**Agent answer (verbatim):**
```
[k8s_diagnose_deployment] UNAVAILABLE: Kubernetes cluster not reachable/configured: ConfigException: Invalid kube-config file. No configuration found.
```

### 4.5 Multi-tool reasoning
*Unavailable systems are listed as UNKNOWN and the conclusion is marked PARTIAL*

**User:** Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.

**Tool calls:**
  - `mlflow_best_run` -> **ok** (13.9 ms, req `130ba1cb1c48`)
  - `dvc_run_provenance` -> **ok** (7.4 ms, req `130ba1cb1c48`)
  - `k8s_diagnose_deployment` -> **unavailable** (0.8 ms, req `130ba1cb1c48`)
  - `k8s_deployed_version` -> **unavailable** (0.7 ms, req `130ba1cb1c48`)
  - `airflow_get_dag_runs` -> **error** (179.3 ms, req `130ba1cb1c48`)

**Agent answer (verbatim):**
```
KNOWN FACTS
- Best run by f1 among 3 recent runs: e90d7961b6dd4b4796afe9c8fe966857 (rf-1794762780), accuracy=0.8800, precision=0.8634, recall=0.7707, f1=0.8144, roc_auc=0.9413.
- Quality gate tag: passed; run age 0.0 days.
- Caveat: selected on a single metric on one hold-out split; validate on production-like data before promotion
- Run e90d7961 used dataset data/customer_churn.csv version data-v2 (md5 b3a5a71f5a021cf320f242f424e2f839), Git commit 11079740. Data matched recorded hash at training time: true.

UNCERTAINTIES
- Deployment health UNKNOWN - [k8s_diagnose_deployment] UNAVAILABLE: Kubernetes cluster not reachable/configured: ConfigException: Invalid kube-config file. No configuration found.
- Deployed version UNKNOWN - [k8s_deployed_version] UNAVAILABLE: Kubernetes cluster not reachable/configured: ConfigException: Invalid kube-config file. No configuration found.
- Recent training workflows UNKNOWN - [airflow_get_dag_runs] ERROR: AIRFLOW_PASSWORD (or AIRFLOW_TOKEN) is not configured
- The deployment does not record which MLflow run its model came from, so 'deployed model == selected best run' cannot be verified from tool data.
- No production-degradation results were supplied; live model quality is unknown.

RECOMMENDED ACTIONS
- Retraining is NOT recommended on the evidence collected (no failed quality gate, no degradation data, within freshness policy of 30 days). This conclusion is PARTIAL because Kubernetes, Airflow evidence was unavailable.
- No retraining was triggered by this request.
```


## 5. Two failure investigations
**F1 - corrupted dataset** (`scripts/inject_failures.py corrupt-dataset`). Evidence: `EXPECTED FAILURE (DataValidationError): dataset contains 343 null values`. Root cause: nulls injected into a copy of the dataset (343 cells). `validate_dataset` raises `AirflowFailException` (no retry); every downstream task is skipped, so nothing is trained or promoted. Resolution: `dvc checkout` the tracked version and re-run.

**F2 - low-quality model** (`low-quality-model`). Evidence: `EXPECTED FAILURE (QualityGateError): quality gate FAILED: f1=0.4633781763826607 < 0.7`. MLflow records the run with `quality_gate=failed` (f1=0.4633); `promote_model` refuses unless the tag is `passed` (`test_failed_quality_gate_blocks_promotion`). Root cause: 45% label noise with a 1-tree depth-1 forest. Resolution: restore hyper-parameters; the agent keeps ranking this run last with `gate=failed`.

**F3 - Kubernetes bad ConfigMap** (`bad-model-path`, needs a cluster). Expected: pods Running but not Ready; kubelet `Unhealthy` events (503 on `/ready`); logs `model file not found ...`; Service with 0 ready endpoints. Diagnosis logic is covered by `test_failure_demo_k8s_bad_model_path_is_diagnosed`.

## 6. Screenshots
Place in `screenshots/`: `dvc_status.png`, `mlflow_runs.png`, `airflow_dag_graph.png`, `kubectl_pods.png`.

## 7. Known limitations and improvements
* "Experiment N" means MLflow experiment id N (latest run).
* The deployment does not record the MLflow run id of its model, so "deployed model == best model" is reported as an uncertainty. Improvement: put `MLFLOW_RUN_ID` in the ConfigMap.
* Single hold-out split, single metric; add cross-validation and drift monitoring.
* Offline planner is keyword-based; LLM mode is the general planner.
* Agent API has only an optional shared key; production needs real authN/Z.
* DVC remote is local; use S3/GCS/Azure for teams.
