# Technical Report — MLOps AI Agent (Assignment 03)

## 1. Architecture
See `architecture.md`. FastAPI → Orchestrator (Claude tool-calling or deterministic offline planner, always labelled in `mode`) → safety executor → four tool modules (DVC, MLflow, Airflow 3, Kubernetes) → evidence-based answer. Training is an Airflow DAG (validate → preprocess → train → evaluate/log → quality gate → promote) writing MLflow runs tagged with DVC provenance; serving is a FastAPI container on Kubernetes (2 replicas, ClusterIP, probes, requests/limits, ConfigMap).

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

Cross-cutting controls: argument schema validation; 20s default timeout; JSON logs with `request_id`, `tool`, `status`, `duration_ms` and secret redaction; confirmation is taken only from the user's request (`confirm` in `/chat`), never from the model; Kubernetes namespace allow-list; Airflow DAG allow-list; DVC/git sub-command allow-list (no shell); retraining never auto-triggered.

## 3. Evidence status (be explicit about what was really executed)
| System | Executed for real in build session? |
|---|---|
| DVC (init, local remote, 2 versions, diff, checkout round-trip) | **Yes** |
| MLflow (sqlite store, 3 runs, registry, best/compare tools) | **Yes** |
| Training pipeline code + both Airflow-stage failures (via `--local`) | **Yes** (pipeline functions); the Airflow *scheduler* run is pending |
| Airflow 3 API adapter | Tested against mock transport only |
| Kubernetes manifests / diagnostics | Analysis tested with fake API objects; **no cluster was available** |
| LLM planner (Anthropic API) | Tested with an injected fake LLM only |

DVC checkout evidence: dataset md5 `f5da0b1b…` (data-v2) → `dvc_checkout rev=data-v1` → md5 `5afc23b9…` (v1 data, `dvc_check_changes` → changed=False vs v1 pointer) → restore pointer + `dvc checkout` → `f5da0b1b…` again.

## 4. Five tested agent conversations (real tool evidence as available)
### 4.1 Dataset provenance
*Status: REAL — DVC/MLflow executed*

**User:** Which dataset version was used for experiment 1?

**Tool calls:**
  - `dvc_run_provenance` → **ok** (2330.5 ms, req `64b2bd390cd9`)

**Agent answer (verbatim):**
```
Run 18ff7be4 used dataset data/customer_churn.csv version data-v2 (md5 f5da0b1b678fc9818fa8594b02988852), Git commit 6dac4c4c. Data matched recorded hash at training time: true.
```

### 4.2 Model selection
*Status: REAL — MLflow executed. Note the agent returns the failed-gate run in the ranking with its gate status instead of hiding it*

**User:** Compare the last five training runs and recommend the best candidate based on F1 score

**Tool calls:**
  - `mlflow_compare_runs` → **ok** (14.1 ms, req `4726bb777b87`)

**Agent answer (verbatim):**
```
Ranking by f1: 1. rf-1794345510 (0586eaca) 0.9730 gate=passed; 2. rf-1794345503 (7ba89205) 0.9670 gate=passed; 3. rf-1794345678 (18ff7be4) 0.1707 gate=failed
Caveat: ranking uses one metric on one hold-out split; a higher score alone does not make a run production-ready.
```

### 4.3 Pipeline execution
*Status: Safety path real (confirmation required). Re-run with `"confirm": true` against live Airflow and paste the run/task states here*

**User:** Trigger the model training pipeline and report whether execution succeeded.

**Tool calls:**
  - `airflow_trigger_training_dag` → **denied** (0.0 ms, req `be75c88845f2`)

**Agent answer (verbatim):**
```
[airflow_trigger_training_dag] DENIED: state-changing operation requires explicit confirmation (confirm=true)
Confirmation required: resend with confirm=true to start the DAG run.
```

### 4.4 Deployment diagnosis
*Status: Cluster unavailable in build sandbox → agent reports UNAVAILABLE and invents nothing (this is test-covered). Re-run on a live cluster and paste output here*

**User:** Why is my prediction API unavailable?

**Tool calls:**
  - `k8s_diagnose_deployment` → **unavailable** (83.3 ms, req `6890033d6208`)

**Agent answer (verbatim):**
```
[k8s_diagnose_deployment] UNAVAILABLE: Kubernetes cluster not reachable/configured: ConfigException: Invalid kube-config file. No configuration found.
```

### 4.5 Multi-tool reasoning
*Status: MLflow + DVC real; Kubernetes/Airflow unavailable → conclusion explicitly marked PARTIAL. Re-run with full infrastructure for the final demo*

**User:** Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.

**Tool calls:**
  - `mlflow_best_run` → **ok** (9.8 ms, req `3fb89f01c22b`)
  - `dvc_run_provenance` → **ok** (4.9 ms, req `3fb89f01c22b`)
  - `k8s_diagnose_deployment` → **unavailable** (0.5 ms, req `3fb89f01c22b`)
  - `k8s_deployed_version` → **unavailable** (0.3 ms, req `3fb89f01c22b`)
  - `airflow_get_dag_runs` → **error** (53.3 ms, req `3fb89f01c22b`)

**Agent answer (verbatim):**
```
KNOWN FACTS
- Best run by f1 among 3 recent runs: 0586eaca081548878c3d5245cfbf5549 (rf-1794345510), accuracy=0.9669, precision=0.9863, recall=0.9600, f1=0.9730, roc_auc=0.9991.
- Quality gate tag: passed; run age 0.0 days.
- Caveat: selected on a single metric on one hold-out split; validate on production-like data before promotion
- Run 0586eaca used dataset data/customer_churn.csv version data-v2 (md5 f5da0b1b678fc9818fa8594b02988852), Git commit 6dac4c4c. Data matched recorded hash at training time: true.

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
**F1 — corrupted dataset (injected: `scripts/inject_failures.py corrupt-dataset --local`).** Symptom: pipeline stops at `validate_dataset`. Evidence: `DataValidationError: dataset contains 70 null values`. Root cause: nulls injected into feature column 3 of a copy of the dataset. In Airflow the task raises `AirflowFailException` (no retry), `preprocess…promote` are never run, so no model is trained or promoted. Resolution: restore the DVC-tracked version (`dvc checkout`), re-run. Agent diagnosis path: `airflow_get_failed_tasks` → `airflow_get_task_logs` (mock-tested in `test_failure_demo_airflow_quality_gate_failure_is_diagnosed`).

**F2 — low-quality model (injected: `low-quality-model --local`).** Symptom: pipeline fails at `quality_gate`. Evidence: MLflow run `18ff7be4` has `f1=0.1707`, tag `quality_gate=failed`; error `quality gate FAILED: f1=0.17 < 0.93`; `promote_model` refuses (`test_failed_quality_gate_blocks_promotion`). Root cause: 45% label noise and a 1-tree depth-1 forest. Resolution: restore hyper-parameters; the earlier passed run remains the only registry candidate. Note the agent still ranks this run last and shows `gate=failed`.

**F3 — Kubernetes bad ConfigMap (`bad-model-path`; run on a live cluster).** Expected: pods `Running` but not Ready; kubelet `Unhealthy` events (HTTP 503 on `/ready`); logs `model file not found at models/does-not-exist.joblib`; Service has 0 ready endpoints. The agent's `analyze_evidence` reports each of these (tested with fake cluster objects in `test_failure_demo_k8s_bad_model_path_is_diagnosed`). Resolution: `restore-k8s`.

## 6. Live infrastructure evidence (to complete on your machine)
Add screenshots to `screenshots/`: `dvc_status.png`, `mlflow_runs.png`, `airflow_dag_graph.png`, `kubectl_pods.png`, and paste live outputs of conversations 4.3–4.5.

## 7. Known limitations and improvements
* Airflow DAG and K8s manifests were not executed in the build sandbox (no Docker/cluster); adjust image tags/env names to your Airflow 3.x patch version if needed.
* Offline planner is keyword-based; the LLM mode is the intended general planner. LLM mode was not run against the live API here.
* "Experiment N" is interpreted as MLflow experiment id N (latest run); a custom naming scheme would need a mapping tag.
* The deployment does not record the MLflow run ID of its model → "deployed model == best model" is reported as an uncertainty. Improvement: add `MLFLOW_RUN_ID` to the ConfigMap/image labels.
* Single hold-out split and one metric; add cross-validation, drift monitoring and a real freshness policy source.
* Agent API has only an optional shared-key; add real authN/Z and per-user permissions for production.
* DVC remote is local; use S3/GCS/Azure for team use.
