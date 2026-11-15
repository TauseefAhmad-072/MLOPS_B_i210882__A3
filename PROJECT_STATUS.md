# PROJECT STATUS (resume file)
Last updated: 2026-11-11

| Phase | Status | Evidence |
|---|---|---|
| P0 foundation + FastAPI (agent + inference) | DONE, executed | tests, `docs/evidence/` |
| P1 DVC + MLflow | DONE, executed for real (local remote, 2 dataset versions, 3 runs incl. failed gate) | `docs/evidence/1_*, 2_*` |
| P2 Airflow DAG + API adapter | CODE DONE; adapter tested with mock transport; **DAG not executed** (no Docker in build sandbox) | `tests/test_airflow.py` |
| P3 Kubernetes manifests + diagnostics | CODE DONE; analysis tested with fake API objects; **not deployed** (no cluster in build sandbox) | `tests/test_kubernetes.py` |
| P4 integrated agent (LLM + offline planner) | DONE; LLM mode tested with a fake LLM, **not run against live Anthropic API** | `tests/test_orchestrator.py` |
| P5 tests/failures/CI/docs | DONE: 43 pytest tests pass, ruff clean, CI workflow written | `docs/evidence/pytest_results.txt` |

## Remaining on a machine with Docker + Kubernetes (cannot be done from the build sandbox)
1. `docker compose up -d`; trigger DAG; confirm it runs (check `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_*` env names vs your Airflow 3.x patch version; adjust image tag if needed).
2. `./scripts/deploy_k8s.sh`; verify pods Ready; run `scripts/inject_failures.py bad-model-path`.
3. Re-run `python scripts/capture_evidence.py`, take screenshots into `screenshots/` (DVC, MLflow UI, Airflow UI, kubectl).
4. Fill the "Live infrastructure evidence" section in `REPORT.md`.
5. Create the GitHub repo and push (NOT done — needs your permission).
