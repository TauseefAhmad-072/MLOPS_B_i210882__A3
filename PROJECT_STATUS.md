# PROJECT STATUS (resume file)
Dataset: synthetic customer churn (`scripts/make_dataset.py`), gate F1 >= 0.70.

| Phase | Status | Evidence |
|---|---|---|
| P0 foundation + FastAPI | DONE, executed | tests |
| P1 DVC + MLflow | DONE, executed (2 dataset versions, runs f1 0.745 / 0.814, failed-gate run 0.463) | `docs/evidence/` |
| P2 Airflow DAG + adapter | CODE DONE; adapter mock-tested; DAG **not yet run** on Docker | `tests/test_airflow.py` |
| P3 Kubernetes | CODE DONE; diagnostics tested with fake objects; **not yet deployed** | `tests/test_kubernetes.py` |
| P4 agent (LLM + offline planner) | DONE; LLM mode tested with fake LLM only | `tests/test_orchestrator.py` |
| P5 tests / failures / CI / docs | DONE: 43 tests pass, ruff clean, CI workflow | `docs/evidence/pytest_results.txt` |

## TODO after pushing (needs Docker + Kubernetes)
1. `docker compose up -d --build`; trigger DAG; screenshot graph.
2. `./scripts/deploy_k8s.sh`; failure `bad-model-path`; screenshots.
3. `python scripts/capture_evidence.py && python scripts/build_report.py`; commit.
4. Verify GitHub Actions is green on the first PR/push.
