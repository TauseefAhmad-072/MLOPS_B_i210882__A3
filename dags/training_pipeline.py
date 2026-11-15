"""Airflow 3 training DAG: validate -> preprocess -> train -> evaluate/log (MLflow) -> quality gate -> promote.
Failed validation or gate stops everything downstream (promote never runs). Retries on transient failures only;
data-quality / gate failures raise AirflowFailException (no retry)."""
from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

import pendulum
from airflow.exceptions import AirflowFailException
from airflow.sdk import DAG, get_current_context, task

PROJECT = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.environ.setdefault("REPO_ROOT", PROJECT)

log = logging.getLogger(__name__)
WORK = "/tmp/ml_work"

default_args = {"retries": 2, "retry_delay": timedelta(seconds=20), "retry_exponential_backoff": True}


def _conf() -> dict:
    return get_current_context()["dag_run"].conf or {}


def _hp() -> dict:
    from training.pipeline import DEFAULT_HP
    return {**DEFAULT_HP, **(_conf().get("hyperparams") or {})}


def _dataset() -> str:
    return _conf().get("dataset_path") or os.environ.get("DVC_DATASET", "data/customer_churn.csv")


with DAG(dag_id="training_pipeline", schedule=None, start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
         catchup=False, default_args=default_args, tags=["mlops", "assignment-03"],
         doc_md=__doc__) as dag:

    @task
    def validate_dataset() -> dict:
        from training import pipeline as p
        ds = _dataset()
        try:
            res = p.validate_dataset(os.path.join(PROJECT, ds))
        except p.DataValidationError as e:
            log.error("dataset validation failed: %s", e)
            raise AirflowFailException(f"dataset validation failed: {e}") from e  # no retry, stops downstream
        log.info("dataset valid: %s", res)
        return {**res, "dataset": ds}

    @task
    def preprocess_data(v: dict) -> dict:
        from training import pipeline as p
        return p.preprocess(v["path"], WORK, _hp())

    @task
    def train_model(split: dict) -> str:
        from training import pipeline as p
        return p.train_model(split["train"], WORK, _hp())

    @task
    def evaluate_and_log(model_path: str, split: dict, v: dict) -> dict:
        from training import pipeline as p
        ctx = get_current_context()
        return p.evaluate_and_log(split["test"], model_path, _hp(), v["dataset"], split,
                                  tags={"airflow.dag_run_id": ctx["dag_run"].run_id})

    @task
    def quality_gate(res: dict) -> dict:
        from training import pipeline as p
        try:
            return p.quality_gate(res["run_id"])
        except p.QualityGateError as e:
            log.error("%s", e)
            raise AirflowFailException(str(e)) from e

    @task
    def promote_model(gate: dict) -> dict:
        from training import pipeline as p
        return p.promote(gate["run_id"])

    v = validate_dataset()
    s = preprocess_data(v)
    m = train_model(s)
    e = evaluate_and_log(m, s, v)
    g = quality_gate(e)
    promote_model(g)
