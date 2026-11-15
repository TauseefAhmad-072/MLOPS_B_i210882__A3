"""Central configuration. Everything comes from environment variables (no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _list(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str
    airflow_base_url: str
    airflow_username: str
    airflow_password: str
    airflow_token: str
    training_dag_id: str
    approved_dags: tuple[str, ...]
    k8s_namespace: str
    allowed_namespaces: tuple[str, ...]
    k8s_deployment: str
    tool_timeout_s: float
    f1_gate: float
    model_max_age_days: int
    dvc_dataset: str
    anthropic_api_key: str
    llm_model: str
    agent_api_key: str


def get_settings() -> Settings:
    """Read env on every call so tests/monkeypatching work."""
    e = os.environ.get
    ns = e("K8S_NAMESPACE", "mlops-agent")
    dag = e("TRAINING_DAG_ID", "training_pipeline")
    return Settings(
        repo_root=Path(e("REPO_ROOT", str(ROOT))),
        mlflow_tracking_uri=e("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"),
        mlflow_experiment=e("MLFLOW_EXPERIMENT", "churn-classifier"),
        airflow_base_url=e("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/"),
        airflow_username=e("AIRFLOW_USERNAME", "admin"),
        airflow_password=e("AIRFLOW_PASSWORD", ""),
        airflow_token=e("AIRFLOW_TOKEN", ""),
        training_dag_id=dag,
        approved_dags=_list(e("APPROVED_DAGS", dag)),
        k8s_namespace=ns,
        allowed_namespaces=_list(e("ALLOWED_NAMESPACES", ns)),
        k8s_deployment=e("K8S_DEPLOYMENT", "ml-inference-api"),
        tool_timeout_s=float(e("TOOL_TIMEOUT_S", "20")),
        f1_gate=float(e("F1_GATE", "0.70")),
        model_max_age_days=int(e("MODEL_MAX_AGE_DAYS", "30")),
        dvc_dataset=e("DVC_DATASET", "data/customer_churn.csv"),
        anthropic_api_key=e("ANTHROPIC_API_KEY", ""),
        llm_model=e("LLM_MODEL", "claude-sonnet-4-6"),
        agent_api_key=e("AGENT_API_KEY", ""),
    )
