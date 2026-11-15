"""Pipeline stages shared by the Airflow DAG and the local CLI.
validate -> preprocess -> train -> evaluate/log to MLflow -> quality gate -> promote."""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agent.config import get_settings

TARGET = "target"
MODEL_NAME = "churn-classifier"
DEFAULT_HP = {"n_estimators": 100, "max_depth": None, "seed": 42, "test_size": 0.25, "label_noise": 0.0}


class DataValidationError(ValueError):
    pass


class QualityGateError(ValueError):
    pass


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def file_md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - DVC uses md5 content hashes
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_provenance(dataset: str, root: Path | None = None) -> dict:
    """Recorded DVC provenance for a dataset path (md5 from .dvc, git commit/tag of that .dvc file)."""
    root = root or get_settings().repo_root
    dvc_file = root / f"{dataset}.dvc"
    recorded_md5 = None
    if dvc_file.exists():
        recorded_md5 = yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    data_file = root / dataset
    actual = file_md5(data_file) if data_file.exists() else None
    commit = _git("log", "-1", "--format=%H", "--", f"{dataset}.dvc", cwd=root)
    tag = None
    if commit:
        tag = _git("tag", "--points-at", commit, "--list", "data-v*", cwd=root)
        tag = tag.splitlines()[0] if tag else None
    return {"dvc.dataset_path": dataset, "dvc.dataset_md5": recorded_md5 or "unknown",
            "dvc.git_commit": commit or "unknown", "dvc.dataset_version": tag or "untagged",
            "dvc.data_matches_recorded": str(bool(actual and actual == recorded_md5)).lower()}


def validate_dataset(path: str | Path, min_rows: int = 100) -> dict:
    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"dataset not found: {p}")
    df = pd.read_csv(p)
    if TARGET not in df.columns:
        raise DataValidationError(f"missing target column '{TARGET}'")
    if len(df) < min_rows:
        raise DataValidationError(f"too few rows: {len(df)} < {min_rows}")
    nulls = int(df.isna().sum().sum())
    if nulls:
        raise DataValidationError(f"dataset contains {nulls} null values")
    if df[TARGET].nunique() < 2:
        raise DataValidationError("target has fewer than 2 classes")
    return {"rows": len(df), "columns": df.shape[1], "path": str(p)}


def preprocess(path: str | Path, workdir: str | Path, hp: dict) -> dict:
    df = pd.read_csv(path)
    train, test = train_test_split(df, test_size=hp["test_size"], random_state=hp["seed"], stratify=df[TARGET])
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    train.to_csv(work / "train.csv", index=False)
    test.to_csv(work / "test.csv", index=False)
    return {"train": str(work / "train.csv"), "test": str(work / "test.csv"), "n_train": len(train), "n_test": len(test)}


def train_model(train_csv: str, workdir: str | Path, hp: dict) -> str:
    df = pd.read_csv(train_csv)
    y = df[TARGET].copy()
    if hp.get("label_noise", 0) > 0:  # used by the 'low-quality model' failure injection
        flip = y.sample(frac=hp["label_noise"], random_state=hp["seed"]).index
        y.loc[flip] = 1 - y.loc[flip]
    model = RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], random_state=hp["seed"])
    model.fit(df.drop(columns=[TARGET]), y)
    out = Path(workdir) / "model.joblib"
    joblib.dump(model, out)
    return str(out)


def evaluate_and_log(test_csv: str, model_path: str, hp: dict, dataset: str, split: dict,
                     tags: dict | None = None) -> dict:
    from mlflow import MlflowClient

    df = pd.read_csv(test_csv)
    model = joblib.load(model_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
    metrics = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
               "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}

    s = get_settings()
    client = MlflowClient(tracking_uri=s.mlflow_tracking_uri)
    exp = client.get_experiment_by_name(s.mlflow_experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(s.mlflow_experiment)
    run = client.create_run(exp_id, run_name=f"rf-{int(time.time())}")
    rid = run.info.run_id
    for k, v in {**hp, "n_train": split["n_train"], "n_test": split["n_test"], "model_type": "RandomForest"}.items():
        client.log_param(rid, k, v)
    for k, v in metrics.items():
        client.log_metric(rid, k, float(v))
    for k, v in {**dataset_provenance(dataset), "quality_gate": "pending", **(tags or {})}.items():
        client.set_tag(rid, k, v)
    client.log_artifact(rid, model_path)
    client.set_terminated(rid, "FINISHED")
    return {"run_id": rid, "experiment_id": exp_id, "metrics": {k: round(float(v), 4) for k, v in metrics.items()}}


def quality_gate(run_id: str, threshold: float | None = None) -> dict:
    from mlflow import MlflowClient

    s = get_settings()
    threshold = s.f1_gate if threshold is None else threshold
    client = MlflowClient(tracking_uri=s.mlflow_tracking_uri)
    f1 = client.get_run(run_id).data.metrics.get("f1")
    if f1 is None or f1 < threshold:
        client.set_tag(run_id, "quality_gate", "failed")
        raise QualityGateError(f"quality gate FAILED: f1={f1} < {threshold}")
    client.set_tag(run_id, "quality_gate", "passed")
    return {"run_id": run_id, "f1": f1, "threshold": threshold, "passed": True}


def promote(run_id: str) -> dict:
    """Register the model version. Only ever called after the quality gate passed."""
    from mlflow import MlflowClient

    s = get_settings()
    client = MlflowClient(tracking_uri=s.mlflow_tracking_uri)
    if client.get_run(run_id).data.tags.get("quality_gate") != "passed":
        raise QualityGateError("refusing to promote a run that has not passed the quality gate")
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:  # noqa: BLE001 - already exists
        pass
    run = client.get_run(run_id)
    mv = client.create_model_version(MODEL_NAME, f"{run.info.artifact_uri}/model.joblib", run_id)
    return {"model": MODEL_NAME, "version": mv.version}


def run_all(dataset: str | None = None, hp: dict | None = None, workdir: str | None = None) -> dict:
    s = get_settings()
    dataset = dataset or s.dvc_dataset
    hp = {**DEFAULT_HP, **(hp or {})}
    work = Path(workdir or s.repo_root / "artifacts_work")
    path = Path(dataset) if Path(dataset).is_absolute() else s.repo_root / dataset
    v = validate_dataset(path)
    split = preprocess(path, work, hp)
    model = train_model(split["train"], work, hp)
    res = evaluate_and_log(split["test"], model, hp, dataset, split)
    gate = quality_gate(res["run_id"])
    return {"validation": v, **res, "gate": gate, "promotion": promote(res["run_id"])}
