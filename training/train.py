"""Local CLI: run the whole pipeline without Airflow, or export a model for the Docker image."""
import argparse
import json
import sys
from pathlib import Path

import joblib

from training import pipeline as p

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=None)
ap.add_argument("--n-estimators", type=int, default=100)
ap.add_argument("--max-depth", type=int, default=None)
ap.add_argument("--label-noise", type=float, default=0.0)
ap.add_argument("--export-only", action="store_true", help="train + save models/model.joblib, no MLflow")
a = ap.parse_args()
hp = {"n_estimators": a.n_estimators, "max_depth": a.max_depth, "label_noise": a.label_noise}

if a.export_only:
    from agent.config import get_settings
    s = get_settings()
    path = s.repo_root / (a.dataset or s.dvc_dataset)
    full = {**p.DEFAULT_HP, **hp}
    split = p.preprocess(path, "/tmp/export_work", full)
    m = p.train_model(split["train"], "/tmp/export_work", full)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(joblib.load(m), "models/model.joblib")
    print("exported models/model.joblib")
    sys.exit(0)

try:
    print(json.dumps(p.run_all(a.dataset, hp), indent=2))
except (p.DataValidationError, p.QualityGateError) as e:
    print(f"PIPELINE STOPPED: {e}", file=sys.stderr)
    sys.exit(2)
