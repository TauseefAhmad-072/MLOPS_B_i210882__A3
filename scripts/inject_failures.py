"""Deliberate failure injection.
  corrupt-dataset   F1: dataset with nulls  -> validate_dataset task fails, downstream skipped
  low-quality-model F2: label noise + tiny model -> quality_gate task fails, promote never runs
  bad-model-path    F3 (k8s): ConfigMap points to a missing model -> pods Running but never Ready
  restore-k8s       undo F3
Add --local to run the real pipeline code without Airflow; otherwise the failure is injected through Airflow/kubectl."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.config import get_settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["corrupt-dataset", "low-quality-model", "bad-model-path", "restore-k8s"])
ap.add_argument("--local", action="store_true")
a = ap.parse_args()
root = get_settings().repo_root

if a.mode == "corrupt-dataset":
    import pandas as pd
    df = pd.read_csv(root / "data/customer_churn.csv")
    df.iloc[::7, 3] = None  # inject nulls
    df.to_csv(root / "data/corrupted.csv", index=False)
    conf = {"dataset_path": "data/corrupted.csv"}
elif a.mode == "low-quality-model":
    conf = {"hyperparams": {"n_estimators": 1, "max_depth": 1, "label_noise": 0.45}}

if a.mode in ("corrupt-dataset", "low-quality-model"):
    if a.local:
        from training import pipeline as p
        out = root / "docs/evidence"
        out.mkdir(parents=True, exist_ok=True)
        try:
            msg = json.dumps(p.run_all(conf.get("dataset_path"), conf.get("hyperparams")), indent=2)
        except (p.DataValidationError, p.QualityGateError) as e:
            msg = f"EXPECTED FAILURE ({type(e).__name__}): {e}"
        print(msg)
        (out / f"failure_{a.mode}.txt").write_text(msg + "\n")
    else:
        from agent.tools import execute_tool
        r = execute_tool("airflow_trigger_training_dag", {"conf": conf, "confirm": True})
        print(r.model_dump_json(indent=2))
        print("Now ask the agent: 'Why did the training pipeline fail?'")
elif a.mode == "bad-model-path":
    subprocess.run(["kubectl", "apply", "-f", "kubernetes/failures/bad-model-path.yaml"], cwd=root, check=True)
    subprocess.run(["kubectl", "-n", "mlops-agent", "rollout", "restart", "deploy/ml-inference-api"], cwd=root, check=True)
    print("Injected. Ask the agent: 'Why is my prediction API unavailable?'")
else:
    subprocess.run(["kubectl", "apply", "-f", "kubernetes/configmap.yaml"], cwd=root, check=True)
    subprocess.run(["kubectl", "-n", "mlops-agent", "rollout", "restart", "deploy/ml-inference-api"], cwd=root, check=True)
