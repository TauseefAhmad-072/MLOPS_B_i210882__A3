import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from training import pipeline as p


def _df(n=200, nulls=False):
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    df["target"] = (df["f0"] + 0.5 * df["f1"] + rng.normal(0, 0.3, n) > 0).astype(int)
    if nulls:
        df.iloc[0, 0] = None
    return df


def test_validation_rejects_null_data_and_short_data(tmp_path):
    f = tmp_path / "d.csv"
    _df(nulls=True).to_csv(f, index=False)
    with pytest.raises(p.DataValidationError, match="null"):
        p.validate_dataset(f)
    _df(20).to_csv(f, index=False)
    with pytest.raises(p.DataValidationError, match="too few"):
        p.validate_dataset(f)


def test_failed_quality_gate_blocks_promotion(env):
    from mlflow import MlflowClient
    c = MlflowClient()
    exp = c.create_experiment("churn-classifier")
    r = c.create_run(exp)
    c.log_metric(r.info.run_id, "f1", 0.2)
    with pytest.raises(p.QualityGateError):
        p.quality_gate(r.info.run_id)
    assert c.get_run(r.info.run_id).data.tags["quality_gate"] == "failed"
    with pytest.raises(p.QualityGateError, match="refusing to promote"):
        p.promote(r.info.run_id)


def test_inference_api_ready_only_with_model(monkeypatch, tmp_path):
    import api.main as m
    monkeypatch.setattr(m, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(m.app) as c:
        assert c.get("/health").status_code == 200       # liveness unaffected
        assert c.get("/ready").status_code == 503        # readiness fails -> pod removed from Service
        assert c.post("/predict", json={"features": [1.0]}).status_code == 503
    from sklearn.ensemble import RandomForestClassifier
    df = _df()
    mp = tmp_path / "m.joblib"
    joblib.dump(RandomForestClassifier(n_estimators=5).fit(df.drop(columns="target"), df["target"]), mp)
    monkeypatch.setattr(m, "MODEL_PATH", str(mp))
    with TestClient(m.app) as c:
        assert c.get("/ready").status_code == 200
        ok = c.post("/predict", json={"features": df.drop(columns="target").iloc[0].tolist()})
        assert ok.status_code == 200 and ok.json()["prediction"] in (0, 1)
        assert c.post("/predict", json={"features": [1.0]}).status_code == 422
        assert c.get("/version").json()["app_version"]
