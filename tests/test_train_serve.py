import pytest
from fastapi.testclient import TestClient

from mlops.data import make_synthetic
from mlops.train import MODEL_NAME, export_production, promote, train_and_register


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow")
    uri = (root / "mlruns").as_uri()
    df = make_synthetic(3000, seed=0)
    run_id, version, auc = train_and_register(df, uri)
    return root, uri, run_id, version, auc


def test_train_registers_and_promotes(registry):
    from mlflow import MlflowClient

    root, uri, run_id, version, auc = registry
    assert auc > 0.8
    assert promote(uri, version, auc) is True
    client = MlflowClient(uri, registry_uri=uri)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "production").version) == version
    assert client.get_run(run_id).data.metrics["val_auc"] == pytest.approx(auc)


def test_promotion_rejects_worse_model(registry):
    root, uri, run_id, version, auc = registry
    promote(uri, version, auc)
    assert promote(uri, version, auc - 0.05) is False  # not better than current production
    assert promote(uri, version, 0.5) is False  # below floor


def test_serve_predict(registry, monkeypatch):
    root, uri, *_ = registry
    model_dir = export_production(uri, root / "serving_model")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from mlops.serve import app

    rows = make_synthetic(5, seed=3).drop(columns="income").to_dict("records")
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"rows": rows})
        assert r.status_code == 200
        out = r.json()["predictions"]
        assert len(out) == 5 and all(0 <= p["probability"] <= 1 for p in out)
        assert client.post("/predict", json={"rows": [{"age": 30}]}).status_code == 422
        bad = dict(rows[0], sex="robot")
        assert client.post("/predict", json={"rows": [bad]}).status_code == 422
