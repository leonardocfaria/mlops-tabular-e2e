import json

from mlops.data import make_synthetic
from mlops.pipeline import run


def test_pipeline_end_to_end(tmp_path):
    summary = run(tmp_path, n=3000)
    assert summary["promoted"] is True
    assert (tmp_path / "mlruns").is_dir()
    assert (tmp_path / "serving_model" / "MLmodel").is_file()
    ok = json.loads((tmp_path / "reports" / "drift_reference_like.json").read_text())
    shifted = json.loads((tmp_path / "reports" / "drift_shifted.json").read_text())
    assert ok["retrain"] is False and shifted["retrain"] is True
    assert (tmp_path / "reports" / "drift_shifted.md").read_text().startswith("# Drift report")


def test_pipeline_checks_user_batch(tmp_path):
    batch = tmp_path / "incoming.csv"
    make_synthetic(500, seed=9, shift=1.0).drop(columns="income").to_csv(batch, index=False)
    assert run(tmp_path, n=3000, batch=str(batch))["retrain"] == {"incoming": True}
