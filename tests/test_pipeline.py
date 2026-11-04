import json

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
