"""One command: python -m mlops.pipeline [--out DIR] [--data data/raw/adult.csv]"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load, make_synthetic, validate
from .drift import drift_report, write_report
from .train import export_production, promote, train_and_register


def run(out: str | Path = ".", data: str | None = None, n: int = 20000, batch: str | None = None) -> dict:
    out = Path(out).resolve()
    uri = (out / "mlruns").as_uri()

    df = load(data, n=n, seed=0)
    validate(df)
    run_id, version, auc = train_and_register(df, uri)
    promoted = promote(uri, version, auc)
    model_dir = export_production(uri, out / "serving_model")

    if batch:  # a real incoming batch (CSV, same schema, target optional)
        batches = {Path(batch).stem: pd.read_csv(batch)}
    else:  # demo: one batch like the training data, one shifted
        batches = {"reference_like": make_synthetic(2000, seed=1), "shifted": make_synthetic(2000, seed=2, shift=1.0)}
    retrain = {}
    for name, rows in batches.items():
        report = drift_report(df, rows)
        write_report(report, out / "reports", f"drift_{name}")
        retrain[name] = report["retrain"]

    summary = {"run_id": run_id, "version": version, "val_auc": round(auc, 4), "promoted": promoted,
               "serving_model": str(model_dir), "retrain": retrain}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=".")
    p.add_argument("--data", default="data/raw/adult.csv", help="used if the file exists, else synthetic data")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--batch", help="CSV of incoming rows to check for drift (default: demo batches)")
    a = p.parse_args()
    run(a.out, a.data, a.rows, a.batch)
