"""Drift monitor: PSI + two-sample KS per feature, with a retrain flag."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .data import CATEGORICAL, NUMERIC

PSI_RETRAIN = 0.2  # common rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant


def _psi_from_fractions(p, q, eps=1e-4):
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(ref, cur, bins: int = 10) -> float:
    """PSI of a numeric feature, bins from reference quantiles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    return _psi_from_fractions(p, q)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    q = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_fractions(p, q)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    features = {}
    for c in NUMERIC:
        ks = ks_2samp(ref[c], cur[c])
        features[c] = {"psi": psi(ref[c], cur[c]), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}
    for c in CATEGORICAL:
        features[c] = {"psi": psi_categorical(ref[c], cur[c])}
    drifted = sorted(c for c, f in features.items() if f["psi"] > PSI_RETRAIN)
    return {"n_ref": len(ref), "n_cur": len(cur), "psi_threshold": PSI_RETRAIN,
            "features": features, "drifted": drifted, "retrain": bool(drifted)}


def write_report(report: dict, out_dir: str | Path, name: str = "drift_report") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2))
    lines = [f"# Drift report ({report['n_cur']} rows vs {report['n_ref']} reference)", "",
             f"**Retrain: {'YES' if report['retrain'] else 'no'}** (PSI > {report['psi_threshold']}: "
             f"{', '.join(report['drifted']) or 'none'})", "",
             "| feature | PSI | KS stat | KS p-value |", "|---|---|---|---|"]
    for c, f in report["features"].items():
        ks = f"{f['ks_stat']:.3f} | {f['ks_pvalue']:.2g}" if "ks_stat" in f else "- | -"
        lines.append(f"| {c} | {f['psi']:.3f} | {ks} |")
    md = out / f"{name}.md"
    md.write_text("\n".join(lines) + "\n")
    return md
