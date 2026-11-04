"""Adult-like census data: seeded synthetic generator, loader and schema validation."""
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "workclass": ["Private", "Self-emp", "Gov", "Without-pay"],
    "marital_status": ["Married", "Never-married", "Divorced", "Widowed"],
    "occupation": ["Exec", "Prof", "Craft", "Sales", "Service", "Clerical"],
    "sex": ["Male", "Female"],
}
RANGES = {"age": (17, 90), "education_num": (1, 16), "hours_per_week": (1, 99), "capital_gain": (0, 99999)}
NUMERIC = list(RANGES)
CATEGORICAL = list(CATEGORIES)
TARGET = "income"  # 1 = ">50K"


def make_synthetic(n: int, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Draw n rows. `shift` > 0 moves age/hours up and changes the occupation mix (covariate drift)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38 + 12 * shift, 13, n), 17, 90).round()
    edu = np.clip(rng.normal(10, 2.5, n), 1, 16).round()
    hours = np.clip(rng.normal(40 + 8 * shift, 11, n), 1, 99).round()
    gain = np.where(rng.random(n) < 0.08, rng.exponential(8000, n), 0).clip(0, 99999).round()
    occ_p = np.array([0.15, 0.15, 0.2, 0.15, 0.2, 0.15]) + shift * np.array([0.2, 0, -0.1, 0, 0, -0.1])
    df = pd.DataFrame({
        "age": age,
        "education_num": edu,
        "hours_per_week": hours,
        "capital_gain": gain,
        "workclass": rng.choice(CATEGORIES["workclass"], n, p=[0.7, 0.12, 0.15, 0.03]),
        "marital_status": rng.choice(CATEGORIES["marital_status"], n, p=[0.47, 0.33, 0.14, 0.06]),
        "occupation": rng.choice(CATEGORIES["occupation"], n, p=occ_p / occ_p.sum()),
        "sex": rng.choice(CATEGORIES["sex"], n, p=[0.67, 0.33]),
    })
    logit = (
        -9.5 + 0.04 * df.age + 0.38 * df.education_num + 0.03 * df.hours_per_week
        + 0.0003 * df.capital_gain + 1.4 * (df.marital_status == "Married")
        + 0.7 * df.occupation.isin(["Exec", "Prof"]) + 0.3 * (df.sex == "Male")
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def load(path: str | Path | None = None, n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Load a real CSV with the SCHEMA columns if given and present, else the synthetic data."""
    if path and Path(path).exists():
        return pd.read_csv(path)
    return make_synthetic(n, seed)


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Raise ValueError listing every schema problem found."""
    errors = []
    cols = NUMERIC + CATEGORICAL + ([TARGET] if require_target else [])
    errors += [f"missing column {c}" for c in cols if c not in df]
    for c in (c for c in cols if c in df):
        if df[c].isna().any():
            errors.append(f"{c}: {int(df[c].isna().sum())} nulls")
    for c, (lo, hi) in RANGES.items():
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"{c}: not numeric")
        elif c in df and ((df[c] < lo) | (df[c] > hi)).any():
            errors.append(f"{c}: values out of range [{lo}, {hi}]")
    for c, allowed in CATEGORIES.items():
        if c in df:
            bad = set(df[c].dropna().unique()) - set(allowed)
            if bad:
                errors.append(f"{c}: unexpected categories {sorted(map(str, bad))}")
    if require_target and TARGET in df and not set(df[TARGET].dropna().unique()) <= {0, 1}:
        errors.append(f"{TARGET}: must be 0/1")
    if errors:
        raise ValueError("; ".join(errors))
