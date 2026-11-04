import numpy as np
import pytest

from mlops.data import make_synthetic, validate
from mlops.drift import drift_report, psi


def test_synthetic_is_deterministic_and_valid():
    a, b = make_synthetic(500, seed=1), make_synthetic(500, seed=1)
    assert a.equals(b)
    validate(a)
    assert 0.1 < a["income"].mean() < 0.4


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.drop(columns=["age"]), "missing column age"),
        (lambda d: d.assign(sex="robot"), "unexpected categories"),
        (lambda d: d.assign(age=-3), "out of range"),
        (lambda d: d.assign(hours_per_week=np.nan), "nulls"),
    ],
)
def test_validate_rejects_bad_data(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        validate(mutate(make_synthetic(200, seed=0)))


def test_psi_near_zero_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_constant_reference_still_detects_change():
    assert psi(np.zeros(100), np.ones(100)) > 0.2


def test_drift_report_validates_incoming_batch():
    with pytest.raises(ValueError, match="missing column age"):
        drift_report(make_synthetic(100), make_synthetic(100).drop(columns="age"))


def test_drift_report_flags_shifted_batch():
    ref = make_synthetic(4000, seed=0)
    same = drift_report(ref, make_synthetic(2000, seed=1))
    shifted = drift_report(ref, make_synthetic(2000, seed=2, shift=1.0))
    assert not same["retrain"]
    assert shifted["retrain"]
    assert shifted["features"]["age"]["psi"] > 0.2
    assert shifted["features"]["age"]["ks_pvalue"] < 0.01
