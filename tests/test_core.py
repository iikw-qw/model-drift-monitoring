import numpy as np
import pandas as pd
import pytest

from src import core, data


def test_psi_of_identical_samples_is_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert core.psi_numeric(a, a) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 8000)
    small = core.psi_numeric(ref, rng.normal(0.1, 1, 8000))
    large = core.psi_numeric(ref, rng.normal(1.0, 1, 8000))
    assert small < 0.1 < 0.25 < large


def test_psi_categorical():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    same = pd.Series(["a"] * 500 + ["b"] * 500)
    shifted = pd.Series(["a"] * 900 + ["b"] * 100)
    assert core.psi_categorical(ref, same) == pytest.approx(0.0, abs=1e-9)
    assert core.psi_categorical(ref, shifted) > 0.25


def test_status_thresholds():
    assert core.psi_status(0.05) == "stable"
    assert core.psi_status(0.15) == "warning"
    assert core.psi_status(0.4) == "alert"


def test_monitor_detects_injected_drift():
    stream = data.generate_stream(weeks=18, per_week=2000, drift_start=8, seed=3)
    report, base_auc = core.monitor(stream)
    early = report.loc[6:8, "max_psi"].max()
    late = report.loc[16:, "max_psi"].max()
    assert late > early
    assert core.first_alert_week(report, "max_psi", 0.25, after=6) is not None
    assert report.loc[16:, "auc"].mean() < base_auc
