"""Tests for SCADACleaner."""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing.cleaner import SCADACleaner

CONFIG = {
    "dataset": {
        "v_cut_in_ms": 3.5,
        "v_cut_out_ms": 25.0,
        "rated_power_kw": 2050.0,
        "power_overrating_tol": 0.05,
        "gap_interpolate_min": 30,
        "pitch_curtailment_threshold_deg": 88.0,
        "sampling_interval_min": 10,
    },
    "physics": {"rho_air_kg_m3": 1.225},
    "features": {
        "scada_cols": ["wind_speed_ms", "active_power_kw", "pitch_angle_deg"],
    },
}


def _make_df(n=50):
    idx = pd.date_range("2021-01-01", periods=n, freq="10T")
    return pd.DataFrame({
        "wind_speed_ms":   np.full(n, 10.0),
        "active_power_kw": np.full(n, 1500.0),
        "pitch_angle_deg": np.full(n, 2.0),
        "ambient_temp_c":  np.full(n, 5.0),
    }, index=idx)


@pytest.fixture
def cleaner():
    return SCADACleaner(CONFIG)


def test_removes_below_cut_in(cleaner):
    df = _make_df()
    df.iloc[0, df.columns.get_loc("wind_speed_ms")] = 2.0  # below cut-in
    cleaned = cleaner.clean(df)
    assert len(cleaned) == len(df) - 1


def test_removes_above_cut_out(cleaner):
    df = _make_df()
    df.iloc[0, df.columns.get_loc("wind_speed_ms")] = 30.0  # above cut-out
    cleaned = cleaner.clean(df)
    assert len(cleaned) == len(df) - 1


def test_removes_curtailment(cleaner):
    df = _make_df()
    df.iloc[5, df.columns.get_loc("pitch_angle_deg")] = 90.0  # curtailed
    cleaned = cleaner.clean(df)
    assert len(cleaned) == len(df) - 1


def test_removes_overrating(cleaner):
    df = _make_df()
    df.iloc[3, df.columns.get_loc("active_power_kw")] = 2200.0  # > 2050 × 1.05
    cleaned = cleaner.clean(df)
    assert len(cleaned) == len(df) - 1


def test_normalisation_range(cleaner):
    df = _make_df()
    cleaned = cleaner.clean(df)
    numeric = cleaned.select_dtypes(include=[np.number])
    assert (numeric.min() >= -1e-6).all()
    assert (numeric.max() <= 1 + 1e-6).all()
