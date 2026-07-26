"""Tests for evaluation metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    mean_absolute_error, root_mean_squared_error,
    r_squared, mean_absolute_percentage_error,
    physics_violation_rate,
)


def test_mae_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_rmse_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert root_mean_squared_error(y, y) == pytest.approx(0.0)


def test_r2_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction():
    y = np.array([100.0, 200.0, 300.0])
    y_pred = np.full_like(y, y.mean())
    assert r_squared(y, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_mape_excludes_low_power():
    y_true = np.array([10.0, 100.0, 200.0])
    y_pred = np.array([20.0, 110.0, 180.0])
    mape = mean_absolute_percentage_error(y_true, y_pred, min_power_kw=50.0)
    # Only samples with y_true >= 50 included: [100, 200]
    expected = 100 * np.mean([10 / 100, 20 / 200])
    assert mape == pytest.approx(expected, rel=1e-4)


def test_pvr_no_violations():
    v = np.array([8.0, 10.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * v ** 3 / 1000.0
    y_pred = p_betz * 0.5
    pvr = physics_violation_rate(y_pred, v)
    assert pvr == pytest.approx(0.0)


def test_pvr_full_violation():
    v = np.array([8.0, 10.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * v ** 3 / 1000.0
    y_pred = p_betz * 2.0
    pvr = physics_violation_rate(y_pred, v)
    assert pvr == pytest.approx(100.0)
