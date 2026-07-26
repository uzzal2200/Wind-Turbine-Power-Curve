"""Tests for CUSUMMonitor."""

import numpy as np
import pytest
from src.monitoring.cusum import CUSUMMonitor

CONFIG = {
    "cusum":   {"k_factor": 0.5, "h_factor": 4.0, "sigma0": 0.0625},
    "physics": {"rho_air_kg_m3": 1.225, "rotor_area_m2": 6647.61},
}


@pytest.fixture
def monitor():
    return CUSUMMonitor(CONFIG)


def test_no_alert_for_healthy_residual(monitor):
    """Constant small residual should never trigger an alert."""
    n  = 200
    v  = np.full(n, 10.0)
    # R ≈ 0.03 (healthy range); CUSUM stays near 0
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 10.0 ** 3 / 1000.0
    p_hat  = np.full(n, p_betz * 0.4)
    p_act  = p_hat * (1 - 0.03)  # ~3 % residual
    monitor.calibrate(p_hat[:50], p_act[:50], v[:50])
    result = monitor.run(p_hat, p_act, v)
    assert result.alert_mask.sum() == 0


def test_alert_triggered_for_large_residual(monitor):
    """Sudden large residual (icing onset) should trigger CUSUM alert."""
    n  = 200
    v  = np.full(n, 10.0)
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 10.0 ** 3 / 1000.0
    p_hat  = np.full(n, p_betz * 0.4)
    p_act  = p_hat.copy()
    # Simulate icing drop at step 100: actual power drops 50 %
    p_act[100:] = p_hat[100:] * 0.5
    monitor.calibrate(p_hat[:50], p_act[:50], v[:50])
    result = monitor.run(p_hat, p_act, v)
    assert result.alert_mask[120:].any(), "Expected alert after icing onset"


def test_residuals_non_negative(monitor):
    n = 50
    v = np.full(n, 8.0)
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 8.0 ** 3 / 1000.0
    p_hat  = np.full(n, p_betz * 0.4)
    p_act  = p_hat * 0.9
    monitor.calibrate(p_hat, p_act, v)
    result = monitor.run(p_hat, p_act, v)
    assert (result.residuals >= 0).all()


def test_cusum_non_negative(monitor):
    n = 50
    v = np.full(n, 8.0)
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 8.0 ** 3 / 1000.0
    p_hat  = np.full(n, p_betz * 0.4)
    p_act  = p_hat * 0.9
    monitor.calibrate(p_hat, p_act, v)
    result = monitor.run(p_hat, p_act, v)
    assert (result.cusum_vals >= 0).all()
