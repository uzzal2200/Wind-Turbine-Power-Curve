"""Tests for PhysicsLoss — core contribution of the paper."""

import pytest
import torch
from src.models.physics_loss import PhysicsLoss


@pytest.fixture
def loss_fn():
    return PhysicsLoss(
        lambda_betz=0.8, lambda_cp=0.5, lambda_smooth=0.3,
        rho=1.225, rotor_area=6647.61,
    )


def test_data_loss_perfect_prediction(loss_fn):
    p = torch.tensor([1000.0, 1500.0, 2000.0])
    v = torch.tensor([8.0, 10.0, 12.0])
    result = loss_fn(p, p, v)
    assert result["data"].item() == pytest.approx(0.0, abs=1e-5)


def test_betz_loss_zero_when_compliant(loss_fn):
    v = torch.tensor([10.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 10.0 ** 3 / 1000.0
    p_safe = torch.tensor([p_betz * 0.9])  # 10 % below Betz → no violation
    result = loss_fn(p_safe, p_safe, v)
    assert result["betz"].item() == pytest.approx(0.0, abs=1e-5)


def test_betz_loss_positive_when_violated(loss_fn):
    v = torch.tensor([10.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * 10.0 ** 3 / 1000.0
    p_viol = torch.tensor([p_betz * 1.1])  # 10 % above Betz → should penalise
    result = loss_fn(p_viol, p_viol, v)
    assert result["betz"].item() > 0.0


def test_cp_loss_zero_when_cp_in_range(loss_fn):
    v = torch.tensor([8.0])
    # Cp ≈ 0.4 (within (0, 0.593])
    p_avail = 0.5 * 1.225 * 6647.61 * 8.0 ** 3 / 1000.0
    p = torch.tensor([0.4 * p_avail])
    result = loss_fn(p, p, v)
    assert result["cp"].item() == pytest.approx(0.0, abs=1e-4)


def test_total_loss_non_negative(loss_fn):
    v = torch.rand(32) * 20 + 3.0
    p = torch.rand(32) * 2000.0
    result = loss_fn(p, p * 0.95, v)
    assert result["total"].item() >= 0.0


def test_pvr_all_compliant(loss_fn):
    v = torch.tensor([8.0, 10.0, 12.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * v ** 3 / 1000.0
    p_safe = p_betz * 0.5
    pvr = loss_fn.physics_violation_rate(p_safe, v)
    assert pvr == pytest.approx(0.0, abs=1e-6)


def test_pvr_all_violated(loss_fn):
    v = torch.tensor([8.0, 10.0, 12.0])
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * v ** 3 / 1000.0
    p_viol = p_betz * 2.0
    pvr = loss_fn.physics_violation_rate(p_viol, v)
    assert pvr == pytest.approx(1.0, abs=1e-6)
