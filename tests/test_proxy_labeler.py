"""Tests for ProxyLabeler."""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing.proxy_labeler import ProxyLabeler

CONFIG = {
    "proxy_labeling": {
        "temp_threshold_c":    2.0,
        "pr_mild_threshold":   0.90,
        "pr_severe_threshold": 0.70,
        "majority_vote_steps": 3,
    }
}


@pytest.fixture
def labeler():
    return ProxyLabeler(CONFIG)


def _make_df(temps, prs):
    df = pd.DataFrame({
        "ambient_temp_c": temps,
        "pr_6h":          prs,
    }, index=pd.date_range("2021-01-01", periods=len(temps), freq="10T"))
    return df


def test_all_healthy_warm(labeler):
    df = _make_df([5.0] * 10, [0.95] * 10)
    out = labeler.label(df)
    assert (out["icing_label"] == 0).all()


def test_all_healthy_cold_high_pr(labeler):
    df = _make_df([-3.0] * 10, [0.95] * 10)
    out = labeler.label(df)
    assert (out["icing_label"] == 0).all()


def test_mild_icing(labeler):
    # Single timestep, no smoother effect with window=3 (need ≥3)
    df = _make_df([-5.0] * 5, [0.80] * 5)
    out = labeler.label(df)
    assert (out["icing_label"] == 1).all()


def test_severe_icing(labeler):
    df = _make_df([-5.0] * 5, [0.50] * 5)
    out = labeler.label(df)
    assert (out["icing_label"] == 2).all()


def test_output_column_present(labeler):
    df = _make_df([1.0] * 6, [0.85] * 6)
    out = labeler.label(df)
    assert "icing_label" in out.columns


def test_labels_integer_type(labeler):
    df = _make_df([1.0] * 6, [0.85] * 6)
    out = labeler.label(df)
    assert out["icing_label"].dtype in (np.int32, np.int64, int)
