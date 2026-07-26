"""
metrics.py
==========
All evaluation metrics reported in the paper (Table II and III).

Regression
----------
  MAE   — Mean Absolute Error (kW)
  RMSE  — Root Mean Squared Error (kW)
  R²    — Coefficient of determination
  MAPE  — Mean Absolute Percentage Error (%), samples where P < 50 kW excluded
  PVR   — Physics Violation Rate (%), fraction of predictions above Betz limit

Classification
--------------
  Per-class precision, recall, F1 for icing severity (Healthy / Mild / Severe).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

_CLASS_NAMES = ["Healthy", "Mild", "Severe"]


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE in kW."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE in kW."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, min_power_kw: float = 50.0
) -> float:
    """
    MAPE (%), excluding samples where y_true < min_power_kw.

    The exclusion of near-zero-power samples follows IEC 61400-12-1
    and matches the paper's evaluation protocol.
    """
    mask = y_true >= min_power_kw
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def physics_violation_rate(
    y_pred:      np.ndarray,
    wind_speed:  np.ndarray,
    rho:         float = 1.225,
    rotor_area:  float = 6647.61,
) -> float:
    """
    PVR (%) — fraction of predictions exceeding the Betz limit.

    P_Betz_max(v) = (16/27) · (1/2) · ρ · A · v³  [kW]
    """
    p_betz_max = (16.0 / 27.0) * 0.5 * rho * rotor_area * wind_speed ** 3 / 1000.0
    violations = (y_pred > p_betz_max).sum()
    return float(100.0 * violations / len(y_pred))


def regression_report(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    wind_speed: np.ndarray,
    model_name: str = "Model",
    rho:        float = 1.225,
    rotor_area: float = 6647.61,
) -> dict:
    """
    Compute all five regression metrics and log a formatted table row.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape, pvr.
    """
    results = {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2":   r_squared(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "pvr":  physics_violation_rate(y_pred, wind_speed, rho, rotor_area),
    }

    logger.info(
        "%-20s  MAE=%6.1f kW  RMSE=%6.1f kW  R²=%.3f  MAPE=%5.1f%%  PVR=%4.1f%%",
        model_name,
        results["mae"], results["rmse"],
        results["r2"], results["mape"], results["pvr"],
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def icing_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Per-class precision, recall, F1 for icing severity classification.

    Parameters
    ----------
    y_true, y_pred : int arrays with values 0 (Healthy), 1 (Mild), 2 (Severe).

    Returns
    -------
    dict keyed by class name with sub-dicts {precision, recall, f1}.
    """
    report_str = classification_report(
        y_true, y_pred,
        target_names=_CLASS_NAMES,
        zero_division=0,
    )
    logger.info("\n%s — Icing Classification Report:\n%s", model_name, report_str)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    return {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(_CLASS_NAMES, prec, rec, f1)
    }


def print_results_table(results_dict: dict[str, dict]) -> None:
    """
    Print a formatted results table matching Table II in the paper.

    Parameters
    ----------
    results_dict : {model_name: {mae, rmse, r2, mape, pvr}}
    """
    header = f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'MAPE':>8} {'PVR':>7}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in results_dict.items():
        print(
            f"{name:<22} {m['mae']:>7.1f} {m['rmse']:>7.1f} "
            f"{m['r2']:>7.3f} {m['mape']:>7.1f}% {m['pvr']:>6.1f}%"
        )
    print(sep)
