"""
feature_engineer.py
===================
Derives the three health-vector features required by DA-PINN:

  1. delta_t_blade    — Blade temperature differential (max - min across blades).
  2. ai_pitch         — Pitch asymmetry index (std of per-blade pitch angles).
  3. pr_6h            — 6-hour rolling median performance ratio.

Also computes the scalar performance ratio PR(t) = P_actual / P_expected(v),
using the MM92 manufacturer power curve for P_expected.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MM92 manufacturer power curve (lookup table, kW vs wind speed m/s)
# Source: Senvion MM92 specification sheet
# ─────────────────────────────────────────────────────────────────────────────

_MM92_CURVE: dict[float, float] = {
    3.0:  10.0,  3.5:  36.0,  4.0:  80.0,  4.5: 140.0,
    5.0: 210.0,  5.5: 300.0,  6.0: 410.0,  6.5: 540.0,
    7.0: 700.0,  7.5: 870.0,  8.0:1050.0,  8.5:1230.0,
    9.0:1400.0,  9.5:1570.0, 10.0:1720.0, 10.5:1840.0,
   11.0:1940.0, 11.5:1990.0, 12.0:2030.0, 12.5:2050.0,
   13.0:2050.0, 25.0:2050.0,
}

_CURVE_SPEEDS = np.array(sorted(_MM92_CURVE.keys()))
_CURVE_POWERS = np.array([_MM92_CURVE[v] for v in _CURVE_SPEEDS])


def manufacturer_power_kw(wind_speed_ms: np.ndarray) -> np.ndarray:
    """Interpolate MM92 manufacturer power curve at given wind speeds (kW)."""
    return np.interp(wind_speed_ms, _CURVE_SPEEDS, _CURVE_POWERS)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Add health-vector features and performance ratio to a cleaned SCADA DataFrame.

    Parameters
    ----------
    config : dict
        Top-level config dict.
    """

    def __init__(self, config: dict) -> None:
        feat = config["features"]
        self.pr_window: int = feat["pr_rolling_window_steps"]

    # ── Public API ─────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append all engineered features in-place.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned, normalised SCADA DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame extended with new feature columns.
        """
        df = df.copy()
        df = self._add_delta_t_blade(df)
        df = self._add_ai_pitch(df)
        df = self._add_performance_ratio(df)
        df = self._add_pr_6h(df)
        logger.info("Feature engineering complete. New columns: delta_t_blade, "
                    "ai_pitch, pr, pr_6h.")
        return df

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _add_delta_t_blade(df: pd.DataFrame) -> pd.DataFrame:
        """
        ΔT_blade = max(blade_temp_*) − min(blade_temp_*) across the three blades.
        Falls back to 0.0 if blade temperature columns are missing.
        """
        blade_cols = [c for c in df.columns if c.startswith("blade_temp_")]
        if len(blade_cols) < 2:
            logger.warning("Fewer than 2 blade_temp_* columns found; delta_t_blade set to 0.")
            df["delta_t_blade"] = 0.0
        else:
            df["delta_t_blade"] = df[blade_cols].max(axis=1) - df[blade_cols].min(axis=1)
        return df

    @staticmethod
    def _add_ai_pitch(df: pd.DataFrame) -> pd.DataFrame:
        """
        AI_pitch = standard deviation of per-blade pitch angles.
        Uses a single pitch_angle_deg column with std=0 if only one is available.
        """
        pitch_cols = [c for c in df.columns if "pitch" in c.lower()]
        if len(pitch_cols) == 0:
            logger.warning("No pitch column found; ai_pitch set to 0.")
            df["ai_pitch"] = 0.0
        elif len(pitch_cols) == 1:
            df["ai_pitch"] = 0.0          # Single pitch sensor → no asymmetry signal
        else:
            df["ai_pitch"] = df[pitch_cols].std(axis=1)
        return df

    @staticmethod
    def _add_performance_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """PR(t) = P_actual(t) / P_expected(v(t))."""
        if "wind_speed_ms" not in df.columns or "active_power_kw" not in df.columns:
            raise KeyError("wind_speed_ms and active_power_kw are required for PR.")

        p_exp = manufacturer_power_kw(df["wind_speed_ms"].to_numpy())
        # Avoid division by very small expected power (below cut-in)
        p_exp_safe = np.where(p_exp < 1.0, 1.0, p_exp)
        df["pr"] = (df["active_power_kw"].to_numpy() / p_exp_safe).clip(0.0, 1.5)
        return df

    def _add_pr_6h(self, df: pd.DataFrame) -> pd.DataFrame:
        """PR_6h = rolling median of PR over `pr_window` steps (60 min default)."""
        if "pr" not in df.columns:
            raise KeyError("PR column must be computed before PR_6h.")
        df["pr_6h"] = (
            df["pr"]
            .rolling(window=self.pr_window, min_periods=1, center=False)
            .median()
        )
        return df
