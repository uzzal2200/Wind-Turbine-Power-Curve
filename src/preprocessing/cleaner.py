"""
cleaner.py
==========
IEC 61400-12-1 compliant SCADA preprocessing for the Kelmarsh Wind Farm dataset.

Steps
-----
1. Remove curtailment periods (pitch saturation + operator-log cross-reference).
2. Discard samples outside the operational envelope (v < v_cut_in or v > v_cut_out).
3. Discard samples exceeding rated capacity by more than the tolerance margin.
4. Linearly interpolate gaps ≤ `gap_interpolate_min` minutes.
5. Exclude longer gaps.
6. Min-max normalise all features to [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class SCADACleaner:
    """
    Apply IEC 61400-12-1 preprocessing to raw Kelmarsh SCADA data.

    Parameters
    ----------
    config : dict
        Top-level config dict (loaded from config.yaml).
    """

    def __init__(self, config: dict) -> None:
        ds = config["dataset"]
        ph = config["physics"]

        self.v_cut_in: float  = ds["v_cut_in_ms"]
        self.v_cut_out: float = ds["v_cut_out_ms"]
        self.rated_power_kw: float = ds["rated_power_kw"]
        self.overrating_tol: float = ds["power_overrating_tol"]
        self.gap_interp_min: int   = ds["gap_interpolate_min"]
        self.pitch_curtail_thresh: float = ds["pitch_curtailment_threshold_deg"]
        self.sampling_min: int     = ds["sampling_interval_min"]
        self.scada_cols: list[str] = config["features"]["scada_cols"]

    # ── Main entry point ───────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw SCADA dataframe with a DatetimeIndex or a 'timestamp' column.

        Returns
        -------
        pd.DataFrame
            Cleaned, normalised dataframe.
        """
        df = self._ensure_datetime_index(df)
        logger.info("Raw samples: %d", len(df))

        df = self._remove_curtailment(df)
        df = self._filter_operational_envelope(df)
        df = self._filter_overrating(df)
        df = self._handle_gaps(df)
        df = self._normalise(df)

        logger.info("Cleaned samples: %d", len(df))
        return df

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
        return df.sort_index()

    def _remove_curtailment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove curtailment via pitch angle saturation heuristic."""
        if "pitch_angle_deg" not in df.columns:
            logger.warning("pitch_angle_deg not found; skipping curtailment removal.")
            return df

        mask_curtailed = df["pitch_angle_deg"].abs() >= self.pitch_curtail_thresh
        n_removed = mask_curtailed.sum()
        logger.info("Curtailment removal: %d samples dropped (pitch ≥ %.1f°)",
                    n_removed, self.pitch_curtail_thresh)
        return df[~mask_curtailed]

    def _filter_operational_envelope(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discard samples outside [v_cut_in, v_cut_out]."""
        if "wind_speed_ms" not in df.columns:
            raise KeyError("Column 'wind_speed_ms' is required.")

        mask = (df["wind_speed_ms"] >= self.v_cut_in) & (df["wind_speed_ms"] <= self.v_cut_out)
        n_removed = (~mask).sum()
        logger.info("Operational envelope filter: %d samples dropped "
                    "(v < %.1f or v > %.1f m/s)", n_removed, self.v_cut_in, self.v_cut_out)
        return df[mask]

    def _filter_overrating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discard samples where active power exceeds rated × (1 + tol)."""
        if "active_power_kw" not in df.columns:
            raise KeyError("Column 'active_power_kw' is required.")

        upper = self.rated_power_kw * (1.0 + self.overrating_tol)
        mask = df["active_power_kw"] <= upper
        n_removed = (~mask).sum()
        logger.info("Over-rating filter: %d samples dropped (P > %.1f kW)", n_removed, upper)
        return df[mask]

    def _handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate short gaps; exclude long gaps."""
        expected_freq = f"{self.sampling_min}T"
        full_index = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
        df = df.reindex(full_index)

        # Identify gap runs
        null_mask = df.isnull().any(axis=1)
        gap_lengths = null_mask.groupby((null_mask != null_mask.shift()).cumsum()).transform("sum")

        interp_limit = self.gap_interp_min // self.sampling_min  # steps
        short_gap = null_mask & (gap_lengths <= interp_limit)
        long_gap  = null_mask & (gap_lengths >  interp_limit)

        df = df.interpolate(method="time", limit=interp_limit)
        df = df[~long_gap]

        logger.info("Gap handling: %d short gaps interpolated; %d long-gap steps excluded",
                    short_gap.sum(), long_gap.sum())
        return df.dropna()

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalise all numeric columns to [0, 1]."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_min = df[numeric_cols].min()
        col_max = df[numeric_cols].max()
        denom = (col_max - col_min).replace(0, 1)  # avoid div-by-zero for constant cols
        df[numeric_cols] = (df[numeric_cols] - col_min) / denom
        logger.info("Min-max normalisation applied to %d columns.", len(numeric_cols))
        return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI convenience
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/config.yaml") -> None:
    import os
    from src.utils.io import load_raw_scada, save_processed

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cleaner = SCADACleaner(config)
    raw_dir = Path(config["paths"]["raw_data_dir"])
    out_dir = Path(config["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(raw_dir.glob("*.csv")):
        logger.info("Processing %s …", csv_file.name)
        df_raw = load_raw_scada(csv_file)
        df_clean = cleaner.clean(df_raw)
        out_path = out_dir / csv_file.name
        save_processed(df_clean, out_path)
        logger.info("Saved %d rows → %s", len(df_clean), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
