"""io.py — Data loading and saving helpers for the Kelmarsh SCADA dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_scada(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["# Date and time"])
    df = df.rename(columns={"# Date and time": "timestamp"})
    return df


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_processed(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)
