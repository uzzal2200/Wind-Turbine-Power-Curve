"""
dataset.py
==========
PyTorch Dataset wrappers for DA-PINN training.

Classes
-------
SCADADataset     — Point-wise dataset for MLP / PINN training.
SequenceDataset  — Sliding-window dataset for LSTM baseline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SCADADataset(Dataset):
    """
    Point-wise PyTorch dataset for MLP-based models.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix (already normalised).
    y : np.ndarray, shape (N,)
        Target power values in kW (physical scale).
    wind_speed_col : int
        Index of the wind_speed_ms column in X (used to return raw wind speed
        for physics loss calculations). Default 0.
    wind_speed_raw : np.ndarray | None
        Physical-scale wind speed array (N,). If provided, this overrides
        the in-X extraction, which is useful when X is already normalised
        and physical values need to be preserved separately.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wind_speed_col: int = 0,
        wind_speed_raw: np.ndarray | None = None,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if wind_speed_raw is not None:
            self.wind_speed_raw = torch.tensor(wind_speed_raw, dtype=torch.float32)
        else:
            # Fall back: extract from X (may be normalised)
            self.wind_speed_raw = self.X[:, wind_speed_col].clone()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.wind_speed_raw[idx]


class SequenceDataset(Dataset):
    """
    Sliding-window dataset for the LSTM baseline.

    Parameters
    ----------
    X          : np.ndarray, shape (N, F)
    y          : np.ndarray, shape (N,)
    seq_length : int   Number of time steps in each window.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 12) -> None:
        if len(X) <= seq_length:
            raise ValueError(
                f"Dataset length ({len(X)}) must be greater than seq_length ({seq_length})."
            )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.y) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_length]          # (seq_len, F)
        y_val = self.y[idx + self.seq_length - 1]            # label at last step
        return x_seq, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Splitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df,
    train_years: list[int],
    val_years:   list[int],
    test_years:  list[int],
) -> tuple:
    """
    Split a DatetimeIndex-ed DataFrame into train / val / test subsets
    based on calendar year, preventing data leakage.

    Returns
    -------
    (df_train, df_val, df_test) — three DataFrames.
    """
    import pandas as pd

    train_mask = df.index.year.isin(train_years)
    val_mask   = df.index.year.isin(val_years)
    test_mask  = df.index.year.isin(test_years)

    return df[train_mask], df[val_mask], df[test_mask]
