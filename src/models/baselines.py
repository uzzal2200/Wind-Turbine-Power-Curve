"""
baselines.py
============
Five baseline models evaluated in the paper:

  1. PolynomialRegression   — Degree-5 polynomial (wind speed only).
  2. GPRBaseline            — Gaussian Process Regression (RBF kernel).
  3. XGBoostBaseline        — Gradient-boosted ensemble.
  4. LSTMBaseline           — 2-layer LSTM with sequence length 12.
  5. PlainMLPBaseline       — Identical MLP backbone to DA-PINN, λ₁=λ₂=λ₃=0.

All expose a unified sklearn-style interface: .fit(X, y) / .predict(X).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Polynomial Regression (degree-5, wind speed only)
# ─────────────────────────────────────────────────────────────────────────────

class PolynomialRegression:
    """Degree-5 polynomial fit on wind speed alone."""

    def __init__(self, degree: int = 5) -> None:
        self.degree = degree
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        from sklearn.linear_model import Ridge

        self._pipeline = Pipeline([
            ("poly",  PolynomialFeatures(degree=self.degree, include_bias=True)),
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1e-3)),
        ])
        # Use only the first column (wind speed) by convention
        self._pipeline.fit(X[:, :1], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._pipeline.predict(X[:, :1])

    @classmethod
    def from_config(cls, config: dict) -> "PolynomialRegression":
        return cls(degree=config["baselines"]["polynomial"]["degree"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPR Baseline
# ─────────────────────────────────────────────────────────────────────────────

class GPRBaseline:
    """
    Gaussian Process Regression with RBF + WhiteKernel.
    Warning: O(n³) — use a subsample for large datasets.
    """

    def __init__(self, max_samples: int = 5_000) -> None:
        self.max_samples = max_samples
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRBaseline":
        if len(X) > self.max_samples:
            idx = np.random.choice(len(X), self.max_samples, replace=False)
            X, y = X[idx], y[idx]
            logger.warning("GPR: subsampled to %d points (O(n³) complexity).", self.max_samples)
        self._gpr.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._gpr.predict(X)

    @classmethod
    def from_config(cls, config: dict) -> "GPRBaseline":
        return cls()


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost Baseline
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostBaseline:
    """XGBoost gradient-boosted regressor."""

    def __init__(
        self,
        n_estimators:  int   = 500,
        max_depth:     int   = 6,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
    ) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @classmethod
    def from_config(cls, config: dict) -> "XGBoostBaseline":
        cfg = config["baselines"]["xgboost"]
        return cls(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg.get("subsample", 0.8),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMModule(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # last step


class LSTMBaseline:
    """2-layer LSTM (trained with PyTorch, standard MSE loss)."""

    def __init__(
        self,
        input_dim:   int   = 12,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        seq_length:  int   = 12,
        dropout:     float = 0.2,
        lr:          float = 1e-3,
        batch_size:  int   = 256,
        max_epochs:  int   = 100,
        patience:    int   = 10,
        device:      str   = "cpu",
    ) -> None:
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience   = patience
        self.device     = torch.device(device)

        self._model = _LSTMModule(input_dim, hidden_size, num_layers, dropout).to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._criterion = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMBaseline":
        from src.training.dataset import SequenceDataset
        from torch.utils.data import DataLoader

        dataset    = SequenceDataset(X, y, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss  = float("inf")
        patience   = 0

        self._model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self._optimizer.zero_grad()
                loss = self._criterion(self._model(xb), yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience  = 0
            else:
                patience += 1
            if patience >= self.patience:
                logger.info("LSTM early stopping at epoch %d", epoch + 1)
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from src.training.dataset import SequenceDataset
        from torch.utils.data import DataLoader

        dataset    = SequenceDataset(X, np.zeros(len(X)), self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds: list[np.ndarray] = []

        self._model.eval()
        with torch.no_grad():
            for xb, _ in dataloader:
                preds.append(self._model(xb.to(self.device)).cpu().numpy())
        return np.concatenate(preds)

    @classmethod
    def from_config(cls, config: dict) -> "LSTMBaseline":
        cfg = config["baselines"]["lstm"]
        return cls(
            input_dim=len(config["features"]["scada_cols"]),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            seq_length=cfg["seq_length"],
            dropout=cfg["dropout"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plain MLP (DA-PINN backbone, no physics loss)
# ─────────────────────────────────────────────────────────────────────────────

class PlainMLPBaseline:
    """
    Identical MLP backbone to DA-PINN, trained with MSE only (λ₁=λ₂=λ₃=0).
    Uses the Trainer class for consistency.
    """

    def __init__(self, config: dict) -> None:
        from src.models.da_pinn import DAPINN

        # Override physics weights to zero
        cfg = config.copy()
        cfg["loss"] = {"lambda_betz": 0.0, "lambda_cp": 0.0, "lambda_smooth": 0.0}
        self._model  = DAPINN.from_config(cfg)
        self._config = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlainMLPBaseline":
        from src.training.trainer import Trainer
        trainer = Trainer(self._model, self._config)
        trainer.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from src.training.dataset import SCADADataset
        from torch.utils.data import DataLoader

        dataset    = SCADADataset(X, np.zeros(len(X)))
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
        preds: list[np.ndarray] = []
        device = next(self._model.parameters()).device

        self._model.eval()
        with torch.no_grad():
            for xb, _ in dataloader:
                preds.append(self._model(xb.to(device)).cpu().numpy())
        return np.concatenate(preds)

    @classmethod
    def from_config(cls, config: dict) -> "PlainMLPBaseline":
        return cls(config)
