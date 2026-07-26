"""
trainer.py
==========
Training loop for DA-PINN with:
  • Physics-constrained composite loss
  • Adam optimiser (lr 1e-3)
  • Early stopping (patience 20)
  • Best-model checkpointing
  • Structured epoch logging
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.models.da_pinn import DAPINN
from src.models.physics_loss import PhysicsLoss
from src.training.dataset import SCADADataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Tracks validation loss and signals when to stop."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-5) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """
    Orchestrates DA-PINN training.

    Parameters
    ----------
    model  : DAPINN
    config : dict   Top-level project config.
    """

    def __init__(self, model: DAPINN, config: dict) -> None:
        self.model  = model
        self.config = config

        # Device
        device_str = config["training"].get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        self.model.to(self.device)

        # Optimiser
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
        )

        # Loss
        loss_cfg = config["loss"]
        phys_cfg = config["physics"]
        self.criterion = PhysicsLoss(
            lambda_betz=loss_cfg["lambda_betz"],
            lambda_cp=loss_cfg["lambda_cp"],
            lambda_smooth=loss_cfg["lambda_smooth"],
            rho=phys_cfg["rho_air_kg_m3"],
            rotor_area=phys_cfg["rotor_area_m2"],
            cp_max=phys_cfg["cp_max"],
            v_mono_min=phys_cfg["v_mono_min_ms"],
            v_mono_max=phys_cfg["v_mono_max_ms"],
            delta_v=phys_cfg["delta_v_ms"],
        )

        # Training settings
        t = config["training"]
        self.batch_size  = t["batch_size"]
        self.max_epochs  = t["max_epochs"]
        self.patience    = t["patience"]
        self.num_workers = t.get("num_workers", 0)
        self.seed        = t.get("seed", 42)

        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_pvr": [], "val_pvr": [],
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        wind_speed_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wind_speed_val: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train DA-PINN.

        Parameters
        ----------
        X_train, y_train, wind_speed_train : Training arrays.
        X_val, y_val, wind_speed_val       : Validation arrays (optional).
            If not provided, 10 % of training data is held out.

        Returns
        -------
        Training history dict.
        """
        torch.manual_seed(self.seed)

        train_dataset = SCADADataset(X_train, y_train, wind_speed_raw=wind_speed_train)

        if X_val is not None:
            val_dataset = SCADADataset(X_val, y_val, wind_speed_raw=wind_speed_val)
        else:
            n_val = max(1, int(0.1 * len(train_dataset)))
            n_tr  = len(train_dataset) - n_val
            train_dataset, val_dataset = random_split(
                train_dataset, [n_tr, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=self.num_workers,
        )

        early_stop = EarlyStopping(patience=self.patience)
        logger.info("Training DA-PINN on %s | %d train / %d val samples",
                    self.device, len(train_dataset), len(val_dataset))

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_pvr = self._run_epoch(train_loader, training=True)
            vl_loss, vl_pvr = self._run_epoch(val_loader,   training=False)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_pvr"].append(tr_pvr)
            self.history["val_pvr"].append(vl_pvr)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "tr_pvr=%.2f%%  vl_pvr=%.2f%%  (%.1fs)",
                epoch, self.max_epochs, tr_loss, vl_loss,
                tr_pvr * 100, vl_pvr * 100, elapsed,
            )

            if early_stop.step(vl_loss, self.model):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        early_stop.restore_best(self.model)
        self._save_checkpoint("best_model.pt")
        logger.info("Training complete. Best val loss: %.4f", early_stop.best_loss)
        return self.history

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> tuple[float, float]:
        """One full pass. Returns (avg_loss, avg_pvr)."""
        self.model.train(training)
        total_loss = 0.0
        total_pvr  = 0.0
        n_batches  = 0

        phys_cfg = self.config["physics"]
        delta_v  = phys_cfg["delta_v_ms"]

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for xb, yb, vb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                vb = vb.to(self.device)

                p_hat = self.model(xb)

                # Compute p_hat at (v + Δv) for monotonicity loss
                p_hat_shifted = self._predict_shifted(xb, vb, delta_v)

                losses = self.criterion(p_hat, yb, vb, p_hat_shifted)
                loss   = losses["total"]

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                pvr = self.criterion.physics_violation_rate(p_hat.detach(), vb)
                total_loss += loss.item()
                total_pvr  += pvr
                n_batches  += 1

        return total_loss / n_batches, total_pvr / n_batches

    def _predict_shifted(
        self,
        xb: torch.Tensor,
        vb: torch.Tensor,
        delta_v: float,
    ) -> torch.Tensor:
        """
        Forward pass with wind speed bumped by Δv.
        Assumes wind_speed is column 0 of xb (un-normalised values stored in vb).
        We create a modified input where the raw v column increases by Δv.
        Since X is normalised, we skip column-level bumping and use the
        separate vb tensor only for the physics loss — p_hat_shifted is
        computed by bumping the first column of xb by a small normalised amount.
        """
        # Approximate: bump first feature column (wind speed, normalised) slightly.
        # This is a first-order approximation; for exact results use the raw scale.
        xb_shifted = xb.clone()
        # Fractional bump: delta_v / v_cut_out (25 m/s)
        v_cut_out = self.config["dataset"]["v_cut_out_ms"]
        xb_shifted[:, 0] = xb_shifted[:, 0] + delta_v / v_cut_out
        xb_shifted[:, 0] = xb_shifted[:, 0].clamp(0.0, 1.0)
        return self.model(xb_shifted)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        logger.info("Checkpoint saved → %s", path)

    @staticmethod
    def load_checkpoint(path: str, config: dict) -> DAPINN:
        """Load a saved model from a checkpoint file."""
        ckpt  = torch.load(path, map_location="cpu")
        model = DAPINN.from_config(ckpt.get("config", config))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
