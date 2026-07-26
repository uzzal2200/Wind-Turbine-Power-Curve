"""
da_pinn.py
==========
DA-PINN backbone: a four-layer MLP receiving a 15-dimensional input
(12 SCADA-derived features + 3-component health conditioning vector).

Architecture (Section III-E of paper)
--------------------------------------
  Input  (15)  →  Linear(64) → ReLU → Linear(128) → ReLU
               →  Linear(64) → ReLU → Linear(1)    → P̂ (scalar, kW)

The health vector h(t) = [ΔT_blade, PR_6h, AI_pitch]ᵀ is concatenated
to the 12 SCADA features before the first linear layer, enabling
degradation-state-aware prediction without per-class models.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DAPINN(nn.Module):
    """
    Degradation-Aware Physics-Informed Neural Network.

    Parameters
    ----------
    input_dim    : int             Total input dimensionality (SCADA + health).
    hidden_dims  : Sequence[int]   Widths of hidden layers.
    activation   : str             "relu" | "gelu" | "tanh".
    dropout      : float           Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        input_dim:   int           = 15,
        hidden_dims: Sequence[int] = (64, 128, 64),
        activation:  str           = "relu",
        dropout:     float         = 0.0,
    ) -> None:
        super().__init__()

        act_cls = self._resolve_activation(activation)
        layers: list[nn.Module] = []

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        # Output layer: single scalar (predicted power in kW, unnormalised)
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim)
            Concatenated [SCADA features | health vector].

        Returns
        -------
        torch.Tensor, shape (N,)
            Predicted active power in kW.
        """
        return self.network(x).squeeze(-1)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_activation(name: str) -> type[nn.Module]:
        mapping = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if name not in mapping:
            raise ValueError(f"Unsupported activation '{name}'. Choose from {list(mapping)}.")
        return mapping[name]

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ── Convenience ────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> "DAPINN":
        """Instantiate from the project config dict."""
        m = config["model"]
        return cls(
            input_dim=m["input_dim"],
            hidden_dims=m["hidden_dims"],
            activation=m.get("activation", "relu"),
            dropout=m.get("dropout", 0.0),
        )
