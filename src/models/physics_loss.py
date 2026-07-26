"""
physics_loss.py
===============
Composite physics-constrained loss for DA-PINN (Eq. 3–7 in paper).

    L_total = L_data + λ₁·L_Betz + λ₂·L_Cp + λ₃·L_smooth

Components
----------
L_data    — Mean squared error between prediction and measured power.
L_Betz    — Penalty for predictions exceeding the Betz limit.
L_Cp      — Penalty for power-coefficient predictions outside (0, 0.593].
L_smooth  — Monotonicity regularisation in the power-ramp region [3.5, 12.5] m/s.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Composite physics-constrained loss function.

    Parameters
    ----------
    lambda_betz   : float  Weight for the Betz limit penalty term.
    lambda_cp     : float  Weight for the Cₚ constraint term.
    lambda_smooth : float  Weight for the monotonicity regularisation term.
    rho           : float  Air density in kg/m³ (default 1.225).
    rotor_area    : float  Rotor swept area in m² (default 6647.61 for MM92).
    cp_max        : float  Maximum physically admissible Cₚ (Betz limit ≈ 0.593).
    v_mono_min    : float  Lower wind speed for monotonicity region (m/s).
    v_mono_max    : float  Upper wind speed for monotonicity region (m/s).
    delta_v       : float  Wind speed increment for monotonicity check (m/s).
    """

    def __init__(
        self,
        lambda_betz:   float = 0.8,
        lambda_cp:     float = 0.5,
        lambda_smooth: float = 0.3,
        rho:           float = 1.225,
        rotor_area:    float = 6647.61,
        cp_max:        float = 0.593,
        v_mono_min:    float = 3.5,
        v_mono_max:    float = 12.5,
        delta_v:       float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_betz   = lambda_betz
        self.lambda_cp     = lambda_cp
        self.lambda_smooth = lambda_smooth
        self.rho           = rho
        self.rotor_area    = rotor_area
        self.cp_max        = cp_max
        self.v_mono_min    = v_mono_min
        self.v_mono_max    = v_mono_max
        self.delta_v       = delta_v

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        p_hat: torch.Tensor,        # (N,) predicted power [kW]
        p_true: torch.Tensor,       # (N,) measured power  [kW]
        wind_speed: torch.Tensor,   # (N,) wind speed      [m/s] (un-normalised)
        p_hat_shifted: torch.Tensor | None = None,  # (N,) predicted at v + Δv
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Parameters
        ----------
        p_hat         : Predicted power output (kW), shape (N,).
        p_true        : Ground-truth power output (kW), shape (N,).
        wind_speed    : Wind speed (m/s) — must be in physical units, NOT normalised.
        p_hat_shifted : DA-PINN output at wind_speed + delta_v (for L_smooth).
                        If None, L_smooth is set to 0.

        Returns
        -------
        dict with keys: total, data, betz, cp, smooth (all scalar tensors).
        """
        l_data   = self._data_loss(p_hat, p_true)
        l_betz   = self._betz_loss(p_hat, wind_speed)
        l_cp     = self._cp_loss(p_hat, wind_speed)
        l_smooth = self._smooth_loss(p_hat, p_hat_shifted, wind_speed)

        total = (
            l_data
            + self.lambda_betz   * l_betz
            + self.lambda_cp     * l_cp
            + self.lambda_smooth * l_smooth
        )

        return {
            "total":  total,
            "data":   l_data,
            "betz":   l_betz,
            "cp":     l_cp,
            "smooth": l_smooth,
        }

    # ── Loss components ────────────────────────────────────────────────────

    @staticmethod
    def _data_loss(p_hat: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
        """L_data = (1/N) Σ (p̂ᵢ - pᵢ)²  (Eq. 4)."""
        return torch.mean((p_hat - p_true) ** 2)

    def _betz_loss(
        self, p_hat: torch.Tensor, wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        L_Betz = (1/N) Σ max(0, p̂ᵢ - P_Betz_max(vᵢ))²  (Eq. 5).

        P_Betz_max(v) = (16/27) · (1/2) · ρ · A · v³  [kW]
        """
        p_betz_max = self._betz_limit_kw(wind_speed)
        violation  = torch.clamp(p_hat - p_betz_max, min=0.0)
        return torch.mean(violation ** 2)

    def _cp_loss(
        self, p_hat: torch.Tensor, wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        L_Cp = (1/N) Σ [max(0, Cₚᵢ − 0.593)² + max(0, −Cₚᵢ)²]  (Eq. 6).

        Cₚᵢ = p̂ᵢ / (0.5 · ρ · A · vᵢ³)
        """
        p_avail = self._available_power_kw(wind_speed)
        # Avoid division by near-zero available power at very low wind speeds
        p_avail_safe = torch.clamp(p_avail, min=1.0)
        cp = p_hat / p_avail_safe

        upper_violation = torch.clamp(cp - self.cp_max, min=0.0)
        lower_violation = torch.clamp(-cp, min=0.0)
        return torch.mean(upper_violation ** 2 + lower_violation ** 2)

    def _smooth_loss(
        self,
        p_hat: torch.Tensor,
        p_hat_shifted: torch.Tensor | None,
        wind_speed: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_smooth = (1/N) Σ max(0, p̂(vᵢ) − p̂(vᵢ + Δv))²  (Eq. 7).

        Applied only within the power-ramp region [v_mono_min, v_mono_max].
        Returns 0 if p_hat_shifted is not provided.
        """
        if p_hat_shifted is None:
            return torch.tensor(0.0, device=p_hat.device, dtype=p_hat.dtype)

        ramp_mask = (wind_speed >= self.v_mono_min) & (wind_speed <= self.v_mono_max)
        if ramp_mask.sum() == 0:
            return torch.tensor(0.0, device=p_hat.device, dtype=p_hat.dtype)

        non_monotone = torch.clamp(p_hat[ramp_mask] - p_hat_shifted[ramp_mask], min=0.0)
        return torch.mean(non_monotone ** 2)

    # ── Physics helpers ────────────────────────────────────────────────────

    def _available_power_kw(self, wind_speed: torch.Tensor) -> torch.Tensor:
        """(1/2) · ρ · A · v³, converted to kW."""
        return 0.5 * self.rho * self.rotor_area * wind_speed ** 3 / 1000.0

    def _betz_limit_kw(self, wind_speed: torch.Tensor) -> torch.Tensor:
        """(16/27) · (1/2) · ρ · A · v³, in kW."""
        return (16.0 / 27.0) * self._available_power_kw(wind_speed)

    # ── Convenience: violation rate ─────────────────────────────────────────

    @torch.no_grad()
    def physics_violation_rate(
        self, p_hat: torch.Tensor, wind_speed: torch.Tensor
    ) -> float:
        """
        Fraction of predictions exceeding the Betz limit (PVR in paper).

        Returns
        -------
        float in [0, 1].
        """
        p_betz_max = self._betz_limit_kw(wind_speed)
        violations = (p_hat > p_betz_max).float().sum()
        return (violations / len(p_hat)).item()
