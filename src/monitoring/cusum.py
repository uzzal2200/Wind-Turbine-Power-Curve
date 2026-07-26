"""
cusum.py
========
Physics-residual severity monitoring via CUSUM control chart (Eq. 8–9 in paper).

Physics residual
----------------
    R(t) = |P̂(t) − P_actual(t)| / P_Betz_max(t)

During healthy operation R(t) ≈ 0.02–0.05.
Icing causes actual power to drop while the physics-constrained prediction
remains near the aerodynamic envelope, systematically elevating R(t).

CUSUM
-----
    S(t) = max(0, S(t−1) + R(t) − μ₀ − k)

Alert when S(t) ≥ h = 4σ₀ (default h = 0.25, σ₀ = 0.0625).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CUSUMState:
    """Holds the running CUSUM state between calls."""
    S: float = 0.0              # current cumulative sum
    alert: bool = False         # True if S ≥ h

@dataclass
class CUSUMResult:
    """Stores per-timestep monitoring output."""
    residuals:  np.ndarray = field(default_factory=lambda: np.array([]))
    cusum_vals: np.ndarray = field(default_factory=lambda: np.array([]))
    alert_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    alert_times: list[int] = field(default_factory=list)   # indices of alert crossings
    mean_lead_time_steps: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CUSUMMonitor class
# ─────────────────────────────────────────────────────────────────────────────

class CUSUMMonitor:
    """
    Detect icing-induced degradation via a one-sided CUSUM chart applied
    to the normalised physics residual R(t).

    Parameters
    ----------
    config : dict   Top-level project config.
    """

    def __init__(self, config: dict) -> None:
        cusum = config["cusum"]
        phys  = config["physics"]

        self.k_factor: float = cusum["k_factor"]    # allowance = k_factor × σ₀
        self.h_factor: float = cusum["h_factor"]    # alert     = h_factor × σ₀
        self.sigma0:   float = cusum["sigma0"]      # calibrated std; updated by .calibrate()

        self.rho:        float = phys["rho_air_kg_m3"]
        self.rotor_area: float = phys["rotor_area_m2"]

        self._k: float = self.k_factor * self.sigma0
        self._h: float = self.h_factor * self.sigma0
        self._mu0: float = 0.0   # baseline mean; updated by .calibrate()

        self._state = CUSUMState()

    # ── Calibration ────────────────────────────────────────────────────────

    def calibrate(
        self,
        p_hat_healthy:    np.ndarray,
        p_actual_healthy: np.ndarray,
        wind_speed_healthy: np.ndarray,
    ) -> dict:
        """
        Estimate μ₀ and σ₀ from confirmed healthy SCADA periods.

        Parameters
        ----------
        p_hat_healthy       : DA-PINN predictions over healthy windows (kW).
        p_actual_healthy    : Measured power over healthy windows (kW).
        wind_speed_healthy  : Wind speed over healthy windows (m/s, physical).

        Returns
        -------
        dict with keys mu0, sigma0, k, h.
        """
        residuals = self._compute_residual(p_hat_healthy, p_actual_healthy, wind_speed_healthy)
        self._mu0   = float(np.mean(residuals))
        self.sigma0 = float(np.std(residuals))
        self._k     = self.k_factor * self.sigma0
        self._h     = self.h_factor * self.sigma0

        logger.info(
            "CUSUM calibrated: μ₀=%.4f  σ₀=%.4f  k=%.4f  h=%.4f",
            self._mu0, self.sigma0, self._k, self._h,
        )
        return {"mu0": self._mu0, "sigma0": self.sigma0, "k": self._k, "h": self._h}

    # ── Online monitoring ──────────────────────────────────────────────────

    def run(
        self,
        p_hat:       np.ndarray,
        p_actual:    np.ndarray,
        wind_speed:  np.ndarray,
        reset_state: bool = True,
    ) -> CUSUMResult:
        """
        Apply CUSUM to a time series of predictions.

        Parameters
        ----------
        p_hat      : DA-PINN predicted power (kW), shape (N,).
        p_actual   : Measured active power (kW),   shape (N,).
        wind_speed : Physical wind speed (m/s),     shape (N,).
        reset_state: If True, reset S(t) to 0 before the run.

        Returns
        -------
        CUSUMResult with per-timestep residuals, CUSUM values, and alert mask.
        """
        if reset_state:
            self._state = CUSUMState()

        residuals  = self._compute_residual(p_hat, p_actual, wind_speed)
        n          = len(residuals)
        cusum_vals = np.zeros(n)
        alert_mask = np.zeros(n, dtype=bool)

        S = self._state.S
        for t in range(n):
            S = max(0.0, S + residuals[t] - self._mu0 - self._k)
            cusum_vals[t] = S
            alert_mask[t] = S >= self._h

        self._state.S     = S
        self._state.alert = bool(alert_mask[-1]) if n > 0 else False

        # Identify alert crossings (first step in each alert run)
        alert_times = self._find_alert_crossings(alert_mask)

        result = CUSUMResult(
            residuals=residuals,
            cusum_vals=cusum_vals,
            alert_mask=alert_mask,
            alert_times=alert_times,
        )
        if alert_times:
            logger.info("CUSUM: %d alert crossing(s) detected.", len(alert_times))
        return result

    # ── Lead-time evaluation ───────────────────────────────────────────────

    def compute_lead_times(
        self,
        alert_times:           list[int],
        confirmed_event_steps: list[int],
        max_match_window:      int = 200,  # ≈33 h at 10-min intervals
    ) -> list[int]:
        """
        Match each CUSUM alert to the nearest subsequent confirmed icing event
        and return the lead time in steps.

        Parameters
        ----------
        alert_times           : Step indices of CUSUM alert crossings.
        confirmed_event_steps : Step indices of confirmed maintenance events.
        max_match_window      : Maximum look-ahead steps for a match.

        Returns
        -------
        List of lead times (steps) for matched events.
        """
        lead_times: list[int] = []
        used_events: set[int] = set()

        for alert_t in alert_times:
            for event_t in confirmed_event_steps:
                if event_t in used_events:
                    continue
                lead = event_t - alert_t
                if 0 < lead <= max_match_window:
                    lead_times.append(lead)
                    used_events.add(event_t)
                    break

        if lead_times:
            avg_steps = np.mean(lead_times)
            avg_hours = avg_steps * 10 / 60  # 10-min intervals → hours
            logger.info(
                "CUSUM lead-time: %d matched events, avg %.1f steps (%.1f h)",
                len(lead_times), avg_steps, avg_hours,
            )
        return lead_times

    # ── Private helpers ────────────────────────────────────────────────────

    def _compute_residual(
        self,
        p_hat:      np.ndarray,
        p_actual:   np.ndarray,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """R(t) = |P̂(t) − P_actual(t)| / P_Betz_max(t)."""
        p_betz_max = self._betz_limit_kw(wind_speed)
        p_betz_safe = np.where(p_betz_max < 1.0, 1.0, p_betz_max)
        return np.abs(p_hat - p_actual) / p_betz_safe

    def _betz_limit_kw(self, wind_speed: np.ndarray) -> np.ndarray:
        return (16.0 / 27.0) * 0.5 * self.rho * self.rotor_area * wind_speed ** 3 / 1000.0

    @staticmethod
    def _find_alert_crossings(alert_mask: np.ndarray) -> list[int]:
        """Return indices where alert transitions from False → True."""
        crossings: list[int] = []
        prev = False
        for i, cur in enumerate(alert_mask):
            if cur and not prev:
                crossings.append(i)
            prev = cur
        return crossings
