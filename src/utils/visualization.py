"""
visualization.py
================
Plotting utilities reproducing all nine paper figures.

Figure map
----------
  Fig 1  — 90-day winter SCADA overview (EDA time series)
  Fig 2  — Wind speed vs active power, seasonal scatter
  Fig 3  — Performance ratio distribution + temperature proxy labels
  Fig 4  — System architecture diagram (static, skip)
  Fig 5  — DA-PINN architecture diagram (static, skip)
  Fig 6  — Predicted vs actual scatter for all six models
  Fig 7  — Betz compliance: Plain MLP vs DA-PINN
  Fig 8  — Icing-induced power curve shift
  Fig 9  — Icing detection timeline (CUSUM panels A/B/C)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Consistent colour scheme
PALETTE = {
    "healthy": "#2ECC71",
    "mild":    "#F39C12",
    "severe":  "#E74C3C",
    "betz":    "#E74C3C",
    "mfr":     "#2C3E50",
    "pred":    "#3498DB",
    "actual":  "#2C3E50",
}


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Winter SCADA time series overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda_timeseries(
    df: pd.DataFrame,
    icing_start_day: int = 28,
    icing_end_day:   int = 44,
    out_path: str = "figures/figure1_eda_timeseries.png",
) -> None:
    """Reproduce Fig. 1: 90-day winter overview."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    days = np.arange(len(df)) * 10 / (60 * 24)  # 10-min steps → fractional days

    axes[0].plot(days, df["wind_speed_ms"], lw=0.6, color="#2980B9", alpha=0.8)
    axes[0].set_ylabel("Wind Speed\n(m/s)")

    axes[1].plot(days, df["active_power_kw"] / 1e3, lw=0.6, color="#27AE60", alpha=0.8)
    axes[1].set_ylabel("Active Power\n(MW)")

    axes[2].plot(days, df["ambient_temp_c"], lw=0.6, color="#E67E22", alpha=0.8)
    axes[2].axhline(2.0, ls="--", lw=0.8, color="k", label="2°C threshold")
    axes[2].set_ylabel("Ambient Temp\n(°C)")
    axes[2].legend(fontsize=7, loc="upper right")

    if "pr_6h" in df.columns:
        pr_col = "pr_6h"
    elif "pr" in df.columns:
        pr_col = "pr"
    else:
        pr_col = None

    if pr_col:
        axes[3].plot(days, df[pr_col], lw=0.6, color="#8E44AD", alpha=0.8)
        axes[3].axhline(0.90, ls="--", lw=0.8, color="#F39C12", label="PR=0.90")
        axes[3].axhline(0.70, ls="--", lw=0.8, color="#E74C3C", label="PR=0.70")
        axes[3].set_ylabel("Perf. Ratio")
        axes[3].set_ylim(0, 1.2)
        axes[3].legend(fontsize=7, loc="upper right")
    axes[3].set_xlabel("Day of Winter Season")

    # Shade confirmed icing event
    for ax in axes:
        ax.axvspan(icing_start_day, icing_end_day, alpha=0.15,
                   color="#E74C3C", label="Confirmed icing event")

    # Add annotation box on top panel
    axes[0].annotate(
        f"Confirmed icing event\n(Days {icing_start_day}–{icing_end_day})",
        xy=((icing_start_day + icing_end_day) / 2, axes[0].get_ylim()[1] * 0.9),
        fontsize=7, ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E74C3C", lw=0.8),
    )

    fig.suptitle("90-Day Winter SCADA Overview (Jan–Mar 2021)", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Seasonal scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal_scatter(
    df: pd.DataFrame,
    v_cut_out: float = 25.0,
    rated_power: float = 2050.0,
    out_path: str = "figures/figure2_scatter_seasonal.png",
) -> None:
    """Reproduce Fig. 2: wind speed vs active power by season."""
    from src.preprocessing.feature_engineer import manufacturer_power_kw

    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season_colors = {"Spring": "#27AE60", "Summer": "#F1C40F",
                     "Autumn": "#E67E22", "Winter": "#2980B9"}

    fig, ax = plt.subplots(figsize=(8, 5))
    df = df.copy()
    df["season"] = df.index.month.map(season_map)

    for season, grp in df.groupby("season"):
        ax.scatter(grp["wind_speed_ms"], grp["active_power_kw"],
                   s=1, alpha=0.3, color=season_colors[season], label=season)

    v_range = np.linspace(0, v_cut_out, 300)
    ax.plot(v_range, manufacturer_power_kw(v_range), "k-", lw=1.5, label="MM92 Mfr. Curve")
    p_betz = (16 / 27) * 0.5 * 1.225 * 6647.61 * v_range ** 3 / 1000.0
    ax.plot(v_range, p_betz.clip(0, rated_power * 1.1),
            "--", color=PALETTE["betz"], lw=1.2, label="Betz Limit (59.3%)")

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Active Power (kW)")
    ax.set_xlim(0, v_cut_out)
    ax.set_ylim(0, rated_power * 1.15)
    ax.axvline(12.5, ls=":", lw=0.8, color="gray")
    ax.text(12.7, 200, "Rated 12.5 m/s", fontsize=7, color="gray")
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title("Wind Speed vs Active Power — Seasonal Operating Conditions")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Performance ratio distribution + proxy labels
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_labels(
    df: pd.DataFrame,
    pr_col: str = "pr_6h",
    temp_col: str = "ambient_temp_c",
    label_col: str = "icing_label",
    out_path: str = "figures/figure3_pr_labels.png",
) -> None:
    """Reproduce Fig. 3: PR distribution and temperature vs PR coloured by label."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    label_colors = {0: PALETTE["healthy"], 1: PALETTE["mild"], 2: PALETTE["severe"]}
    label_names  = {0: "Healthy", 1: "Mild Icing", 2: "Severe Icing"}

    # (a) PR histogram
    for cls in [0, 1, 2]:
        subset = df.loc[df[label_col] == cls, pr_col]
        axes[0].hist(subset, bins=80, alpha=0.6, color=label_colors[cls],
                     label=label_names[cls], density=False)
    axes[0].axvline(0.90, ls="--", lw=1, color=PALETTE["mild"], label="PR=0.90")
    axes[0].axvline(0.70, ls="--", lw=1, color=PALETTE["severe"], label="PR=0.70")
    axes[0].set_xlabel("Performance Ratio")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) PR Distribution with Icing Thresholds")
    axes[0].legend(fontsize=8)

    # (b) Temperature vs PR scatter
    for cls in [0, 1, 2]:
        subset = df[df[label_col] == cls]
        axes[1].scatter(subset[temp_col], subset[pr_col],
                        s=1, alpha=0.3, color=label_colors[cls], label=label_names[cls])
    axes[1].axvline(2.0, ls="--", lw=1, color="k", label="2°C threshold")
    axes[1].axhline(0.90, ls="--", lw=0.8, color=PALETTE["mild"])
    axes[1].axhline(0.70, ls="--", lw=0.8, color=PALETTE["severe"])
    axes[1].set_xlabel("Ambient Temperature (°C)")
    axes[1].set_ylabel("Performance Ratio")
    axes[1].set_title("(b) Temperature vs PR coloured by Proxy Label")
    axes[1].legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — All-models scatter (hexbin)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_models_scatter(
    y_true:     np.ndarray,
    preds_dict: dict[str, np.ndarray],   # {model_name: y_pred}
    wind_speed: np.ndarray,
    rated_power: float = 2050.0,
    rho: float = 1.225,
    rotor_area: float = 6647.61,
    out_path: str = "figures/figure6_all_models_r2_ranked.png",
) -> None:
    """Reproduce Fig. 6: predicted vs actual hexbin for all models, ranked by R²."""
    from src.evaluation.metrics import r_squared, physics_violation_rate

    # Sort by R²
    ranked = sorted(preds_dict.items(),
                    key=lambda kv: r_squared(y_true, kv[1]))

    n_models = len(ranked)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    p_max = rated_power * 1.2
    ref   = np.linspace(0, p_max, 300)
    p_betz = (16 / 27) * 0.5 * rho * rotor_area * ref ** 3 / 1000.0

    for ax, (name, y_pred) in zip(axes, ranked):
        r2  = r_squared(y_true, y_pred)
        pvr = physics_violation_rate(y_pred, wind_speed, rho, rotor_area)
        hb  = ax.hexbin(y_true, y_pred, gridsize=60, bins="log",
                        cmap="YlOrRd", mincnt=1)
        ax.plot(ref, ref, "k--", lw=0.8, label="y=x")
        ax.plot(ref, p_betz.clip(0, p_max), "--", color=PALETTE["betz"],
                lw=0.8, label="Betz ceiling")
        ax.set_xlim(0, p_max)
        ax.set_ylim(0, p_max)
        ax.set_title(f"{name}\nR²={r2:.3f}  PVR={pvr:.1f}%", fontsize=8)
        ax.set_xlabel("Actual (kW)", fontsize=7)
        ax.set_ylabel("Predicted (kW)", fontsize=7)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    plt.suptitle("Predicted vs Actual Power — All Models (log₁₀ hexbin)", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — CUSUM detection timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_cusum_timeline(
    p_hat:       np.ndarray,
    p_actual:    np.ndarray,
    cusum_vals:  np.ndarray,
    icing_labels: np.ndarray,
    alert_threshold: float = 0.25,
    maintenance_step: int  = 48 * 6,  # 48 h at 10-min intervals
    out_path: str = "figures/figure9_detection_timeline.png",
) -> None:
    """Reproduce Fig. 9: three-panel icing detection timeline."""
    hours = np.arange(len(p_hat)) * 10 / 60

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(3, 1, hspace=0.35)

    # Panel A — Power
    ax_a = fig.add_subplot(gs[0])
    ax_a.plot(hours, p_actual / 1e3, color=PALETTE["actual"], lw=0.8, label="Actual Power")
    ax_a.plot(hours, p_hat   / 1e3, color=PALETTE["pred"],   lw=0.8, ls="--",
              label="DA-PINN Expected")
    ax_a.set_ylabel("Active Power (MW)")
    ax_a.set_title("(A) Power Divergence")
    ax_a.legend(fontsize=8, loc="upper right")

    # Panel B — CUSUM
    ax_b = fig.add_subplot(gs[1])
    ax_b.plot(hours, cusum_vals, color="#8E44AD", lw=1.0, label="CUSUM S(t)")
    ax_b.axhline(alert_threshold, ls="--", color=PALETTE["betz"], lw=1.0,
                 label=f"Threshold h={alert_threshold}")
    ax_b.axvline(hours[maintenance_step], ls=":", color="k", lw=0.8,
                 label=f"Maintenance (Hour {hours[maintenance_step]:.0f})")
    ax_b.set_ylabel("CUSUM S(t)")
    ax_b.set_title("(B) CUSUM Physics Residual")
    ax_b.legend(fontsize=8)

    # Panel C — Icing state
    ax_c = fig.add_subplot(gs[2])
    label_colors = [PALETTE["healthy"], PALETTE["mild"], PALETTE["severe"]]
    label_names  = ["Healthy", "Mild", "Severe"]
    for cls in range(3):
        mask = icing_labels == cls
        ax_c.fill_between(hours, cls, cls + 1,
                          where=mask, alpha=0.6, color=label_colors[cls],
                          label=label_names[cls])
    ax_c.set_yticks([0.5, 1.5, 2.5])
    ax_c.set_yticklabels(label_names, fontsize=8)
    ax_c.set_ylabel("Icing State")
    ax_c.set_title("(C) Proxy Icing State")
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.set_xlabel("Time (Hours)")

    plt.suptitle("Icing Detection Timeline — DA-PINN CUSUM Monitor", fontweight="bold")
    save_fig(fig, out_path)
