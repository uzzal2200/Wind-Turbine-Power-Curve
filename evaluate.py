#!/usr/bin/env python
"""
evaluate.py
===========
Evaluate DA-PINN (and optionally all baselines) on the test set (2023–2024).
Prints the metrics table from Table II / III of the paper.

Usage
-----
  python evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
  python evaluate.py --config configs/config.yaml --all --save-tables
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.evaluation.metrics import regression_report, print_results_table
from src.training.trainer import Trainer
from src.utils.io import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DA-PINN on the test set")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--all",        action="store_true",
                        help="Evaluate all baseline models as well.")
    parser.add_argument("--save-tables", action="store_true",
                        help="Save metric tables to outputs/predictions/metrics.json")
    return parser.parse_args()


def load_test_arrays(config: dict):
    """Load test-split arrays — reuses the same logic as train.py."""
    import pandas as pd
    from src.preprocessing.cleaner import SCADACleaner
    from src.training.dataset import chronological_split
    from src.utils.io import load_processed

    proc_dir = Path(config["paths"]["processed_data_dir"])
    dfs = [load_processed(f) for f in sorted(proc_dir.glob("*.csv"))]
    df  = pd.concat(dfs).sort_index()

    ds    = config["dataset"]
    feats = config["features"]
    all_cols = feats["scada_cols"] + feats["health_vector_cols"]

    _, _, df_test = chronological_split(
        df,
        train_years=ds["train_years"],
        val_years=ds["val_years"],
        test_years=ds["test_years"],
    )

    X_test  = df_test[all_cols].to_numpy(dtype=np.float32)
    y_test  = df_test["active_power_kw"].to_numpy(dtype=np.float32)
    ws_test = df_test["wind_speed_ms"].to_numpy(dtype=np.float32)
    labels_test = df_test["icing_label"].to_numpy(dtype=np.int32) if "icing_label" in df_test else None
    return X_test, y_test, ws_test, labels_test


def predict_da_pinn(checkpoint: str, config: dict, X_test: np.ndarray) -> np.ndarray:
    model = Trainer.load_checkpoint(checkpoint, config)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        return model(X_t).numpy()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        "da_pinn.eval",
        log_file=str(Path(config["paths"]["log_dir"]) / "evaluate.log"),
    )

    X_test, y_test, ws_test, _ = load_test_arrays(config)
    phys = config["physics"]
    all_results: dict[str, dict] = {}

    # DA-PINN
    logger.info("Evaluating DA-PINN …")
    y_pred_pinn = predict_da_pinn(args.checkpoint, config, X_test)
    all_results["DA-PINN (Prop.)"] = regression_report(
        y_test, y_pred_pinn, ws_test, "DA-PINN (Prop.)",
        rho=phys["rho_air_kg_m3"], rotor_area=phys["rotor_area_m2"],
    )

    if args.all:
        logger.info("Evaluating baseline models …")
        from src.models.baselines import (
            PolynomialRegression, XGBoostBaseline, PlainMLPBaseline
        )
        # Add baseline evaluation here with pre-trained models if available.
        # For brevity, only structure is shown — load saved models from disk.
        logger.warning("Baseline evaluation requires pre-trained models in outputs/.")

    print("\n" + "=" * 65)
    print("POWER PREDICTION PERFORMANCE — Kelmarsh Test Set (2023–2024)")
    print_results_table(all_results)

    if args.save_tables:
        out = Path(config["paths"]["predictions_dir"])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Metrics saved → %s/metrics.json", out)


if __name__ == "__main__":
    main()
