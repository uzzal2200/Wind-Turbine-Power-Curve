#!/usr/bin/env python
"""
train.py
========
CLI entry point for training DA-PINN (and optionally all baselines).

Usage
-----
  python train.py --config configs/config.yaml
  python train.py --config configs/config.yaml --all-baselines
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from src.models.da_pinn import DAPINN
from src.preprocessing.cleaner import SCADACleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.proxy_labeler import ProxyLabeler
from src.training.dataset import chronological_split
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.io import load_config, load_raw_scada, save_processed, load_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA-PINN")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--all-baselines", action="store_true",
                        help="Also train all five baseline models.")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing if processed data already exists.")
    return parser.parse_args()


def preprocess(config: dict, logger: logging.Logger) -> None:
    """Run full preprocessing pipeline and save results."""
    cleaner   = SCADACleaner(config)
    feat_eng  = FeatureEngineer(config)
    labeler   = ProxyLabeler(config)

    raw_dir = Path(config["paths"]["raw_data_dir"])
    out_dir = Path(config["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(raw_dir.glob("*.csv")):
        logger.info("Preprocessing %s …", csv_file.name)
        df_raw   = load_raw_scada(csv_file)
        df_clean = cleaner.clean(df_raw)
        df_feat  = feat_eng.transform(df_clean)
        df_label = labeler.label(df_feat)
        out_path = out_dir / csv_file.name
        save_processed(df_label, out_path)
        logger.info("Saved %d rows → %s", len(df_label), out_path)


def load_splits(config: dict):
    """Load processed data and return (X_train, y_train, ws_train, X_val, …, X_test, …)."""
    import pandas as pd

    proc_dir = Path(config["paths"]["processed_data_dir"])
    dfs = [load_processed(f) for f in sorted(proc_dir.glob("*.csv"))]
    df  = pd.concat(dfs).sort_index()

    ds    = config["dataset"]
    feats = config["features"]
    all_cols = feats["scada_cols"] + feats["health_vector_cols"]
    target   = "active_power_kw"
    ws_col   = "wind_speed_ms"

    df_train, df_val, df_test = chronological_split(
        df,
        train_years=ds["train_years"],
        val_years=ds["val_years"],
        test_years=ds["test_years"],
    )

    def arrays(split):
        X  = split[all_cols].to_numpy(dtype=np.float32)
        y  = split[target].to_numpy(dtype=np.float32)
        ws = split[ws_col].to_numpy(dtype=np.float32)
        return X, y, ws

    return arrays(df_train), arrays(df_val), arrays(df_test)


def train_da_pinn(config: dict, splits, logger: logging.Logger) -> None:
    (X_tr, y_tr, ws_tr), (X_vl, y_vl, ws_vl), _ = splits
    model   = DAPINN.from_config(config)
    trainer = Trainer(model, config)
    logger.info("DA-PINN — %d trainable parameters", model.count_parameters())
    trainer.fit(X_tr, y_tr, ws_tr, X_vl, y_vl, ws_vl)


def train_baselines(config: dict, splits, logger: logging.Logger) -> None:
    from src.models.baselines import (
        PolynomialRegression, GPRBaseline,
        XGBoostBaseline, LSTMBaseline, PlainMLPBaseline,
    )

    (X_tr, y_tr, ws_tr), _, _ = splits

    baselines = [
        ("Polynomial Regression", PolynomialRegression.from_config(config)),
        ("GPR",                   GPRBaseline.from_config(config)),
        ("XGBoost",               XGBoostBaseline.from_config(config)),
        ("LSTM",                  LSTMBaseline.from_config(config)),
        ("Plain MLP",             PlainMLPBaseline.from_config(config)),
    ]

    for name, model in baselines:
        logger.info("Training %s …", name)
        model.fit(X_tr, y_tr)
        logger.info("%s training complete.", name)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        "da_pinn.train",
        log_file=str(Path(config["paths"]["log_dir"]) / "train.log"),
    )

    if not args.skip_preprocessing:
        preprocess(config, logger)

    splits = load_splits(config)

    logger.info("=== Training DA-PINN ===")
    train_da_pinn(config, splits, logger)

    if args.all_baselines:
        logger.info("=== Training Baseline Models ===")
        train_baselines(config, splits, logger)

    logger.info("All training runs complete.")


if __name__ == "__main__":
    main()
