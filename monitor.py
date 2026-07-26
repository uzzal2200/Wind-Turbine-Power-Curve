#!/usr/bin/env python
"""
monitor.py
==========
Run the CUSUM physics-residual icing monitor on a SCADA CSV feed.

Usage
-----
  # Calibrate on validation set, then monitor test set
  python monitor.py --config configs/config.yaml \
                    --checkpoint outputs/checkpoints/best_model.pt \
                    --calibrate

  # Monitor a new SCADA file
  python monitor.py --config configs/config.yaml \
                    --checkpoint outputs/checkpoints/best_model.pt \
                    --input data/raw/new_scada.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.monitoring.cusum import CUSUMMonitor
from src.training.trainer import Trainer
from src.utils.io import load_config, load_processed
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUSUM icing monitor")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--input",      default=None,
                        help="Path to new SCADA CSV for monitoring. "
                             "Defaults to test split if not specified.")
    parser.add_argument("--calibrate",  action="store_true",
                        help="Re-calibrate CUSUM on validation healthy periods.")
    return parser.parse_args()


def predict(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32)
            preds.append(model(xb).numpy())
    return np.concatenate(preds)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        "da_pinn.monitor",
        log_file=str(Path(config["paths"]["log_dir"]) / "monitor.log"),
    )

    model   = Trainer.load_checkpoint(args.checkpoint, config)
    monitor = CUSUMMonitor(config)

    ds    = config["dataset"]
    feats = config["features"]
    all_cols = feats["scada_cols"] + feats["health_vector_cols"]

    import pandas as pd
    proc_dir = Path(config["paths"]["processed_data_dir"])
    dfs = [load_processed(f) for f in sorted(proc_dir.glob("*.csv"))]
    df  = pd.concat(dfs).sort_index()

    if args.calibrate:
        logger.info("Calibrating CUSUM on validation healthy periods …")
        df_val = df[df.index.year.isin(ds["val_years"])]
        healthy = df_val[df_val.get("icing_label", pd.Series(0, index=df_val.index)) == 0]
        X_h  = healthy[all_cols].to_numpy(dtype=np.float32)
        ws_h = healthy["wind_speed_ms"].to_numpy(dtype=np.float32)
        y_h  = healthy["active_power_kw"].to_numpy(dtype=np.float32)
        ph   = predict(model, X_h)
        monitor.calibrate(ph, y_h, ws_h)

    # Choose monitoring input
    if args.input:
        from src.preprocessing.cleaner import SCADACleaner
        from src.preprocessing.feature_engineer import FeatureEngineer
        from src.preprocessing.proxy_labeler import ProxyLabeler
        from src.utils.io import load_raw_scada

        logger.info("Loading new SCADA file: %s", args.input)
        df_new  = load_raw_scada(args.input)
        df_new  = SCADACleaner(config).clean(df_new)
        df_new  = FeatureEngineer(config).transform(df_new)
        df_mon  = ProxyLabeler(config).label(df_new)
    else:
        logger.info("Using test split (2023–2024) for monitoring …")
        df_mon = df[df.index.year.isin(ds["test_years"])]

    X_mon  = df_mon[all_cols].to_numpy(dtype=np.float32)
    ws_mon = df_mon["wind_speed_ms"].to_numpy(dtype=np.float32)
    y_mon  = df_mon["active_power_kw"].to_numpy(dtype=np.float32)

    logger.info("Running CUSUM monitor on %d timesteps …", len(X_mon))
    p_hat  = predict(model, X_mon)
    result = monitor.run(p_hat, y_mon, ws_mon)

    n_alerts = len(result.alert_times)
    logger.info("Monitoring complete: %d alert crossing(s) detected.", n_alerts)

    for i, t in enumerate(result.alert_times):
        logger.info("  Alert %d at step %d (Hour %.1f)", i + 1, t, t * 10 / 60)

    # Save residuals
    out_dir = Path(config["paths"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    out_df = pd.DataFrame({
        "timestamp":  df_mon.index,
        "p_actual":   y_mon,
        "p_hat":      p_hat,
        "residual":   result.residuals,
        "cusum":      result.cusum_vals,
        "alert":      result.alert_mask.astype(int),
    })
    out_path = out_dir / "cusum_monitor_output.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("CUSUM output saved → %s", out_path)


if __name__ == "__main__":
    main()
