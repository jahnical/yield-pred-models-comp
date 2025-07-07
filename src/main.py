#!/usr/bin/env python3
"""
Launch-pad for comparative maize-yield experiments.

Usage examples
--------------
➜ python main.py vit
➜ python main.py vit_lstm --tune
➜ python main.py xgb --csv data/field_images.csv
"""
from __future__ import annotations
# ───────────────── standard lib ─────────────────
from   dataclasses import dataclass
from   pathlib     import Path
import argparse, sys, textwrap, logging
# ───────────────── third-party ──────────────────
import numpy as np, pandas as pd
import xgboost as xgb
import torch
# project imports
from data_util          import get_data_splits, compute_metrics
from vit_yield.training import train_model, tune_hyperparams, tune_resnet_hyperparams
from vit_yield.vit_regressor import ViTYieldRegressor, VitConfig
from cnn_yield.resnet_regressor import ResNetYieldRegressor, CNNCfg
from xgb_utils import train_xgb, tune_xgb_optuna
from linear_utils import train_linear, tune_linear

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────
# 1│ Dataclass-based runtime configuration
# ─────────────────────────────────────────────────────────────
@dataclass
class CLIConfig:
    model:        str
    tune:         bool = False
    csv:          Path = Path("/home/matthew/Projects/comparative-yield-models/Yield_GEE_S2_ByField/field_images_grouped.csv")
    data_root:    Path = Path("/home/matthew/Projects/comparative-yield-models/")
    temp_csv:     Path = Path("/home/matthew/Projects/comparative-yield-models/Yield_GEE_TEMP_ByField/field_temps.csv")
    soil_root:  Path = Path("/home/matthew/Projects/comparative-yield-models/Yield_GEE_SoilOrg/")

# ─────────────────────────────────────────────────────────────
# 2│ Argument-parser that builds CLIConfig
#    (mutually-exclusive flags, automatic help formatting)
# ─────────────────────────────────────────────────────────────
def parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Run a single model (optionally with Optuna tuning) on all
            leave-one-field-out splits and print per-fold metrics.
            """),
    )
    parser.add_argument(
        "model",
        choices=["vit", "vit_lstm", "resnet", "resnet_lstm", "xgb", "linear"],
        help="Model/experiment to run",
    )
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyper-parameter search")
    parser.add_argument("--csv",  type=Path, help="CSV with field -> image paths")
    parser.add_argument("--data_root", type=Path, help="Directory with imagery TIFFs")

    ns = parser.parse_args()
    return CLIConfig(
        model=ns.model,
        tune=ns.tune,
        csv=ns.csv or CLIConfig.csv,
        data_root=ns.data_root or CLIConfig.data_root,
    )

# ─────────────────────────────────────────────────────────────
# 3│ Factory/dispatch table (no long if-elif chains)
# ─────────────────────────────────────────────────────────────
def build_and_train(cfg: CLIConfig,
                    X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray):
    """Return trained model + metrics dict."""
    # ---------- Vision-Transformer family ----------
    if cfg.model in {"vit", "vit_lstm"}:
        vit_cfg = VitConfig(lstm=cfg.model == "vit_lstm")
        if cfg.tune and cfg.model == "vit_lstm":
            best = tune_hyperparams(X_tr, y_tr, 20)
            vit_cfg = VitConfig(lstm_hidden=best.params['hidden'], lstm_layers=best.params['layers'], lstm=True)      # Optuna -> dataclass
        model = ViTYieldRegressor(vit_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- CNN(-LSTM) ----------
    elif cfg.model in {"resnet", "resnet_lstm"}:
        cnn_cfg = CNNCfg(lstm=cfg.model == "resnet_lstm")
        if cfg.tune and cfg.model == "resnet_lstm":
            best = tune_resnet_hyperparams(X_tr, y_tr, 20)
            cnn_cfg = CNNCfg(lstm_hidden=best.params["lstm_hidden"],
                             lstm_layers=best.params["lstm_layers"], lstm=True)  # Optuna -> dataclass
        model = ResNetYieldRegressor(cnn_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- xgb ----------
    elif cfg.model == "xgb":
        best = tune_xgb_optuna(X_tr, y_tr) if cfg.tune else {}
        print(f"Best params: {best}")  # debug info
        model, metrics = train_xgb(X_tr, y_tr, X_va, y_va, params=best)

    # ---------- Linear regression ----------
    elif cfg.model == "linear":
        best = tune_linear(X_tr, y_tr) if cfg.tune else {}
        model, metrics = train_linear(X_tr, y_tr, X_va, y_va, **best)


    else:  # should never happen due to argparse choices
        raise ValueError(f"Unknown model {cfg.model}")

    return model, metrics

# ─────────────────────────────────────────────────────────────
# 4│ Main routine
# ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Reading CSV and building LOFO splits …")
    
    temp_df = pd.read_csv(cfg.temp_csv)
    images_df      = pd.read_csv(cfg.csv)
    images_splits  = get_data_splits(images_df, temp_df, cfg.data_root, cfg.model, (20241201, 20250228))    
    
    results = []

    for k, (X_tr, y_tr, X_va, y_va, test_field) in enumerate(images_splits, 1):
        logging.info(f"Fold {k}/{len(images_splits)}  —  left-out field: {test_field}")
        model, metrics = build_and_train(cfg, X_tr, y_tr, X_va, y_va)
        
        print(f"Example model predictions (first 5) Y value: {y_va[:5]}")
        if cfg.model in {"vit", "vit_lstm", "resnet", "resnet_lstm"}:
            X_tensor = torch.from_numpy(X_va[:5]).float().to('cuda' if torch.cuda.is_available() else 'cpu')
            print(model(X_tensor))
        elif cfg.model == "xgb":
            print(model.predict(xgb.DMatrix(X_va[:5])))
        elif cfg.model == "linear":
            X_tensor = torch.from_numpy(X_va[:5]).float().to('cuda' if torch.cuda.is_available() else 'cpu')
            print(model(X_tensor))
        else:
            raise ValueError(f"Unknown model {cfg.model}")
        
        results.append(metrics)
        print(f"Fold {k:02d} | {metrics}")
        
    # ────────────────────────────────────────────────
    # Display per-fold metrics
    print("\n=== Per-fold metrics ===")
    for k, m in enumerate(results, 1):
        print(f"Fold {k:02d} | {m}")  
        
    # ─┐ aggregate & print summary
    #   └──────────────────────────────────────────────
    df_metrics = pd.DataFrame(results)
    print("\n=== Cross-validated summary ===")
    print(df_metrics.describe().loc[["mean", "std"]])
    
    out_csv = Path(cfg.data_root, f"metrics_{cfg.model}.csv")
    df_metrics.to_csv(out_csv, index_label="fold")
    print(f"\nSaved fold metrics ➜ {out_csv}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user ✋")