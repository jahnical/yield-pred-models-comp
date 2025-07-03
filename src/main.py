#!/usr/bin/env python3
"""
Launch-pad for comparative maize-yield experiments.

Usage examples
--------------
➜ python main.py vit
➜ python main.py vit_lstm --tune
➜ python main.py xgboost --csv data/field_images.csv
"""
from __future__ import annotations
# ───────────────── standard lib ─────────────────
from   dataclasses import dataclass
from   pathlib     import Path
import argparse, sys, textwrap, logging
# ───────────────── third-party ──────────────────
import numpy as np, pandas as pd
# project imports
from data_util          import get_data_splits, compute_metrics
from vit_yield.training import train_model, tune_hyperparams
from vit_yield.vit_regressor import ViTYieldRegressor, VitConfig
from cnn_yield.resnet_regressor import ResNetYieldRegressor, CNNCfg
from tabular            import train_xgboost, tune_xgboost
from linear_regression  import train_linear_regression, tune_linear_regression

# ─────────────────────────────────────────────────────────────
# 1│ Dataclass-based runtime configuration
# ─────────────────────────────────────────────────────────────
@dataclass
class CLIConfig:
    model:        str
    tune:         bool = False
    csv:          Path = Path("/home/matthew/Documents/yield/imagery/Yield_GEE_S2_ByField/field_images.csv")
    data_root:    Path = Path("/home/matthew/Documents/yield/imagery/Yield_GEE_S2_ByField/")

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
        choices=["vit", "vit_lstm", "resnet", "resnet_lstm", "xgboost", "linear"],
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
            vit_cfg = VitConfig(**best.params)      # Optuna -> dataclass
        model = ViTYieldRegressor(vit_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- CNN(-LSTM) ----------
    elif cfg.model in {"resnet", "resnet_lstm"}:
        cnn_cfg = CNNCfg(lstm=cfg.model == "resnet_lstm")
        if cfg.tune and cfg.model == "resnet_lstm":
            best = tune_hyperparams(X_tr, y_tr, 20)
            cnn_cfg = CNNCfg(**best.params)
        model = ResNetYieldRegressor(cnn_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- XGBoost ----------
    elif cfg.model == "xgboost":
        best = tune_xgboost(X_tr, y_tr) if cfg.tune else {}
        model, _, metrics = train_xgboost(X_tr, y_tr, X_va, y_va, param_grid=best)

    # ---------- Linear regression ----------
    elif cfg.model == "linear":
        best = tune_linear_regression(X_tr, y_tr) if cfg.tune else {}
        model, _, metrics = train_linear_regression(X_tr, y_tr, X_va, y_va, **best)

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

    df      = pd.read_csv(cfg.csv)
    splits  = get_data_splits(df, cfg.data_root, cfg.model)
    results = []

    for k, (X_tr, y_tr, X_va, y_va, test_field) in enumerate(splits, 1):
        logging.info(f"Fold {k}/{len(splits)}  —  left-out field: {test_field}")
        _, metrics = build_and_train(cfg, X_tr, y_tr, X_va, y_va)
        results.append(metrics)
        print(f"Fold {k:02d} | {metrics}")

    # ─┐ aggregate & print summary
    #   └──────────────────────────────────────────────
    df_metrics = pd.DataFrame(results)
    print("\n=== Cross-validated summary ===")
    print(df_metrics.describe().loc[["mean", "std"]])

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user ✋")