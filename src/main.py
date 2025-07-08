#!/usr/bin/env python3
"""
Launch‑pad for comparative maize‑yield experiments, now extended with
**inference‑time benchmarking**.  The script still trains each model on
nested leave‑one‑field‑out (LOFO) splits but, after training, records the
average milliseconds per sample required for forward inference on the
validation set.  This enables the accuracy‑versus‑efficiency trade‑off
analysis discussed in Section 4.

Usage examples
-------------------------
➜ python main.py vit
➜ python main.py vit_lstm --tune
➜ python main.py xgb --csv data/field_images.csv                  
"""
from __future__ import annotations
# ───────────────── standard lib ─────────────────
from   dataclasses import dataclass
from   pathlib     import Path
import argparse, sys, textwrap, logging, time
# ───────────────── third‑party ──────────────────
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
# Helper: average inference time (ms) on a validation set
# ─────────────────────────────────────────────────────────────

def avg_infer_time_ms(model, X_val: np.ndarray, *, device: str = "cuda", batch: int = 64) -> float:
    """Return mean milliseconds per sample for forward passes on *X_val*.
    Handles PyTorch modules and XGBoost Booster objects.  The caller must
    ensure the model is already on the desired device.
    """
    n = len(X_val)

    # ── XGBoost (CPU‑only) ───────────────────────────────
    if isinstance(model, xgb.Booster):
        dm = xgb.DMatrix(X_val)
        t0 = time.perf_counter()
        _ = model.predict(dm)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000 / n

    # ── PyTorch (CPU/GPU) ────────────────────────────────
    torch_dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval(); model.to(torch_dev)
    X_tensor = torch.from_numpy(X_val).float()
    dl = torch.utils.data.DataLoader(X_tensor, batch_size=batch, shuffle=False)

    # GPU timing needs explicit synchronisation
    if torch_dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for xb in dl:
            _ = model(xb.to(torch_dev))
    if torch_dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n

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
    soil_root:    Path = Path("/home/matthew/Projects/comparative-yield-models/Yield_GEE_SoilOrg/")
    device:       str  = "cuda"   # add a quick switch for CPU timing

# ─────────────────────────────────────────────────────────────
# 2│ Argument-parser that builds CLIConfig
# ─────────────────────────────────────────────────────────────

def parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Train a single model (optionally with Optuna tuning) on all
            leave‑one‑field‑out splits, report per‑fold metrics **and**
            average inference time (ms per sample).
            """),
    )
    parser.add_argument(
        "model",
        choices=["vit", "vit_lstm", "resnet", "resnet_lstm", "xgb", "linear"],
        help="Model/experiment to run",
    )
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyper‑parameter search")
    parser.add_argument("--csv",  type=Path, help="CSV with field -> image paths")
    parser.add_argument("--data_root", type=Path, help="Directory with imagery TIFFs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device for PyTorch models")

    ns = parser.parse_args()
    return CLIConfig(
        model=ns.model,
        tune=ns.tune,
        csv=ns.csv or CLIConfig.csv,
        data_root=ns.data_root or CLIConfig.data_root,
        device=ns.device,
    )

# ─────────────────────────────────────────────────────────────
# 3│ Factory/dispatch table (no long if‑elif chains)
# ─────────────────────────────────────────────────────────────

def build_and_train(cfg: CLIConfig,
                    X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray):
    """Return trained model + metrics dict (RMSE, MAE, R²)."""
    # ---------- Vision‑Transformer family ----------
    if cfg.model in {"vit", "vit_lstm"}:
        vit_cfg = VitConfig(lstm=cfg.model == "vit_lstm")
        if cfg.tune and cfg.model == "vit_lstm":
            best = tune_hyperparams(X_tr, y_tr, 20)
            vit_cfg = VitConfig(lstm_hidden=best.params["hidden"],
                                lstm_layers=best.params["layers"], lstm=True)
        model = ViTYieldRegressor(vit_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- CNN(-LSTM) ----------
    elif cfg.model in {"resnet", "resnet_lstm"}:
        cnn_cfg = CNNCfg(lstm=cfg.model == "resnet_lstm")
        if cfg.tune and cfg.model == "resnet_lstm":
            best = tune_resnet_hyperparams(X_tr, y_tr, 20)
            cnn_cfg = CNNCfg(**best.params)  # Optuna -> dataclass
        model = ResNetYieldRegressor(cnn_cfg)
        model, metrics = train_model(model, X_tr, y_tr, X_va, y_va)[0:2]

    # ---------- xgb ----------
    elif cfg.model == "xgb":
        best = tune_xgb_optuna(X_tr, y_tr) if cfg.tune else {}
        model, metrics = train_xgb(X_tr, y_tr, X_va, y_va, params=best)

    # ---------- Linear regression ----------
    elif cfg.model == "linear":
        best = tune_linear(X_tr, y_tr) if cfg.tune else {}
        model, metrics = train_linear(X_tr, y_tr, X_va, y_va, **best)

    else:
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
        logging.info(f"Fold {k}/{len(images_splits)}  —  left‑out field: {test_field}")
        model, metrics = build_and_train(cfg, X_tr, y_tr, X_va, y_va)

        # ── Timing ─────────────────────────────────────────
        infer_ms = avg_infer_time_ms(model, X_va, device=cfg.device)
        metrics["infer_ms"] = infer_ms

        logging.info(f"Fold {k:02d} | {metrics} | {infer_ms:.2f} ms/sample")
        results.append(metrics)

    # ────────────────────────────────────────────────
    # Display per‑fold metrics
    print("\n=== Per‑fold metrics ===")
    for k, m in enumerate(results, 1):
        print(f"Fold {k:02d} | {m}")

    # ─┐ aggregate & print summary
    df_metrics = pd.DataFrame(results)
    print("\n=== Cross‑validated summary (mean ± SD) ===")
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
