from __future__ import annotations
from typing   import Tuple, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch import amp                                   # new AMP namespace ✔
from tqdm.auto import tqdm
import optuna

from vit_yield.vit_regressor import ViTYieldRegressor, VitConfig

# ──────────────────────────────────────────────────────────────
#  Dataset + metrics
# ──────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """Holds (X, y) where X is (N,C,H,W) or (N,T,C,H,W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int: return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae  = float(np.abs(y_true - y_pred).mean())
    return {"RMSE": rmse, "MAE": mae}

# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────
@dataclass
class TrainCfg:
    epochs:    int   = 50
    lr:        float = 1e-3
    batch:     int   = 32
    patience:  int   = 5
    clip_grad: float = 1.0

def train_model(
    model: nn.Module,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X:   np.ndarray,
    val_y:   np.ndarray,
    cfg:     TrainCfg = TrainCfg(),
    device:  str      = "cuda",
) -> Tuple[nn.Module, Dict[str, float]]:

    dev  = torch.device(device if torch.cuda.is_available() else "cpu")
    pin  = dev.type == "cuda"
    wkr  = 4 if pin else 0

    train_ds = PatchDataset(train_X, train_y)
    val_ds   = PatchDataset(val_X,   val_y)
    tl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                    pin_memory=pin, num_workers=wkr)
    vl = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False,
                    pin_memory=pin, num_workers=wkr)

    model.to(dev)
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    scaler= amp.GradScaler(device=dev.type)          # new API ✔ :contentReference[oaicite:3]{index=3}
    loss_fn = nn.MSELoss()

    best_rmse, wait, best_state = float("inf"), cfg.patience, None

    for ep in range(1, cfg.epochs + 1):
        # ── train ────────────────────────────────────────────
        model.train(); running = 0.0
        for xb, yb in tqdm(tl, leave=False):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=dev.type):       # new API ✔ :contentReference[oaicite:4]{index=4}
                loss = loss_fn(model(xb).squeeze(), yb)
            scaler.scale(loss).backward()
            if cfg.clip_grad:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt); scaler.update()
            running += loss.item() * xb.size(0)
        sch.step()
        train_rmse = (running / len(train_ds)) ** 0.5

        # ── validate ────────────────────────────────────────
        model.eval(); val_loss = 0.0
        with torch.no_grad(), amp.autocast(device_type=dev.type):
            for xb, yb in vl:
                pred = model(xb.to(dev)).squeeze().cpu()
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_rmse = (val_loss / len(val_ds)) ** 0.5
        print(f"Ep {ep:03d} ┊ train {train_rmse:.3f} ┊ val {val_rmse:.3f}")

        # ── early-stopping logic ────────────────────────────
        if val_rmse < best_rmse:
            best_rmse, wait = val_rmse, cfg.patience
            best_state = {k: v.to("cpu") for k, v in model.state_dict().items()}
        else:
            wait -= 1
            if wait == 0:
                print("Early-stopping triggered."); break

    model.load_state_dict(best_state)
    return model, {"RMSE": best_rmse}

# ──────────────────────────────────────────────────────────────
#  Optuna tuning
# ──────────────────────────────────────────────────────────────
def tune_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    device:   str = "cuda",
) -> optuna.Study:

    def objective(trial: optuna.Trial):
        cfg_vit = VitConfig(
            lstm=True,
            lstm_hidden=trial.suggest_int("hidden", 64, 256),
            lstm_layers=trial.suggest_int("layers", 1, 3),
        )
        cfg_train = TrainCfg(
            epochs   = trial.suggest_int("epochs", 20, 80),
            lr       = trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            batch    = 32,
            patience = 6,
        )

        model = ViTYieldRegressor(cfg_vit)
        # simple hold-out split (80/20)
        n = len(X); split = int(0.8 * n)
        mdl, metrics = train_model(
            model,
            X[:split], y[:split],
            X[split:], y[split:],
            cfg=cfg_train,
            device=device,
        )
        trial.report(metrics["RMSE"], step=cfg_train.epochs)
        return metrics["RMSE"]

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study