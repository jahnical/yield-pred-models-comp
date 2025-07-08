"""
Reproducibility & Workflow Documentation
=======================================

This project is designed for full reproducibility of comparative yield modeling experiments. To ensure that reviewers and collaborators can exactly reproduce your results, follow these steps:

1. **Environment Setup**
   - Clone the repository.
   - Use the provided `requirements.txt` to install all dependencies:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
   - The project uses fixed versions for all dependencies. If you use a different environment, ensure all package versions match those in `requirements.txt`.

2. **Random Seeds**
   - All scripts set `np.random.seed(42)` and `torch.manual_seed(42)` for deterministic results.
   - For full determinism in PyTorch, you may also set:
     ```python
     torch.use_deterministic_algorithms(True)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     ```
   - Note: Some GPU operations may still introduce minor non-determinism.

3. **Data**
   - Place all required data (e.g., Sentinel-2 TIFFs, CSVs) in the specified folders as described in the README.
   - Use the same CSVs and imagery as referenced in your experiments.

4. **Running Experiments**
   - Use the main script as described in the README and docstrings:
     ```bash
     python src/main.py vit --csv data/field_images.csv --data_root data/Yield_GEE_S2_ByField
     ```
   - For hyperparameter tuning, add `--tune`.
   - All results, metrics, and model checkpoints are saved with clear naming conventions.

5. **Results & Reporting**
   - All metrics are computed using the same helper functions (`metrics.py`) for consistency.
   - Cross-validation splits are deterministic and based on field names.
   - Inference-time benchmarking is included for efficiency comparisons.

6. **Version Control**
   - All code, configuration, and scripts are tracked in git.
   - Any changes to the workflow should be committed and described in commit messages.

7. **License**
   - The project is released under GPL-3.0. All derivative works must also be open source under the same license.

8. **Contact**
   - For questions or issues, open an issue on the repository or contact the maintainer.

By following these steps, reviewers and collaborators can reproduce all results and analyses exactly as reported.
"""

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
from cnn_yield.resnet_regressor import CNNCfg, ResNetYieldRegressor
import optuna

from vit_yield.vit_regressor import ViTYieldRegressor, VitConfig
from metrics import compute_metrics  # <-- central helper (RMSE, MAE, R²)

torch.manual_seed(42)  # for reproducibility
np.random.seed(42)     # for reproducibility

# ──────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """Holds (X, y) where X is (N,C,H,W) or (N,T,C,H,W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int: return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────
@dataclass
class TrainCfg:
    epochs:    int   = 50
    lr:        float = 1e-3
    batch:     int   = 32
    patience:  int   = 10
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
    scaler= amp.GradScaler(device=dev.type)          # new API ✔
    loss_fn = nn.MSELoss()

    best_rmse, wait, best_state = float("inf"), cfg.patience, None

    for ep in range(1, cfg.epochs + 1):
        # ── train ────────────────────────────────────────────
        model.train(); running = 0.0
        for xb, yb in tqdm(tl, leave=False):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=dev.type):
                loss = loss_fn(model(xb).squeeze(), yb)
            scaler.scale(loss).backward()
            if cfg.clip_grad:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt); scaler.update()
            running += loss.item() * xb.size(0)
        sch.step()
        train_rmse = (running / len(train_ds)) ** 0.5

        def validate(model):
            # ── validate ────────────────────────────────────────
            model.eval(); y_preds, y_true = [], []
            with torch.no_grad(), amp.autocast(device_type=dev.type):
                for xb, yb in vl:
                    pred = model(xb.to(dev)).squeeze().cpu()
                    y_preds.append(pred.numpy())
                    y_true.append(yb.numpy())
            y_pred_np = np.concatenate(y_preds)
            y_true_np = np.concatenate(y_true)
            return compute_metrics(y_true_np, y_pred_np)
        
        val_metrics = validate(model)
        val_rmse = val_metrics["rmse"]

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
    
    return model, validate(model)  # returns RMSE, MAE, R²

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
        trial.report(metrics["rmse"], step=cfg_train.epochs)
        return metrics["rmse"]

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial


#Hyper parameter tuning for ResNet model
def tune_resnet_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    device:   str = "cuda",
) -> optuna.Study:
    """Optuna hyper-parameter tuning for ResNet model."""
    
    def objective(trial: optuna.Trial):
        cfg_cnn = CNNCfg.from_optuna_trial(trial)

        model = ResNetYieldRegressor(cfg_cnn)
        # simple hold-out split (80/20)
        n = len(X); split = int(0.8 * n)
        mdl, metrics = train_model(
            model,
            X[:split], y[:split],
            X[split:], y[split:],
            cfg=TrainCfg(),
            device=device,
        )
        trial.report(metrics["rmse"], step=TrainCfg.epochs)
        return metrics["rmse"]

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial