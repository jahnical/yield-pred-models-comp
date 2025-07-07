# xgb_utils.py
from __future__ import annotations
from pathlib import Path
import numpy as np, optuna, xgboost as xgb
from sklearn.model_selection import train_test_split
from metrics import compute_metrics           # <- your existing helper

# --- utils ----------------------------------------------------
def _gpu_available() -> bool:
    """True if xgboost was compiled with CUDA and at least one device exists."""
    try:
        import cupy, os  # cupy ships with xgboost wheels that have GPU support
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
    
# ---------- defaults pulled from recent RS-yield papers ----------
BASE_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method= "hist",#"gpu_hist" if _gpu_available() else "hist",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=800,
)

def _dmat(X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y)

# --------------------------------------------------------------
def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    params:  dict | None = None,
    early:   int  = 80,                # ≈10 % of default 800 trees
):
    """Train XGBoost regressor with early stopping; returns (model, metrics)."""
    prm = BASE_PARAMS | (params or {})
    tr_d, va_d = _dmat(X_train, y_train), _dmat(X_val, y_val)

    model = xgb.train(
        prm,
        tr_d,
        num_boost_round=prm.pop("n_estimators"),
        evals=[(va_d, "val")],
        early_stopping_rounds=early,
        verbose_eval=False,
    )

    y_pred = model.predict(va_d, iteration_range=(0, model.best_iteration + 1))
    return model, compute_metrics(y_val, y_pred)


# --------------------------------------------------------------
def tune_xgb_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
):
    """Optuna Bayesian search; returns best param-dict merged into BASE_PARAMS."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        prm = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1200, step=100),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-2, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
        }
        model, m = train_xgb(X_tr, y_tr, X_val, y_val, prm)
        trial.report(m["rmse"], step=0)
        return m["rmse"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return BASE_PARAMS | study.best_trial.params
