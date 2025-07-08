# linear_utils.py
from __future__ import annotations
import numpy as np, torch, optuna
from sklearn.model_selection import train_test_split
from metrics import compute_metrics

torch.manual_seed(42)  # for reproducibility
np.random.seed(42)     # for reproducibility

# ────────────── model definition ──────────────
class LinearReg(torch.nn.Module):
    def __init__(self, d_in: int):       # y = Wx + b
        super().__init__()
        self.linear = torch.nn.Linear(d_in, 1)

    def forward(self, x):                # (B,d) → (B,1)
        return self.linear(x).squeeze(1)

# ────────────── single-run training ───────────
def train_linear(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    lr: float = 1e-3,
    wd: float = 0.0,
    epochs: int = 200,
    patience: int = 25,
    device: str | torch.device | None = None,
):

    dev   = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net   = LinearReg(X_tr.shape[1]).to(dev)
    opt   = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.MSELoss()

    XtrT  = torch.from_numpy(X_tr).float().to(dev)
    ytrT  = torch.from_numpy(y_tr).float().to(dev)
    XvaT  = torch.from_numpy(X_va).float().to(dev)
    yvaT  = torch.from_numpy(y_va).float().to(dev)

    best_rmse, wait, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        net.train();  opt.zero_grad()
        loss = lossf(net(XtrT), ytrT)
        loss.backward();  opt.step()

        net.eval()
        with torch.no_grad():
            rmse = float(torch.sqrt(lossf(net(XvaT), yvaT)))
        if rmse < best_rmse - 1e-4:
            best_rmse, best_state, wait = rmse, net.state_dict(), 0
        else:
            wait += 1
            if wait >= patience: break

    net.load_state_dict(best_state)
    y_pred = net(XvaT).detach().cpu().numpy()
    print(f"LinearReg: {ep} epochs, best RMSE={best_rmse:.4f}, predicted {y_pred[:5]} true: {y_va[:5]}")  # debug info
    return net, compute_metrics(y_va, y_pred)

# ────────────── Optuna hyper-parameter search ──────────
def tune_linear(
    X: np.ndarray, y: np.ndarray,
    n_trials: int = 40,
    seed: int = 42,
):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    def obj(trial: optuna.Trial):
        params = dict(
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            wd = trial.suggest_float("wd", 1e-8, 1e-1, log=True),
            epochs = trial.suggest_int("epochs", 100, 400),
        )
        _, m = train_linear(X_tr, y_tr, X_va, y_va, **params)
        return m["rmse"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    return study.best_params