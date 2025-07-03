import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from data_util import compute_metrics

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        return self.linear(x)

def train_linear_regression(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3):
    model = LinearRegressionModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(torch.from_numpy(X_train).float())
        loss = loss_fn(y_pred.squeeze(), torch.from_numpy(y_train).float())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_val).float()).squeeze().numpy()
    metrics = compute_metrics(y_val, y_pred)
    return model, y_pred, metrics

def tune_linear_regression(X, y, n_trials=20):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 50, 300)
        model = LinearRegressionModel(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(torch.from_numpy(X).float())
            loss = loss_fn(y_pred.squeeze(), torch.from_numpy(y).float())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(X).float()).squeeze().numpy()
        rmse = ((y_pred - y) ** 2).mean() ** 0.5
        return rmse
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
