import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from data_util import compute_metrics
from sklearn.model_selection import train_test_split

def train_xgboost(X_train, y_train, X_val, y_val, param_grid=None):
    if param_grid is None:
        param_grid = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
        }
    model = xgb.XGBRegressor(**param_grid)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    return model, y_pred, metrics

def tune_xgboost(X, y, n_trials=20):
    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        param = {
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'reg:squarederror',
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
