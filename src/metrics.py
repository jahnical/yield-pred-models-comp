# metrics.py
"""Small helper to compute standard regression metrics."""

from typing import Dict
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE and R² in one dict (ready for DataFrame())."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}
