import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: mean(|y - y_hat|)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return np.nan
    return float(np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask])))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt(mean((y - y_hat)^2))"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2)))

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy: (1/N) * sum(sign(y_hat) == sign(y))
    Excludes exact zero actual returns from DA calculation.
    For exact zero predictions (e.g. naive zero return), sign(0) is 0, which yields False when compared to non-zero actuals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0.0)
    if not np.any(valid_mask):
        return np.nan
        
    actual_sign = np.sign(y_true[valid_mask])
    pred_sign = np.sign(y_pred[valid_mask])
    
    correct_dir = (actual_sign == pred_sign)
    return float(np.mean(correct_dir))

def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, commodity: str) -> Dict[str, Any]:
    """Computes MAE, RMSE, and Directional Accuracy summary dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    n_obs = int(np.sum(valid_mask))
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    da = calculate_directional_accuracy(y_true, y_pred)
    
    return {
        "Commodity": commodity,
        "Model": model_name,
        "N": n_obs,
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da
    }
