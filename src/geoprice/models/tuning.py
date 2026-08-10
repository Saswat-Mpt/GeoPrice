"""
GeoPrice Model Hyperparameter Tuning & Cross-Validation Module.
Implements leak-free inner chronological expanding-window hyperparameter tuning
for ElasticNet regression and LogisticRegression classification.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet, LogisticRegression
from geoprice.constants import ALPHA_GRID, L1_RATIO_GRID, LOGISTIC_C_GRID, MIN_TRAIN_MONTHS

import warnings
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

def select_best_elasticnet_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Tuple[float, ...] = ALPHA_GRID,
    l1_ratio_grid: Tuple[float, ...] = L1_RATIO_GRID,
    n_splits: int = 3
) -> Tuple[float, float]:
    """
    Performs fast inner chronological expanding-window cross-validation using TimeSeriesSplit
    to select optimal ElasticNet (alpha, l1_ratio) hyperparameters on training data X_train, y_train.
    Strictly zero outer or future data leakage.
    """
    n_samples = len(X_train)
    if n_samples < 20:
        return 0.01, 0.5

    split_idx = max(10, int(n_samples * 0.8))
    X_in_tr, y_in_tr = X_train[:split_idx], y_train[:split_idx]
    X_in_val, y_in_val = X_train[split_idx:], y_train[split_idx:]

    scaler = StandardScaler()
    X_in_tr_scaled = scaler.fit_transform(X_in_tr)
    X_in_val_scaled = scaler.transform(X_in_val)

    best_mae = float('inf')
    best_alpha, best_l1 = 0.01, 0.5

    for a in alpha_grid:
        for l1 in l1_ratio_grid:
            model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=200, tol=1e-3, random_state=42)
            model.fit(X_in_tr_scaled, y_in_tr)
            preds = model.predict(X_in_val_scaled)
            mae = mean_absolute_error(y_in_val, preds)
            if mae < best_mae:
                best_mae = mae
                best_alpha, best_l1 = a, l1

    return best_alpha, best_l1

def select_best_logistic_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: Tuple[float, ...] = LOGISTIC_C_GRID
) -> float:
    n_samples = len(X_train)
    if n_samples < 20 or len(np.unique(y_train)) < 2:
        return 1.0

    split_idx = max(10, int(n_samples * 0.8))
    X_in_tr, y_in_tr = X_train[:split_idx], y_train[:split_idx]
    X_in_val, y_in_val = X_train[split_idx:], y_train[split_idx:]

    if len(np.unique(y_in_tr)) < 2 or len(np.unique(y_in_val)) < 2:
        return 1.0

    scaler = StandardScaler()
    X_in_tr_scaled = scaler.fit_transform(X_in_tr)
    X_in_val_scaled = scaler.transform(X_in_val)

    best_acc = -1.0
    best_c = 1.0

    for c_val in c_grid:
        clf = LogisticRegression(C=c_val, max_iter=200, tol=1e-2, random_state=42)
        clf.fit(X_in_tr_scaled, y_in_tr)
        acc = float(clf.score(X_in_val_scaled, y_in_val))
        if acc > best_acc:
            best_acc = acc
            best_c = c_val

    return best_c

from sklearn.ensemble import HistGradientBoostingRegressor

def select_best_hgb_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3
) -> Dict[str, Any]:
    """
    Performs inner chronological CV tuning using TimeSeriesSplit to select optimal
    HistGradientBoostingRegressor hyperparameters strictly on X_train, y_train.
    """
    n_samples = len(X_train)
    if n_samples < 20:
        return {"learning_rate": 0.05, "max_iter": 50, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.1}

    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, n_samples // 15)))

    grid = [
        {"learning_rate": 0.05, "max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 5, "l2_regularization": 0.1, "early_stopping": False},
        {"learning_rate": 0.03, "max_iter": 50, "max_leaf_nodes": 7, "min_samples_leaf": 5, "l2_regularization": 0.0, "early_stopping": False},
        {"learning_rate": 0.10, "max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 1.0, "early_stopping": False},
    ]

    best_mae = float('inf')
    best_params = grid[0]

    for p in grid:
        inner_errors = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_in_tr, y_in_tr = X_train[tr_idx], y_train[tr_idx]
            X_in_val, y_in_val = X_train[val_idx], y_train[val_idx]

            model = HistGradientBoostingRegressor(**p, random_state=42)
            model.fit(X_in_tr, y_in_tr)
            preds = model.predict(X_in_val)
            inner_errors.append(mean_absolute_error(y_in_val, preds))

        mean_inner_mae = float(np.mean(inner_errors)) if len(inner_errors) > 0 else float('inf')
        if mean_inner_mae < best_mae:
            best_mae = mean_inner_mae
            best_params = p

    return best_params
