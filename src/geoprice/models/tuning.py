"""
GeoPrice Model Hyperparameter Tuning & Cross-Validation Module.
Implements leak-free inner chronological cross-validation using TimeSeriesSplit
for ElasticNet regression, LogisticRegression classification, and HistGradientBoosting.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from geoprice.constants import ALPHA_GRID, L1_RATIO_GRID, LOGISTIC_C_GRID


def select_best_elasticnet_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Tuple[float, ...] = ALPHA_GRID,
    l1_ratio_grid: Tuple[float, ...] = L1_RATIO_GRID,
    n_splits: int = 3
) -> Tuple[float, float]:
    """
    Inner chronological cross-validation using TimeSeriesSplit to select optimal
    ElasticNet (alpha, l1_ratio). Strictly zero outer or future data leakage.

    Returns (best_alpha, best_l1_ratio) selected by lowest mean validation MAE
    across chronological folds.
    """
    n_samples = len(X_train)
    if n_samples < 20:
        return 0.01, 0.5

    effective_splits = min(n_splits, max(2, n_samples // 15))
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    best_mae = float('inf')
    best_alpha, best_l1 = 0.01, 0.5

    for a in alpha_grid:
        for l1 in l1_ratio_grid:
            fold_maes = []
            for tr_idx, val_idx in tscv.split(X_train):
                X_in_tr, y_in_tr = X_train[tr_idx], y_train[tr_idx]
                X_in_val, y_in_val = X_train[val_idx], y_train[val_idx]

                scaler = StandardScaler()
                X_in_tr_s = scaler.fit_transform(X_in_tr)
                X_in_val_s = scaler.transform(X_in_val)

                model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=200, tol=1e-3, random_state=42)
                model.fit(X_in_tr_s, y_in_tr)
                preds = model.predict(X_in_val_s)
                fold_maes.append(mean_absolute_error(y_in_val, preds))

            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_alpha, best_l1 = a, l1

    return best_alpha, best_l1


def select_best_logistic_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: Tuple[float, ...] = LOGISTIC_C_GRID,
    n_splits: int = 3
) -> float:
    """
    Inner chronological cross-validation using TimeSeriesSplit to select optimal
    LogisticRegression C parameter. Scored by balanced accuracy to prevent
    class-imbalance bias.

    Returns best_C selected by highest mean validation balanced accuracy.
    """
    n_samples = len(X_train)
    if n_samples < 20 or len(np.unique(y_train)) < 2:
        return 1.0

    effective_splits = min(n_splits, max(2, n_samples // 15))
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    best_score = -1.0
    best_c = 1.0

    for c_val in c_grid:
        fold_scores = []
        valid_folds = 0
        for tr_idx, val_idx in tscv.split(X_train):
            X_in_tr, y_in_tr = X_train[tr_idx], y_train[tr_idx]
            X_in_val, y_in_val = X_train[val_idx], y_train[val_idx]

            if len(np.unique(y_in_tr)) < 2 or len(np.unique(y_in_val)) < 2:
                continue

            scaler = StandardScaler()
            X_in_tr_s = scaler.fit_transform(X_in_tr)
            X_in_val_s = scaler.transform(X_in_val)

            clf = LogisticRegression(C=c_val, max_iter=200, tol=1e-4, random_state=42)
            clf.fit(X_in_tr_s, y_in_tr)
            y_pred = clf.predict(X_in_val_s)
            fold_scores.append(balanced_accuracy_score(y_in_val, y_pred))
            valid_folds += 1

        if valid_folds > 0:
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_c = c_val

    return best_c


def select_best_hgb_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3
) -> Dict[str, Any]:
    """
    Inner chronological cross-validation using TimeSeriesSplit to select optimal
    HistGradientBoostingRegressor hyperparameters. Uses a candidate grid of 6 representative
    configurations for explainability and fast evaluation.

    Returns dict of best hyperparameters selected by lowest mean validation MAE.
    """
    n_samples = len(X_train)
    default_params = {
        "learning_rate": 0.05, "max_iter": 50, "max_leaf_nodes": 15,
        "min_samples_leaf": 20, "l2_regularization": 0.1, "early_stopping": False
    }
    if n_samples < 30:
        return default_params

    effective_splits = min(n_splits, max(2, n_samples // 15))
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    grid = [
        {"learning_rate": 0.03, "max_iter": 50,  "max_leaf_nodes": 7,  "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"learning_rate": 0.03, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 50,  "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 7,  "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"learning_rate": 0.10, "max_iter": 50,  "max_leaf_nodes": 31, "min_samples_leaf": 10, "l2_regularization": 0.0},
    ]

    # Add early_stopping: False to all entries
    for p in grid:
        p["early_stopping"] = False

    best_mae = float('inf')
    best_params = grid[0].copy()

    for p in grid:
        fold_maes = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_in_tr, y_in_tr = X_train[tr_idx], y_train[tr_idx]
            X_in_val, y_in_val = X_train[val_idx], y_train[val_idx]

            model = HistGradientBoostingRegressor(**p, random_state=42)
            model.fit(X_in_tr, y_in_tr)
            preds = model.predict(X_in_val)
            fold_maes.append(mean_absolute_error(y_in_val, preds))

        mean_mae = float(np.mean(fold_maes))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = p.copy()

    return best_params
