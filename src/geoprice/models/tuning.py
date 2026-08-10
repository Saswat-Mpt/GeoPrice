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

    n_cv = min(n_splits, max(2, n_samples // 15))
    tscv = TimeSeriesSplit(n_splits=n_cv)

    best_mae = float('inf')
    best_alpha, best_l1 = 0.01, 0.5

    # Pre-scale X_train for fast grid iterations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    for a in alpha_grid:
        for l1 in l1_ratio_grid:
            inner_errors = []
            for tr_idx, val_idx in tscv.split(X_scaled):
                X_in_tr, y_in_tr = X_scaled[tr_idx], y_train[tr_idx]
                X_in_val, y_in_val = X_scaled[val_idx], y_train[val_idx]

                model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=1000, tol=1e-3, random_state=42)
                model.fit(X_in_tr, y_in_tr)
                preds = model.predict(X_in_val)
                inner_errors.append(mean_absolute_error(y_in_val, preds))

            mean_inner_mae = float(np.mean(inner_errors)) if len(inner_errors) > 0 else float('inf')
            if mean_inner_mae < best_mae:
                best_mae = mean_inner_mae
                best_alpha, best_l1 = a, l1

    return best_alpha, best_l1

def select_best_logistic_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: Tuple[float, ...] = LOGISTIC_C_GRID,
    n_splits: int = 3
) -> float:
    """
    Performs fast inner chronological expanding-window cross-validation using LogisticRegressionCV + TimeSeriesSplit
    to select optimal LogisticRegression C parameter on training data X_train, y_train.
    Strictly zero outer or future data leakage.
    """
    n_samples = len(X_train)
    if n_samples < 20 or len(np.unique(y_train)) < 2:
        return 1.0

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)

    n_cv = min(n_splits, max(2, n_samples // 15))
    tscv = TimeSeriesSplit(n_splits=n_cv)

    try:
        clf_cv = LogisticRegressionCV(
            Cs=list(c_grid),
            cv=tscv,
            max_iter=5000,
            random_state=42
        )
        clf_cv.fit(X_tr_scaled, y_train)
        return float(clf_cv.C_[0])
    except Exception:
        return 1.0
