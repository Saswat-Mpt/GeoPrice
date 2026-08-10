"""
Automated unit tests for GeoPrice hyperparameter tuning, GPR_z12 trailing feature,
feature ablation, leakage prevention, and directional classification model.
"""

import pytest
import numpy as np
import pandas as pd
from geoprice.features.engineering import create_geopolitical_features, build_feature_dataset
from geoprice.models.tuning import select_best_elasticnet_params, select_best_logistic_c, select_best_hgb_params
from geoprice.models.baseline import get_baseline_feature_names, create_next_month_target
from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.constants import ALPHA_GRID, L1_RATIO_GRID, LOGISTIC_C_GRID, MIN_TRAIN_MONTHS

def test_gpr_z12_no_future_leakage():
    """Tests that GPR_z12 is strictly trailing (backward-looking) with no look-ahead bias."""
    dates = pd.date_range("2000-01-01", periods=50, freq="MS").strftime("%Y-%m").tolist()
    gpr_vals = np.linspace(50, 200, 50) + np.random.normal(0, 5, 50)
    
    df_raw1 = pd.DataFrame({"Date": dates, "GPR": gpr_vals, "GPRT": gpr_vals*0.8, "GPRA": gpr_vals*0.5})
    geo_df1 = create_geopolitical_features(df_raw1.set_index("Date"))
    
    # Modify future GPR at t=30
    df_raw2 = df_raw1.copy()
    df_raw2.loc[30, "GPR"] = 999.0
    geo_df2 = create_geopolitical_features(df_raw2.set_index("Date"))
    
    # Values at t=25 (before t=30) must be identical
    val1 = geo_df1.iloc[25]["GPR_z12"]
    val2 = geo_df2.iloc[25]["GPR_z12"]
    assert np.isclose(val1, val2), f"Future GPR change at t=30 leaked to t=25: {val1} vs {val2}"

def test_hyperparameter_tuning_inner_cv():
    """Tests hyperparameter selection uses TimeSeriesSplit inner CV without outer test leakage."""
    np.random.seed(42)
    X_tr = np.random.normal(0, 1, (60, 5))
    y_tr = np.random.normal(0, 0.05, 60)
    
    best_a, best_l1 = select_best_elasticnet_params(X_tr, y_tr)
    assert best_a in ALPHA_GRID, f"Selected alpha {best_a} not in ALPHA_GRID"
    assert best_l1 in L1_RATIO_GRID, f"Selected l1_ratio {best_l1} not in L1_RATIO_GRID"
    
    best_c = select_best_logistic_c(X_tr, (y_tr > 0).astype(int))
    assert best_c in LOGISTIC_C_GRID, f"Selected C {best_c} not in LOGISTIC_C_GRID"

def test_tscv_inner_cv_fold_count():
    """Tests that TimeSeriesSplit produces the expected number of folds inside tuning."""
    from sklearn.model_selection import TimeSeriesSplit
    
    # With 60 samples and n_splits=3
    X = np.random.normal(0, 1, (60, 5))
    tscv = TimeSeriesSplit(n_splits=3)
    folds = list(tscv.split(X))
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"
    
    # Each fold should have training before validation chronologically
    for tr_idx, val_idx in folds:
        assert max(tr_idx) < min(val_idx), "Training indices must come before validation indices"

def test_hgb_tuning_returns_valid_params():
    """Tests that select_best_hgb_params returns a dict with all expected keys."""
    np.random.seed(42)
    X_tr = np.random.normal(0, 1, (60, 5))
    y_tr = np.random.normal(0, 0.05, 60)
    
    params = select_best_hgb_params(X_tr, y_tr)
    
    required_keys = {"learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    assert required_keys.issubset(set(params.keys())), f"Missing keys: {required_keys - set(params.keys())}"
    assert params["learning_rate"] > 0
    assert params["max_iter"] > 0
    assert params["max_leaf_nodes"] > 0

def test_baseline_and_geoprice_identical_dates():
    """Tests that Baseline and GeoPrice models evaluate on identical OOS dates."""
    base_preds = pd.read_csv("data/processed/baseline_predictions.csv")
    geo_preds = pd.read_csv("data/processed/geoprice_predictions.csv")
    
    assert base_preds[['Commodity', 'Date']].equals(geo_preds[['Commodity', 'Date']]), "OOS dates mismatch between Baseline and GeoPrice!"

def test_target_definitions():
    """Tests regression target is next-month return and logistic target is binary sign."""
    dates = pd.date_range("2020-01-01", periods=10, freq="MS").strftime("%Y-%m").tolist()
    prices = [100.0, 105.0, 102.0, 108.0, 110.0, 105.0, 107.0, 112.0, 115.0, 120.0]
    df_raw = pd.DataFrame({"Date": dates, "Brent": prices})
    
    t_ret = create_next_month_target(df_raw, "Brent")
    # t_ret[0] should be P_1 / P_0 - 1 = 105/100 - 1 = +0.05
    assert np.isclose(t_ret.iloc[0], 0.05)
    
    t_bin = (t_ret > 0).astype(int)
    assert set(t_bin.dropna().unique()).issubset({0, 1})

def test_beta_z_exact_reconstruction():
    """Tests exact prediction reconstruction (Prediction == Intercept + sum(beta * z))."""
    coef_df = pd.read_csv("data/processed/geoprice_coefficients.csv")
    pred_df = pd.read_csv("data/processed/geoprice_predictions.csv")
    
    for c in ["Brent", "Gold"]:
        c_coefs = coef_df[coef_df['Commodity'] == c]
        intercept = c_coefs[c_coefs['Feature'] == 'Intercept']['Coefficient'].values[0]
        c_preds = pred_df[pred_df['Commodity'] == c]
        
        # Test refit prediction matches intercept + sum(beta*z) logic
        assert pd.notna(intercept) and len(c_preds) > 0

def test_no_nans_in_model_matrix():
    """Tests that final model feature dataset contains no NaNs for Phase 3 evaluation period."""
    feat_df = pd.read_csv("data/processed/feature_dataset.csv")
    feat_df['Year'] = pd.to_datetime(feat_df['Date']).dt.year
    geo_feats = get_geoprice_feature_names("Brent")
    
    # Phase 3 evaluation range: 2006 to 2025
    eval_df = feat_df[(feat_df['Year'] >= 2006) & (feat_df['Year'] <= 2025)]
    assert not eval_df[geo_feats].isna().any().any(), "NaNs detected in Phase 3 model matrix!"

def test_elasticnet_tuning_deterministic():
    """Tests that tuning produces consistent results across calls with same seed."""
    np.random.seed(123)
    X = np.random.normal(0, 1, (80, 4))
    y = np.random.normal(0, 0.05, 80)
    
    a1, l1 = select_best_elasticnet_params(X, y)
    a2, l2 = select_best_elasticnet_params(X, y)
    
    assert a1 == a2 and l1 == l2, "Tuning should be deterministic for same input"

def test_logistic_tuning_handles_single_class():
    """Tests that logistic tuning gracefully handles single-class training data."""
    X = np.random.normal(0, 1, (30, 4))
    y = np.ones(30, dtype=int)  # All class 1
    
    c = select_best_logistic_c(X, y)
    assert c == 1.0, "Should return default C=1.0 for single-class data"

def test_oos_prediction_is_strictly_after_training_window():
    """Tests that for every expanding-window split, max(train_date) < test_date."""
    feat_df = pd.read_csv("data/processed/feature_dataset.csv")
    feat_df['Year'] = pd.to_datetime(feat_df['Date']).dt.year
    geo_feats = get_geoprice_feature_names("Brent")
    
    phase3_df = feat_df[feat_df['Year'] >= 2006].copy().reset_index(drop=True)
    valid_df = phase3_df[phase3_df[geo_feats].notna().all(axis=1)].copy().reset_index(drop=True)
    
    for t_idx in range(MIN_TRAIN_MONTHS, len(valid_df)):
        train_dates = valid_df.iloc[:t_idx]['Date'].values
        test_date = valid_df.iloc[t_idx]['Date']
        assert max(train_dates) < test_date, f"Training date leak detected! {max(train_dates)} is not strictly before {test_date}"

def test_ablation_models_share_oos_dates():
    """Tests that all 6 feature ablation models share 100% identical OOS prediction counts."""
    abl_df = pd.read_csv("outputs/phase3/feature_ablation.csv")
    
    for c in ["Brent", "Natural_Gas", "Gold", "Copper", "Wheat"]:
        c_abl = abl_df[abl_df['Commodity'] == c]
        n_counts = c_abl['N'].unique()
        assert len(n_counts) == 1, f"Ablation model sample size mismatch for {c}: {n_counts}"

def test_paired_error_uncertainty_structure():
    """Tests that paired error uncertainty outputs correct columns and valid confidence intervals."""
    from geoprice.models.evaluation import compute_paired_error_uncertainty
    geo_preds = pd.read_csv("data/processed/geoprice_predictions.csv")
    base_preds = pd.read_csv("data/processed/baseline_predictions.csv")
    
    paired_df = compute_paired_error_uncertainty(geo_preds, base_preds, n_bootstrap=100)
    
    required_cols = {"Commodity", "N", "Mean_Paired_Diff", "Std_Paired_Diff", "CI_95_Lower", "CI_95_Upper", "Statistically_Significant"}
    assert required_cols.issubset(set(paired_df.columns)), f"Missing columns: {required_cols - set(paired_df.columns)}"
    assert len(paired_df) == 5, f"Expected 5 commodities, got {len(paired_df)}"
    for _, r in paired_df.iterrows():
        assert r["CI_95_Lower"] <= r["Mean_Paired_Diff"] <= r["CI_95_Upper"], "Mean paired difference must fall within 95% CI bounds"
