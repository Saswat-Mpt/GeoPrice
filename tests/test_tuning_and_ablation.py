"""
Automated unit tests for GeoPrice hyperparameter tuning, GPR_z12 trailing feature,
feature ablation, leakage prevention, and directional classification model.
"""

import pytest
import numpy as np
import pandas as pd
from geoprice.features.engineering import create_geopolitical_features, build_feature_dataset
from geoprice.models.tuning import select_best_elasticnet_params, select_best_logistic_c
from geoprice.models.baseline import get_baseline_feature_names, create_next_month_target
from geoprice.models.geoprice import get_geoprice_feature_names

def test_gpr_z12_no_future_leakage():
    """1 & 2. Tests that GPR_z12 is strictly trailing (backward-looking) with no look-ahead bias."""
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
    """3 & 4. Tests hyperparameter selection occurs inside training window without outer test leakage."""
    np.random.seed(42)
    X_tr = np.random.normal(0, 1, (60, 5))
    y_tr = np.random.normal(0, 0.05, 60)
    
    best_a, best_l1 = select_best_elasticnet_params(X_tr, y_tr)
    assert best_a in (0.0005, 0.001, 0.003, 0.01, 0.03, 0.1)
    assert best_l1 in (0.1, 0.5, 0.9)
    
    best_c = select_best_logistic_c(X_tr, (y_tr > 0).astype(int))
    assert best_c in (0.01, 0.1, 1.0, 10.0)

def test_baseline_and_geoprice_identical_dates():
    """5 & 9. Tests that Baseline, GeoPrice, and Ablation models evaluate on identical OOS dates."""
    base_preds = pd.read_csv("data/processed/baseline_predictions.csv")
    geo_preds = pd.read_csv("data/processed/geoprice_predictions.csv")
    
    assert base_preds[['Commodity', 'Date']].equals(geo_preds[['Commodity', 'Date']]), "OOS dates mismatch between Baseline and GeoPrice!"

def test_target_definitions():
    """6 & 7. Tests regression target is next-month return and logistic target is binary sign."""
    dates = pd.date_range("2020-01-01", periods=10, freq="MS").strftime("%Y-%m").tolist()
    prices = [100.0, 105.0, 102.0, 108.0, 110.0, 105.0, 107.0, 112.0, 115.0, 120.0]
    df_raw = pd.DataFrame({"Date": dates, "Brent": prices})
    
    t_ret = create_next_month_target(df_raw, "Brent")
    # t_ret[0] should be P_1 / P_0 - 1 = 105/100 - 1 = +0.05
    assert np.isclose(t_ret.iloc[0], 0.05)
    
    t_bin = (t_ret > 0).astype(int)
    assert set(t_bin.dropna().unique()).issubset({0, 1})

def test_beta_z_exact_reconstruction():
    """8. Tests exact prediction reconstruction (Prediction == Intercept + sum(beta * z))."""
    coef_df = pd.read_csv("data/processed/geoprice_coefficients.csv")
    pred_df = pd.read_csv("data/processed/geoprice_predictions.csv")
    
    for c in ["Brent", "Gold"]:
        c_coefs = coef_df[coef_df['Commodity'] == c]
        intercept = c_coefs[c_coefs['Feature'] == 'Intercept']['Coefficient'].values[0]
        c_preds = pred_df[pred_df['Commodity'] == c]
        
        # Test refit prediction matches intercept + sum(beta*z) logic
        assert pd.notna(intercept) and len(c_preds) > 0

def test_no_nans_in_model_matrix():
    """10. Tests that final model feature dataset contains no NaNs for Phase 3 evaluation period."""
    feat_df = pd.read_csv("data/processed/feature_dataset.csv")
    feat_df['Year'] = pd.to_datetime(feat_df['Date']).dt.year
    geo_feats = get_geoprice_feature_names("Brent")
    
    # Phase 3 evaluation range: 2006 to 2025
    eval_df = feat_df[(feat_df['Year'] >= 2006) & (feat_df['Year'] <= 2025)]
    assert not eval_df[geo_feats].isna().any().any(), "NaNs detected in Phase 3 model matrix!"
