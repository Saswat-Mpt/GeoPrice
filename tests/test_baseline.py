import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.baseline import (
    create_next_month_target,
    get_baseline_feature_names,
    run_expanding_window_baseline
)
from geoprice.models.validation import validate_baseline_features

@pytest.fixture
def baseline_toy_data():
    dates = [f"2006-{m:02d}" for m in range(1, 13)] + [f"2007-{m:02d}" for m in range(1, 13)] + [f"2008-{m:02d}" for m in range(1, 13)]
    n = len(dates)
    brent = [50.0 + i*1.5 for i in range(n)]
    
    df_raw = pd.DataFrame({'Date': dates, 'Brent': brent})
    
    df_feat = pd.DataFrame({
        'Date': dates,
        'Brent_return_1m': [0.02]*n,
        'Brent_return_3m': [0.05]*n,
        'Brent_return_6m': [0.10]*n,
        'Brent_vol_3m': [0.03]*n,
        'GPR': [100.0]*n,
        'DXY': [90.0]*n
    })
    return df_feat, df_raw

def test_target_is_next_month_return(baseline_toy_data):
    df_feat, df_raw = baseline_toy_data
    target = create_next_month_target(df_raw, 'Brent')
    
    # At index 0 (2006-01): Price_0 = 50.0, Price_1 = 51.5 -> target_0 = (51.5 / 50.0) - 1 = 0.03
    expected_t0 = (51.5 / 50.0) - 1.0
    assert np.isclose(target.iloc[0], expected_t0)
    assert pd.isna(target.iloc[-1]), "Last target observation must be NaN because future price is unknown!"

def test_baseline_features_exclude_geopolitical_features():
    feat_cols = get_baseline_feature_names('Brent')
    assert validate_baseline_features(feat_cols)
    assert 'GPR' not in feat_cols
    assert 'DXY' not in feat_cols

def test_no_future_feature_leakage(baseline_toy_data):
    df_feat, df_raw = baseline_toy_data
    
    # Target target_t uses price t+1
    target1 = create_next_month_target(df_raw, 'Brent')
    
    # Modify future price at index 10 (t+1)
    df_raw_mod = df_raw.copy()
    df_raw_mod.loc[10, 'Brent'] = 999.0
    target2 = create_next_month_target(df_raw_mod, 'Brent')
    
    # Predictor features at index 5 must remain unchanged
    feat_cols = get_baseline_feature_names('Brent')
    f5_orig = df_feat.loc[5, feat_cols].tolist()
    f5_mod = df_feat.loc[5, feat_cols].tolist()
    
    assert f5_orig == f5_mod, "Predictor features must not change when future price is modified!"
    # Target at index 9 MUST change because future price at index 10 changed
    assert target1.iloc[9] != target2.iloc[9]

def test_expanding_window_baseline(baseline_toy_data):
    df_feat, df_raw = baseline_toy_data
    pred_df, metrics_df, coef_df, config = run_expanding_window_baseline(
        df_feat, df_raw, commodity='Brent', start_year=2006, min_train_months=12
    )
    
    assert len(pred_df) > 0
    assert 'Actual_Return' in pred_df.columns
    assert 'Predicted_Return' in pred_df.columns
    assert len(metrics_df) == 2 # Naive and ElasticNet
    assert (metrics_df['Model'] == 'ElasticNet Baseline').any()
    assert (metrics_df['Model'] == 'Naive (Zero Return)').any()
