import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.geoprice import (
    get_geoprice_feature_names,
    get_gpr_only_feature_names,
    run_expanding_window_geoprice
)

@pytest.fixture
def geoprice_toy_data():
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
        'GPR': [100.0 + i for i in range(n)],
        'GPR_change': [5.0]*n,
        'GPR_lag1': [95.0]*n,
        'GPR_lag3': [90.0]*n,
        'GPRT': [110.0]*n,
        'GPRA': [90.0]*n,
        'DXY': [90.0]*n
    })
    return df_feat, df_raw

def test_geoprice_feature_count():
    feats = get_geoprice_feature_names('Brent')
    assert len(feats) == 11
    assert 'GPR' in feats
    assert 'DXY' in feats
    assert 'Brent_return_1m' in feats

def test_gpr_only_feature_count():
    feats = get_gpr_only_feature_names('Brent')
    assert len(feats) == 5
    assert 'GPR' in feats
    assert 'DXY' not in feats

def test_geoprice_expanding_window(geoprice_toy_data):
    df_feat, df_raw = geoprice_toy_data
    pred_df, metrics_df, coef_df, ablation_df, config = run_expanding_window_geoprice(
        df_feat, df_raw, commodity='Brent', start_year=2006, min_train_months=12
    )
    
    assert len(pred_df) > 0
    assert len(metrics_df) == 1
    assert len(ablation_df) == 3 # Baseline, GPR_only, GeoPrice
    assert len(coef_df) == 12 # 11 features + Intercept
