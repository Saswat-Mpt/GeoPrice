import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.features.engineering import build_feature_dataset
from geoprice.features.availability import apply_gpr_availability_rule, apply_dxy_availability_rule

@pytest.fixture
def base_toy_data():
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    prices = [100.0 + i*2 for i in range(12)]
    gpr = [80.0 + i*5 for i in range(12)]
    df = pd.DataFrame({
        'Date': dates,
        'GPR': gpr,
        'GPRT': gpr,
        'GPRA': gpr,
        'Brent': prices,
        'Natural_Gas': prices,
        'Gold': prices,
        'Copper': prices,
        'Wheat': prices,
        'DXY': [100.0]*12
    })
    return df

def test_no_future_leakage(base_toy_data):
    """
    CRITICAL ANTI-LEAKAGE TEST:
    Calculate features for baseline dataset.
    Modify future values at month t+1 (index 5).
    Confirm that feature values at month t (index 4) do NOT change at all.
    """
    t_idx = 4
    
    # 1. Baseline features
    df1 = base_toy_data.copy()
    feat1 = build_feature_dataset(df1)
    row_t1 = feat1.iloc[t_idx].copy()
    
    # 2. Modify future observations at t+1 (index 5)
    df2 = base_toy_data.copy()
    df2.loc[5, 'Brent'] = 9999.0
    df2.loc[5, 'GPR'] = 8888.0
    df2.loc[5, 'DXY'] = 7777.0
    
    feat2 = build_feature_dataset(df2)
    row_t2 = feat2.iloc[t_idx].copy()
    
    # 3. Assert feature at t_idx is identical
    for col in feat1.columns:
        if col != 'Date' and pd.notna(row_t1[col]):
            assert np.isclose(row_t1[col], row_t2[col]), f"LEAKAGE DETECTED! Feature '{col}' at month {t_idx} changed when future month {t_idx+1} was modified."

def test_gpr_point_in_time_rule(base_toy_data):
    df_lag0 = apply_gpr_availability_rule(base_toy_data, release_lag_months=0)
    assert np.isclose(df_lag0['GPR_pit'].iloc[5], base_toy_data['GPR'].iloc[5])
    
    df_lag1 = apply_gpr_availability_rule(base_toy_data, release_lag_months=1)
    assert np.isclose(df_lag1['GPR_pit'].iloc[5], base_toy_data['GPR'].iloc[4])

def test_dxy_point_in_time_rule(base_toy_data):
    df_lag0 = apply_dxy_availability_rule(base_toy_data, release_lag_months=0)
    assert np.isclose(df_lag0['DXY_pit'].iloc[5], base_toy_data['DXY'].iloc[5])
    
    df_lag1 = apply_dxy_availability_rule(base_toy_data, release_lag_months=1)
    assert np.isclose(df_lag1['DXY_pit'].iloc[5], base_toy_data['DXY'].iloc[4])
