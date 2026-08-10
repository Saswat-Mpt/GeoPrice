import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.features.engineering import (
    calculate_returns,
    calculate_rolling_volatility,
    create_geopolitical_features,
    build_feature_dataset,
    COMMODITIES
)

@pytest.fixture
def sample_data():
    """Generates small deterministic toy DataFrame for testing feature formulas."""
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    prices = [100.0, 105.0, 110.0, 104.0, 108.0, 112.0, 120.0, 115.0, 118.0, 122.0, 125.0, 130.0]
    gpr = [80.0, 85.0, 90.0, 120.0, 110.0, 95.0, 100.0, 105.0, 130.0, 125.0, 115.0, 110.0]
    gprt = [75.0, 80.0, 85.0, 115.0, 105.0, 90.0, 95.0, 100.0, 125.0, 120.0, 110.0, 105.0]
    gpra = [85.0, 90.0, 95.0, 125.0, 115.0, 100.0, 105.0, 110.0, 135.0, 130.0, 120.0, 115.0]
    dxy = [98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5]
    
    df = pd.DataFrame({
        'Date': dates,
        'GPR': gpr,
        'GPRT': gprt,
        'GPRA': gpra,
        'Brent': prices,
        'Natural_Gas': prices,
        'Gold': prices,
        'Copper': prices,
        'Wheat': prices,
        'DXY': dxy
    })
    return df

def test_return_1m(sample_data):
    ret_df = calculate_returns(sample_data, 'Brent')
    assert f"Brent_return_1m" in ret_df.columns
    # Month 1 (idx 1): 105 / 100 - 1 = 0.05
    assert np.isclose(ret_df.iloc[1]['Brent_return_1m'], 0.05)
    # Month 0 should be NaN
    assert np.isnan(ret_df.iloc[0]['Brent_return_1m'])

def test_return_3m(sample_data):
    ret_df = calculate_returns(sample_data, 'Brent')
    assert f"Brent_return_3m" in ret_df.columns
    # Month 3 (idx 3): 104 / 100 - 1 = 0.04
    assert np.isclose(ret_df.iloc[3]['Brent_return_3m'], 0.04)
    # Month 0-2 should be NaN
    assert ret_df['Brent_return_3m'].iloc[:3].isna().all()

def test_return_6m(sample_data):
    ret_df = calculate_returns(sample_data, 'Brent')
    assert f"Brent_return_6m" in ret_df.columns
    # Month 6 (idx 6): 120 / 100 - 1 = 0.20
    assert np.isclose(ret_df.iloc[6]['Brent_return_6m'], 0.20)
    # Month 0-5 should be NaN
    assert ret_df['Brent_return_6m'].iloc[:6].isna().all()

def test_rolling_volatility(sample_data):
    vol_df = calculate_rolling_volatility(sample_data, 'Brent')
    assert "Brent_vol_3m" in vol_df.columns
    # Month 0-2 should be NaN (min_periods=3)
    assert vol_df['Brent_vol_3m'].iloc[:2].isna().all()
    # Month 3 should have std dev of 1M returns for month 1, 2, 3
    ret_1m = (sample_data['Brent'] / sample_data['Brent'].shift(1)) - 1.0
    expected_std = ret_1m.iloc[1:4].std(ddof=1)
    actual_std = vol_df.iloc[3]['Brent_vol_3m']
    assert np.isclose(actual_std, expected_std)

def test_gpr_change(sample_data):
    geo_df = create_geopolitical_features(sample_data)
    assert "GPR_change" in geo_df.columns
    # Month 1: 85.0 - 80.0 = 5.0
    assert np.isclose(geo_df.iloc[1]['GPR_change'], 5.0)
    # Month 3: 120.0 - 90.0 = 30.0
    assert np.isclose(geo_df.iloc[3]['GPR_change'], 30.0)

def test_gpr_lags(sample_data):
    geo_df = create_geopolitical_features(sample_data)
    assert "GPR_lag1" in geo_df.columns
    assert "GPR_lag3" in geo_df.columns
    # Month 1 lag1: 80.0
    assert np.isclose(geo_df.iloc[1]['GPR_lag1'], 80.0)
    # Month 3 lag3: 80.0
    assert np.isclose(geo_df.iloc[3]['GPR_lag3'], 80.0)

def test_feature_dataset_structure(sample_data):
    feat_df = build_feature_dataset(sample_data)
    assert len(feat_df) == len(sample_data)
    # Total columns: Date + 7 common + 20 commodity (4x5) = 28
    assert len(feat_df.columns) == 28
    for c in COMMODITIES:
        for f in [f"{c}_return_1m", f"{c}_return_3m", f"{c}_return_6m", f"{c}_vol_3m"]:
            assert f in feat_df.columns
    for f in ['GPR', 'GPR_change', 'GPR_lag1', 'GPR_lag3', 'GPRT', 'GPRA', 'DXY']:
        assert f in feat_df.columns
