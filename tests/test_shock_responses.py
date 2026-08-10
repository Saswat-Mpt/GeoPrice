import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shock_responses import calculate_forward_commodity_responses

@pytest.fixture
def response_toy_data():
    dates = [f"2020-{m:02d}" for m in range(1, 10)]
    brent = [100.0, 105.0, 110.0, 120.0, 114.0, 108.0, 105.0, 102.0, 100.0]
    df = pd.DataFrame({
        'Date': dates,
        'GPR': [80.0]*9,
        'Brent': brent,
        'Natural_Gas': brent,
        'Gold': brent,
        'Copper': brent,
        'Wheat': brent
    })
    episodes = pd.DataFrame([{
        'episode_id': 1,
        'representative_shock_date': '2020-02', # Month 2 (idx 1), Brent = 105.0
        'representative_GPR': 80.0,
        'representative_GPR_change': 10.0,
        'raw_shock_count': 1,
        'cluster_dates': '2020-02'
    }, {
        'episode_id': 2,
        'representative_shock_date': '2020-08', # Month 8 (idx 7), near end
        'representative_GPR': 80.0,
        'representative_GPR_change': 10.0,
        'raw_shock_count': 1,
        'cluster_dates': '2020-08'
    }])
    return df, episodes

def test_forward_1m_return(response_toy_data):
    df, episodes = response_toy_data
    responses, summary = calculate_forward_commodity_responses(episodes, df)
    
    # Episode 1 (2020-02, P=105.0): +1M is 2020-03 (P=110.0) -> 110/105 - 1 = 0.047619
    ep1_1m = responses.iloc[0]['Brent_1m']
    expected_1m = (110.0 / 105.0) - 1.0
    assert np.isclose(ep1_1m, expected_1m)

def test_forward_2m_return(response_toy_data):
    df, episodes = response_toy_data
    responses, summary = calculate_forward_commodity_responses(episodes, df)
    
    # Episode 1 (2020-02, P=105.0): +2M is 2020-04 (P=120.0) -> 120/105 - 1 = 0.142857
    ep1_2m = responses.iloc[0]['Brent_2m']
    expected_2m = (120.0 / 105.0) - 1.0
    assert np.isclose(ep1_2m, expected_2m)

def test_forward_3m_return(response_toy_data):
    df, episodes = response_toy_data
    responses, summary = calculate_forward_commodity_responses(episodes, df)
    
    # Episode 1 (2020-02, P=105.0): +3M is 2020-05 (P=114.0) -> 114/105 - 1 = 0.085714
    ep1_3m = responses.iloc[0]['Brent_3m']
    expected_3m = (114.0 / 105.0) - 1.0
    assert np.isclose(ep1_3m, expected_3m)

def test_incomplete_forward_windows(response_toy_data):
    df, episodes = response_toy_data
    responses, summary = calculate_forward_commodity_responses(episodes, df)
    
    # Episode 2 (2020-08, index 7, max index 8):
    # +1M (2020-09, index 8) is valid
    # +2M and +3M are beyond index 8 -> should be NaN
    ep2_1m = responses.iloc[1]['Brent_1m']
    ep2_2m = responses.iloc[1]['Brent_2m']
    ep2_3m = responses.iloc[1]['Brent_3m']
    
    assert pd.notna(ep2_1m)
    assert pd.isna(ep2_2m)
    assert pd.isna(ep2_3m)
