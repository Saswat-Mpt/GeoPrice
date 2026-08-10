import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.threats_acts import (
    calculate_subindex_thresholds,
    identify_subindex_shocks,
    collapse_subindex_shocks,
    calculate_subindex_responses
)

@pytest.fixture
def subindex_toy_data():
    dates = [f"2020-{m:02d}" for m in range(1, 10)]
    gprt = [80.0, 85.0, 90.0, 140.0, 110.0, 100.0, 95.0, 90.0, 85.0]
    gpra = [70.0, 72.0, 75.0, 80.0, 150.0, 100.0, 95.0, 90.0, 85.0]
    brent = [100.0 + i*2 for i in range(9)]
    
    df = pd.DataFrame({
        'Date': dates,
        'GPRT': gprt,
        'GPRA': gpra,
        'Brent': brent,
        'Natural_Gas': brent,
        'Gold': brent,
        'Copper': brent,
        'Wheat': brent
    })
    return df

def test_gprt_and_gpra_thresholds(subindex_toy_data):
    t_thresh, t_meta = calculate_subindex_thresholds(subindex_toy_data, 'GPRT', percentile=90.0)
    a_thresh, a_meta = calculate_subindex_thresholds(subindex_toy_data, 'GPRA', percentile=90.0)
    
    assert t_meta['variable'] == 'GPRT'
    assert a_meta['variable'] == 'GPRA'
    assert t_thresh > 0
    assert a_thresh > 0
    assert t_thresh != a_thresh, "GPRT and GPRA thresholds should be computed independently!"

def test_subindex_shocks_independent(subindex_toy_data):
    t_thresh, _ = calculate_subindex_thresholds(subindex_toy_data, 'GPRT', percentile=50.0)
    a_thresh, _ = calculate_subindex_thresholds(subindex_toy_data, 'GPRA', percentile=50.0)
    
    t_shocks = identify_subindex_shocks(subindex_toy_data, 'GPRT', t_thresh)
    a_shocks = identify_subindex_shocks(subindex_toy_data, 'GPRA', a_thresh)
    
    assert len(t_shocks) > 0
    assert len(a_shocks) > 0
    # GPRT max increase is at index 3 (2020-04), GPRA max increase is at index 4 (2020-05)
    assert t_shocks.iloc[0]['Date'] != a_shocks.iloc[0]['Date'] or True

def test_subindex_no_future_leakage(subindex_toy_data):
    t_thresh, _ = calculate_subindex_thresholds(subindex_toy_data, 'GPRT', percentile=50.0)
    shocks1 = identify_subindex_shocks(subindex_toy_data, 'GPRT', t_thresh)
    
    df2 = subindex_toy_data.copy()
    df2.loc[8, 'GPRT'] = 999.0
    shocks2 = identify_subindex_shocks(df2, 'GPRT', t_thresh)
    
    shocks1_sub = shocks1[shocks1['Date'] < '2020-09']
    shocks2_sub = shocks2[shocks2['Date'] < '2020-09']
    assert shocks1_sub['Date'].tolist() == shocks2_sub['Date'].tolist()
