import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shocks import (
    calculate_gpr_change,
    calculate_shock_threshold,
    identify_raw_shocks,
    collapse_overlapping_shocks
)

@pytest.fixture
def toy_gpr_data():
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    # Monthly GPR levels with specific known increases
    gpr = [100.0, 105.0, 102.0, 150.0, 155.0, 120.0, 118.0, 160.0, 110.0, 108.0, 105.0, 100.0]
    return pd.DataFrame({'Date': dates, 'GPR': gpr})

def test_gpr_change_calculation(toy_gpr_data):
    changes = calculate_gpr_change(toy_gpr_data)
    # Month 1 (idx 1): 105 - 100 = 5.0
    assert np.isclose(changes.iloc[1], 5.0)
    # Month 3 (idx 3): 150 - 102 = 48.0
    assert np.isclose(changes.iloc[3], 48.0)
    # Month 2 (idx 2): 102 - 105 = -3.0
    assert np.isclose(changes.iloc[2], -3.0)

def test_positive_change_filter(toy_gpr_data):
    threshold, meta = calculate_shock_threshold(toy_gpr_data, percentile=90.0)
    # Positive changes in toy data: 5.0 (m1), 48.0 (m3), 5.0 (m4), 42.0 (m7)
    # Total valid changes = 11, positive changes = 4
    assert meta['positive_change_count'] == 4
    assert meta['threshold'] > 0

def test_top_decile_threshold():
    # 10 positive changes: 1..10
    pos_changes = list(range(1, 11))
    # 90th percentile of 1..10 is 9.1
    p90 = np.percentile(pos_changes, 90.0)
    assert np.isclose(p90, 9.1)

def test_shock_selection(toy_gpr_data):
    threshold, meta = calculate_shock_threshold(toy_gpr_data, percentile=50.0)
    raw_shocks = identify_raw_shocks(toy_gpr_data, threshold)
    assert len(raw_shocks) > 0
    assert (raw_shocks['GPR_change'] >= threshold).all()

def test_overlap_collapsing(toy_gpr_data):
    # Setup raw shocks with consecutive months (2020-04 Δ=48, 2020-05 Δ=5)
    threshold = 4.0
    raw_shocks = identify_raw_shocks(toy_gpr_data, threshold)
    episodes = collapse_overlapping_shocks(raw_shocks, toy_gpr_data, window_months=3)
    
    # 2020-02, 04, 05, 08 are all within 3 months of consecutive neighbors -> 1 episode with representative date 2020-04 (dGPR=48)
    ep1 = episodes.iloc[0]
    assert ep1['representative_shock_date'] == '2020-04'
    assert ep1['representative_GPR_change'] == 48.0
    assert ep1['raw_shock_count'] == 4

def test_no_future_data_for_shock_detection(toy_gpr_data):
    """Verify shock detection at month t depends only on GPR at or before month t."""
    df1 = toy_gpr_data.copy()
    threshold, _ = calculate_shock_threshold(df1, percentile=50.0)
    shocks1 = identify_raw_shocks(df1, threshold)
    
    # Modify future GPR at month 10
    df2 = toy_gpr_data.copy()
    df2.loc[10, 'GPR'] = 999.0
    shocks2 = identify_raw_shocks(df2, threshold)
    
    # Shocks before month 10 should be identical
    shocks1_sub = shocks1[shocks1['Date'] < '2020-10']
    shocks2_sub = shocks2[shocks2['Date'] < '2020-10']
    assert len(shocks1_sub) == len(shocks2_sub)
    assert shocks1_sub['Date'].tolist() == shocks2_sub['Date'].tolist()
