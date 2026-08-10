import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.regimes import (
    calculate_gpr_regime_thresholds,
    assign_gpr_regimes,
    get_current_gpr_state,
    build_regime_episodes,
    REGIMES
)

@pytest.fixture
def regime_toy_data():
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    # Values designed to test regime cutoffs: LOW < 50, MODERATE 50-75, HIGH 75-90, EXTREME > 90
    gpr = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    brent = [100.0 + i*2 for i in range(12)]
    return pd.DataFrame({
        'Date': dates,
        'GPR': gpr,
        'GPRT': gpr,
        'GPRA': gpr,
        'Brent': brent,
        'Natural_Gas': brent,
        'Gold': brent,
        'Copper': brent,
        'Wheat': brent
    })

def test_regime_thresholds(regime_toy_data):
    thresholds, meta = calculate_gpr_regime_thresholds(regime_toy_data)
    assert thresholds['P50'] < thresholds['P75'] < thresholds['P90']
    assert meta['total_months'] == 12

def test_regime_assignment(regime_toy_data):
    thresholds, _ = calculate_gpr_regime_thresholds(regime_toy_data)
    df_reg = assign_gpr_regimes(regime_toy_data, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    
    assert 'GPR_regime' in df_reg.columns
    # Check boundary behavior
    low_mask = df_reg['GPR'] < thresholds['P50']
    assert (df_reg.loc[low_mask, 'GPR_regime'] == 'LOW').all()
    
    extreme_mask = df_reg['GPR'] > thresholds['P90']
    assert (df_reg.loc[extreme_mask, 'GPR_regime'] == 'EXTREME').all()

def test_current_regime(regime_toy_data):
    thresholds, _ = calculate_gpr_regime_thresholds(regime_toy_data)
    df_reg = assign_gpr_regimes(regime_toy_data, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    curr = get_current_gpr_state(df_reg, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    
    assert curr['current_date'] == '2020-12'
    assert curr['current_GPR'] == 100.0
    assert curr['current_GPR_regime'] == 'EXTREME'

def test_regime_episode_creation(regime_toy_data):
    thresholds, _ = calculate_gpr_regime_thresholds(regime_toy_data)
    df_reg = assign_gpr_regimes(regime_toy_data, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    episodes = build_regime_episodes(df_reg)
    
    assert isinstance(episodes, pd.DataFrame)
    assert 'representative_GPR' in episodes.columns
    assert len(episodes) > 0
