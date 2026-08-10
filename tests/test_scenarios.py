import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.scenarios.lookup import get_historical_scenario
from geoprice.analysis.shock_responses import COMMODITIES
from geoprice.analysis.regimes import REGIMES

def test_valid_regimes_and_commodities():
    for c in COMMODITIES:
        for r in REGIMES:
            res = get_historical_scenario(c, r, conflict_reference="None")
            assert res['commodity'] == c
            assert res['selected_regime'] == r
            assert 'regime_stats' in res
            assert res['mode'] == "HISTORICAL SCENARIO LOOKUP (NON-ML)"

def test_invalid_regime_rejected():
    with pytest.raises(ValueError):
        get_historical_scenario("Brent", "SUPER_HIGH")

def test_invalid_commodity_rejected():
    with pytest.raises(ValueError):
        get_historical_scenario("UnicornOil", "HIGH")

def test_conflict_reference_lookup():
    res = get_historical_scenario("Brent", "HIGH", conflict_reference="Major-conflict reference")
    assert "conflict_stats" in res
    assert "conflict_1m_median_pct" in res['conflict_stats']
    assert pd.notna(res['conflict_stats']['conflict_1m_median_pct'])

def test_no_ml_prediction_in_scenario_mode():
    res = get_historical_scenario("Gold", "EXTREME", conflict_reference="Major-conflict reference")
    assert "predicted_return_decimal" not in res
    assert "predicted_direction" not in res
    assert "feature_weights" not in res
    assert "HISTORICAL" in res['mode']
