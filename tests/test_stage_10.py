import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.inference.pipeline import (
    get_latest_features,
    predict_next_month,
    get_current_risk_context
)
from geoprice.analysis.shock_responses import COMMODITIES

def test_model_artifacts_exist():
    for c in COMMODITIES:
        mpath = f"models/{c.lower()}_model.joblib"
        assert os.path.exists(mpath), f"Production model artifact '{mpath}' missing!"
    assert os.path.exists("models/model_metadata.json")

def test_get_latest_features():
    for c in COMMODITIES:
        latest_date, feat_vec, feat_dict = get_latest_features(c)
        assert pd.notna(latest_date)
        assert len(feat_dict) == 11
        assert f"{c}_return_1m" in feat_dict
        assert "GPR" in feat_dict
        assert "DXY" in feat_dict

def test_predict_next_month():
    for c in COMMODITIES:
        res = predict_next_month(c)
        assert res['commodity'] == c
        assert not np.isnan(res['predicted_return_decimal'])
        assert res['predicted_direction'] in ["UP", "DOWN"]
        assert len(res['feature_weights']) == 11

def test_get_current_risk_context():
    for c in COMMODITIES:
        ctx = get_current_risk_context(c)
        assert ctx['commodity'] == c
        assert ctx['current_GPR_regime'] in ["LOW", "MODERATE", "HIGH", "EXTREME"]
        assert isinstance(ctx['is_gpr_shock'], bool)
        assert pd.notna(ctx['analogue_1m_median_pct'])

def test_missing_model_error_handling():
    with pytest.raises(FileNotFoundError):
        predict_next_month("InvalidCommodity")

def test_current_gpr_delta_uses_previous_gpr_month():
    df_aligned = pd.read_csv("data/processed/monthly_aligned.csv").set_index('Date')
    ctx = get_current_risk_context("Brent")
    latest_date = ctx['latest_date']
    
    if latest_date in df_aligned.index:
        curr_loc = df_aligned.index.get_loc(latest_date)
        if curr_loc > 0:
            prev_date = df_aligned.index[curr_loc - 1]
            gpr_curr = df_aligned.loc[latest_date, 'GPR']
            gpr_prev = df_aligned.loc[prev_date, 'GPR']
            expected_delta = float(gpr_curr - gpr_prev)
            assert np.isclose(ctx['latest_delta_gpr'], expected_delta)
            assert not np.isclose(ctx['latest_delta_gpr'], 0.0)

def test_forecast_target_month():
    res = predict_next_month("Brent")
    origin = pd.to_datetime(res['forecast_origin_date'])
    expected_target = str((origin + pd.DateOffset(months=1)).to_period('M'))
    assert res['target_month'] == expected_target, f"Target month {res['target_month']} != expected {expected_target}"
