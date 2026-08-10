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
