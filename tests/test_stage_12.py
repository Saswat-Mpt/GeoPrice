import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.interpretation.contributions import (
    explain_current_forecast,
    evaluate_all_commodity_interpretations
)
from geoprice.analysis.shock_responses import COMMODITIES

def test_explain_current_forecast_reconstruction():
    for c in COMMODITIES:
        exp = explain_current_forecast(c)
        assert exp['reconstruction_pass'], f"Prediction reconstruction check failed for {c}!"
        assert abs(exp['reconstructed_prediction'] - exp['model_prediction']) < 1e-10
        assert len(exp['contributions_df']) == 11

def test_evaluate_all_commodity_interpretations():
    m_coef, c_contrib, top_contrib, recon, summary = evaluate_all_commodity_interpretations()
    assert len(recon) == 5
    assert recon['Reconstruction_Pass'].all()
    assert 'Contribution' in c_contrib.columns
    assert 'Standardized_Value' in c_contrib.columns
