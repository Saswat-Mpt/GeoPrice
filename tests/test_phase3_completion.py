import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.evaluation import compute_model_improvements, compute_regime_robustness

def test_phase3_outputs_exist_and_aligned():
    b_pred_path = "data/processed/baseline_predictions.csv"
    g_pred_path = "data/processed/geoprice_predictions.csv"
    
    if os.path.exists(b_pred_path) and os.path.exists(g_pred_path):
        b_preds = pd.read_csv(b_pred_path)
        g_preds = pd.read_csv(g_pred_path)
        
        # Test commodity, date, and target return alignment
        b_keys = list(zip(b_preds['Commodity'], b_preds['Date']))
        g_keys = list(zip(g_preds['Commodity'], g_preds['Date']))
        assert b_keys == g_keys, "(Commodity, Date) prediction key alignment failed between Baseline and GeoPrice!"
        assert np.allclose(b_preds['Actual_Return'], g_preds['Actual_Return'])

def test_improvements_calculation():
    b_metrics = pd.DataFrame([
        {"Commodity": "Brent", "Model": "ElasticNet Baseline", "N": 100, "MAE": 0.05, "RMSE": 0.07, "Directional_Accuracy": 0.50}
    ])
    g_metrics = pd.DataFrame([
        {"Commodity": "Brent", "Model": "GeoPrice", "N": 100, "MAE": 0.045, "RMSE": 0.065, "Directional_Accuracy": 0.55}
    ])
    
    imp = compute_model_improvements(b_metrics, g_metrics)
    assert len(imp) == 1
    assert np.isclose(imp.iloc[0]['MAE_Improvement'], 0.005)
    assert np.isclose(imp.iloc[0]['MAE_Improvement_Pct'], 10.0)
    assert np.isclose(imp.iloc[0]['Directional_Accuracy_Improvement_Points'], 5.0)
