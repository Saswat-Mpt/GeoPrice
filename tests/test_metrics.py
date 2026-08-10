import pytest
import numpy as np

from geoprice.models.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_directional_accuracy,
    evaluate_all_metrics
)

def test_mae_calculation():
    y_true = np.array([0.05, -0.02, 0.03, 0.01])
    y_pred = np.array([0.04, -0.01, 0.05, 0.00])
    
    # Absolute errors: 0.01, 0.01, 0.02, 0.01 -> mean = 0.0125
    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 0.0125)

def test_rmse_calculation():
    y_true = np.array([0.04, -0.02])
    y_pred = np.array([0.01, 0.02])
    
    # Errors: 0.03, -0.04 -> Sq errors: 0.0009, 0.0016 -> mean = 0.00125 -> sqrt = 0.035355...
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(0.00125))

def test_directional_accuracy():
    y_true = np.array([0.05, -0.02, 0.03, -0.04])
    y_pred = np.array([0.01, -0.01, -0.02, -0.05])
    
    # Directions:
    # 1: + / + -> match
    # 2: - / - -> match
    # 3: + / - -> mismatch
    # 4: - / - -> match
    # DA = 3/4 = 0.75
    da = calculate_directional_accuracy(y_true, y_pred)
    assert np.isclose(da, 0.75)

def test_evaluate_all_metrics():
    y_true = np.array([0.02, -0.01, 0.04])
    y_pred = np.array([0.01, -0.02, 0.03])
    res = evaluate_all_metrics(y_true, y_pred, "TestModel", "Brent")
    
    assert res['Commodity'] == "Brent"
    assert res['Model'] == "TestModel"
    assert res['N'] == 3
    assert not np.isnan(res['MAE'])
    assert not np.isnan(res['RMSE'])
    assert res['Directional_Accuracy'] == 1.0
