import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from geoprice.models.metrics import evaluate_all_metrics

BASELINE_FEATURE_SUFFIXES = ['return_1m', 'return_3m', 'return_6m', 'vol_3m']

def get_baseline_feature_names(commodity: str) -> List[str]:
    """Returns the exact 4 commodity-history feature names for a given commodity."""
    return [f"{commodity}_{suffix}" for suffix in BASELINE_FEATURE_SUFFIXES]

def create_next_month_target(df_raw: pd.DataFrame, commodity: str) -> pd.Series:
    """
    Creates target y_t = next month's decimal return from month t to month t+1:
    y_t = P_(t+1) / P_t - 1
    
    Target at forecast origin month t (e.g. 2020-06) is July 2020 return (P_July / P_June - 1).
    """
    df = df_raw.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
        
    price = df[commodity]
    next_price = price.shift(-1)
    target = (next_price / price) - 1.0
    return target

def build_baseline_pipeline(alpha: float = 0.01, l1_ratio: float = 0.5) -> Pipeline:
    """
    Builds scikit-learn Pipeline of StandardScaler -> ElasticNet.
    StandardScaler is fitted ONLY on training data inside the pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42))
    ])

def run_expanding_window_baseline(
    df_features: pd.DataFrame,
    df_raw: pd.DataFrame,
    commodity: str,
    start_year: int = 2006,
    min_train_months: int = 48,
    alpha: float = 0.01,
    l1_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Runs out-of-sample expanding-window cross-validation for Baseline model on a single commodity.
    
    Features: 4 commodity history features ONLY.
    Target: Next-month decimal return (y_t).
    Window: Phase 3 DXY-supported period (2006 onward).
    
    Returns:
    1. Out-of-sample predictions DataFrame
    2. Metrics summary DataFrame (Naive vs ElasticNet)
    3. Feature coefficients DataFrame
    4. Model configuration metadata dict
    """
    feat_cols = get_baseline_feature_names(commodity)
    
    # 1. Align features and target
    df_feat = df_features.copy()
    if 'Date' in df_feat.columns:
        df_feat = df_feat.set_index('Date')
        
    target_series = create_next_month_target(df_raw, commodity)
    
    # Combined dataset for slicing
    data = df_feat[feat_cols].copy()
    data['Target'] = target_series
    
    # Filter to Phase 3 window (2006-01 onward) and drop NaNs in predictors/target
    data = data.reset_index()
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    phase3_data = data[data['Year'] >= start_year].copy().reset_index(drop=True)
    
    # Valid rows with complete predictor features and target
    valid_mask = phase3_data[feat_cols].notna().all(axis=1) & phase3_data['Target'].notna()
    dataset = phase3_data[valid_mask].copy().reset_index(drop=True)
    
    if len(dataset) <= min_train_months + 1:
        raise ValueError(f"Insufficient observations for expanding window CV ({len(dataset)} valid rows).")
        
    # 2. Expanding-Window Out-of-Sample Prediction Loop
    predictions = []
    
    for t_idx in range(min_train_months, len(dataset)):
        train_df = dataset.iloc[:t_idx]
        test_row = dataset.iloc[t_idx]
        
        X_train = train_df[feat_cols].values
        y_train = train_df['Target'].values
        
        X_test = test_row[feat_cols].values.reshape(1, -1)
        y_test = test_row['Target']
        forecast_date = test_row['Date']
        
        # Fit pipeline ONLY on current training fold
        pipeline = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
        pipeline.fit(X_train, y_train)
        
        # Out-of-sample prediction
        pred_elastic = float(pipeline.predict(X_test)[0])
        pred_naive = 0.0 # Naive zero return prediction
        
        predictions.append({
            "Date": forecast_date,
            "Commodity": commodity,
            "Actual_Return": float(y_test),
            "Predicted_Return": pred_elastic,
            "Naive_Predicted_Return": pred_naive,
            "Absolute_Error": float(abs(y_test - pred_elastic)),
            "Squared_Error": float((y_test - pred_elastic) ** 2),
            "Actual_Direction": int(np.sign(y_test)),
            "Predicted_Direction": int(np.sign(pred_elastic)),
            "Correct_Direction": bool(np.sign(y_test) == np.sign(pred_elastic))
        })
        
    pred_df = pd.DataFrame(predictions)
    
    # 3. Calculate Metrics
    y_actual = pred_df['Actual_Return'].values
    y_elastic = pred_df['Predicted_Return'].values
    y_naive = pred_df['Naive_Predicted_Return'].values
    
    m_elastic = evaluate_all_metrics(y_actual, y_elastic, "ElasticNet Baseline", commodity)
    m_naive = evaluate_all_metrics(y_actual, y_naive, "Naive (Zero Return)", commodity)
    
    metrics_df = pd.DataFrame([m_naive, m_elastic])
    
    # 4. Extract Final Refit Model Coefficients
    X_full = dataset[feat_cols].values
    y_full = dataset['Target'].values
    final_pipeline = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
    final_pipeline.fit(X_full, y_full)
    
    coefs = final_pipeline.named_steps['model'].coef_
    intercept = float(final_pipeline.named_steps['model'].intercept_)
    
    coef_rows = [{"Commodity": commodity, "Feature": "Intercept", "Coefficient": intercept}]
    for fname, cval in zip(feat_cols, coefs):
        coef_rows.append({"Commodity": commodity, "Feature": fname, "Coefficient": float(cval)})
        
    coef_df = pd.DataFrame(coef_rows)
    
    config = {
        "commodity": commodity,
        "model": "ElasticNet Baseline",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "start_year": start_year,
        "min_train_months": min_train_months,
        "total_oos_predictions": len(pred_df),
        "feature_columns": feat_cols,
        "target_definition": "y_t = P_(t+1)/P_t - 1"
    }
    
    return pred_df, metrics_df, coef_df, config
