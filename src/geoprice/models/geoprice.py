import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from geoprice.models.baseline import (
    get_baseline_feature_names,
    create_next_month_target,
    build_baseline_pipeline
)
from geoprice.models.metrics import evaluate_all_metrics

from geoprice.constants import GEOPOLITICAL_FEATURES, MACRO_FEATURES

COMMON_GEOPOLITICAL_FEATURES = list(GEOPOLITICAL_FEATURES)
MACRO_CONTROL_FEATURES = list(MACRO_FEATURES)

def get_geoprice_feature_names(commodity: str) -> List[str]:
    """Returns the full 11-feature list for GeoPrice model (4 commodity history + 6 GPR + 1 DXY)."""
    comm_feats = get_baseline_feature_names(commodity)
    return comm_feats + COMMON_GEOPOLITICAL_FEATURES + MACRO_CONTROL_FEATURES

def get_gpr_only_feature_names(commodity: str) -> List[str]:
    """Returns 5-feature list for GPR-only ablation model (4 commodity history + 1 GPR)."""
    comm_feats = get_baseline_feature_names(commodity)
    return comm_feats + ['GPR']

def run_expanding_window_geoprice(
    df_features: pd.DataFrame,
    df_raw: pd.DataFrame,
    commodity: str,
    start_year: int = 2006,
    min_train_months: int = 48,
    alpha: float = 0.01,
    l1_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Runs out-of-sample expanding-window cross-validation for GeoPrice model (11 features)
    and GPR-only ablation model (5 features) on a single commodity.
    
    Uses identical date filtering, target definition, expanding window splits, and pipeline as Stage 7.
    """
    full_feats = get_geoprice_feature_names(commodity)
    gpr_only_feats = get_gpr_only_feature_names(commodity)
    baseline_feats = get_baseline_feature_names(commodity)

    df_feat = df_features.copy()
    if 'Date' in df_feat.columns:
        df_feat = df_feat.set_index('Date')

    target_series = create_next_month_target(df_raw, commodity)

    data = df_feat[full_feats].copy()
    data['Target'] = target_series

    data = data.reset_index()
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    phase3_data = data[data['Year'] >= start_year].copy().reset_index(drop=True)

    valid_mask = phase3_data[full_feats].notna().all(axis=1) & phase3_data['Target'].notna()
    dataset = phase3_data[valid_mask].copy().reset_index(drop=True)

    if len(dataset) <= min_train_months + 1:
        raise ValueError(f"Insufficient observations for expanding window CV ({len(dataset)} valid rows).")

    predictions = []
    ablation_preds_gpr_only = []

    for t_idx in range(min_train_months, len(dataset)):
        train_df = dataset.iloc[:t_idx]
        test_row = dataset.iloc[t_idx]
        
        y_train = train_df['Target'].values
        y_test = test_row['Target']
        forecast_date = test_row['Date']

        # 1. GeoPrice Model (11 features)
        X_train_geo = train_df[full_feats].values
        X_test_geo = test_row[full_feats].values.reshape(1, -1)
        
        pipeline_geo = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
        pipeline_geo.fit(X_train_geo, y_train)
        pred_geo = float(pipeline_geo.predict(X_test_geo)[0])

        predictions.append({
            "Date": forecast_date,
            "Commodity": commodity,
            "Actual_Return": float(y_test),
            "Predicted_Return": pred_geo,
            "Absolute_Error": float(abs(y_test - pred_geo)),
            "Squared_Error": float((y_test - pred_geo) ** 2),
            "Actual_Direction": int(np.sign(y_test)),
            "Predicted_Direction": int(np.sign(pred_geo)),
            "Correct_Direction": bool(np.sign(y_test) == np.sign(pred_geo))
        })

        # 2. GPR-only Ablation Model (5 features)
        X_train_gpr = train_df[gpr_only_feats].values
        X_test_gpr = test_row[gpr_only_feats].values.reshape(1, -1)
        
        pipeline_gpr = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
        pipeline_gpr.fit(X_train_gpr, y_train)
        pred_gpr = float(pipeline_gpr.predict(X_test_gpr)[0])

        ablation_preds_gpr_only.append(pred_gpr)

    pred_df = pd.DataFrame(predictions)
    
    # 3. Calculate GeoPrice Metrics
    y_actual = pred_df['Actual_Return'].values
    y_geo = pred_df['Predicted_Return'].values
    m_geo = evaluate_all_metrics(y_actual, y_geo, "GeoPrice", commodity)
    metrics_df = pd.DataFrame([m_geo])

    # 4. Calculate Ablation Metrics (Baseline vs GPR_only vs GeoPrice)
    # Re-evaluate Baseline on exact same dataset
    base_preds = []
    for t_idx in range(min_train_months, len(dataset)):
        train_df = dataset.iloc[:t_idx]
        test_row = dataset.iloc[t_idx]
        X_train_base = train_df[baseline_feats].values
        X_test_base = test_row[baseline_feats].values.reshape(1, -1)
        p_base = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
        p_base.fit(X_train_base, train_df['Target'].values)
        base_preds.append(float(p_base.predict(X_test_base)[0]))

    m_base_ablation = evaluate_all_metrics(y_actual, np.array(base_preds), "Baseline", commodity)
    m_gpr_ablation = evaluate_all_metrics(y_actual, np.array(ablation_preds_gpr_only), "GPR_only", commodity)
    m_geo_ablation = evaluate_all_metrics(y_actual, y_geo, "GeoPrice", commodity)

    m_base_ablation["Feature_Set"] = "Baseline"
    m_gpr_ablation["Feature_Set"] = "GPR_only"
    m_geo_ablation["Feature_Set"] = "GeoPrice"

    ablation_df = pd.DataFrame([m_base_ablation, m_gpr_ablation, m_geo_ablation])

    # 5. Extract GeoPrice Final Refit Model Coefficients
    X_full = dataset[full_feats].values
    y_full = dataset['Target'].values
    final_pipeline = build_baseline_pipeline(alpha=alpha, l1_ratio=l1_ratio)
    final_pipeline.fit(X_full, y_full)

    coefs = final_pipeline.named_steps['model'].coef_
    intercept = float(final_pipeline.named_steps['model'].intercept_)

    coef_rows = [{"Commodity": commodity, "Feature": "Intercept", "Coefficient": intercept, "Feature_Group": "Intercept"}]
    for fname, cval in zip(full_feats, coefs):
        if fname in baseline_feats:
            group = "Commodity History"
        elif fname == "DXY":
            group = "Macro"
        else:
            group = "Geopolitical"
        coef_rows.append({"Commodity": commodity, "Feature": fname, "Coefficient": float(cval), "Feature_Group": group})

    coef_df = pd.DataFrame(coef_rows)

    config = {
        "commodity": commodity,
        "model": "GeoPrice Model",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "feature_count": len(full_feats),
        "total_oos_predictions": len(pred_df),
        "feature_columns": full_feats
    }

    return pred_df, metrics_df, coef_df, ablation_df, config
