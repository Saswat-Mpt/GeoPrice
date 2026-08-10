import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.inference.pipeline import get_latest_features
from geoprice.analysis.shock_responses import COMMODITIES

MODELS_DIR = "models"

def get_feature_group(col_name: str, commodity: str) -> str:
    """Classifies feature into COMMODITY_HISTORY, GEOPOLITICAL, or MACRO_CONTROL."""
    if col_name in [f"{commodity}_return_1m", f"{commodity}_return_3m", f"{commodity}_return_6m", f"{commodity}_vol_3m"]:
        return "COMMODITY_HISTORY"
    elif col_name == "DXY":
        return "MACRO_CONTROL"
    else:
        return "GEOPOLITICAL"

def explain_current_forecast(commodity: str) -> Dict[str, Any]:
    """
    Explains the latest GeoPrice forecast for a commodity by calculating feature contributions:
    Contribution_j = beta_j * z_j
    
    where:
    beta_j = fitted ElasticNet coefficient
    z_j = standardized feature value computed using trained StandardScaler.
    
    Validates exact reconstruction: Prediction == Intercept + sum(Contributions) within 1e-10.
    """
    model_path = os.path.join(MODELS_DIR, f"{commodity.lower()}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at '{model_path}'. Run scripts/retrain_models.py.")

    pipeline = joblib.load(model_path)
    scaler = pipeline.named_steps['scaler']
    model_obj = pipeline.named_steps['model']

    latest_date, feat_vector, feat_dict = get_latest_features(commodity)
    feat_cols = get_geoprice_feature_names(commodity)

    X_raw = feat_vector[feat_cols].values
    
    # 1. Transform current feature vector using trained scaler (z_j)
    z_scaled = scaler.transform(X_raw)[0]
    
    # 2. Extract coefficients (beta_j) and intercept (beta_0)
    beta_coefs = model_obj.coef_
    intercept = float(model_obj.intercept_)

    # 3. Calculate contributions (beta_j * z_j)
    contributions = beta_coefs * z_scaled
    sum_contributions = float(np.sum(contributions))
    reconstructed_pred = intercept + sum_contributions

    # 4. Actual Model Prediction
    actual_pred = float(pipeline.predict(X_raw)[0])
    reconstruction_diff = abs(reconstructed_pred - actual_pred)

    assert reconstruction_diff < 1e-10, f"Reconstruction check failed for {commodity}! Diff={reconstruction_diff}"

    # 5. Build detailed feature contribution records
    contrib_rows = []
    for idx, (col_name, raw_val, z_val, beta_val, c_val) in enumerate(zip(feat_cols, X_raw[0], z_scaled, beta_coefs, contributions)):
        grp = get_feature_group(col_name, commodity)
        c_dir = "Positive" if c_val > 1e-10 else ("Negative" if c_val < -1e-10 else "Zero")
        
        contrib_rows.append({
            "Commodity": commodity,
            "Feature": col_name,
            "Feature_Group": grp,
            "Raw_Value": float(raw_val),
            "Standardized_Value": float(z_val),
            "Coefficient": float(beta_val),
            "Contribution": float(c_val),
            "Absolute_Contribution": float(abs(c_val)),
            "Contribution_Direction": c_dir,
            "Is_Zero_Coefficient": bool(abs(beta_val) < 1e-10)
        })

    contrib_df = pd.DataFrame(contrib_rows)
    ranked_df = contrib_df.sort_values('Absolute_Contribution', ascending=False).reset_index(drop=True)
    ranked_df['Rank'] = range(1, len(ranked_df) + 1)

    return {
        "commodity": commodity,
        "forecast_origin_date": latest_date,
        "intercept": intercept,
        "sum_of_contributions": sum_contributions,
        "reconstructed_prediction": reconstructed_pred,
        "model_prediction": actual_pred,
        "reconstruction_difference": reconstruction_diff,
        "reconstruction_pass": bool(reconstruction_diff < 1e-10),
        "contributions_df": contrib_df,
        "ranked_contributions_df": ranked_df
    }

def evaluate_all_commodity_interpretations() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates consolidated Stage 12 interpretation datasets across all 5 commodities."""
    model_coef_rows = []
    curr_contrib_rows = []
    top_contrib_rows = []
    reconstruct_rows = []
    summary_rows = []

    for c in COMMODITIES:
        exp = explain_current_forecast(c)
        
        # 1. Model-wide coefficients
        for idx, row in exp['contributions_df'].iterrows():
            model_coef_rows.append({
                "Commodity": c,
                "Feature": row['Feature'],
                "Feature_Group": row['Feature_Group'],
                "Coefficient": row['Coefficient'],
                "Absolute_Coefficient": float(abs(row['Coefficient'])),
                "Is_Zero": row['Is_Zero_Coefficient']
            })

            curr_contrib_rows.append({
                "Commodity": c,
                "Feature": row['Feature'],
                "Feature_Group": row['Feature_Group'],
                "Raw_Value": row['Raw_Value'],
                "Standardized_Value": row['Standardized_Value'],
                "Coefficient": row['Coefficient'],
                "Contribution": row['Contribution']
            })

        for idx, row in exp['ranked_contributions_df'].iterrows():
            top_contrib_rows.append({
                "Commodity": c,
                "Rank": row['Rank'],
                "Feature": row['Feature'],
                "Feature_Group": row['Feature_Group'],
                "Standardized_Value": row['Standardized_Value'],
                "Coefficient": row['Coefficient'],
                "Contribution": row['Contribution'],
                "Contribution_Direction": row['Contribution_Direction']
            })

        # Reconstruction check
        reconstruct_rows.append({
            "Commodity": c,
            "Intercept": exp['intercept'],
            "Sum_of_Contributions": exp['sum_of_contributions'],
            "Reconstructed_Prediction": exp['reconstructed_prediction'],
            "Model_Prediction": exp['model_prediction'],
            "Difference": exp['reconstruction_difference'],
            "Reconstruction_Pass": exp['reconstruction_pass']
        })

        # Summary stats
        cdf = exp['contributions_df']
        n_zero = int(cdf['Is_Zero_Coefficient'].sum())
        n_nonzero = len(cdf) - n_zero
        
        pos_c = cdf.loc[cdf['Coefficient'].idxmax()]
        neg_c = cdf.loc[cdf['Coefficient'].idxmin()]
        
        pos_contrib = cdf.loc[cdf['Contribution'].idxmax()]
        neg_contrib = cdf.loc[cdf['Contribution'].idxmin()]

        summary_rows.append({
            "Commodity": c,
            "Number_of_nonzero_coefficients": n_nonzero,
            "Number_of_zero_coefficients": n_zero,
            "Largest_positive_coefficient_feature": pos_c['Feature'],
            "Largest_positive_coefficient": pos_c['Coefficient'],
            "Largest_negative_coefficient_feature": neg_c['Feature'],
            "Largest_negative_coefficient": neg_c['Coefficient'],
            "Largest_current_positive_contribution_feature": pos_contrib['Feature'],
            "Largest_current_positive_contribution": pos_contrib['Contribution'],
            "Largest_current_negative_contribution_feature": neg_contrib['Feature'],
            "Largest_current_negative_contribution": neg_contrib['Contribution']
        })

    return (
        pd.DataFrame(model_coef_rows),
        pd.DataFrame(curr_contrib_rows),
        pd.DataFrame(top_contrib_rows),
        pd.DataFrame(reconstruct_rows),
        pd.DataFrame(summary_rows)
    )
