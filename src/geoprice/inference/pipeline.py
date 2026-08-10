import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.analysis.shock_responses import COMMODITIES

MODELS_DIR = "models"
DATA_DIR = "data/processed"

def get_latest_features(commodity: str) -> Tuple[str, pd.DataFrame, Dict[str, float]]:
    """Loads latest forecast origin date and 11 feature values for a given commodity."""
    feat_path = os.path.join(DATA_DIR, "feature_dataset.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Validated dataset '{feat_path}' missing. Run scripts/update_data.py.")

    df = pd.read_csv(feat_path)
    feat_cols = get_geoprice_feature_names(commodity)
    
    # Valid rows with complete predictor features
    valid_df = df.dropna(subset=feat_cols).sort_values('Date')
    if len(valid_df) == 0:
        raise ValueError(f"No valid feature rows found for commodity '{commodity}'.")

    latest_row = valid_df.iloc[-1]
    latest_date = str(latest_row['Date'])
    
    feat_dict = {col: float(latest_row[col]) for col in feat_cols}
    feat_vector = pd.DataFrame([feat_dict])
    
    return latest_date, feat_vector, feat_dict

def predict_next_month(commodity: str) -> Dict[str, Any]:
    """
    Runs production inference for next-month commodity return using trained ElasticNet pipeline artifact.
    Does NOT retrain models. Loads saved .joblib artifact from models/.
    """
    model_path = os.path.join(MODELS_DIR, f"{commodity.lower()}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at '{model_path}'. Run scripts/retrain_models.py.")

    pipeline = joblib.load(model_path)
    latest_date, feat_vector, feat_dict = get_latest_features(commodity)
    
    feat_cols = get_geoprice_feature_names(commodity)
    X_input = feat_vector[feat_cols].values

    pred_return = float(pipeline.predict(X_input)[0])
    pred_dir = "UP" if pred_return > 0 else "DOWN"

    # Next month label calculation
    p_date = pd.to_datetime(latest_date)
    target_month = str((p_date + pd.DateOffset(months=1)).to_period('M'))

    # Extract model standardized coefficients
    model_obj = pipeline.named_steps['model']
    coefs = model_obj.coef_
    intercept = float(model_obj.intercept_)

    weights = []
    for col_name, cval in zip(feat_cols, coefs):
        if col_name in [f"{commodity}_return_1m", f"{commodity}_return_3m", f"{commodity}_return_6m", f"{commodity}_vol_3m"]:
            grp = "Commodity History"
        elif col_name == "DXY":
            grp = "Macro Control"
        else:
            grp = "Geopolitical Risk"
            
        weights.append({
            "feature": col_name,
            "group": grp,
            "coefficient": float(cval),
            "abs_weight": float(abs(cval))
        })

    weights_df = pd.DataFrame(weights).sort_values('abs_weight', ascending=False).reset_index(drop=True)

    return {
        "commodity": commodity,
        "forecast_origin_date": latest_date,
        "target_month": target_month,
        "predicted_return_decimal": pred_return,
        "predicted_return_pct": pred_return * 100.0,
        "predicted_direction": pred_dir,
        "intercept": intercept,
        "feature_values": feat_dict,
        "feature_weights": weights_df
    }

def get_current_risk_context(commodity: str) -> Dict[str, Any]:
    """Retrieves current geopolitical risk regime, subindex percentiles, analogue statistics, and shock status."""
    current_regime_path = os.path.join(DATA_DIR, "current_gpr_regime.json")
    regime_sum_path = os.path.join(DATA_DIR, "regime_summary.csv")
    shock_meta_path = os.path.join(DATA_DIR, "gpr_shock_threshold.json")
    aligned_path = os.path.join(DATA_DIR, "monthly_aligned.csv")

    if not os.path.exists(current_regime_path) or not os.path.exists(regime_sum_path):
        raise FileNotFoundError("Phase 2 analytical outputs missing. Run scripts/update_data.py.")

    with open(current_regime_path) as f:
        curr_state = json.load(f)

    with open(shock_meta_path) as f:
        shock_meta = json.load(f)

    reg_sum = pd.read_csv(regime_sum_path)
    df_aligned = pd.read_csv(aligned_path).set_index('Date')

    latest_date = curr_state['current_date']
    curr_regime = curr_state['current_GPR_regime']
    
    # Extract historical analogue +1M and +3M medians for this commodity and regime
    sub_reg = reg_sum[(reg_sum['Commodity'] == commodity) & (reg_sum['Regime'] == curr_regime)]
    
    m1_row = sub_reg[sub_reg['Horizon'] == '+1M']
    m3_row = sub_reg[sub_reg['Horizon'] == '+3M']
    
    m1_median = float(m1_row['Median'].iloc[0]) if len(m1_row) > 0 else np.nan
    m3_median = float(m3_row['Median'].iloc[0]) if len(m3_row) > 0 else np.nan
    n_episodes = int(m1_row['N'].iloc[0]) if len(m1_row) > 0 else 0

    # Calculate shock status for latest month relative to immediately preceding month
    if latest_date in df_aligned.index:
        latest_gpr = float(df_aligned.loc[latest_date, 'GPR'])
        curr_loc = df_aligned.index.get_loc(latest_date)
        if curr_loc > 0:
            prev_date = df_aligned.index[curr_loc - 1]
            prev_gpr = float(df_aligned.loc[prev_date, 'GPR'])
        else:
            prev_gpr = latest_gpr
    else:
        latest_gpr = float(curr_state['current_GPR'])
        prev_gpr = latest_gpr

    latest_dgpr = float(latest_gpr - prev_gpr)
    
    is_shock = bool(latest_dgpr >= shock_meta['threshold'])

    current_price = float(df_aligned.loc[latest_date, commodity]) if (latest_date in df_aligned.index and commodity in df_aligned.columns) else np.nan

    return {
        "commodity": commodity,
        "current_price": current_price,
        "latest_date": latest_date,
        "current_GPR": curr_state['current_GPR'],
        "current_GPR_percentile": curr_state['current_GPR_percentile'],
        "current_GPR_regime": curr_regime,
        "current_GPRT": curr_state['current_GPRT'],
        "current_GPRT_percentile": curr_state['current_GPRT_percentile'],
        "current_GPRA": curr_state['current_GPRA'],
        "current_GPRA_percentile": curr_state['current_GPRA_percentile'],
        "latest_delta_gpr": latest_dgpr,
        "shock_threshold": shock_meta['threshold'],
        "is_gpr_shock": is_shock,
        "analogue_1m_median_pct": m1_median * 100.0 if pd.notna(m1_median) else np.nan,
        "analogue_3m_median_pct": m3_median * 100.0 if pd.notna(m3_median) else np.nan,
        "analogue_episodes_count": n_episodes
    }
