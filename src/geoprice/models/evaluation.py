import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from geoprice.models.metrics import calculate_mae, calculate_rmse, calculate_directional_accuracy
from geoprice.analysis.shock_responses import COMMODITIES

def compute_model_improvements(base_metrics_df: pd.DataFrame, geo_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Computes MAE, RMSE, and Directional Accuracy improvements of GeoPrice over Baseline."""
    imp_rows = []
    commodities_to_process = [c for c in COMMODITIES if c in base_metrics_df['Commodity'].values]
    if not commodities_to_process:
        commodities_to_process = list(base_metrics_df['Commodity'].unique())
    
    for c in commodities_to_process:
        b_row = base_metrics_df[(base_metrics_df['Commodity'] == c) & (base_metrics_df['Model'] == 'ElasticNet Baseline')].iloc[0]
        g_row = geo_metrics_df[geo_metrics_df['Commodity'] == c].iloc[0]
        
        b_mae, g_mae = b_row['MAE'], g_row['MAE']
        b_rmse, g_rmse = b_row['RMSE'], g_row['RMSE']
        b_da, g_da = b_row['Directional_Accuracy'], g_row['Directional_Accuracy']
        
        mae_imp = b_mae - g_mae
        mae_imp_pct = (mae_imp / b_mae) * 100.0 if b_mae > 0 else np.nan
        
        rmse_imp = b_rmse - g_rmse
        rmse_imp_pct = (rmse_imp / b_rmse) * 100.0 if b_rmse > 0 else np.nan
        
        da_imp_pts = (g_da - b_da) * 100.0
        
        imp_rows.append({
            "Commodity": c,
            "Baseline_MAE": b_mae,
            "GeoPrice_MAE": g_mae,
            "MAE_Improvement": mae_imp,
            "MAE_Improvement_Pct": mae_imp_pct,
            "Baseline_RMSE": b_rmse,
            "GeoPrice_RMSE": g_rmse,
            "RMSE_Improvement": rmse_imp,
            "RMSE_Improvement_Pct": rmse_imp_pct,
            "Baseline_Directional_Accuracy": b_da,
            "GeoPrice_Directional_Accuracy": g_da,
            "Directional_Accuracy_Improvement_Points": da_imp_pts
        })
        
    return pd.DataFrame(imp_rows)

def compute_paired_error_uncertainty(
    geo_preds_df: pd.DataFrame,
    base_preds_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Computes paired monthly absolute error differences d_t = |y_t - y_hat_base| - |y_t - y_hat_geo|
    and 95% bootstrap confidence intervals for mean(d_t).
    Positive mean_d indicates GeoPrice error reduction relative to Baseline.
    """
    np.random.seed(seed)
    rows = []
    
    merged = pd.merge(
        geo_preds_df,
        base_preds_df.rename(columns={"Predicted_Return": "Baseline_Predicted"}),
        on=['Date', 'Commodity', 'Actual_Return']
    )
    
    for c in COMMODITIES:
        sub = merged[merged['Commodity'] == c].copy()
        if len(sub) == 0:
            continue
            
        base_err = np.abs(sub['Actual_Return'] - sub['Baseline_Predicted'])
        geo_err = np.abs(sub['Actual_Return'] - sub['Predicted_Return'])
        d_t = (base_err - geo_err).values  # Positive means GeoPrice MAE is smaller
        
        mean_d = float(np.mean(d_t))
        std_d = float(np.std(d_t, ddof=1))
        
        # Bootstrap 95% CI
        boot_means = []
        n_samples = len(d_t)
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(d_t, size=n_samples, replace=True)
            boot_means.append(np.mean(boot_sample))
            
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        
        is_stat_sig = bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))
        
        rows.append({
            "Commodity": c,
            "N": n_samples,
            "Mean_Paired_Diff": mean_d,
            "Std_Paired_Diff": std_d,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
            "Statistically_Significant": is_stat_sig
        })
        
    return pd.DataFrame(rows)

def compute_regime_robustness(
    geo_preds_df: pd.DataFrame,
    base_preds_df: pd.DataFrame,
    df_regimes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subgroup analysis 1: High/Extreme vs Low/Moderate GPR Regimes.
    Sensitivity analysis 2: Excluding EXTREME GPR Regime Months.
    """
    reg_map = df_regimes.set_index('Date')['GPR_regime'].to_dict()
    
    geo_df = geo_preds_df.copy()
    base_df = base_preds_df.copy().rename(columns={"Predicted_Return": "Baseline_Predicted"})
    
    merged = pd.merge(geo_df, base_df[['Date', 'Commodity', 'Baseline_Predicted']], on=['Date', 'Commodity'])
    merged['GPR_regime'] = merged['Date'].map(reg_map)
    
    rob_rows = []
    
    for c in COMMODITIES:
        sub_c = merged[merged['Commodity'] == c]
        
        # High/Extreme subgroup
        high_mask = sub_c['GPR_regime'].isin(['HIGH', 'EXTREME'])
        low_mask = sub_c['GPR_regime'].isin(['LOW', 'MODERATE'])
        
        # High/Extreme metrics
        sub_high = sub_c[high_mask]
        b_mae_high = calculate_mae(sub_high['Actual_Return'], sub_high['Baseline_Predicted'])
        g_mae_high = calculate_mae(sub_high['Actual_Return'], sub_high['Predicted_Return'])
        
        # Low/Moderate metrics
        sub_low = sub_c[low_mask]
        b_mae_low = calculate_mae(sub_low['Actual_Return'], sub_low['Baseline_Predicted'])
        g_mae_low = calculate_mae(sub_low['Actual_Return'], sub_low['Predicted_Return'])
        
        rob_rows.append({
            "Commodity": c,
            "Subgroup": "HIGH_and_EXTREME_Regimes",
            "N": len(sub_high),
            "Baseline_MAE": b_mae_high,
            "GeoPrice_MAE": g_mae_high,
            "MAE_Diff": b_mae_high - g_mae_high
        })
        rob_rows.append({
            "Commodity": c,
            "Subgroup": "LOW_and_MODERATE_Regimes",
            "N": len(sub_low),
            "Baseline_MAE": b_mae_low,
            "GeoPrice_MAE": g_mae_low,
            "MAE_Diff": b_mae_low - g_mae_low
        })
        
        # Sensitivity check: excluding EXTREME
        sub_no_extreme = sub_c[sub_c['GPR_regime'] != 'EXTREME']
        b_mae_no_ext = calculate_mae(sub_no_extreme['Actual_Return'], sub_no_extreme['Baseline_Predicted'])
        g_mae_no_ext = calculate_mae(sub_no_extreme['Actual_Return'], sub_no_extreme['Predicted_Return'])
        
        rob_rows.append({
            "Commodity": c,
            "Subgroup": "Excluding_EXTREME_Regime",
            "N": len(sub_no_extreme),
            "Baseline_MAE": b_mae_no_ext,
            "GeoPrice_MAE": g_mae_no_ext,
            "MAE_Diff": b_mae_no_ext - g_mae_no_ext
        })

    return pd.DataFrame(rob_rows), merged

def compute_directional_confusion(geo_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Creates confusion matrix summary for GeoPrice directional prediction."""
    rows = []
    for c in COMMODITIES:
        sub = geo_preds_df[(geo_preds_df['Commodity'] == c) & (geo_preds_df['Actual_Return'] != 0.0)].copy()
        
        actual_up = (sub['Actual_Return'] > 0)
        actual_down = (sub['Actual_Return'] < 0)
        pred_up = (sub['Predicted_Return'] > 0)
        pred_down = (sub['Predicted_Return'] < 0)
        
        tp = int((actual_up & pred_up).sum())
        tn = int((actual_down & pred_down).sum())
        fp = int((actual_down & pred_up).sum())
        fn = int((actual_up & pred_down).sum())
        
        rows.append({
            "Commodity": c,
            "Total_N": len(sub),
            "Actual_Up_Pred_Up (TP)": tp,
            "Actual_Down_Pred_Down (TN)": tn,
            "Actual_Down_Pred_Up (FP)": fp,
            "Actual_Up_Pred_Down (FN)": fn,
            "Directional_Accuracy_Pct": float((tp + tn) / len(sub) * 100.0) if len(sub) > 0 else np.nan
        })
    return pd.DataFrame(rows)

def compute_largest_prediction_errors(geo_preds_df: pd.DataFrame, df_aligned: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Extracts top K largest absolute error observations per commodity."""
    df_raw = df_aligned.set_index('Date')
    rows = []
    
    for c in COMMODITIES:
        sub = geo_preds_df[geo_preds_df['Commodity'] == c].copy()
        sub['Abs_Error'] = sub['Absolute_Error']
        top = sub.sort_values('Abs_Error', ascending=False).head(top_k)
        
        for _, r in top.iterrows():
            d = r['Date']
            gpr_val = df_raw.loc[d, 'GPR'] if d in df_raw.index and 'GPR' in df_raw.columns else np.nan
            gprt_val = df_raw.loc[d, 'GPRT'] if d in df_raw.index and 'GPRT' in df_raw.columns else np.nan
            gpra_val = df_raw.loc[d, 'GPRA'] if d in df_raw.index and 'GPRA' in df_raw.columns else np.nan
            
            rows.append({
                "Date": d,
                "Commodity": c,
                "Actual_Return": r['Actual_Return'],
                "Predicted_Return": r['Predicted_Return'],
                "Absolute_Error": r['Abs_Error'],
                "GPR": gpr_val,
                "GPRT": gprt_val,
                "GPRA": gpra_val
            })
            
    return pd.DataFrame(rows)
