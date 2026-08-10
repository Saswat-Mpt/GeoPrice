import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.geoprice import (
    run_expanding_window_geoprice,
    get_geoprice_feature_names
)
from geoprice.models.validation import validate_expanding_window_order
from geoprice.analysis.shock_responses import COMMODITIES

def generate_stage_08_figures(all_preds_df: pd.DataFrame, all_ablation_df: pd.DataFrame, all_coefs_df: pd.DataFrame, output_dir: str = "outputs/figures/phase3"):
    """Generates visualization charts for Stage 8 GeoPrice Model."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Actual vs GeoPrice predicted returns
    fig, axes = plt.subplots(len(COMMODITIES), 1, figsize=(12, 12), sharex=True)
    for idx, c in enumerate(COMMODITIES):
        c_preds = all_preds_df[all_preds_df['Commodity'] == c].copy()
        c_preds['Date_dt'] = pd.to_datetime(c_preds['Date'])
        
        ax = axes[idx]
        ax.plot(c_preds['Date_dt'], c_preds['Actual_Return'] * 100, label='Actual Next-Month Return (%)', color='black', alpha=0.6, linewidth=1.2)
        ax.plot(c_preds['Date_dt'], c_preds['Predicted_Return'] * 100, label='GeoPrice Forecast (%)', color='#d62728', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_title(f"GeoPrice Out-of-Sample Forecasts vs Actuals: {c}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Return (%)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel("Forecast Origin Date", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geoprice_actual_vs_pred.png"), dpi=300)
    plt.close()

    # Figure 2: Baseline vs GPR_only vs GeoPrice MAE (Ablation Chart)
    plt.figure(figsize=(11, 5))
    x = np.arange(len(COMMODITIES))
    width = 0.25

    base_maes = [all_ablation_df[(all_ablation_df['Commodity'] == c) & (all_ablation_df['Feature_Set'] == 'Baseline')]['MAE'].values[0] * 100 for c in COMMODITIES]
    gpr_maes = [all_ablation_df[(all_ablation_df['Commodity'] == c) & (all_ablation_df['Feature_Set'] == 'GPR_only')]['MAE'].values[0] * 100 for c in COMMODITIES]
    geo_maes = [all_ablation_df[(all_ablation_df['Commodity'] == c) & (all_ablation_df['Feature_Set'] == 'GeoPrice')]['MAE'].values[0] * 100 for c in COMMODITIES]

    plt.bar(x - width, base_maes, width, label='Baseline (History Only)', color='#7f7f7f')
    plt.bar(x, gpr_maes, width, label='GPR-Only Ablation', color='#1f77b4')
    plt.bar(x + width, geo_maes, width, label='GeoPrice (Full 11 Features)', color='#d62728')

    plt.title("Ablation Study MAE (%): Baseline vs GPR-Only vs GeoPrice", fontsize=12, fontweight='bold')
    plt.xlabel("Commodity", fontsize=10)
    plt.ylabel("MAE (%)", fontsize=10)
    plt.xticks(x, COMMODITIES)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geoprice_vs_baseline_mae.png"), dpi=300)
    plt.close()

    # Figure 3: Standardized Feature Coefficients for GeoPrice
    plt.figure(figsize=(12, 6))
    non_int = all_coefs_df[all_coefs_df['Feature'] != 'Intercept'].copy()
    
    # Plot feature coefficients for Brent and Gold as primary examples
    brent_coefs = non_int[non_int['Commodity'] == 'Brent']
    gold_coefs = non_int[non_int['Commodity'] == 'Gold']
    
    fx = np.arange(len(brent_coefs))
    plt.bar(fx - 0.2, brent_coefs['Coefficient'], 0.4, label='Brent Coefficients', color='#1f77b4')
    plt.bar(fx + 0.2, gold_coefs['Coefficient'], 0.4, label='Gold Coefficients', color='#ff7f0e')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("GeoPrice Standardized ElasticNet Feature Coefficients (Brent & Gold)", fontsize=12, fontweight='bold')
    plt.xlabel("Feature Name", fontsize=10)
    plt.ylabel("Standardized Coefficient Weight", fontsize=10)
    plt.xticks(fx, brent_coefs['Feature'], rotation=45, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geoprice_coefficients.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 8 GeoPrice Model Forecasting")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 3, Stage 8: GeoPrice Model Forecasting")
    print("=" * 80)

    feat_path = "data/processed/feature_dataset.csv"
    raw_path = "data/processed/monthly_aligned.csv"
    base_preds_path = "data/processed/baseline_predictions.csv"

    if not os.path.exists(feat_path) or not os.path.exists(raw_path) or not os.path.exists(base_preds_path):
        print("Error: Required Stage 7 outputs missing!")
        sys.exit(1)

    df_features = pd.read_csv(feat_path)
    df_raw = pd.read_csv(raw_path)
    base_preds_df = pd.read_csv(base_preds_path)

    print(f"\n[Step 1/5] Loaded dataset: {len(df_features)} total rows.")
    print("-> Features: 11 per commodity (4 history + 6 GPR + 1 DXY)")
    print("-> Target: Next-month commodity decimal return y_t = P_(t+1)/P_t - 1")

    all_preds = []
    all_metrics = []
    all_coefs = []
    all_ablations = []

    print("\n[Step 2/5] Running expanding-window out-of-sample validation for GeoPrice Model...")
    for c in COMMODITIES:
        pred_df, metrics_df, coef_df, ablation_df, config = run_expanding_window_geoprice(
            df_features, df_raw, commodity=c, start_year=2006, min_train_months=48, alpha=0.01, l1_ratio=0.5
        )
        
        # Verify prediction dates match baseline prediction dates exactly
        c_base = base_preds_df[base_preds_df['Commodity'] == c]
        assert list(pred_df['Date']) == list(c_base['Date']), f"Prediction date mismatch for {c} between Baseline and GeoPrice!"
        
        all_preds.append(pred_df)
        all_metrics.append(metrics_df)
        all_coefs.append(coef_df)
        all_ablations.append(ablation_df)
        
        print(f"  -> {c:15s} | OOS Predictions: {len(pred_df)} ({pred_df['Date'].iloc[0]} to {pred_df['Date'].iloc[-1]})")

    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_coefs_df = pd.concat(all_coefs, ignore_index=True)
    all_ablations_df = pd.concat(all_ablations, ignore_index=True)

    print("\n[Step 3/5] Saving Stage 8 output datasets...")
    os.makedirs("data/processed", exist_ok=True)
    
    all_preds_df.to_csv("data/processed/geoprice_predictions.csv", index=False)
    all_metrics_df.to_csv("data/processed/geoprice_metrics.csv", index=False)
    all_coefs_df.to_csv("data/processed/geoprice_coefficients.csv", index=False)
    all_ablations_df.to_csv("data/processed/geoprice_ablation.csv", index=False)

    print("\n[Step 4/5] Generating Stage 8 figures under outputs/figures/phase3/...")
    generate_stage_08_figures(all_preds_df, all_ablations_df, all_coefs_df)

    # Final Stage 8 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 8 FINAL SUMMARY REPORT")
    print("=" * 80)
    print("Ablation Metrics Comparison (Baseline vs GPR_only vs GeoPrice):")
    print("-" * 80)
    
    for c in COMMODITIES:
        print(f"\nCommodity: {c}")
        sub = all_ablations_df[all_ablations_df['Commodity'] == c].copy()
        sub['MAE (%)'] = sub['MAE'] * 100
        sub['RMSE (%)'] = sub['RMSE'] * 100
        sub['DA (%)'] = sub['Directional_Accuracy'] * 100
        print(sub[['Feature_Set', 'N', 'MAE (%)', 'RMSE (%)', 'DA (%)']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Stage 8 complete. GeoPrice Model established. Ready for Stage 9: Final Comparison & Phase 3 Completion.")
    print("=" * 80)

if __name__ == "__main__":
    main()
