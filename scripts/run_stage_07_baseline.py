import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.baseline import (
    run_expanding_window_baseline,
    get_baseline_feature_names
)
from geoprice.models.validation import (
    validate_baseline_features,
    validate_expanding_window_order
)
from geoprice.analysis.shock_responses import COMMODITIES

def generate_stage_07_figures(all_preds_df: pd.DataFrame, all_metrics_df: pd.DataFrame, output_dir: str = "outputs/figures/phase3"):
    """Generates visualization charts for Stage 7 Baseline Forecasting."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Actual vs Predicted next-month return for each commodity
    fig, axes = plt.subplots(len(COMMODITIES), 1, figsize=(12, 12), sharex=True)
    
    for idx, c in enumerate(COMMODITIES):
        c_preds = all_preds_df[all_preds_df['Commodity'] == c].copy()
        c_preds['Date_dt'] = pd.to_datetime(c_preds['Date'])
        
        ax = axes[idx]
        ax.plot(c_preds['Date_dt'], c_preds['Actual_Return'] * 100, label='Actual Next-Month Return (%)', color='black', alpha=0.6, linewidth=1.2)
        ax.plot(c_preds['Date_dt'], c_preds['Predicted_Return'] * 100, label='ElasticNet Baseline Forecast (%)', color='#1f77b4', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_title(f"Baseline Out-of-Sample Forecasts vs Actuals: {c}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Return (%)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel("Forecast Origin Date", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_actual_vs_pred.png"), dpi=300)
    plt.close()

    # Figure 2: Naive vs ElasticNet Baseline MAE Comparison across commodities
    plt.figure(figsize=(10, 5))
    x = np.arange(len(COMMODITIES))
    width = 0.35

    naive_maes = [all_metrics_df[(all_metrics_df['Commodity'] == c) & (all_metrics_df['Model'] == 'Naive (Zero Return)')]['MAE'].values[0] * 100 for c in COMMODITIES]
    elastic_maes = [all_metrics_df[(all_metrics_df['Commodity'] == c) & (all_metrics_df['Model'] == 'ElasticNet Baseline')]['MAE'].values[0] * 100 for c in COMMODITIES]

    plt.bar(x - width/2, naive_maes, width, label='Naive Zero-Return MAE (%)', color='#7f7f7f')
    plt.bar(x + width/2, elastic_maes, width, label='ElasticNet Baseline MAE (%)', color='#1f77b4')

    plt.title("Out-of-Sample Mean Absolute Error (MAE): Naive vs ElasticNet Baseline", fontsize=12, fontweight='bold')
    plt.xlabel("Commodity", fontsize=10)
    plt.ylabel("MAE (%)", fontsize=10)
    plt.xticks(x, COMMODITIES)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_vs_naive_mae.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 7 Baseline Commodity Return Forecasting")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 3, Stage 7: Baseline Commodity Return Forecasting")
    print("=" * 80)

    feat_path = "data/processed/feature_dataset.csv"
    raw_path = "data/processed/monthly_aligned.csv"

    if not os.path.exists(feat_path) or not os.path.exists(raw_path):
        print("Error: Feature dataset or raw monthly aligned dataset missing!")
        sys.exit(1)

    df_features = pd.read_csv(feat_path)
    df_raw = pd.read_csv(raw_path)

    print(f"\n[Step 1/5] Loaded feature dataset: {len(df_features)} total rows.")
    print("-> Target: Next-month commodity decimal return y_t = P_(t+1)/P_t - 1")
    print("-> Window: Phase 3 DXY-supported period (2006 to present)")

    all_preds = []
    all_metrics = []
    all_coefs = []
    all_configs = {}

    print("\n[Step 2/5] Running expanding-window out-of-sample validation for 5 commodities...")
    for c in COMMODITIES:
        feat_cols = get_baseline_feature_names(c)
        assert validate_baseline_features(feat_cols), f"Geopolitical features detected in baseline feature set for {c}!"
        
        pred_df, metrics_df, coef_df, config = run_expanding_window_baseline(
            df_features, df_raw, commodity=c, start_year=2006, min_train_months=48, alpha=0.01, l1_ratio=0.5
        )
        
        assert validate_expanding_window_order(pred_df), f"Time ordering violation in predictions for {c}!"
        
        all_preds.append(pred_df)
        all_metrics.append(metrics_df)
        all_coefs.append(coef_df)
        all_configs[c] = config
        
        print(f"  -> {c:15s} | OOS Predictions: {len(pred_df)} ({pred_df['Date'].iloc[0]} to {pred_df['Date'].iloc[-1]})")

    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_coefs_df = pd.concat(all_coefs, ignore_index=True)

    print("\n[Step 3/5] Saving Stage 7 output datasets...")
    os.makedirs("data/processed", exist_ok=True)
    
    all_preds_df.to_csv("data/processed/baseline_predictions.csv", index=False)
    all_metrics_df.to_csv("data/processed/baseline_metrics.csv", index=False)
    all_coefs_df.to_csv("data/processed/baseline_coefficients.csv", index=False)
    
    with open("data/processed/baseline_model_config.json", "w") as f:
        json.dump(all_configs, f, indent=4)

    print("\n[Step 4/5] Generating Phase 3 baseline plots under outputs/figures/phase3/...")
    generate_stage_07_figures(all_preds_df, all_metrics_df)

    # Final Stage 7 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 7 FINAL SUMMARY REPORT")
    print("=" * 80)
    print("Out-of-Sample Performance Comparison (Naive vs ElasticNet Baseline):")
    print("-" * 80)
    
    for c in COMMODITIES:
        print(f"\nCommodity: {c}")
        sub = all_metrics_df[all_metrics_df['Commodity'] == c].copy()
        sub['MAE (%)'] = sub['MAE'] * 100
        sub['RMSE (%)'] = sub['RMSE'] * 100
        sub['DA (%)'] = sub['Directional_Accuracy'] * 100
        print(sub[['Model', 'N', 'MAE (%)', 'RMSE (%)', 'DA (%)']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Figures generated under outputs/figures/phase3/:")
    print("  1. baseline_actual_vs_pred.png")
    print("  2. baseline_vs_naive_mae.png")
    print("\nStage 7 complete. Baseline benchmark established. Ready for Stage 8: GeoPrice Model.")
    print("=" * 80)

if __name__ == "__main__":
    main()
