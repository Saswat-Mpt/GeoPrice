import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.interpretation.contributions import evaluate_all_commodity_interpretations
from geoprice.analysis.shock_responses import COMMODITIES

def generate_stage_12_figures(model_coef_df: pd.DataFrame, curr_contrib_df: pd.DataFrame, output_dir: str = "outputs/figures/phase4"):
    """Generates visualization charts for Stage 12 Interpretation."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Model-wide coefficients by commodity
    plt.figure(figsize=(12, 6))
    non_int = model_coef_df[model_coef_df['Feature'] != 'Intercept'].copy()
    
    brent_coefs = non_int[non_int['Commodity'] == 'Brent']
    gold_coefs = non_int[non_int['Commodity'] == 'Gold']
    
    fx = np.arange(len(brent_coefs))
    plt.bar(fx - 0.2, brent_coefs['Coefficient'], 0.4, label='Brent Coefficients', color='#1f77b4')
    plt.bar(fx + 0.2, gold_coefs['Coefficient'], 0.4, label='Gold Coefficients', color='#ff7f0e')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Stage 12 Model-Wide Standardized ElasticNet Coefficients (beta_j)", fontsize=12, fontweight='bold')
    plt.xlabel("Feature Name", fontsize=10)
    plt.ylabel("Standardized Coefficient Weight (beta_j)", fontsize=10)
    plt.xticks(fx, brent_coefs['Feature'], rotation=45, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geoprice_model_coefficients.png"), dpi=300)
    plt.close()

    # Figure 2: Current Forecast Feature Contributions (beta_j * z_j) for Brent
    plt.figure(figsize=(10, 5))
    brent_contribs = curr_contrib_df[curr_contrib_df['Commodity'] == 'Brent'].sort_values('Contribution')
    
    colors = ['#d62728' if c < 0 else '#2ca02c' for c in brent_contribs['Contribution']]
    plt.barh(brent_contribs['Feature'], brent_contribs['Contribution'] * 100, color=colors)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    
    plt.title("Current Forecast Feature Contributions (beta_j * z_j) for Brent (2026-07)", fontsize=12, fontweight='bold')
    plt.xlabel("Return Contribution (% points)", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "current_forecast_contributions.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 12 Coefficient Interpretation & Explanation")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 4, Stage 12: Coefficient Interpretation (beta * z)")
    print("=" * 80)

    # 1. Evaluate Interpretations across 5 commodities
    print("\n[Step 1/3] Extracting coefficients and calculating current contributions (beta * z)...")
    model_coef_df, curr_contrib_df, top_contrib_df, reconstruct_df, summary_df = evaluate_all_commodity_interpretations()

    # 2. Prediction Reconstruction Audit
    print("\n[Step 2/3] Auditing exact prediction reconstruction (Prediction == Intercept + sum(beta * z))...")
    all_reconstructions_pass = bool(reconstruct_df['Reconstruction_Pass'].all())
    for idx, r in reconstruct_df.iterrows():
        status = "PASS [PASS]" if r['Reconstruction_Pass'] else "FAIL [FAIL]"
        print(f"  -> {r['Commodity']:15s} | Intercept: {r['Intercept']:+.4f} | Sum(beta*z): {r['Sum_of_Contributions']:+.4f} | Pred: {r['Model_Prediction']:+.4f} | Status: {status}")

    assert all_reconstructions_pass, "Prediction reconstruction check failed for one or more commodities!"

    # 3. Save Outputs & Generate Figures
    print("\n[Step 3/3] Saving Stage 12 output datasets and generating figures...")
    os.makedirs("data/processed", exist_ok=True)
    
    model_coef_df.to_csv("data/processed/geoprice_coefficients_interpretation.csv", index=False)
    curr_contrib_df.to_csv("data/processed/current_forecast_contributions.csv", index=False)
    top_contrib_df.to_csv("data/processed/top_current_contributors.csv", index=False)
    reconstruct_df.to_csv("data/processed/forecast_reconstruction_check.csv", index=False)
    summary_df.to_csv("data/processed/coefficient_summary.csv", index=False)

    generate_stage_12_figures(model_coef_df, curr_contrib_df)

    # Final Stage 12 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 12 FINAL SUMMARY REPORT")
    print("=" * 80)
    print("Top Current Feature Contributors (beta * z) by Commodity:")
    print("-" * 80)
    for c in COMMODITIES:
        print(f"\nCommodity: {c}")
        top3 = top_contrib_df[top_contrib_df['Commodity'] == c].head(3)
        print(top3[['Rank', 'Feature', 'Feature_Group', 'Standardized_Value', 'Coefficient', 'Contribution', 'Contribution_Direction']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Figures generated under outputs/figures/phase4/:")
    print("  1. geoprice_model_coefficients.png")
    print("  2. current_forecast_contributions.png")
    print("\nStage 12 complete. Model interpretability validated. Ready for Stage 13: Final Dashboard.")
    print("=" * 80)

if __name__ == "__main__":
    main()
