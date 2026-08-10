import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shocks import (
    calculate_gpr_change,
    calculate_shock_threshold,
    identify_raw_shocks,
    collapse_overlapping_shocks
)
from geoprice.analysis.shock_responses import calculate_forward_commodity_responses, COMMODITIES
from geoprice.analysis.validation import validate_shock_analysis

def generate_figures(df_full: pd.DataFrame, raw_shocks: pd.DataFrame, threshold_meta: dict, summary_df: pd.DataFrame, output_dir: str = "outputs/figures"):
    """Generates 3 required analysis figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: GPR level over time with raw shock months marked
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(df_full['Date']), df_full['GPR'], label='GPR Index', color='#1f77b4', linewidth=1.5)
    
    raw_dates = pd.to_datetime(raw_shocks['Date'])
    raw_gprs = raw_shocks['GPR']
    plt.scatter(raw_dates, raw_gprs, color='#d62728', zorder=5, label='Top-Decile GPR Shocks', s=35)
    
    plt.title("Caldara-Iacoviello Geopolitical Risk Index (GPR) & Identified Shock Events (1985-2026)", fontsize=12, fontweight='bold')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("GPR Level", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpr_shocks_timeline.png"), dpi=300)
    plt.close()

    # Figure 2: Distribution of positive ΔGPR values with 90th percentile marked
    gpr_changes = calculate_gpr_change(df_full).dropna()
    pos_changes = gpr_changes[gpr_changes > 0]
    
    plt.figure(figsize=(9, 5))
    plt.hist(pos_changes, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.axvline(threshold_meta['threshold'], color='#d62728', linestyle='--', linewidth=2, label=f"90th Pct Threshold ({threshold_meta['threshold']:.2f})")
    plt.title("Distribution of Positive Monthly GPR Increases (ΔGPR > 0)", fontsize=12, fontweight='bold')
    plt.xlabel("Monthly Increase ΔGPR", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpr_delta_distribution.png"), dpi=300)
    plt.close()

    # Figure 3: Commodity response summary chart showing median +1M/+2M/+3M response
    plt.figure(figsize=(10, 5))
    horizons = ['+1M', '+2M', '+3M']
    x = np.arange(len(COMMODITIES))
    width = 0.25

    for i, h in enumerate(horizons):
        sub_df = summary_df[summary_df['Horizon'] == h]
        medians = [sub_df[sub_df['Commodity'] == c]['Median'].values[0] * 100 for c in COMMODITIES]
        plt.bar(x + i*width, medians, width, label=f"{h} Median Return (%)")

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Historical Commodity Responses Following Geopolitical Shocks (Median %)", fontsize=12, fontweight='bold')
    plt.xlabel("Commodity", fontsize=10)
    plt.ylabel("Median Cumulative Return (%)", fontsize=10)
    plt.xticks(x + width, COMMODITIES)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "commodity_shock_responses.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 3 Shock Identification & Response Analysis")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 2, Stage 3: Geopolitical Shock Identification & Commodity Response")
    print("=" * 80)

    aligned_path = "data/processed/monthly_aligned.csv"
    if not os.path.exists(aligned_path):
        print(f"Error: Stage 1 output file '{aligned_path}' not found!")
        sys.exit(1)

    # 1. Load Data & Determine Analysis Window
    print(f"\n[Step 1/6] Loading Stage 1 validated monthly dataset...")
    df_aligned = pd.read_csv(aligned_path)
    
    # Filter to GPR active range (1985-01 onward)
    valid_df = df_aligned.dropna(subset=['GPR', 'Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']).copy()
    valid_start = valid_df['Date'].iloc[0]
    valid_end = valid_df['Date'].iloc[-1]
    print(f"-> Total dataset months:   {len(df_aligned)} ({df_aligned['Date'].iloc[0]} to {df_aligned['Date'].iloc[-1]})")
    print(f"-> Analysis window:        {len(valid_df)} months ({valid_start} to {valid_end})")

    # 2. Calculate Delta GPR & 90th Percentile Threshold
    print("\n[Step 2/6] Calculating Delta GPR and 90th-percentile shock threshold...")
    threshold, thresh_meta = calculate_shock_threshold(valid_df, percentile=90.0)
    print(f"-> Positive Delta GPR count:    {thresh_meta['positive_change_count']}")
    print(f"-> 90th-Percentile Cutoff: {threshold:.4f}")

    # 3. Identify Raw Shocks
    print("\n[Step 3/6] Identifying raw top-decile GPR shock months...")
    raw_shocks = identify_raw_shocks(valid_df, threshold)
    print(f"-> Raw shock months found: {len(raw_shocks)}")

    # 4. Collapse Overlapping Shocks
    print("\n[Step 4/6] Collapsing overlapping shocks (within 3 months)...")
    episodes_df = collapse_overlapping_shocks(raw_shocks, valid_df, window_months=3)
    print(f"-> Non-overlapping shock episodes: {len(episodes_df)}")
    avg_shocks_per_ep = len(raw_shocks) / len(episodes_df) if len(episodes_df) > 0 else 0
    print(f"-> Avg raw shocks per episode:    {avg_shocks_per_ep:.2f}")

    # 5. Calculate Forward Commodity Responses
    print("\n[Step 5/6] Measuring forward commodity responses (+1M, +2M, +3M)...")
    responses_df, summary_df = calculate_forward_commodity_responses(episodes_df, valid_df)
    
    # 6. Save Processed Outputs & Figures
    print("\n[Step 6/6] Saving shock analysis outputs and generating figures...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/gpr_shock_threshold.json", "w") as f:
        json.dump(thresh_meta, f, indent=4)
        
    raw_shocks.to_csv("data/processed/raw_gpr_shocks.csv", index=False)
    episodes_df.to_csv("data/processed/shock_episodes.csv", index=False)
    responses_df.to_csv("data/processed/shock_responses.csv", index=False)
    summary_df.to_csv("data/processed/shock_summary.csv", index=False)

    generate_figures(valid_df, raw_shocks, thresh_meta, summary_df)
    
    val_results = validate_shock_analysis(raw_shocks, episodes_df, responses_df, threshold)
    print(f"-> Validation status: {'PASS [PASS]' if val_results['overall_pass'] else 'FAIL'}")

    # Final Stage 3 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 3 FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Analysis Window:    {valid_start} to {valid_end} ({len(valid_df)} months)")
    print(f"Delta GPR Formula:  Delta GPR_t = GPR_t - GPR_(t-1)")
    print(f"Positive Delta GPRs:{thresh_meta['positive_change_count']}")
    print(f"90th Pct Threshold: {threshold:.4f}")
    print(f"Raw Shock Months:   {len(raw_shocks)}")
    print(f"Shock Episodes:     {len(episodes_df)} (Earliest: {episodes_df['representative_shock_date'].iloc[0]}, Latest: {episodes_df['representative_shock_date'].iloc[-1]})")

    print("\nCommodity Forward Return Summary (Representative Shock Episodes):")
    print("-" * 80)
    for c in COMMODITIES:
        print(f"\nCommodity: {c}")
        sub = summary_df[summary_df['Commodity'] == c]
        print(sub.to_string(index=False))

    print("\n" + "=" * 80)
    print("Figures generated under outputs/figures/:")
    print("  1. gpr_shocks_timeline.png")
    print("  2. gpr_delta_distribution.png")
    print("  3. commodity_shock_responses.png")
    print("\nStage 3 complete. Ready for Stage 4: Threats vs Acts.")
    print("=" * 80)

if __name__ == "__main__":
    main()
