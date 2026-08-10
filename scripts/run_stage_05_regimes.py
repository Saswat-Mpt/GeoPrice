import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.regimes import (
    calculate_gpr_regime_thresholds,
    assign_gpr_regimes,
    get_current_gpr_state,
    build_regime_episodes,
    calculate_analogue_responses,
    build_regime_scenario_lookup,
    REGIMES
)
from geoprice.analysis.shock_responses import COMMODITIES

def generate_regime_figures(df_regimes: pd.DataFrame, current_state: dict, analogue_sum: pd.DataFrame, thresholds: dict, output_dir: str = "outputs/figures"):
    """Generates visualization charts for GPR Regimes and Historical Analogue."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: GPR history with horizontal P50, P75, and P90 thresholds
    plt.figure(figsize=(12, 5))
    dates = pd.to_datetime(df_regimes['Date'])
    plt.plot(dates, df_regimes['GPR'], label='GPR Index', color='#1f77b4', linewidth=1.5)

    plt.axhline(thresholds['P50'], color='#2ca02c', linestyle='--', linewidth=1.5, label=f"P50 Threshold ({thresholds['P50']:.1f}) [LOW/MOD]")
    plt.axhline(thresholds['P75'], color='#ff7f0e', linestyle='--', linewidth=1.5, label=f"P75 Threshold ({thresholds['P75']:.1f}) [MOD/HIGH]")
    plt.axhline(thresholds['P90'], color='#d62728', linestyle='--', linewidth=1.5, label=f"P90 Threshold ({thresholds['P90']:.1f}) [HIGH/EXTREME]")

    plt.title("GPR Level Time Series & Empirical Regime Cutoff Boundaries (1985-2026)", fontsize=12, fontweight='bold')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("GPR Level", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpr_regimes_timeline.png"), dpi=300)
    plt.close()

    # Figure 2: Current GPR Position relative to historical GPR distribution
    gpr_vals = df_regimes['GPR'].dropna().values
    plt.figure(figsize=(8, 5))
    plt.hist(gpr_vals, bins=35, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.axvline(current_state['current_GPR'], color='#d62728', linewidth=2.5, label=f"Current GPR ({current_state['current_GPR']:.1f} | {current_state['current_GPR_percentile']:.0f}th Pct -> {current_state['current_GPR_regime']})")

    plt.title("Current GPR Position Relative to Historical Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("GPR Level", fontsize=10)
    plt.ylabel("Historical Frequency", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpr_current_position.png"), dpi=300)
    plt.close()

    # Figure 3: Current-regime median commodity response at +1M and +3M
    plt.figure(figsize=(10, 5))
    x = np.arange(len(COMMODITIES))
    width = 0.35

    m1 = [analogue_sum[(analogue_sum['Commodity'] == c) & (analogue_sum['Horizon'] == '+1M')]['Median'].values[0] * 100 for c in COMMODITIES]
    m3 = [analogue_sum[(analogue_sum['Commodity'] == c) & (analogue_sum['Horizon'] == '+3M')]['Median'].values[0] * 100 for c in COMMODITIES]

    plt.bar(x - width/2, m1, width, label='+1M Median Return (%)', color='#2ca02c')
    plt.bar(x + width/2, m3, width, label='+3M Median Return (%)', color='#1f77b4')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"Historical Commodity Responses During '{current_state['current_GPR_regime']}' GPR Regime", fontsize=12, fontweight='bold')
    plt.xlabel("Commodity", fontsize=10)
    plt.ylabel("Median Cumulative Return (%)", fontsize=10)
    plt.xticks(x, COMMODITIES)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "current_regime_analogue_responses.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 5 GPR Regime & Historical Analogue Analysis")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 2, Stage 5: GPR Regime & Historical Analogue")
    print("=" * 80)

    aligned_path = "data/processed/monthly_aligned.csv"
    if not os.path.exists(aligned_path):
        print(f"Error: '{aligned_path}' not found!")
        sys.exit(1)

    df_aligned = pd.read_csv(aligned_path)
    valid_df = df_aligned.dropna(subset=['GPR', 'Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']).copy()
    
    print(f"\n[Step 1/6] Loaded dataset: {len(valid_df)} months ({valid_df['Date'].iloc[0]} to {valid_df['Date'].iloc[-1]})")

    # 1. Calculate Regime Cutoff Boundaries
    print("\n[Step 2/6] Calculating GPR empirical regime thresholds (P50, P75, P90)...")
    thresholds, thresh_meta = calculate_gpr_regime_thresholds(valid_df)
    df_regimes = assign_gpr_regimes(valid_df, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    
    print(f"-> P50 Cutoff (LOW/MODERATE):    {thresholds['P50']:.2f}")
    print(f"-> P75 Cutoff (MODERATE/HIGH):  {thresholds['P75']:.2f}")
    print(f"-> P90 Cutoff (HIGH/EXTREME):   {thresholds['P90']:.2f}")

    # 2. Current GPR State
    print("\n[Step 3/6] Identifying current GPR risk regime...")
    current_state = get_current_gpr_state(df_regimes, thresholds['P50'], thresholds['P75'], thresholds['P90'])
    print(f"-> Latest Date:            {current_state['current_date']}")
    print(f"-> Current GPR Level:      {current_state['current_GPR']:.2f}")
    print(f"-> Current GPR Percentile: {current_state['current_GPR_percentile']:.1f}th percentile")
    print(f"-> Current Risk Regime:    {current_state['current_GPR_regime']}")

    # 3. Build Representative Regime Episodes
    print("\n[Step 4/6] Grouping regime months into representative regime episodes...")
    episodes_df = build_regime_episodes(df_regimes)
    print(f"-> Total regime episodes built: {len(episodes_df)}")

    # 4. Historical Analogue Responses for Current Regime
    print(f"\n[Step 5/6] Evaluating historical analogue responses for '{current_state['current_GPR_regime']}' regime...")
    analogue_df, analogue_sum = calculate_analogue_responses(episodes_df, valid_df, current_state['current_GPR_regime'])
    scenario_lookup = build_regime_scenario_lookup(episodes_df, valid_df)

    # 5. Save Processed Outputs & Figures
    print("\n[Step 6/6] Saving output CSVs/JSONs and generating visualization figures...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/gpr_regime_thresholds.json", "w") as f:
        json.dump(thresh_meta, f, indent=4)
    with open("data/processed/current_gpr_regime.json", "w") as f:
        json.dump(current_state, f, indent=4)

    df_regimes[['Date', 'GPR', 'GPR_percentile', 'GPR_regime']].to_csv("data/processed/gpr_regime_months.csv", index=False)
    episodes_df.to_csv("data/processed/gpr_regime_episodes.csv", index=False)
    analogue_df.to_csv("data/processed/current_regime_analogue.csv", index=False)
    analogue_sum.to_csv("data/processed/regime_summary.csv", index=False)
    scenario_lookup.to_csv("data/processed/regime_scenario_lookup.csv", index=False)

    generate_regime_figures(df_regimes, current_state, analogue_sum, thresholds)

    # Final Stage 5 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 5 FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"GPR Empirical Boundaries: P50={thresholds['P50']:.1f} | P75={thresholds['P75']:.1f} | P90={thresholds['P90']:.1f}")
    print(f"Current GPR Situation:   GPR={current_state['current_GPR']:.2f} ({current_state['current_GPR_percentile']:.0f}th pct) -> REGIME: {current_state['current_GPR_regime']}")
    print(f"Contextual Subindices:   GPRT={current_state['current_GPRT']:.2f} ({current_state['current_GPRT_percentile']:.0f}th pct) | GPRA={current_state['current_GPRA']:.2f} ({current_state['current_GPRA_percentile']:.0f}th pct)")
    print(f"Representative Episodes: {len(analogue_df)} episodes in {current_state['current_GPR_regime']} regime (Coverage: {analogue_df['representative_date'].iloc[0]} to {analogue_df['representative_date'].iloc[-1]})")

    print(f"\nHistorical Commodity Responses in '{current_state['current_GPR_regime']}' Regime:")
    print("-" * 80)
    for c in COMMODITIES:
        sub = analogue_sum[analogue_sum['Commodity'] == c]
        print(sub.to_string(index=False))

    print("\n" + "=" * 80)
    print("Stage 5 complete. Ready for Stage 6: Major Conflict Reference Cases.")
    print("=" * 80)

if __name__ == "__main__":
    main()
