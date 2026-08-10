import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.threats_acts import (
    calculate_subindex_thresholds,
    identify_subindex_shocks,
    collapse_subindex_shocks,
    calculate_subindex_responses
)
from geoprice.analysis.shock_responses import COMMODITIES

def generate_threats_acts_figures(df_full: pd.DataFrame, threat_shocks: pd.DataFrame, act_shocks: pd.DataFrame, threat_sum: pd.DataFrame, act_sum: pd.DataFrame, output_dir: str = "outputs/figures"):
    """Generates visualization charts for Threats vs Acts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: GPRT and GPRA time series with shock points
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(df_full['Date']), df_full['GPRT'], label='GPRT (Threats)', color='#1f77b4', linewidth=1.5, alpha=0.8)
    plt.plot(pd.to_datetime(df_full['Date']), df_full['GPRA'], label='GPRA (Acts)', color='#ff7f0e', linewidth=1.5, alpha=0.8)
    
    plt.scatter(pd.to_datetime(threat_shocks['Date']), threat_shocks['GPRT'], color='#1f77b4', marker='^', s=40, label='Threat Shocks')
    plt.scatter(pd.to_datetime(act_shocks['Date']), act_shocks['GPRA'], color='#ff7f0e', marker='s', s=40, label='Act Shocks')
    
    valid_start = df_full['Date'].iloc[0]
    valid_end = df_full['Date'].iloc[-1]
    plt.title(f"Geopolitical Threats (GPRT) vs Geopolitical Acts (GPRA) Shocks ({valid_start} to {valid_end})", fontsize=12, fontweight='bold')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Subindex Level", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gprt_gpra_timeline.png"), dpi=300)
    plt.close()

    # Figure 2: Threat vs Act median commodity response
    plt.figure(figsize=(11, 5))
    x = np.arange(len(COMMODITIES))
    width = 0.35

    threat_3m_medians = [threat_sum[(threat_sum['Commodity'] == c) & (threat_sum['Horizon'] == '+3M')]['Median'].values[0] * 100 for c in COMMODITIES]
    act_3m_medians = [act_sum[(act_sum['Commodity'] == c) & (act_sum['Horizon'] == '+3M')]['Median'].values[0] * 100 for c in COMMODITIES]

    plt.bar(x - width/2, threat_3m_medians, width, label='Threat Shocks (+3M Median %)', color='#1f77b4')
    plt.bar(x + width/2, act_3m_medians, width, label='Act Shocks (+3M Median %)', color='#ff7f0e')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Commodity +3M Responses: Geopolitical Threats (GPRT) vs Realized Acts (GPRA)", fontsize=12, fontweight='bold')
    plt.xlabel("Commodity", fontsize=10)
    plt.ylabel("Median +3M Cumulative Return (%)", fontsize=10)
    plt.xticks(x, COMMODITIES)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threats_vs_acts_responses.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 4 Threats vs Acts Analysis")
    args = parser.parse_args()

    print("=" * 80)
    print("Running GeoPrice — Phase 2, Stage 4: Threats vs Acts Analysis")
    print("=" * 80)

    aligned_path = "data/processed/monthly_aligned.csv"
    if not os.path.exists(aligned_path):
        print(f"Error: '{aligned_path}' not found!")
        sys.exit(1)

    df_aligned = pd.read_csv(aligned_path)
    valid_df = df_aligned.dropna(subset=['GPRT', 'GPRA', 'Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']).copy()
    valid_start = valid_df['Date'].iloc[0]
    valid_end = valid_df['Date'].iloc[-1]
    
    print(f"\n[Step 1/5] Loaded data: {len(valid_df)} months ({valid_start} to {valid_end})")

    # 1. Threat Shocks (GPRT)
    print("\n[Step 2/5] Analyzing Geopolitical Threats (GPRT) shocks...")
    t_thresh, t_meta = calculate_subindex_thresholds(valid_df, 'GPRT', percentile=90.0)
    raw_threats = identify_subindex_shocks(valid_df, 'GPRT', t_thresh)
    threat_episodes = collapse_subindex_shocks(raw_threats, 'GPRT', window_months=3)
    threat_resp, threat_sum = calculate_subindex_responses(threat_episodes, valid_df, 'GPRT')
    
    print(f"-> GPRT 90th Pct Cutoff:  {t_thresh:.4f}")
    print(f"-> Raw Threat Shocks:     {len(raw_threats)}")
    print(f"-> Threat Episodes:       {len(threat_episodes)}")

    # 2. Act Shocks (GPRA)
    print("\n[Step 3/5] Analyzing Geopolitical Acts (GPRA) shocks...")
    a_thresh, a_meta = calculate_subindex_thresholds(valid_df, 'GPRA', percentile=90.0)
    raw_acts = identify_subindex_shocks(valid_df, 'GPRA', a_thresh)
    act_episodes = collapse_subindex_shocks(raw_acts, 'GPRA', window_months=3)
    act_resp, act_sum = calculate_subindex_responses(act_episodes, valid_df, 'GPRA')

    print(f"-> GPRA 90th Pct Cutoff:  {a_thresh:.4f}")
    print(f"-> Raw Act Shocks:        {len(raw_acts)}")
    print(f"-> Act Episodes:          {len(act_episodes)}")

    # 3. Combine Outputs
    print("\n[Step 4/5] Combining responses and calculating descriptive differences...")
    all_responses = pd.concat([threat_resp, act_resp], ignore_index=True)
    all_summary = pd.concat([threat_sum, act_sum], ignore_index=True)

    # 4. Save Outputs
    print("\n[Step 5/5] Saving Stage 4 output files and figures...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/gprt_shock_threshold.json", "w") as f:
        json.dump(t_meta, f, indent=4)
    with open("data/processed/gpra_shock_threshold.json", "w") as f:
        json.dump(a_meta, f, indent=4)

    raw_threats.to_csv("data/processed/raw_gprt_shocks.csv", index=False)
    raw_acts.to_csv("data/processed/raw_gpra_shocks.csv", index=False)
    threat_episodes.to_csv("data/processed/gprt_shock_episodes.csv", index=False)
    act_episodes.to_csv("data/processed/gpra_shock_episodes.csv", index=False)
    all_responses.to_csv("data/processed/threats_acts_responses.csv", index=False)
    all_summary.to_csv("data/processed/threats_acts_summary.csv", index=False)

    generate_threats_acts_figures(valid_df, raw_threats, raw_acts, threat_sum, act_sum)

    # Final Stage 4 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 4 FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"GPRT Threat Threshold: {t_thresh:.4f} ({len(raw_threats)} raw shocks -> {len(threat_episodes)} episodes)")
    print(f"GPRA Act Threshold:    {a_thresh:.4f} ({len(raw_acts)} raw shocks -> {len(act_episodes)} episodes)")
    
    print("\nDescriptive Summary (+3M Median Return Comparison):")
    print("-" * 80)
    for c in COMMODITIES:
        t_med = threat_sum[(threat_sum['Commodity'] == c) & (threat_sum['Horizon'] == '+3M')]['Median'].values[0]
        a_med = act_sum[(act_sum['Commodity'] == c) & (act_sum['Horizon'] == '+3M')]['Median'].values[0]
        diff = t_med - a_med
        print(f"  {c:15s} | Threat +3M: {t_med*100:6.2f}% | Act +3M: {a_med*100:6.2f}% | Diff (Threat-Act): {diff*100:6.2f}%")

    print("\n" + "=" * 80)
    print("Stage 4 complete. Ready for Stage 5: Current GPR Regime & Historical Analogue.")
    print("=" * 80)

if __name__ == "__main__":
    main()
