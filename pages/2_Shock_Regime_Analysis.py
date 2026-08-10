import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shock_responses import COMMODITIES

st.set_page_config(
    page_title="GeoPrice — Shock & Regime Analysis",
    page_icon="📜",
    layout="wide"
)

st.title("📜 GeoPrice — Shock & Regime Analysis")
st.markdown("##### Historical Geopolitical Event Studies, Threat vs Act Differences, and Empirical Risk Regimes")

st.markdown("---")

from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data/processed")

GPR_THRESHOLD_FILE = DATA_DIR / "gpr_shock_threshold.json"
GPRT_THRESHOLD_FILE = DATA_DIR / "gprt_shock_threshold.json"
GPRA_THRESHOLD_FILE = DATA_DIR / "gpra_shock_threshold.json"
SHOCK_EPISODES_FILE = DATA_DIR / "shock_episodes.csv"
ALIGNED_FILE = DATA_DIR / "monthly_aligned.csv"

required_files = [
    GPR_THRESHOLD_FILE,
    GPRT_THRESHOLD_FILE,
    GPRA_THRESHOLD_FILE,
    SHOCK_EPISODES_FILE,
]

missing_files = [str(f) for f in required_files if not f.exists()]

if missing_files:
    st.error(
        "Required Phase 2 outputs are missing. "
        "Run Stages 3–4 before launching the dashboard."
    )
    st.stop()

with open(GPR_THRESHOLD_FILE, "r", encoding="utf-8") as f:
    gpr_info = json.load(f)

with open(GPRT_THRESHOLD_FILE, "r", encoding="utf-8") as f:
    gprt_info = json.load(f)

with open(GPRA_THRESHOLD_FILE, "r", encoding="utf-8") as f:
    gpra_info = json.load(f)

shock_episodes = pd.read_csv(SHOCK_EPISODES_FILE)

gpr_thresh = float(gpr_info["threshold"])
gprt_thresh = float(gprt_info["threshold"])
gpra_thresh = float(gpra_info["threshold"])

episodes_count = len(shock_episodes)

# Derive common analysis window dates from monthly aligned dataset
if ALIGNED_FILE.exists():
    df_aligned = pd.read_csv(ALIGNED_FILE)
    valid_df = df_aligned.dropna(subset=['GPR', 'Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat'])
    start_date = valid_df['Date'].iloc[0]
    end_date = valid_df['Date'].iloc[-1]
else:
    start_date = "1992-01"
    end_date = "2026-06"

# 1. Systematic Shock Analysis
st.subheader("1. Systematic GPR Shock Analysis (Phase 2 Stage 3)")
st.write(f"Top-decile GPR increases ($\\Delta GPR_t = GPR_t - GPR_{{t-1}} \\ge {gpr_thresh:.2f}$), collapsed into {episodes_count} non-overlapping episodes ({start_date} → {end_date}).")

shock_sum_path = "data/processed/shock_summary.csv"
if os.path.exists(shock_sum_path):
    shock_sum = pd.read_csv(shock_sum_path).copy()
    shock_sum['Mean (%)'] = (shock_sum['Mean'] * 100).round(2).astype(str) + "%"
    shock_sum['Median (%)'] = (shock_sum['Median'] * 100).round(2).astype(str) + "%"
    shock_sum['Min (%)'] = (shock_sum['Min'] * 100).round(2).astype(str) + "%"
    shock_sum['Max (%)'] = (shock_sum['Max'] * 100).round(2).astype(str) + "%"
    
    st.dataframe(shock_sum[['Commodity', 'Horizon', 'N', 'Mean (%)', 'Median (%)', 'Min (%)', 'Max (%)']], use_container_width=True)

st.markdown("---")

# 2. Threats vs Acts Comparison
st.subheader("2. Geopolitical Threats (GPRT) vs Realized Acts (GPRA) Analysis (Stage 4)")
st.write(f"Comparing commodity responses following Threat Shocks ($\\Delta GPRT \\ge {gprt_thresh:.2f}$) versus Realized Act Shocks ($\\Delta GPRA \\ge {gpra_thresh:.2f}$).")

ta_sum_path = "data/processed/threats_acts_summary.csv"
if os.path.exists(ta_sum_path):
    ta_sum = pd.read_csv(ta_sum_path).copy()
    ta_sum['Mean (%)'] = (ta_sum['Mean'] * 100).round(2).astype(str) + "%"
    ta_sum['Median (%)'] = (ta_sum['Median'] * 100).round(2).astype(str) + "%"
    
    st.dataframe(ta_sum[['Subindex', 'Commodity', 'Horizon', 'N', 'Mean (%)', 'Median (%)']], use_container_width=True)

st.markdown("---")

# 3. Current Regime Analogue & All-Regime Lookup
st.subheader("3. Current GPR Regime & Historical Analogue (Stage 5)")
reg_lookup_path = "data/processed/regime_scenario_lookup.csv"
if os.path.exists(reg_lookup_path):
    reg_df = pd.read_csv(reg_lookup_path).copy()
    reg_df['Mean (%)'] = (reg_df['Mean'] * 100).round(2).astype(str) + "%"
    reg_df['Median (%)'] = (reg_df['Median'] * 100).round(2).astype(str) + "%"
    
    st.dataframe(reg_df[['Regime', 'Commodity', 'Horizon', 'N', 'Mean (%)', 'Median (%)']], use_container_width=True)

st.markdown("---")

# 4. Major Conflict Case Studies
st.subheader("4. Documented Major Conflict Reference Cases (Stage 6)")
st.write("Systematically mapped conflict reference episodes anchored to Stage 3 shock dates.")

conf_path = "data/processed/conflict_reference_cases.csv"
if os.path.exists(conf_path):
    conf_df = pd.read_csv(conf_path)
    st.dataframe(
        conf_df[['conflict_name', 'representative_shock_date', 'GPR', 'GPR_change', 'Brent_1m', 'Gold_1m', 'Natural_Gas_1m', 'source']],
        use_container_width=True
    )

st.markdown("---")

# 5. Out-of-Sample Performance Baseline vs GeoPrice vs Naive
st.subheader("5. Headline Out-of-Sample Results (Stage 9)")
comp_path = "data/processed/final_model_comparison.csv"
if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path).copy()
    comp_df['MAE (%)'] = (comp_df['MAE'] * 100).round(2).astype(str) + "%"
    comp_df['RMSE (%)'] = (comp_df['RMSE'] * 100).round(2).astype(str) + "%"
    
    comp_df['Directional Accuracy'] = comp_df.apply(
        lambda r: "N/A" if r['Model'] == "Naive" else f"{round(r['Directional_Accuracy'] * 100, 1)}%",
        axis=1
    )
    
    st.dataframe(
        comp_df[['Commodity', 'Model', 'N', 'MAE (%)', 'RMSE (%)', 'Directional Accuracy']],
        use_container_width=True
    )
