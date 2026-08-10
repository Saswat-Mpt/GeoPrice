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

# 1. Systematic Shock Analysis
st.subheader("1. Systematic GPR Shock Analysis (Phase 2 Stage 3)")
st.write("Top-decile GPR increases ($\\Delta GPR_t = GPR_t - GPR_{t-1} \\ge 37.49$), collapsed into 21 non-overlapping episodes (1985–2026).")

shock_sum_path = "data/processed/shock_summary.csv"
if os.path.exists(shock_sum_path):
    shock_sum = pd.read_csv(shock_sum_path)
    shock_sum['Mean (%)'] = (shock_sum['Mean'] * 100).round(2)
    shock_sum['Median (%)'] = (shock_sum['Median'] * 100).round(2)
    shock_sum['Min (%)'] = (shock_sum['Min'] * 100).round(2)
    shock_sum['Max (%)'] = (shock_sum['Max'] * 100).round(2)
    
    st.dataframe(shock_sum[['Commodity', 'Horizon', 'N', 'Mean (%)', 'Median (%)', 'Min (%)', 'Max (%)']], use_container_width=True)

st.markdown("---")

# 2. Threats vs Acts Comparison
st.subheader("2. Geopolitical Threats (GPRT) vs Realized Acts (GPRA) Analysis (Stage 4)")
st.write("Comparing commodity responses following Threat Shocks ($\\Delta GPRT \\ge 46.42$) versus Realized Act Shocks ($\\Delta GPRA \\ge 37.20$).")

ta_sum_path = "data/processed/threats_acts_summary.csv"
if os.path.exists(ta_sum_path):
    ta_sum = pd.read_csv(ta_sum_path)
    ta_sum['Mean (%)'] = (ta_sum['Mean'] * 100).round(2)
    ta_sum['Median (%)'] = (ta_sum['Median'] * 100).round(2)
    
    st.dataframe(ta_sum[['Subindex', 'Commodity', 'Horizon', 'N', 'Mean (%)', 'Median (%)']], use_container_width=True)

st.markdown("---")

# 3. Current Regime Analogue & All-Regime Lookup
st.subheader("3. Current GPR Regime & Historical Analogue (Stage 5)")
reg_lookup_path = "data/processed/regime_scenario_lookup.csv"
if os.path.exists(reg_lookup_path):
    reg_df = pd.read_csv(reg_lookup_path)
    reg_df['Mean (%)'] = (reg_df['Mean'] * 100).round(2)
    reg_df['Median (%)'] = (reg_df['Median'] * 100).round(2)
    
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
