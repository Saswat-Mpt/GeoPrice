import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.inference.pipeline import get_current_risk_context, predict_next_month
from geoprice.analysis.shock_responses import COMMODITIES

st.set_page_config(
    page_title="GeoPrice — Market Overview",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GeoPrice — Market Overview")
st.markdown("##### Historical Evidence + Geopolitical Context + Machine Learning Forecast")

# 1. Data Freshness Banner
st.info(
    "🗓️ **Data Vintage & Horizon Information**\n\n"
    "- **Geopolitical Risk Indices (GPR/GPRT/GPRA):** Monthly data through `2026-07`\n"
    "- **Commodity Prices & Macro Control (DXY):** Monthly data through `2026-07`\n"
    "- **Forecast Target Horizon:** Next-month return `2026-08` ($y_t = P_{t+1}/P_t - 1$)\n\n"
    "*Note: GeoPrice uses official monthly public data series. Figures reflect latest available monthly observations.*"
)

# 2. Current GPR Snapshot
try:
    ctx_brent = get_current_risk_context("Brent")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current GPR Level", f"{ctx_brent['current_GPR']:.1f}", delta=f"{ctx_brent['current_GPR_percentile']:.0f}th Percentile")
    with c2:
        st.metric("GPRT (Threats)", f"{ctx_brent['current_GPRT']:.1f}", delta=f"{ctx_brent['current_GPRT_percentile']:.0f}th Percentile")
    with c3:
        st.metric("GPRA (Acts)", f"{ctx_brent['current_GPRA']:.1f}", delta=f"{ctx_brent['current_GPRA_percentile']:.0f}th Percentile")
    with c4:
        st.metric("Current Risk Regime", ctx_brent['current_GPR_regime'], delta="Empirical Bucket")
except Exception as e:
    st.error(f"Error loading GPR state: {str(e)}")

st.markdown("---")

# 3. Five-Commodity Summary Table
st.subheader("1. Five-Commodity Current Summary Table")
summary_rows = []

for c in COMMODITIES:
    try:
        ctx = get_current_risk_context(c)
        fcast = predict_next_month(c)
        
        summary_rows.append({
            "Commodity": c,
            "Latest Price": f"${ctx['current_price']:.2f}" if pd.notna(ctx['current_price']) else "N/A",
            "GPR Regime": ctx['current_GPR_regime'],
            "Historical +1M Median": f"{ctx['analogue_1m_median_pct']:+.2f}%",
            "GeoPrice ML +1M Forecast": f"{fcast['predicted_return_pct']:+.2f}%",
            "Direction": f"{'▲ UP' if fcast['predicted_direction'] == 'UP' else '▼ DOWN'}"
        })
    except Exception as e:
        pass

if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.markdown("---")

# 4. Headline Model Comparison (Phase 3 Out-of-Sample Performance)
st.subheader("2. Out-of-Sample Model Performance (Baseline vs GeoPrice)")
comp_path = "data/processed/final_model_comparison.csv"
if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path)
    comp_df['MAE (%)'] = (comp_df['MAE'] * 100).round(2)
    comp_df['RMSE (%)'] = (comp_df['RMSE'] * 100).round(2)
    comp_df['Directional Accuracy (%)'] = (comp_df['Directional_Accuracy'] * 100).round(1)
    
    st.dataframe(
        comp_df[['Commodity', 'Model', 'N', 'MAE (%)', 'RMSE (%)', 'Directional Accuracy (%)']],
        use_container_width=True
    )
