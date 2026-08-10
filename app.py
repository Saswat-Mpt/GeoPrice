import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from geoprice.inference.pipeline import get_current_risk_context, predict_next_month
from geoprice.analysis.shock_responses import COMMODITIES

st.set_page_config(
    page_title="GeoPrice — Market Overview",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .disclaimer { font-size: 0.8rem; color: #94A3B8; margin-top: 2rem; border-top: 1px solid #E2E8F0; padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# Dynamic date derivation from dataset
aligned_path = "data/processed/monthly_aligned.csv"
if os.path.exists(aligned_path):
    df_aligned = pd.read_csv(aligned_path)
    latest_gpr_date = df_aligned.dropna(subset=['GPR'])['Date'].iloc[-1]
    latest_dxy_date = df_aligned.dropna(subset=['DXY'])['Date'].iloc[-1]
    
    # Common commodity cutoff vs Gold extra month
    comm_common_latest = df_aligned.dropna(subset=['Brent', 'Natural_Gas', 'Copper', 'Wheat'])['Date'].iloc[-1]
    gold_latest = df_aligned.dropna(subset=['Gold'])['Date'].iloc[-1]
    
    feat_path_check = 'data/processed/feature_dataset.csv'
    if os.path.exists(feat_path_check):
        df_feat = pd.read_csv(feat_path_check)
        forecast_origin = df_feat.dropna(subset=['GPR','DXY','Brent_return_1m'])['Date'].iloc[-1]
        target_period = str((pd.to_datetime(forecast_origin) + pd.DateOffset(months=1)).to_period('M'))
    else:
        forecast_origin = latest_gpr_date
        target_period = 'Next Month'
else:
    latest_gpr_date = "Latest Available Month"
    latest_dxy_date = "Latest Available Month"
    comm_common_latest = "Latest Available Month"
    gold_latest = "Latest Available Month"
    forecast_origin = "Latest Available Month"
    target_period = "Next Month"

# Sidebar
st.sidebar.title("🌍 GeoPrice")
st.sidebar.markdown("**Geopolitical Risk-Aware Commodity Outlook**")
st.sidebar.markdown("---")

selected_commodity = st.sidebar.selectbox(
    "Select Commodity:",
    options=COMMODITIES,
    index=0,
    help="Select one of 5 primary commodities to inspect current outlook."
)

# Commodity-specific forecast origin and target
try:
    c_fcast_info = predict_next_month(selected_commodity)
    c_origin = c_fcast_info['forecast_origin_date']
    c_target = c_fcast_info['target_month']
except Exception:
    c_origin = forecast_origin
    c_target = target_period

st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture:**")
st.sidebar.markdown("- **Pipeline**: `StandardScaler -> ElasticNet`")
st.sidebar.markdown("- **Validation**: Expanding-Window OOS CV")
st.sidebar.markdown(f"- **Data Vintage**: Through `{c_origin}` ({selected_commodity})")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='disclaimer'><b>Disclaimer:</b> GeoPrice provides model-based forecasts and historical context for analytical purposes only. It is not investment advice and does not predict geopolitical events.</div>",
    unsafe_allow_html=True
)

# Page Header
st.markdown("<div class='main-header'>GeoPrice — Market Overview</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Historical evidence + geopolitical context + machine-learning forecast</div>", unsafe_allow_html=True)

# 1. Data Freshness Banner (Dynamic Dates & Commodity Breakdown)
st.info(
    f"**Data Vintage & Horizon Information**\n\n"
    f"- **GPR/GPRT/GPRA:** Monthly data through `{latest_gpr_date}` *(Common commodity regime analysis cutoff: `{comm_common_latest}`)*\n"
    f"- **Commodity Prices:** Brent, Natural Gas, Copper, Wheat through `{comm_common_latest}` | Gold through `{gold_latest}`\n"
    f"- **Macro Control (DXY):** Monthly data through `{latest_dxy_date}`\n"
    f"- **Selected Commodity ({selected_commodity}) Forecast Origin:** `{c_origin}` | **Target Month:** `{c_target}`\n\n"
    f"*Release-aware availability rule applied; full historical vintage reconstruction not performed.*"
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
        st.metric("Current Risk Regime", ctx_brent['current_GPR_regime'], delta="Empirical Percentile Bucket")
except Exception as e:
    st.error(f"Error loading GPR state: {str(e)}")

st.markdown("---")

# 3. Five-Commodity Current Summary Table
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

# 4. Out-of-Sample Headline Results Table (Baseline vs GeoPrice)
st.subheader("2. Out-of-Sample Model Performance (Baseline vs GeoPrice)")
comp_path = "data/processed/final_model_comparison.csv"
if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path).copy()
    comp_df['MAE (%)'] = (comp_df['MAE'] * 100).round(2).astype(str) + "%"
    comp_df['RMSE (%)'] = (comp_df['RMSE'] * 100).round(2).astype(str) + "%"
    
    # Render Naive Directional Accuracy as N/A
    comp_df['Directional Accuracy'] = comp_df.apply(
        lambda r: "N/A" if r['Model'] == "Naive" else f"{round(r['Directional_Accuracy'] * 100, 1)}%",
        axis=1
    )
    
    st.dataframe(
        comp_df[['Commodity', 'Model', 'N', 'MAE (%)', 'RMSE (%)', 'Directional Accuracy']],
        use_container_width=True
    )
