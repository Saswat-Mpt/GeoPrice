import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from geoprice.inference.pipeline import (
    predict_next_month,
    get_current_risk_context
)
from geoprice.scenarios.lookup import (
    get_historical_scenario,
    VALID_CONFLICT_REFERENCES
)
from geoprice.analysis.shock_responses import COMMODITIES
from geoprice.analysis.regimes import REGIMES

st.set_page_config(
    page_title="GeoPrice — Geopolitical Risk-Aware Commodity Outlook",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .disclaimer { font-size: 0.8rem; color: #94A3B8; margin-top: 2rem; border-top: 1px solid #E2E8F0; padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌍 GeoPrice")
st.sidebar.markdown("**Geopolitical Risk-Aware Commodity Outlook**")
st.sidebar.markdown("---")

selected_commodity = st.sidebar.selectbox(
    "Select Commodity:",
    options=COMMODITIES,
    index=0,
    help="Select one of 5 primary commodities to inspect forecasts and historical risk analogues."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Information:**")
st.sidebar.markdown("- **Pipeline**: `StandardScaler -> ElasticNet`")
st.sidebar.markdown("- **Validation**: Expanding-Window OOS CV")
st.sidebar.markdown("- **Data Timeline**: 1960–2026 (Phase 3: 2006–2026)")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='disclaimer'><b>Disclaimer:</b> GeoPrice provides model-based forecasts and historical context for analytical purposes only. It is not investment advice and does not predict geopolitical events.</div>",
    unsafe_allow_html=True
)

# Main Application Body
st.markdown("<div class='main-header'>GeoPrice Commodity Outlook</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Evaluating commodity returns through classical ML and empirical geopolitical risk regimes</div>", unsafe_allow_html=True)

# Attempt to load inference & context
try:
    context = get_current_risk_context(selected_commodity)
    forecast = predict_next_month(selected_commodity)
    has_models = True
except FileNotFoundError as e:
    st.error(f"⚠️ {str(e)}")
    has_models = False

if has_models:
    # 1. Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_str = f"${context['current_price']:.2f}" if pd.notna(context['current_price']) else "N/A"
        st.metric("Latest Price", price_str, delta=f"Date: {context['latest_date']}")

    with col2:
        st.metric("GPR Index Level", f"{context['current_GPR']:.1f}", delta=f"{context['current_GPR_percentile']:.0f}th Percentile")

    with col3:
        regime = context['current_GPR_regime']
        st.metric("Current GPR Regime", regime, delta=f"Subindices: GPRT {context['current_GPRT_percentile']:.0f}th | GPRA {context['current_GPRA_percentile']:.0f}th")

    with col4:
        ret_pct = forecast['predicted_return_pct']
        direction = forecast['predicted_direction']
        dir_symbol = "▲" if direction == "UP" else "▼"
        st.metric("GeoPrice ML Forecast", f"{ret_pct:+.2f}%", delta=f"{dir_symbol} {direction} (Target: {forecast['target_month']})")

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Forecast & Risk Overview",
        "📜 Historical Risk Analysis",
        "🔮 Historical Scenario Explorer",
        "📊 Model Diagnostics & Explainability"
    ])

    # TAB 1: FORECAST & RISK OVERVIEW
    with tab1:
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("1. GeoPrice ML Forecast (Next-Month Return)")
            st.info(
                f"**Model-Implied Return ({forecast['target_month']}):** `{forecast['predicted_return_pct']:+.2f}%`\n\n"
                f"**Forecasted Direction:** `{forecast['predicted_direction']}`\n\n"
                f"*Generated via expanding-window ElasticNet model fitted on past commodity history, GPR/GPRT/GPRA indices, and DXY.*"
            )
            
            st.markdown("##### Forecast Origin & Horizon")
            st.write(f"- **Forecast Origin Date ($t$):** `{forecast['forecast_origin_date']}`")
            st.write(f"- **Target Return Horizon ($t+1$):** `{forecast['target_month']}`")
            st.write(f"- **Formula:** $y_t = \\frac{{P_{{t+1}}}}{{P_t}} - 1$")

        with c_right:
            st.subheader("2. Historical Regime Reference (Empirical Analogue)")
            reg = context['current_GPR_regime']
            m1_ret = context['analogue_1m_median_pct']
            m3_ret = context['analogue_3m_median_pct']
            n_eps = context['analogue_episodes_count']
            
            st.success(
                f"**Current Risk Regime:** `{reg}`\n\n"
                f"**Historical +1M Median Return:** `{m1_ret:+.2f}%` (across {n_eps} representative episodes)\n\n"
                f"**Historical +3M Median Return:** `{m3_ret:+.2f}%`\n\n"
                f"*Historical analogue statistics reflect empirical past responses under similar GPR regimes and are separate from ML forecasts.*"
            )

            # Shock Status Badge
            st.markdown("##### Geopolitical Shock Status")
            if context['is_gpr_shock']:
                st.error(f"⚡ **GPR SHOCK DETECTED**: Monthly $\\Delta GPR$ = `+{context['latest_delta_gpr']:.1f}` (Exceeds 90th pct threshold `{context['shock_threshold']:.1f}`)")
            else:
                st.write(f"✓ **NORMAL DRIFT**: Monthly $\\Delta GPR$ = `{context['latest_delta_gpr']:+.1f}` (Below shock threshold `{context['shock_threshold']:.1f}`)")

    # TAB 2: HISTORICAL RISK ANALYSIS
    with tab2:
        st.subheader("Historical GPR Index & Commodity Price History (1985–2026)")
        
        aligned_path = "data/processed/monthly_aligned.csv"
        if os.path.exists(aligned_path):
            df_aligned = pd.read_csv(aligned_path)
            dates = pd.to_datetime(df_aligned['Date'])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            
            # Commodity Price Chart
            ax1.plot(dates, df_aligned[selected_commodity], color='#1f77b4', linewidth=1.5, label=f"{selected_commodity} Price")
            ax1.set_title(f"Historical {selected_commodity} Monthly Price", fontsize=11, fontweight='bold')
            ax1.set_ylabel("USD / Unit", fontsize=9)
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax1.legend(loc='upper left')
            
            # GPR Level Chart
            ax2.plot(dates, df_aligned['GPR'], color='#d62728', linewidth=1.5, label="GPR Index")
            ax2.set_title("Caldara-Iacoviello Geopolitical Risk (GPR) Index", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Date", fontsize=10)
            ax2.set_ylabel("GPR Level", fontsize=9)
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.legend(loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("Documented Major Conflict Reference Cases")
        conflict_path = "data/processed/conflict_reference_cases.csv"
        if os.path.exists(conflict_path):
            conf_df = pd.read_csv(conflict_path)
            st.dataframe(
                conf_df[['conflict_name', 'representative_shock_date', 'GPR', 'GPR_change', f'{selected_commodity}_1m', f'{selected_commodity}_3m', 'source']],
                use_container_width=True
            )

    # TAB 3: HISTORICAL SCENARIO EXPLORER (STAGE 11)
    with tab3:
        st.subheader("🔮 Historical Scenario Explorer (Manual Mode)")
        st.warning("⚠️ **HISTORICAL REFERENCE ONLY — NOT A MACHINE LEARNING FORECAST**\n\nThis explorer evaluates empirical historical commodity responses under user-selected GPR regimes and major conflict reference cases. It does NOT call the ElasticNet ML model or construct synthetic GPR values.")
        
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            scenario_regime = st.selectbox(
                "Select Geopolitical Risk Regime:",
                options=REGIMES,
                index=2, # HIGH by default
                help="Select empirical GPR level regime (LOW < P50, MODERATE P50-P75, HIGH P75-P90, EXTREME > P90)."
            )
        with sc_col2:
            scenario_conflict = st.selectbox(
                "Select Conflict Reference:",
                options=VALID_CONFLICT_REFERENCES,
                index=0,
                help="Select whether to overlay historical responses from documented major conflict reference cases."
            )
            
        scenario_data = get_historical_scenario(selected_commodity, scenario_regime, scenario_conflict)
        reg_s = scenario_data['regime_stats']
        
        st.markdown("---")
        st.markdown(f"#### Historical Commodity Response: `{selected_commodity}` in `{scenario_regime}` Regime")
        
        sc_m1, sc_m3 = st.columns(2)
        with sc_m1:
            st.markdown("##### **+1M Horizon Response**")
            st.write(f"- **Median Return:** `{reg_s['1m_median_pct']:+.2f}%`")
            st.write(f"- **Mean Return:** `{reg_s['1m_mean_pct']:+.2f}%`")
            st.write(f"- **Range:** `{reg_s['1m_min_pct']:+.2f}%` to `{reg_s['1m_max_pct']:+.2f}%`")
            st.write(f"- **Sample Size:** `N = {reg_s['1m_n']} episodes`")
            
        with sc_m3:
            st.markdown("##### **+3M Horizon Response**")
            st.write(f"- **Median Return:** `{reg_s['3m_median_pct']:+.2f}%`")
            st.write(f"- **Mean Return:** `{reg_s['3m_mean_pct']:+.2f}%`")
            st.write(f"- **Range:** `{reg_s['3m_min_pct']:+.2f}%` to `{reg_s['3m_max_pct']:+.2f}%`")
            st.write(f"- **Sample Size:** `N = {reg_s['3m_n']} episodes`")

        if scenario_conflict == "Major-conflict reference":
            st.markdown("---")
            st.markdown(f"#### Major-Conflict Historical Reference Case Response")
            conf_s = scenario_data['conflict_stats']
            
            c_m1, c_m3 = st.columns(2)
            with c_m1:
                st.write(f"- **Major Conflict +1M Median:** `{conf_s['conflict_1m_median_pct']:+.2f}%` (`N = {conf_s['conflict_1m_n']}`)")
            with c_m3:
                st.write(f"- **Major Conflict +3M Median:** `{conf_s['conflict_3m_median_pct']:+.2f}%` (`N = {conf_s['conflict_3m_n']}`)")

        st.info(f"💡 **Interpretation:** {scenario_data['interpretation']}")

    # TAB 4: MODEL DIAGNOSTICS & EXPLAINABILITY
    with tab4:
        st.subheader("1. Standardized ElasticNet Feature Weights")
        weights_df = forecast['feature_weights']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#1f77b4' if g == 'Commodity History' else ('#d62728' if g == 'Geopolitical Risk' else '#2ca02c') for g in weights_df['group']]
        ax.barh(weights_df['feature'], weights_df['coefficient'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"GeoPrice Feature Coefficients for {selected_commodity}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Standardized Coefficient Weight", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("2. Out-of-Sample Model Comparison")
            comp_path = "data/processed/final_model_comparison.csv"
            if os.path.exists(comp_path):
                comp_df = pd.read_csv(comp_path)
                sub_comp = comp_df[comp_df['Commodity'] == selected_commodity].copy()
                sub_comp['MAE (%)'] = (sub_comp['MAE'] * 100).round(2)
                sub_comp['RMSE (%)'] = (sub_comp['RMSE'] * 100).round(2)
                sub_comp['DA (%)'] = (sub_comp['Directional_Accuracy'] * 100).round(1)
                st.table(sub_comp[['Model', 'N', 'MAE (%)', 'RMSE (%)', 'DA (%)']])

        with c2:
            st.subheader("3. Feature Ablation Study")
            abl_path = "data/processed/geoprice_ablation.csv"
            if os.path.exists(abl_path):
                abl_df = pd.read_csv(abl_path)
                sub_abl = abl_df[abl_df['Commodity'] == selected_commodity].copy()
                sub_abl['MAE (%)'] = (sub_abl['MAE'] * 100).round(2)
                sub_abl['RMSE (%)'] = (sub_abl['RMSE'] * 100).round(2)
                sub_abl['DA (%)'] = (sub_abl['Directional_Accuracy'] * 100).round(1)
                st.table(sub_abl[['Feature_Set', 'N', 'MAE (%)', 'RMSE (%)', 'DA (%)']])
