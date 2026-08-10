import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.inference.pipeline import predict_next_month, get_current_risk_context
from geoprice.scenarios.lookup import get_historical_scenario, VALID_CONFLICT_REFERENCES
from geoprice.interpretation.contributions import explain_current_forecast
from geoprice.analysis.shock_responses import COMMODITIES
from geoprice.analysis.regimes import REGIMES

st.set_page_config(
    page_title="GeoPrice — Outlook & Scenarios",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 GeoPrice — Commodity Risk Outlook")
st.markdown("##### Next-Month Forecast, Feature Contributions (beta * z), Historical Analogue & Scenario Explorer")

# Controls
c_sel, mode_sel = st.columns([1, 2])
with c_sel:
    selected_commodity = st.selectbox("Select Commodity:", options=COMMODITIES, index=0)
with mode_sel:
    selected_mode = st.radio("Select Mode:", options=["Current Outlook (ML + History)", "Scenario Explorer (Historical Only)"], horizontal=True)

st.markdown("---")

# MODE A: CURRENT OUTLOOK
if "Current Outlook" in selected_mode:
    try:
        context = get_current_risk_context(selected_commodity)
        forecast = predict_next_month(selected_commodity)
        explanation = explain_current_forecast(selected_commodity)
        
        # 1. Headline Forecast & Analogue Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Latest Price", f"${context['current_price']:.2f}" if pd.notna(context['current_price']) else "N/A", delta=f"Date: {context['latest_date']}")
        with k2:
            st.metric("Current GPR Regime", context['current_GPR_regime'], delta=f"GPR Level {context['current_GPR']:.1f}")
        with k3:
            ret_pct = forecast['predicted_return_pct']
            direction = forecast['predicted_direction']
            dir_sym = "▲" if direction == "UP" else "▼"
            st.metric("GeoPrice ML Forecast", f"{ret_pct:+.2f}%", delta=f"{dir_sym} {direction} (Target: {forecast['target_month']})")
        with k4:
            m1 = context['analogue_1m_median_pct']
            st.metric("Historical +1M Median", f"{m1:+.2f}%", delta=f"N={context['analogue_episodes_count']} episodes")

        st.markdown("---")

        # 2. Evidence Agreement & Separation Section
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("1. GeoPrice ML Forecast")
            st.info(
                f"**Target Month ({forecast['target_month']}):** `{forecast['predicted_return_pct']:+.2f}%`\n\n"
                f"**Direction:** `{forecast['predicted_direction']}`\n\n"
                f"*Generated via expanding-window ElasticNet model fitted on past commodity history, GPR/GPRT/GPRA indices, and DXY.*"
            )

        with c_right:
            st.subheader("2. Historical Regime Reference")
            m1_ret = context['analogue_1m_median_pct']
            m3_ret = context['analogue_3m_median_pct']
            
            st.success(
                f"**Regime:** `{context['current_GPR_regime']}`\n\n"
                f"**Historical +1M Median Return:** `{m1_ret:+.2f}%`\n\n"
                f"**Historical +3M Median Return:** `{m3_ret:+.2f}%`\n\n"
                f"*Historical analogue statistics reflect empirical past responses under similar GPR regimes and are separate from ML forecasts.*"
            )

        # Directional Agreement Line
        hist_dir = "UP" if m1_ret > 0 else "DOWN"
        ml_dir = forecast['predicted_direction']
        
        if hist_dir == ml_dir:
            st.success(f"✓ **EVIDENCE AGREEMENT**: Historical regime response ({m1_ret:+.2f}%) and GeoPrice ML estimate ({forecast['predicted_return_pct']:+.2f}%) agree on **{ml_dir}** direction.")
        else:
            st.warning(f"⚡ **EVIDENCE DIVERGENCE**: Historical regime response suggests **{hist_dir}** ({m1_ret:+.2f}%), while GeoPrice ML estimate predicts **{ml_dir}** ({forecast['predicted_return_pct']:+.2f}%).")

        # Model Comparison Table Section
        comp_file = "outputs/phase3/tuning_experiments_comparison.csv"
        if os.path.exists(comp_file):
            comp_df = pd.read_csv(comp_file)
            c_comp = comp_df[comp_df['Commodity'] == selected_commodity].copy()
            if len(c_comp) > 0:
                st.subheader(f"Model Performance Benchmarks for {selected_commodity}")
                st.dataframe(c_comp[['Model', 'N', 'MAE', 'RMSE', 'Directional_Accuracy']], use_container_width=True)

        st.markdown("---")

        # 3. Model Interpretation (Stage 12 beta * z)
        st.subheader("3. Model Explanation — Why did GeoPrice make this forecast?")
        st.write("Feature contribution formula: $y = \\beta_0 + \\sum (\\beta_j \\times z_j)$, where $\\beta_j$ is the ElasticNet coefficient and $z_j$ is the standardized feature value.")
        
        contrib_df = explanation['ranked_contributions_df']
        
        # Display Reconstruction check
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Model Intercept (beta_0)", f"{explanation['intercept']:+.4f}")
        with rc2:
            st.metric("Sum of Contributions (beta*z)", f"{explanation['sum_of_contributions']:+.4f}")
        with rc3:
            st.metric("Exact Reconstructed Return", f"{explanation['reconstructed_prediction']*100:+.2f}%", delta="100% Match Check")

        fig, ax = plt.subplots(figsize=(10, 4))
        sorted_contribs = contrib_df.sort_values('Contribution')
        colors = ['#d62728' if c < 0 else '#2ca02c' for c in sorted_contribs['Contribution']]
        ax.barh(sorted_contribs['Feature'], sorted_contribs['Contribution'] * 100, color=colors)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"Feature Contributions (beta_j * z_j) for {selected_commodity} ({forecast['forecast_origin_date']})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Return Contribution (% points)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(contrib_df[['Rank', 'Feature', 'Feature_Group', 'Raw_Value', 'Standardized_Value', 'Coefficient', 'Contribution', 'Contribution_Direction']], use_container_width=True)

        st.markdown("---")

        # 4. Past-Month Validation Explorer
        st.subheader("4. Explore Past Out-of-Sample Validation Month")
        base_preds_path = "data/processed/geoprice_predictions.csv"
        if os.path.exists(base_preds_path):
            oos_preds = pd.read_csv(base_preds_path)
            c_oos = oos_preds[oos_preds['Commodity'] == selected_commodity].copy()
            
            avail_dates = c_oos['Date'].tolist()
            selected_date = st.selectbox("Select Past Validation Forecast Origin Date:", options=avail_dates, index=len(avail_dates)-1)
            
            match_row = c_oos[c_oos['Date'] == selected_date].iloc[0]
            
            p_actual = match_row['Actual_Return'] * 100
            p_pred = match_row['Predicted_Return'] * 100
            p_err = match_row['Absolute_Error'] * 100
            
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Forecast Origin Date", match_row['Date'])
            with e2:
                st.metric("Model Prediction", f"{p_pred:+.2f}%")
            with e3:
                st.metric("Actual Next-Month Return", f"{p_actual:+.2f}%")
            with e4:
                st.metric("Prediction Absolute Error", f"{p_err:.2f}% pts")

    except Exception as e:
        st.error(f"Error loading outlook data: {str(e)}")

# MODE B: SCENARIO EXPLORER (MANUAL MODE - STAGE 11)
else:
    st.warning("⚠️ **HISTORICAL REFERENCE ONLY — NOT A MACHINE LEARNING FORECAST**\n\nThis explorer evaluates empirical historical commodity responses under user-selected GPR regimes and major conflict reference cases. It does NOT call the ElasticNet ML model or construct synthetic GPR values.")
    
    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        scenario_regime = st.selectbox("Select Geopolitical Risk Regime:", options=REGIMES, index=2)
    with sc_col2:
        scenario_conflict = st.selectbox("Select Conflict Reference:", options=VALID_CONFLICT_REFERENCES, index=0)
        
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
