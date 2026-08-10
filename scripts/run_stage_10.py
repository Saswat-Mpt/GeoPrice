import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.inference.pipeline import predict_next_month, get_current_risk_context
from geoprice.analysis.shock_responses import COMMODITIES

def main():
    print("=" * 80)
    print("Running GeoPrice — Phase 4, Stage 10: Production Pipeline & Inference Check")
    print("=" * 80)

    print("\n[Step 1/3] Verifying trained production model artifacts in models/...")
    for c in COMMODITIES:
        mpath = f"models/{c.lower()}_model.joblib"
        if not os.path.exists(mpath):
            print(f"Error: Model artifact '{mpath}' missing! Run scripts/retrain_models.py.")
            sys.exit(1)
        print(f"  [PASS] Artifact verified: '{mpath}'")

    print("\n[Step 2/3] Running production inference for all 5 commodities...")
    inference_summary = []
    
    for c in COMMODITIES:
        forecast = predict_next_month(c)
        context = get_current_risk_context(c)
        
        inference_summary.append({
            "Commodity": c,
            "Price": context['current_price'],
            "Forecast_Origin": forecast['forecast_origin_date'],
            "Target_Month": forecast['target_month'],
            "GeoPrice_Forecast": f"{forecast['predicted_return_pct']:+.2f}%",
            "Direction": forecast['predicted_direction'],
            "GPR_Level": context['current_GPR'],
            "Regime": context['current_GPR_regime'],
            "Shock_Status": "YES (GPR SHOCK)" if context['is_gpr_shock'] else "NO (NORMAL DRIFT)",
            "Analogue_+1M_Median": f"{context['analogue_1m_median_pct']:+.2f}%",
            "Analogue_+3M_Median": f"{context['analogue_3m_median_pct']:+.2f}%"
        })

    inf_df = pd.DataFrame(inference_summary)

    print("\n" + "=" * 80)
    print("STAGE 10 PRODUCTION INFERENCE SUMMARY")
    print("=" * 80)
    print(inf_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Interactive Streamlit Dashboard Entry Point: app.py")
    print("Startup Command: streamlit run app.py")
    print("=" * 80)
    print("STAGE 10 COMPLETE — GEOPRICE PROJECT FULLY FINISHED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
