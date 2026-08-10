import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from geoprice.analysis.shock_responses import COMMODITIES
from geoprice.analysis.regimes import REGIMES

VALID_CONFLICT_REFERENCES = ["None", "Major-conflict reference"]
DATA_DIR = "data/processed"

def get_historical_scenario(commodity: str, regime: str, conflict_reference: str = "None") -> Dict[str, Any]:
    """
    Historical Scenario Lookup Function (Stage 11).
    Retrieves empirical historical commodity responses under user-selected GPR regimes and conflict reference cases.
    
    GUARANTEE: Pure historical lookup. Does NOT call ElasticNet ML models or construct synthetic GPR values.
    """
    # 1. Input Validation
    if commodity not in COMMODITIES:
        raise ValueError(f"Invalid commodity '{commodity}'. Must be one of {COMMODITIES}.")
        
    regime_upper = str(regime).upper()
    if regime_upper not in REGIMES:
        raise ValueError(f"Invalid regime '{regime}'. Must be one of {REGIMES}.")
        
    if conflict_reference not in VALID_CONFLICT_REFERENCES:
        raise ValueError(f"Invalid conflict reference '{conflict_reference}'. Must be one of {VALID_CONFLICT_REFERENCES}.")

    # 2. Historical Regime Lookup (Phase 2 Stage 5 outputs)
    reg_lookup_path = os.path.join(DATA_DIR, "regime_scenario_lookup.csv")
    if not os.path.exists(reg_lookup_path):
        raise FileNotFoundError(f"Validated scenario lookup table '{reg_lookup_path}' missing. Run scripts/update_data.py.")

    reg_df = pd.read_csv(reg_lookup_path)
    sub_reg = reg_df[(reg_df['Commodity'] == commodity) & (reg_df['Regime'] == regime_upper)]

    m1_row = sub_reg[sub_reg['Horizon'] == '+1M']
    m3_row = sub_reg[sub_reg['Horizon'] == '+3M']

    regime_stats = {
        "1m_n": int(m1_row['N'].iloc[0]) if len(m1_row) > 0 else 0,
        "1m_mean_pct": float(m1_row['Mean'].iloc[0]) * 100.0 if len(m1_row) > 0 and pd.notna(m1_row['Mean'].iloc[0]) else np.nan,
        "1m_median_pct": float(m1_row['Median'].iloc[0]) * 100.0 if len(m1_row) > 0 and pd.notna(m1_row['Median'].iloc[0]) else np.nan,
        "1m_min_pct": float(m1_row['Min'].iloc[0]) * 100.0 if len(m1_row) > 0 and pd.notna(m1_row['Min'].iloc[0]) else np.nan,
        "1m_max_pct": float(m1_row['Max'].iloc[0]) * 100.0 if len(m1_row) > 0 and pd.notna(m1_row['Max'].iloc[0]) else np.nan,
        
        "3m_n": int(m3_row['N'].iloc[0]) if len(m3_row) > 0 else 0,
        "3m_mean_pct": float(m3_row['Mean'].iloc[0]) * 100.0 if len(m3_row) > 0 and pd.notna(m3_row['Mean'].iloc[0]) else np.nan,
        "3m_median_pct": float(m3_row['Median'].iloc[0]) * 100.0 if len(m3_row) > 0 and pd.notna(m3_row['Median'].iloc[0]) else np.nan,
        "3m_min_pct": float(m3_row['Min'].iloc[0]) * 100.0 if len(m3_row) > 0 and pd.notna(m3_row['Min'].iloc[0]) else np.nan,
        "3m_max_pct": float(m3_row['Max'].iloc[0]) * 100.0 if len(m3_row) > 0 and pd.notna(m3_row['Max'].iloc[0]) else np.nan,
    }

    # 3. Conflict Reference Lookup (Phase 2 Stage 6 outputs)
    conflict_stats = {}
    if conflict_reference == "Major-conflict reference":
        conf_summary_path = os.path.join(DATA_DIR, "conflict_reference_summary.csv")
        if not os.path.exists(conf_summary_path):
            raise FileNotFoundError(f"Conflict summary file '{conf_summary_path}' missing. Run scripts/update_data.py.")

        conf_df = pd.read_csv(conf_summary_path)
        sub_conf = conf_df[conf_df['Commodity'] == commodity]
        
        c1_row = sub_conf[sub_conf['Horizon'] == '+1M']
        c3_row = sub_conf[sub_conf['Horizon'] == '+3M']

        conflict_stats = {
            "conflict_1m_n": int(c1_row['N'].iloc[0]) if len(c1_row) > 0 else 0,
            "conflict_1m_median_pct": float(c1_row['Median'].iloc[0]) * 100.0 if len(c1_row) > 0 and pd.notna(c1_row['Median'].iloc[0]) else np.nan,
            "conflict_3m_n": int(c3_row['N'].iloc[0]) if len(c3_row) > 0 else 0,
            "conflict_3m_median_pct": float(c3_row['Median'].iloc[0]) * 100.0 if len(c3_row) > 0 and pd.notna(c3_row['Median'].iloc[0]) else np.nan,
        }

    # 4. Descriptive Interpretation Text
    interp_text = f"Historically, {commodity} exhibited a median +1M return of {regime_stats['1m_median_pct']:+.2f}% and +3M return of {regime_stats['3m_median_pct']:+.2f}% during '{regime_upper}' GPR regime periods (N={regime_stats['1m_n']} episodes)."
    if conflict_reference == "Major-conflict reference":
        interp_text += f" Across documented major-conflict reference cases, the median +1M response was {conflict_stats['conflict_1m_median_pct']:+.2f}% and +3M response was {conflict_stats['conflict_3m_median_pct']:+.2f}% (N={conflict_stats['conflict_1m_n']})."

    return {
        "commodity": commodity,
        "selected_regime": regime_upper,
        "conflict_reference": conflict_reference,
        "mode": "HISTORICAL SCENARIO LOOKUP (NON-ML)",
        "regime_stats": regime_stats,
        "conflict_stats": conflict_stats,
        "interpretation": interp_text
    }

def export_consolidated_scenario_lookup():
    """Exports clean, consolidated scenario lookup dataset data/processed/scenario_lookup.csv."""
    reg_lookup_path = os.path.join(DATA_DIR, "regime_scenario_lookup.csv")
    if os.path.exists(reg_lookup_path):
        reg_df = pd.read_csv(reg_lookup_path)
        out_path = os.path.join(DATA_DIR, "scenario_lookup.csv")
        reg_df.to_csv(out_path, index=False)
        return out_path
    return ""
