import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from geoprice.analysis.shock_responses import COMMODITIES

DOCUMENTED_CONFLICT_CANDIDATES = [
    {
        "conflict_name": "9/11 Terrorist Attacks",
        "event_start": "2001-09",
        "event_end": "2001-10",
        "representative_shock_date": "2001-09",
        "source": "U.S. Department of State / Britannica",
        "notes": "Systematic GPR shock triggered by September 11 terrorist attacks."
    },
    {
        "conflict_name": "2003 Iraq Invasion",
        "event_start": "2003-03",
        "event_end": "2003-05",
        "representative_shock_date": "2003-03",
        "source": "U.N. Security Council Records / Britannica",
        "notes": "Systematic GPR shock during U.S.-led invasion of Iraq."
    },
    {
        "conflict_name": "2014 Russia-Ukraine / Crimea Crisis",
        "event_start": "2014-03",
        "event_end": "2014-04",
        "representative_shock_date": "2014-03",
        "source": "OSCE / International Court of Justice / Britannica",
        "notes": "Systematic GPR shock triggered by annexation of Crimea."
    },
    {
        "conflict_name": "2022 Russia-Ukraine Invasion",
        "event_start": "2022-02",
        "event_end": "2022-04",
        "representative_shock_date": "2022-03",
        "source": "U.N. General Assembly Resolutions / Britannica",
        "notes": "Systematic GPR shock following full-scale invasion of Ukraine."
    }
]

def map_conflict_reference_cases(shock_episodes_df: pd.DataFrame, responses_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maps candidate documented major conflict cases to systematically identified Stage 3 shock episodes.
    Extracts forward commodity cumulative returns (+1M, +2M, +3M) and computes small-sample medians.
    """
    valid_shock_dates = set(shock_episodes_df['representative_shock_date'].tolist())
    
    conflict_rows = []
    for cand in DOCUMENTED_CONFLICT_CANDIDATES:
        rep_date = cand['representative_shock_date']
        
        # Verify candidate exists in Stage 3 systematic shock list
        if rep_date in valid_shock_dates:
            resp_row = responses_df[responses_df['shock_date'] == rep_date]
            
            row = {
                "conflict_name": cand['conflict_name'],
                "event_start": cand['event_start'],
                "event_end": cand['event_end'],
                "representative_shock_date": rep_date,
                "GPR": float(resp_row['GPR'].iloc[0]) if len(resp_row) > 0 else np.nan,
                "GPR_change": float(resp_row['GPR_change'].iloc[0]) if len(resp_row) > 0 else np.nan,
                "source": cand['source'],
                "notes": cand['notes']
            }
            
            if len(resp_row) > 0:
                for c in COMMODITIES:
                    for h in ['1m', '2m', '3m']:
                        col_name = f"{c}_{h}"
                        row[col_name] = float(resp_row[col_name].iloc[0]) if col_name in resp_row.columns else np.nan
                        
            conflict_rows.append(row)

    cases_df = pd.DataFrame(conflict_rows)
    
    # Calculate small-sample median and mean across reference cases
    summary_rows = []
    for c in COMMODITIES:
        for horizon in ['1m', '2m', '3m']:
            col_name = f"{c}_{horizon}"
            if col_name in cases_df.columns:
                series = cases_df[col_name].dropna()
                summary_rows.append({
                    "Commodity": c,
                    "Horizon": f"+{horizon.upper()}",
                    "N": int(len(series)),
                    "Mean": float(series.mean()) if len(series) > 0 else np.nan,
                    "Median": float(series.median()) if len(series) > 0 else np.nan,
                    "Min": float(series.min()) if len(series) > 0 else np.nan,
                    "Max": float(series.max()) if len(series) > 0 else np.nan
                })
                
    summary_df = pd.DataFrame(summary_rows)
    return cases_df, summary_df
