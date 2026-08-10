import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from geoprice.constants import COMMODITIES

def calculate_forward_commodity_responses(episodes_df: pd.DataFrame, df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every representative shock episode month t, calculates forward commodity cumulative returns:
    +1M = P_(t+1) / P_t - 1
    +2M = P_(t+2) / P_t - 1
    +3M = P_(t+3) / P_t - 1
    
    Incomplete forward windows (e.g. near dataset end) return NaN.
    Returns:
    1. Episode-level responses DataFrame
    2. Summary statistics DataFrame (N, mean, median, min, max) per commodity and horizon
    """
    df = df_full.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
        
    dates_list = list(df.index)
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    
    response_rows = []
    
    for idx, ep in episodes_df.iterrows():
        rep_date = ep['representative_shock_date']
        row_dict = {
            "episode_id": ep['episode_id'],
            "shock_date": rep_date,
            "GPR": ep['representative_GPR'],
            "GPR_change": ep['representative_GPR_change'],
            "raw_shock_count": ep['raw_shock_count']
        }
        
        if rep_date in date_to_idx:
            t_idx = date_to_idx[rep_date]
            
            for c in COMMODITIES:
                p_t = df.loc[rep_date, c]
                
                # +1M horizon (t+1)
                if t_idx + 1 < len(dates_list):
                    p_t1 = df.loc[dates_list[t_idx + 1], c]
                    row_dict[f"{c}_1m"] = (p_t1 / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_t1) and p_t > 0) else np.nan
                else:
                    row_dict[f"{c}_1m"] = np.nan
                    
                # +2M horizon (t+2)
                if t_idx + 2 < len(dates_list):
                    p_t2 = df.loc[dates_list[t_idx + 2], c]
                    row_dict[f"{c}_2m"] = (p_t2 / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_t2) and p_t > 0) else np.nan
                else:
                    row_dict[f"{c}_2m"] = np.nan

                # +3M horizon (t+3)
                if t_idx + 3 < len(dates_list):
                    p_t3 = df.loc[dates_list[t_idx + 3], c]
                    row_dict[f"{c}_3m"] = (p_t3 / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_t3) and p_t > 0) else np.nan
                else:
                    row_dict[f"{c}_3m"] = np.nan

        response_rows.append(row_dict)
        
    responses_df = pd.DataFrame(response_rows)
    
    # Calculate Summary Statistics
    summary_rows = []
    for c in COMMODITIES:
        for horizon in ['1m', '2m', '3m']:
            col_name = f"{c}_{horizon}"
            if col_name in responses_df.columns:
                series = responses_df[col_name].dropna()
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
    return responses_df, summary_df
