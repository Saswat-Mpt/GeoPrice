import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def calculate_gpr_change(df: pd.DataFrame) -> pd.Series:
    """
    Calculates absolute monthly change in GPR:
    ΔGPR_t = GPR_t - GPR_(t-1)
    """
    return df['GPR'] - df['GPR'].shift(1)

def calculate_shock_threshold(df: pd.DataFrame, percentile: float = 90.0) -> Tuple[float, Dict[str, Any]]:
    """
    Filters positive GPR changes (ΔGPR > 0) and computes the specified percentile threshold (default 90th).
    Returns threshold value and distribution metadata dictionary.
    """
    gpr_change = calculate_gpr_change(df)
    positive_changes = gpr_change.dropna()[gpr_change.dropna() > 0]
    
    threshold = float(np.percentile(positive_changes, percentile))
    
    metadata = {
        "total_valid_changes": int(gpr_change.notna().sum()),
        "positive_change_count": int(len(positive_changes)),
        "percentile": float(percentile),
        "threshold": float(threshold),
        "min_pos_change": float(positive_changes.min()),
        "max_pos_change": float(positive_changes.max()),
        "median_pos_change": float(positive_changes.median())
    }
    return threshold, metadata

def identify_raw_shocks(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Identifies all months where ΔGPR_t >= threshold.
    Returns DataFrame containing raw shock dates, GPR, GPR_previous, GPR_change, threshold, and rank.
    """
    temp_df = df.copy()
    temp_df['GPR_previous'] = temp_df['GPR'].shift(1)
    temp_df['GPR_change'] = temp_df['GPR'] - temp_df['GPR_previous']
    
    raw_shocks = temp_df[temp_df['GPR_change'] >= threshold].copy()
    raw_shocks['threshold'] = threshold
    raw_shocks['GPR_change_rank'] = raw_shocks['GPR_change'].rank(ascending=False, method='min').astype(int)
    
    raw_shocks = raw_shocks.sort_values('Date').reset_index(drop=True)
    return raw_shocks

def collapse_overlapping_shocks(raw_shocks: pd.DataFrame, df_full: pd.DataFrame, window_months: int = 3) -> pd.DataFrame:
    """
    Collapses raw shock months occurring within window_months of one another into a single episode.
    The representative shock date for each episode is the month with the maximum positive ΔGPR.
    """
    if len(raw_shocks) == 0:
        return pd.DataFrame()

    # Convert Date strings to Period for distance calculation
    raw_shocks = raw_shocks.copy()
    raw_shocks['Period'] = pd.to_datetime(raw_shocks['Date']).dt.to_period('M')
    
    episodes: List[Dict[str, Any]] = []
    current_cluster: List[pd.Series] = []
    
    for idx, row in raw_shocks.iterrows():
        if not current_cluster:
            current_cluster.append(row)
        else:
            prev_row = current_cluster[-1]
            diff_months = (row['Period'].year - prev_row['Period'].year) * 12 + (row['Period'].month - prev_row['Period'].month)
            if diff_months <= window_months:
                current_cluster.append(row)
            else:
                # Close current episode cluster
                episodes.append(_process_cluster(current_cluster, len(episodes) + 1))
                current_cluster = [row]
                
    if current_cluster:
        episodes.append(_process_cluster(current_cluster, len(episodes) + 1))
        
    episodes_df = pd.DataFrame(episodes)
    return episodes_df

def _process_cluster(cluster: List[pd.Series], episode_id: int) -> Dict[str, Any]:
    """Helper to select representative shock date with max ΔGPR from a cluster."""
    cluster_df = pd.DataFrame(cluster)
    rep_row = cluster_df.loc[cluster_df['GPR_change'].idxmax()]
    
    return {
        "episode_id": episode_id,
        "representative_shock_date": rep_row['Date'],
        "episode_start": cluster_df['Date'].iloc[0],
        "episode_end": cluster_df['Date'].iloc[-1],
        "representative_GPR": float(rep_row['GPR']),
        "representative_GPR_change": float(rep_row['GPR_change']),
        "raw_shock_count": int(len(cluster_df)),
        "cluster_dates": ", ".join(cluster_df['Date'].tolist())
    }
