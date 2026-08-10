import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from geoprice.analysis.shocks import _process_cluster
from geoprice.analysis.shock_responses import COMMODITIES

def calculate_subindex_thresholds(df: pd.DataFrame, var_name: str, percentile: float = 90.0) -> Tuple[float, Dict[str, Any]]:
    """Calculates 90th percentile threshold for positive changes of a subindex (GPRT or GPRA)."""
    change = df[var_name] - df[var_name].shift(1)
    pos_changes = change.dropna()[change.dropna() > 0]
    
    threshold = float(np.percentile(pos_changes, percentile))
    meta = {
        "variable": var_name,
        "total_valid_changes": int(change.notna().sum()),
        "positive_change_count": int(len(pos_changes)),
        "percentile": float(percentile),
        "threshold": float(threshold),
        "min_pos_change": float(pos_changes.min()),
        "max_pos_change": float(pos_changes.max()),
        "median_pos_change": float(pos_changes.median())
    }
    return threshold, meta

def identify_subindex_shocks(df: pd.DataFrame, var_name: str, threshold: float) -> pd.DataFrame:
    """Identifies raw shock months where Δsubindex_t >= threshold."""
    temp_df = df.copy()
    temp_df[f"{var_name}_previous"] = temp_df[var_name].shift(1)
    temp_df[f"{var_name}_change"] = temp_df[var_name] - temp_df[f"{var_name}_previous"]
    
    raw_shocks = temp_df[temp_df[f"{var_name}_change"] >= threshold].copy()
    raw_shocks['threshold'] = threshold
    raw_shocks['change_rank'] = raw_shocks[f"{var_name}_change"].rank(ascending=False, method='min').astype(int)
    
    return raw_shocks.sort_values('Date').reset_index(drop=True)

def collapse_subindex_shocks(raw_shocks: pd.DataFrame, var_name: str, window_months: int = 3) -> pd.DataFrame:
    """Collapses nearby subindex shock months within window_months into representative episodes."""
    if len(raw_shocks) == 0:
        return pd.DataFrame()

    raw_shocks = raw_shocks.copy()
    raw_shocks['Period'] = pd.to_datetime(raw_shocks['Date']).dt.to_period('M')
    
    episodes: List[Dict[str, Any]] = []
    current_cluster: List[pd.Series] = []
    
    for idx, row in raw_shocks.iterrows():
        if not current_cluster:
            current_cluster.append(row)
        else:
            prev_row = current_cluster[-1]
            diff = (row['Period'].year - prev_row['Period'].year) * 12 + (row['Period'].month - prev_row['Period'].month)
            if diff <= window_months:
                current_cluster.append(row)
            else:
                episodes.append(_process_subindex_cluster(current_cluster, var_name, len(episodes) + 1))
                current_cluster = [row]
                
    if current_cluster:
        episodes.append(_process_subindex_cluster(current_cluster, var_name, len(episodes) + 1))
        
    return pd.DataFrame(episodes)

def _process_subindex_cluster(cluster: List[pd.Series], var_name: str, episode_id: int) -> Dict[str, Any]:
    cluster_df = pd.DataFrame(cluster)
    change_col = f"{var_name}_change"
    rep_row = cluster_df.loc[cluster_df[change_col].idxmax()]
    
    return {
        "episode_id": episode_id,
        "representative_shock_date": rep_row['Date'],
        "episode_start": cluster_df['Date'].iloc[0],
        "episode_end": cluster_df['Date'].iloc[-1],
        "representative_level": float(rep_row[var_name]),
        "representative_change": float(rep_row[change_col]),
        "raw_shock_count": int(len(cluster_df)),
        "cluster_dates": ", ".join(cluster_df['Date'].tolist())
    }

def calculate_subindex_responses(episodes_df: pd.DataFrame, df_full: pd.DataFrame, var_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates forward commodity returns for subindex shock episodes."""
    df = df_full.copy().set_index('Date')
    dates_list = list(df.index)
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    
    response_rows = []
    for idx, ep in episodes_df.iterrows():
        rep_date = ep['representative_shock_date']
        row_dict = {
            "subindex": var_name,
            "episode_id": ep['episode_id'],
            "shock_date": rep_date,
            "level": ep['representative_level'],
            "change": ep['representative_change']
        }
        
        if rep_date in date_to_idx:
            t_idx = date_to_idx[rep_date]
            for c in COMMODITIES:
                p_t = df.loc[rep_date, c]
                for h_idx, h_name in [(1, '1m'), (2, '2m'), (3, '3m')]:
                    if t_idx + h_idx < len(dates_list):
                        p_th = df.loc[dates_list[t_idx + h_idx], c]
                        row_dict[f"{c}_{h_name}"] = (p_th / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_th) and p_t > 0) else np.nan
                    else:
                        row_dict[f"{c}_{h_name}"] = np.nan
        response_rows.append(row_dict)
        
    responses_df = pd.DataFrame(response_rows)
    
    summary_rows = []
    for c in COMMODITIES:
        for horizon in ['1m', '2m', '3m']:
            col_name = f"{c}_{horizon}"
            if col_name in responses_df.columns:
                series = responses_df[col_name].dropna()
                summary_rows.append({
                    "Subindex": var_name,
                    "Commodity": c,
                    "Horizon": f"+{horizon.upper()}",
                    "N": int(len(series)),
                    "Mean": float(series.mean()) if len(series) > 0 else np.nan,
                    "Median": float(series.median()) if len(series) > 0 else np.nan,
                    "Min": float(series.min()) if len(series) > 0 else np.nan,
                    "Max": float(series.max()) if len(series) > 0 else np.nan
                })
    return responses_df, pd.DataFrame(summary_rows)
