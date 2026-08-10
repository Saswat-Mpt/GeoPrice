import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.stats import percentileofscore

from geoprice.analysis.shock_responses import COMMODITIES

REGIMES = ['LOW', 'MODERATE', 'HIGH', 'EXTREME']

def calculate_gpr_regime_thresholds(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calculates empirical P50, P75, and P90 thresholds for GPR level distribution."""
    gpr_series = df['GPR'].dropna()
    p50 = float(np.percentile(gpr_series, 50.0))
    p75 = float(np.percentile(gpr_series, 75.0))
    p90 = float(np.percentile(gpr_series, 90.0))
    
    thresholds = {"P50": p50, "P75": p75, "P90": p90}
    meta = {
        "analysis_start": str(df['Date'].iloc[0]),
        "analysis_end": str(df['Date'].iloc[-1]),
        "total_months": int(len(gpr_series)),
        "P50": p50,
        "P75": p75,
        "P90": p90,
        "min_GPR": float(gpr_series.min()),
        "max_GPR": float(gpr_series.max()),
        "mean_GPR": float(gpr_series.mean())
    }
    return thresholds, meta

def assign_gpr_regimes(df: pd.DataFrame, p50: float, p75: float, p90: float) -> pd.DataFrame:
    """
    Assigns GPR levels to empirical regimes:
    LOW:      GPR < P50
    MODERATE: P50 <= GPR < P75
    HIGH:     P75 <= GPR <= P90
    EXTREME:  GPR > P90
    """
    res = df.copy()
    gpr_all = res['GPR'].dropna()
    
    # Calculate empirical percentile for every row
    res['GPR_percentile'] = res['GPR'].apply(lambda x: float(percentileofscore(gpr_all, x)) if pd.notna(x) else np.nan)
    
    def get_regime(val):
        if pd.isna(val):
            return np.nan
        if val < p50:
            return 'LOW'
        elif p50 <= val < p75:
            return 'MODERATE'
        elif p75 <= val <= p90:
            return 'HIGH'
        else:
            return 'EXTREME'
            
    res['GPR_regime'] = res['GPR'].apply(get_regime)
    return res

def get_current_gpr_state(df_regimes: pd.DataFrame, p50: float, p75: float, p90: float) -> Dict[str, Any]:
    """Identifies the latest complete GPR observation date, level, percentile, and regime."""
    valid_df = df_regimes.dropna(subset=['GPR']).sort_values('Date')
    latest_row = valid_df.iloc[-1]
    
    gpr_all = valid_df['GPR'].values
    gprt_all = valid_df['GPRT'].dropna().values if 'GPRT' in valid_df.columns else gpr_all
    gpra_all = valid_df['GPRA'].dropna().values if 'GPRA' in valid_df.columns else gpr_all
    
    gpr_val = float(latest_row['GPR'])
    gprt_val = float(latest_row['GPRT']) if 'GPRT' in latest_row and pd.notna(latest_row['GPRT']) else gpr_val
    gpra_val = float(latest_row['GPRA']) if 'GPRA' in latest_row and pd.notna(latest_row['GPRA']) else gpr_val
    
    current_state = {
        "current_date": str(latest_row['Date']),
        "current_GPR": gpr_val,
        "current_GPR_percentile": float(percentileofscore(gpr_all, gpr_val)),
        "current_GPR_regime": str(latest_row['GPR_regime']),
        "current_GPRT": gprt_val,
        "current_GPRT_percentile": float(percentileofscore(gprt_all, gprt_val)),
        "current_GPRA": gpra_val,
        "current_GPRA_percentile": float(percentileofscore(gpra_all, gpra_val))
    }
    return current_state

def build_regime_episodes(df_regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Groups contiguous historical months belonging to the same GPR regime into regime episodes.
    Selects the month with the highest GPR level as the representative date per episode.
    """
    valid_df = df_regimes.dropna(subset=['GPR_regime']).sort_values('Date').copy()
    valid_df['Period'] = pd.to_datetime(valid_df['Date']).dt.to_period('M')
    
    episodes: List[Dict[str, Any]] = []
    current_cluster: List[pd.Series] = []
    
    for idx, row in valid_df.iterrows():
        if not current_cluster:
            current_cluster.append(row)
        else:
            prev_row = current_cluster[-1]
            diff = (row['Period'].year - prev_row['Period'].year) * 12 + (row['Period'].month - prev_row['Period'].month)
            same_regime = (row['GPR_regime'] == prev_row['GPR_regime'])
            
            if diff == 1 and same_regime:
                current_cluster.append(row)
            else:
                episodes.append(_process_regime_cluster(current_cluster, len(episodes) + 1))
                current_cluster = [row]
                
    if current_cluster:
        episodes.append(_process_regime_cluster(current_cluster, len(episodes) + 1))
        
    return pd.DataFrame(episodes)

def _process_regime_cluster(cluster: List[pd.Series], episode_id: int) -> Dict[str, Any]:
    cluster_df = pd.DataFrame(cluster)
    rep_row = cluster_df.loc[cluster_df['GPR'].idxmax()]
    
    return {
        "episode_id": episode_id,
        "regime": rep_row['GPR_regime'],
        "start_date": cluster_df['Date'].iloc[0],
        "end_date": cluster_df['Date'].iloc[-1],
        "representative_date": rep_row['Date'],
        "representative_GPR": float(rep_row['GPR']),
        "month_count": int(len(cluster_df))
    }

def calculate_analogue_responses(episodes_df: pd.DataFrame, df_full: pd.DataFrame, current_regime: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates forward returns (+1M, +3M) for representative analogue episodes in current regime."""
    analogue_episodes = episodes_df[episodes_df['regime'] == current_regime].copy()
    df = df_full.copy().set_index('Date')
    dates_list = list(df.index)
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    
    rows = []
    for idx, ep in analogue_episodes.iterrows():
        rep_date = ep['representative_date']
        
        # Exclude current month if future window is unavailable
        if rep_date in date_to_idx:
            t_idx = date_to_idx[rep_date]
            row_dict = {
                "episode_id": ep['episode_id'],
                "regime": current_regime,
                "representative_date": rep_date,
                "GPR": ep['representative_GPR']
            }
            
            for c in COMMODITIES:
                p_t = df.loc[rep_date, c]
                # +1M
                if t_idx + 1 < len(dates_list):
                    p_t1 = df.loc[dates_list[t_idx + 1], c]
                    row_dict[f"{c}_1m"] = (p_t1 / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_t1) and p_t > 0) else np.nan
                else:
                    row_dict[f"{c}_1m"] = np.nan
                    
                # +3M
                if t_idx + 3 < len(dates_list):
                    p_t3 = df.loc[dates_list[t_idx + 3], c]
                    row_dict[f"{c}_3m"] = (p_t3 / p_t - 1.0) if (pd.notna(p_t) and pd.notna(p_t3) and p_t > 0) else np.nan
                else:
                    row_dict[f"{c}_3m"] = np.nan

            rows.append(row_dict)
            
    analogue_df = pd.DataFrame(rows)
    
    summary_rows = []
    for c in COMMODITIES:
        for horizon in ['1m', '3m']:
            col_name = f"{c}_{horizon}"
            if col_name in analogue_df.columns:
                series = analogue_df[col_name].dropna()
                summary_rows.append({
                    "Regime": current_regime,
                    "Commodity": c,
                    "Horizon": f"+{horizon.upper()}",
                    "N": int(len(series)),
                    "Mean": float(series.mean()) if len(series) > 0 else np.nan,
                    "Median": float(series.median()) if len(series) > 0 else np.nan,
                    "Min": float(series.min()) if len(series) > 0 else np.nan,
                    "Max": float(series.max()) if len(series) > 0 else np.nan
                })
    return analogue_df, pd.DataFrame(summary_rows)

def build_regime_scenario_lookup(episodes_df: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """Builds reusable historical scenario lookup table across all 4 regimes x 5 commodities x (+1M/+3M)."""
    lookup_rows = []
    
    for r in REGIMES:
        _, sum_r = calculate_analogue_responses(episodes_df, df_full, current_regime=r)
        lookup_rows.append(sum_r)
        
    return pd.concat(lookup_rows, ignore_index=True)
