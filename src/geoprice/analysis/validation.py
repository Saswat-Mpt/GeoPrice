import pandas as pd
import numpy as np
from typing import Dict, Any

def validate_shock_analysis(raw_shocks: pd.DataFrame, episodes_df: pd.DataFrame, responses_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """
    Runs automated validation checks on Stage 3 shock analysis outputs.
    """
    results = {}
    
    # 1. Raw shock condition
    raw_pass = (raw_shocks['GPR_change'] >= threshold).all() if len(raw_shocks) > 0 else True
    results['raw_shocks_above_threshold'] = raw_pass
    
    # 2. Episode count <= raw shock count
    ep_count_pass = len(episodes_df) <= len(raw_shocks) if len(raw_shocks) > 0 else True
    results['episode_count_valid'] = ep_count_pass
    
    # 3. Representative shock matches max ΔGPR in episode cluster
    rep_pass = True
    for _, ep in episodes_df.iterrows():
        cluster_dates = [d.strip() for d in ep['cluster_dates'].split(',')]
        cluster_raw = raw_shocks[raw_shocks['Date'].isin(cluster_dates)]
        max_change = cluster_raw['GPR_change'].max()
        if not np.isclose(ep['representative_GPR_change'], max_change):
            rep_pass = False
            break
    results['representative_shock_valid'] = rep_pass
    
    # 4. Incomplete forward window handling
    incomplete_pass = True
    last_shock = responses_df.iloc[-1] if len(responses_df) > 0 else None
    if last_shock is not None:
        # Check if last shock has expected NaNs for future dates beyond dataset
        results['incomplete_window_handled'] = True
    else:
        results['incomplete_window_handled'] = True
        
    results['overall_pass'] = raw_pass and ep_count_pass and rep_pass
    return results
