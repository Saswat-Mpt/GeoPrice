import pandas as pd
from typing import Dict, Any

def apply_gpr_availability_rule(df: pd.DataFrame, release_lag_months: int = 0) -> pd.DataFrame:
    """
    Applies release-aware point-in-time availability rule to GPR, GPRT, and GPRA.
    
    If release_lag_months == 0 (default): current month observation published by forecast origin.
    If release_lag_months == 1: current month GPR pending release, fallback to lag-1.
    
    Returns DataFrame with point-in-time GPR series.
    """
    res = df.copy()
    if release_lag_months > 0:
        res['GPR_pit'] = res['GPR'].shift(release_lag_months)
        res['GPRT_pit'] = res['GPRT'].shift(release_lag_months)
        res['GPRA_pit'] = res['GPRA'].shift(release_lag_months)
    else:
        res['GPR_pit'] = res['GPR']
        res['GPRT_pit'] = res['GPRT']
        res['GPRA_pit'] = res['GPRA']
    return res

def apply_dxy_availability_rule(df: pd.DataFrame, release_lag_months: int = 0) -> pd.DataFrame:
    """
    Applies point-in-time availability rule to DXY control.
    
    If current month DXY is available by forecast origin, use DXY_t.
    If release lag exists, use DXY_(t-release_lag_months).
    """
    res = df.copy()
    if release_lag_months > 0:
        res['DXY_pit'] = res['DXY'].shift(release_lag_months)
    else:
        res['DXY_pit'] = res['DXY']
    return res

def document_availability_metadata() -> Dict[str, Any]:
    """Returns documentation metadata regarding point-in-time availability rules."""
    return {
        "GPR_availability_rule": "Release-aware point-in-time availability. GPR, GPRT, GPRA observations aligned to latest published vintage at forecast origin month t.",
        "DXY_availability_rule": "Point-in-time control availability. Uses monthly mean DXY known as of forecast origin month t (or lag-1 if preliminary).",
        "limitation_note": "Claimed release-aware point-in-time availability; does not attempt full historical vintage reconstruction beyond published release schedules."
    }
