import numpy as np
import pandas as pd
from typing import Dict, Any, List

FORBIDDEN_GEOPOLITICAL_COLS = ['GPR', 'GPR_change', 'GPR_lag1', 'GPR_lag3', 'GPRT', 'GPRA', 'DXY']

def validate_baseline_features(feature_cols: List[str]) -> bool:
    """Verifies baseline feature list contains ONLY commodity-history features (NO GPR/DXY)."""
    for col in feature_cols:
        if any(forbidden in col for forbidden in FORBIDDEN_GEOPOLITICAL_COLS):
            return False
    return True

def validate_expanding_window_order(pred_df: pd.DataFrame) -> bool:
    """Verifies chronological ordering of out-of-sample prediction dates."""
    dates = pd.to_datetime(pred_df['Date'])
    return bool(dates.is_monotonic_increasing)
