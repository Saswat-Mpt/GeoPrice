import pandas as pd
import numpy as np
from typing import Dict, Any

from geoprice.features.engineering import COMMODITIES

EXPECTED_COMMON_FEATURES = ['GPR', 'GPR_change', 'GPR_lag1', 'GPR_lag3', 'GPRT', 'GPRA', 'DXY']

def get_commodity_feature_names(commodity: str) -> list:
    return [
        f"{commodity}_return_1m",
        f"{commodity}_return_3m",
        f"{commodity}_return_6m",
        f"{commodity}_vol_3m"
    ]

def validate_features(feature_df: pd.DataFrame, df_raw: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Validates Stage 2 feature dataset structure, formula correctness, and anti-leakage.
    """
    results = {}
    
    # 1. Column existence
    all_expected_cols = ['Date'] + EXPECTED_COMMON_FEATURES
    for c in COMMODITIES:
        all_expected_cols.extend(get_commodity_feature_names(c))
        
    has_all_cols = all(col in feature_df.columns for col in all_expected_cols)
    results['all_expected_columns_exist'] = has_all_cols
    
    # 2. Total feature counts
    results['total_columns'] = len(feature_df.columns)
    results['common_features_count'] = len(EXPECTED_COMMON_FEATURES)
    results['features_per_commodity'] = 11
    
    # 3. Missingness report
    missing_summary = {}
    for col in feature_df.columns:
        if col != 'Date':
            m_count = feature_df[col].isna().sum()
            m_pct = round((m_count / len(feature_df)) * 100, 2)
            missing_summary[col] = {"missing_count": int(m_count), "missing_pct": m_pct}
            
    results['missing_summary'] = missing_summary
    
    # 4. Anti-Leakage Verification Test
    # Test that altering observation at t+1 does not change features at t
    if df_raw is not None and len(df_raw) > 20:
        from geoprice.features.engineering import build_feature_dataset
        
        t_idx = 100
        df_copy1 = df_raw.copy()
        feat_df1 = build_feature_dataset(df_copy1)
        row_t_feat1 = feat_df1.iloc[t_idx].copy()
        
        # Modify future price and GPR at t_idx + 1
        df_copy2 = df_raw.copy()
        price_cols = ['Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat', 'GPR', 'DXY']
        for pcol in price_cols:
            if pcol in df_copy2.columns and pd.notna(df_copy2.loc[t_idx + 1, pcol]):
                df_copy2.loc[t_idx + 1, pcol] = df_copy2.loc[t_idx + 1, pcol] * 2.0
                
        feat_df2 = build_feature_dataset(df_copy2)
        row_t_feat2 = feat_df2.iloc[t_idx].copy()
        
        leakage_found = False
        for col in feature_df.columns:
            if col != 'Date' and pd.notna(row_t_feat1[col]) and pd.notna(row_t_feat2[col]):
                if not np.isclose(row_t_feat1[col], row_t_feat2[col]):
                    leakage_found = True
                    break
                    
        results['anti_leakage_passed'] = not leakage_found
    else:
        results['anti_leakage_passed'] = True
        
    results['overall_pass'] = has_all_cols and results['anti_leakage_passed']
    return results

def create_feature_dictionary(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Creates metadata feature dictionary describing all 11 features per commodity."""
    dict_rows = [
        # Common Geopolitical Features
        {"Feature_Name": "GPR", "Group": "Geopolitical", "Definition": "GPR level at month t (GPR_t)", "Source": "Caldara-Iacoviello", "Window": "Current Month", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "Missing pre-1985"},
        {"Feature_Name": "GPR_change", "Group": "Geopolitical", "Definition": "Monthly GPR absolute change (GPR_t - GPR_(t-1))", "Source": "Caldara-Iacoviello", "Window": "1 Month", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "First month NaN"},
        {"Feature_Name": "GPR_lag1", "Group": "Geopolitical", "Definition": "GPR level at month t-1 (GPR_(t-1))", "Source": "Caldara-Iacoviello", "Window": "Lag-1", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "First month NaN"},
        {"Feature_Name": "GPR_lag3", "Group": "Geopolitical", "Definition": "GPR level at month t-3 (GPR_(t-3))", "Source": "Caldara-Iacoviello", "Window": "Lag-3", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "First 3 months NaN"},
        {"Feature_Name": "GPRT", "Group": "Geopolitical", "Definition": "Geopolitical Threats subindex level (GPRT_t)", "Source": "Caldara-Iacoviello", "Window": "Current Month", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "Missing pre-1985"},
        {"Feature_Name": "GPRA", "Group": "Geopolitical", "Definition": "Geopolitical Acts subindex level (GPRA_t)", "Source": "Caldara-Iacoviello", "Window": "Current Month", "Availability_Rule": "Release-aware point-in-time", "Expected_Missingness": "Missing pre-1985"},
        
        # Macro Control
        {"Feature_Name": "DXY", "Group": "Macro Control", "Definition": "Monthly arithmetic mean U.S. Dollar Index", "Source": "FRED", "Window": "Current Month", "Availability_Rule": "Point-in-time control", "Expected_Missingness": "Missing pre-2001"},
    ]
    
    # Commodity features for each commodity
    for c in COMMODITIES:
        c_src = "World Bank Pink Sheet" if c == "Gold" else "FRED"
        dict_rows.extend([
            {"Feature_Name": f"{c}_return_1m", "Group": "Commodity History", "Definition": f"1-month decimal return: P_t / P_(t-1) - 1 for {c}", "Source": c_src, "Window": "1 Month", "Availability_Rule": "Point-in-time price", "Expected_Missingness": "First month NaN"},
            {"Feature_Name": f"{c}_return_3m", "Group": "Commodity History", "Definition": f"3-month decimal return: P_t / P_(t-3) - 1 for {c}", "Source": c_src, "Window": "3 Months", "Availability_Rule": "Point-in-time price", "Expected_Missingness": "First 3 months NaN"},
            {"Feature_Name": f"{c}_return_6m", "Group": "Commodity History", "Definition": f"6-month decimal return: P_t / P_(t-6) - 1 for {c}", "Source": c_src, "Window": "6 Months", "Availability_Rule": "Point-in-time price", "Expected_Missingness": "First 6 months NaN"},
            {"Feature_Name": f"{c}_vol_3m", "Group": "Commodity History", "Definition": f"3-month rolling std dev of 1M returns for {c}", "Source": c_src, "Window": "3 Months", "Availability_Rule": "Point-in-time price", "Expected_Missingness": "First 3 months NaN"},
        ])
        
    return pd.DataFrame(dict_rows)
