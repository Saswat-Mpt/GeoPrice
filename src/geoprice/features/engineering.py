import pandas as pd
import numpy as np

COMMODITIES = ['Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']

def calculate_returns(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """
    Calculates 1-month, 3-month, and 6-month decimal returns for a given commodity price series P_t.
    R_1,t = P_t / P_(t-1) - 1
    R_3,t = P_t / P_(t-3) - 1
    R_6,t = P_t / P_(t-6) - 1
    """
    res = pd.DataFrame(index=df.index)
    price = df[commodity]
    
    res[f"{commodity}_return_1m"] = (price / price.shift(1)) - 1.0
    res[f"{commodity}_return_3m"] = (price / price.shift(3)) - 1.0
    res[f"{commodity}_return_6m"] = (price / price.shift(6)) - 1.0
    
    return res

def calculate_rolling_volatility(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """
    Calculates 3-month rolling volatility of monthly commodity returns.
    rolling_vol_3m = sample std dev of previous 3 monthly returns (R_1,t-2, R_1,t-1, R_1,t).
    Uses backward-looking rolling window without future observations.
    """
    res = pd.DataFrame(index=df.index)
    ret_1m = (df[commodity] / df[commodity].shift(1)) - 1.0
    
    # 3-month rolling standard deviation of 1M returns
    res[f"{commodity}_vol_3m"] = ret_1m.rolling(window=3, min_periods=3).std(ddof=1)
    return res

def create_geopolitical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 6 geopolitical risk features from GPR, GPRT, GPRA:
    1. GPR level (GPR_t)
    2. GPR monthly change (GPR_t - GPR_(t-1), absolute change)
    3. GPR lag-1 (GPR_(t-1))
    4. GPR lag-3 (GPR_(t-3))
    5. GPRT level (GPRT_t)
    6. GPRA level (GPRA_t)
    """
    res = pd.DataFrame(index=df.index)
    
    res['GPR'] = df['GPR']
    res['GPR_change'] = df['GPR'] - df['GPR'].shift(1)
    res['GPR_lag1'] = df['GPR'].shift(1)
    res['GPR_lag3'] = df['GPR'].shift(3)
    res['GPRT'] = df['GPRT']
    res['GPRA'] = df['GPRA']
    
    return res

def create_macro_control_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts DXY macro control feature.
    """
    res = pd.DataFrame(index=df.index)
    res['DXY'] = df['DXY']
    return res

def build_feature_dataset(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the master Stage 2 feature dataset.
    Contains:
    - Date
    - 7 Common Features: GPR, GPR_change, GPR_lag1, GPR_lag3, GPRT, GPRA, DXY
    - 20 Commodity Features: 4 features x 5 commodities
      (Brent, Natural_Gas, Gold, Copper, Wheat)
    """
    df = df_aligned.copy()
    
    # Ensure Date column is string and set index for calculation
    if 'Date' in df.columns:
        df = df.set_index('Date')
        
    feature_dfs = []
    
    # 1. Common Geopolitical Features (6)
    geo_df = create_geopolitical_features(df)
    feature_dfs.append(geo_df)
    
    # 2. Common Macro Control (1)
    macro_df = create_macro_control_feature(df)
    feature_dfs.append(macro_df)
    
    # 3. Commodity-Specific Features (4 per commodity)
    for c in COMMODITIES:
        ret_df = calculate_returns(df, c)
        vol_df = calculate_rolling_volatility(df, c)
        feature_dfs.append(ret_df)
        feature_dfs.append(vol_df)
        
    # Combine all feature DataFrames
    features_all = pd.concat(feature_dfs, axis=1)
    
    # Reset index to bring Date back as a column
    result = features_all.reset_index()
    return result
