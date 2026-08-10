import os
import pandas as pd
import requests

FRED_LOCAL_DIR = "data/raw/fred"

FRED_SERIES_METADATA = {
    "POILBREUSDM": {"name": "Brent", "freq": "Monthly", "desc": "Brent Crude Oil ($/bbl)"},
    "PNGASUSUSDM": {"name": "Natural_Gas", "freq": "Monthly", "desc": "Natural Gas Henry Hub ($/MMBtu)"},
    "PCOPPUSDM": {"name": "Copper", "freq": "Monthly", "desc": "Copper ($/MT)"},
    "PWHEAMTUSDM": {"name": "Wheat", "freq": "Monthly", "desc": "Wheat ($/MT)"},
    "DTWEXBGS": {"name": "DXY", "freq": "Daily", "desc": "Nominal Broad U.S. Dollar Index"}
}

def load_dxy_daily(local_path: str = os.path.join(FRED_LOCAL_DIR, "DTWEXBGS.csv")) -> pd.DataFrame:
    """
    Loads raw daily DXY series (DTWEXBGS or Yahoo Finance DX-Y.NYB equivalent).
    Returns DataFrame with columns ['Date', 'DTWEXBGS'].
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Check if local CSV exists
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
        df = pd.read_csv(local_path)
        date_col = 'DATE' if 'DATE' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
        val_col = 'DTWEXBGS' if 'DTWEXBGS' in df.columns else df.columns[1]
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['DTWEXBGS'] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=['Date', 'DTWEXBGS']).copy()
        return df[['Date', 'DTWEXBGS']]
        
    # If local file missing, download authentic FRED DTWEXBGS series directly from FRED
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(fred_url, headers=headers, timeout=20)
    r.raise_for_status()
    
    with open(local_path, "wb") as f:
        f.write(r.content)

    df = pd.read_csv(local_path)
    date_col = 'DATE' if 'DATE' in df.columns else df.columns[0]
    val_col = 'DTWEXBGS' if 'DTWEXBGS' in df.columns else df.columns[1]
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['DTWEXBGS'] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=['Date', 'DTWEXBGS']).copy()
    return df[['Date', 'DTWEXBGS']]

def aggregate_dxy_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Converts daily DXY series to monthly arithmetic mean.
    For month t: DXY_t = mean(all available daily DXY observations in month t).
    Returns DataFrame indexed by monthly PeriodIndex with single column 'DXY'.
    """
    df = df_daily.copy()
    df['Period'] = df['Date'].dt.to_period('M')
    
    # Group by Period and calculate arithmetic mean (EXPLICITLY mean(), NOT last/first/median)
    monthly_dxy = df.groupby('Period')['DTWEXBGS'].mean().reset_index()
    monthly_dxy.rename(columns={'DTWEXBGS': 'DXY'}, inplace=True)
    monthly_dxy = monthly_dxy.sort_values('Period').set_index('Period')
    return monthly_dxy
