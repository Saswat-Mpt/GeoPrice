import os
import pandas as pd
import requests

FRED_LOCAL_DIR = "data/raw/fred"

FRED_COMMODITY_SERIES = {
    "POILBREUSDM": "Brent",
    "PNGASUSUSDM": "Natural_Gas",
    "PCOPPUSDM": "Copper",
    "PWHEAMTUSDM": "Wheat"
}

def load_fred_commodity(series_id: str, name: str, local_dir: str = FRED_LOCAL_DIR) -> pd.DataFrame:
    """Downloads and loads a single FRED monthly commodity series. Returns DataFrame with Period index and one column named `name`."""
    local_path = os.path.join(local_dir, f"{series_id}.csv")
    os.makedirs(local_dir, exist_ok=True)
    
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 100):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    
    df = pd.read_csv(local_path)
    date_col = 'DATE' if 'DATE' in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    df['Period'] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M')
    df[name] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=['Period', name])
    df = df[['Period', name]].drop_duplicates(subset=['Period']).sort_values('Period').set_index('Period')
    return df

def load_fred_commodities() -> pd.DataFrame:
    """Loads Brent, Natural_Gas, Copper, Wheat from FRED. Returns DataFrame indexed by Period."""
    dfs = []
    for series_id, name in FRED_COMMODITY_SERIES.items():
        dfs.append(load_fred_commodity(series_id, name))
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how='outer')
    return result

def load_dxy_daily(local_path: str = os.path.join(FRED_LOCAL_DIR, "DTWEXBGS.csv")) -> pd.DataFrame:
    """
    Loads raw daily DXY series (DTWEXBGS).
    Returns DataFrame with columns ['Date', 'DTWEXBGS'].
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Check if local CSV exists
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 100):
        # If local file missing, download authentic FRED DTWEXBGS series directly from FRED
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(fred_url, headers=headers, timeout=20)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    df = pd.read_csv(local_path)
    date_col = 'DATE' if 'DATE' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
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
