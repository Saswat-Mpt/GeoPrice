import os
import pandas as pd
import requests

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
LOCAL_GPR_PATH = "data/raw/gpr/data_gpr_export.xls"

def download_gpr_if_missing(local_path: str = LOCAL_GPR_PATH) -> str:
    """Download official Caldara-Iacoviello GPR dataset if not present locally."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100000:
        return local_path
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(GPR_URL, headers=headers, timeout=30)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def load_gpr(local_path: str = LOCAL_GPR_PATH) -> pd.DataFrame:
    """
    Loads Caldara-Iacoviello GPR dataset and extracts GPR, GPRT, GPRA.
    Returns DataFrame indexed by monthly PeriodIndex with numeric columns ['GPR', 'GPRT', 'GPRA'].
    """
    file_path = download_gpr_if_missing(local_path)
    
    # Read raw excel file
    df_raw = pd.read_excel(file_path)
    
    # Determine date column
    date_col = 'month' if 'month' in df_raw.columns else ('Date' if 'Date' in df_raw.columns else 'date')
    if date_col not in df_raw.columns:
        raise ValueError(f"Could not find date column in {file_path}")
    
    df_raw['Period'] = pd.to_datetime(df_raw[date_col], errors='coerce').dt.to_period('M')
    df = df_raw.dropna(subset=['Period']).copy()
    
    # Standardize column names (GPR, GPRT, GPRA)
    col_map = {}
    for c in ['GPR', 'GPRT', 'GPRA']:
        if c in df.columns:
            col_map[c] = c
        elif c == 'GPRT' and 'GPR_THREAT' in df.columns:
            col_map['GPR_THREAT'] = 'GPRT'
        elif c == 'GPRA' and 'GPR_ACT' in df.columns:
            col_map['GPR_ACT'] = 'GPRA'
            
    required_vars = ['GPR', 'GPRT', 'GPRA']
    for req in required_vars:
        if req not in col_map.values() and req not in df.columns:
            raise KeyError(f"Required variable '{req}' not found in GPR dataset.")
            
    df = df.rename(columns=col_map)
    df = df[['Period', 'GPR', 'GPRT', 'GPRA']].dropna(subset=['GPR'])
    
    # Ensure numeric types
    for c in ['GPR', 'GPRT', 'GPRA']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.sort_values('Period').drop_duplicates(subset=['Period'])
    df = df.set_index('Period')
    return df
