import os
import re
import pandas as pd
import requests

WB_URL = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
LOCAL_WB_PATH = "data/raw/world_bank/CMO-Historical-Data-Monthly.xlsx"

def download_world_bank_if_missing(local_path: str = LOCAL_WB_PATH) -> str:
    """Download World Bank Pink Sheet Excel file if missing."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 50000:
        return local_path
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # Try direct URL or discovery
    try:
        r = requests.get(WB_URL, headers=headers, timeout=30)
        if r.status_code == 200 and len(r.content) > 50000:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
    except Exception:
        pass

    # Fallback to web scraping
    page_url = "https://www.worldbank.org/en/research/commodity-markets"
    r = requests.get(page_url, headers=headers, timeout=30)
    matches = re.findall(r'https?://[^\s"\']*CMO-Historical-Data-Monthly\.xlsx', r.text)
    if matches:
        r2 = requests.get(matches[0], headers=headers, timeout=30)
        r2.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r2.content)
        return local_path
    else:
        raise RuntimeError("Failed to download World Bank Pink Sheet workbook.")

def inspect_world_bank_gold(local_path: str = LOCAL_WB_PATH) -> dict:
    """
    Mandatory inspection of World Bank workbook for Gold.
    Determines sheet, date format, Gold column, units, date range, missing values.
    """
    file_path = download_world_bank_if_missing(local_path)
    excel = pd.ExcelFile(file_path)
    
    sheet_name = 'Monthly Prices' if 'Monthly Prices' in excel.sheet_names else excel.sheet_names[1]
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    col_names = df_raw.iloc[3].values
    units_names = df_raw.iloc[4].values
    
    gold_col_idx = None
    gold_col_name = None
    gold_unit = None
    
    for idx in range(df_raw.shape[1]):
        c_name = str(col_names[idx]) if pd.notna(col_names[idx]) else ""
        c_unit = str(units_names[idx]) if pd.notna(units_names[idx]) else ""
        combined = f"{c_name} {c_unit}".strip()
        if 'gold' in combined.lower():
            gold_col_idx = idx
            gold_col_name = c_name if c_name else c_unit
            gold_unit = c_unit
            break
            
    if gold_col_idx is None:
        raise ValueError("Gold column could not be identified in World Bank Pink Sheet.")

    # Parse dates
    data_rows = df_raw.iloc[5:].copy()
    data_rows.rename(columns={0: 'Date_Raw'}, inplace=True)
    
    def parse_wb_date(d):
        if pd.isna(d) or not isinstance(d, str):
            return pd.NaT
        d = d.strip()
        if len(d) == 7 and 'M' in d:
            parts = d.split('M')
            return pd.Period(f"{parts[0]}-{parts[1]}", freq='M')
        return pd.NaT

    data_rows['Period'] = data_rows['Date_Raw'].apply(parse_wb_date)
    valid_df = data_rows.dropna(subset=['Period']).copy()
    
    gold_series = pd.to_numeric(valid_df[gold_col_idx], errors='coerce')
    valid_gold = gold_series.dropna()
    
    first_date = valid_df.loc[valid_gold.index[0], 'Period']
    last_date = valid_df.loc[valid_gold.index[-1], 'Period']
    
    return {
        "file_path": file_path,
        "sheet_name": sheet_name,
        "date_column": "Column 0 (Date_Raw)",
        "gold_col_index": gold_col_idx,
        "gold_col_name": gold_col_name,
        "gold_unit": gold_unit,
        "total_rows": len(valid_df),
        "valid_gold_obs": len(valid_gold),
        "missing_gold_obs": gold_series.isna().sum(),
        "first_valid_date": str(first_date),
        "last_valid_date": str(last_date)
    }

def load_world_bank_commodities(local_path: str = LOCAL_WB_PATH) -> pd.DataFrame:
    """
    Loads all monthly commodity series from World Bank Pink Sheet.
    Returns DataFrame indexed by monthly PeriodIndex containing:
    ['Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']
    """
    file_path = download_world_bank_if_missing(local_path)
    df_raw = pd.read_excel(file_path, sheet_name='Monthly Prices', header=None)
    
    col_names = df_raw.iloc[3].values
    units_names = df_raw.iloc[4].values
    
    mapping = {
        'Brent': ['crude oil, brent', 'brent'],
        'Natural_Gas': ['natural gas, us', 'natural gas us'],
        'Gold': ['gold'],
        'Copper': ['copper'],
        'Wheat': ['wheat, us hrw', 'wheat, us srw', 'wheat']
    }
    
    col_indices = {}
    for target_key, patterns in mapping.items():
        found_idx = None
        for idx in range(df_raw.shape[1]):
            c_name = str(col_names[idx]) if pd.notna(col_names[idx]) else ""
            c_unit = str(units_names[idx]) if pd.notna(units_names[idx]) else ""
            combined = f"{c_name} {c_unit}".strip().lower()
            if any(p in combined for p in patterns):
                found_idx = idx
                break
        if found_idx is None:
            raise KeyError(f"Could not find column for commodity {target_key}")
        col_indices[target_key] = found_idx

    data_rows = df_raw.iloc[5:].copy()
    data_rows.rename(columns={0: 'Date_Raw'}, inplace=True)
    
    def parse_wb_date(d):
        if pd.isna(d) or not isinstance(d, str):
            return pd.NaT
        d = d.strip()
        if len(d) == 7 and 'M' in d:
            parts = d.split('M')
            return pd.Period(f"{parts[0]}-{parts[1]}", freq='M')
        return pd.NaT

    data_rows['Period'] = data_rows['Date_Raw'].apply(parse_wb_date)
    valid_df = data_rows.dropna(subset=['Period']).copy()
    
    out_df = pd.DataFrame({'Period': valid_df['Period']})
    for col_name, idx in col_indices.items():
        out_df[col_name] = pd.to_numeric(valid_df[idx], errors='coerce')
        
    out_df = out_df.sort_values('Period').drop_duplicates(subset=['Period'])
    out_df = out_df.set_index('Period')
    return out_df

def load_world_bank_gold(local_path: str = LOCAL_WB_PATH) -> pd.DataFrame:
    """Loads Gold price series from World Bank Pink Sheet. Returns DataFrame indexed by Period with column 'Gold'."""
    # Use the same parsing as load_world_bank_commodities but only extract Gold
    file_path = download_world_bank_if_missing(local_path)
    df_raw = pd.read_excel(file_path, sheet_name='Monthly Prices', header=None)
    col_names = df_raw.iloc[3].values
    units_names = df_raw.iloc[4].values
    
    gold_idx = None
    for idx in range(df_raw.shape[1]):
        c_name = str(col_names[idx]) if pd.notna(col_names[idx]) else ""
        c_unit = str(units_names[idx]) if pd.notna(units_names[idx]) else ""
        combined = f"{c_name} {c_unit}".strip().lower()
        if 'gold' in combined:
            gold_idx = idx
            break
    if gold_idx is None:
        raise KeyError("Could not find Gold column in World Bank Pink Sheet")
    
    data_rows = df_raw.iloc[5:].copy()
    data_rows.rename(columns={0: 'Date_Raw'}, inplace=True)
    
    def parse_wb_date(d):
        if pd.isna(d) or not isinstance(d, str):
            return pd.NaT
        d = d.strip()
        if len(d) == 7 and 'M' in d:
            parts = d.split('M')
            return pd.Period(f"{parts[0]}-{parts[1]}", freq='M')
        return pd.NaT
    
    data_rows['Period'] = data_rows['Date_Raw'].apply(parse_wb_date)
    valid_df = data_rows.dropna(subset=['Period']).copy()
    
    out_df = pd.DataFrame({'Period': valid_df['Period']})
    out_df['Gold'] = pd.to_numeric(valid_df[gold_idx], errors='coerce')
    out_df = out_df.sort_values('Period').drop_duplicates(subset=['Period']).set_index('Period')
    return out_df
