import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.data.gpr import load_gpr
from geoprice.data.world_bank import inspect_world_bank_gold, load_world_bank_commodities
from geoprice.data.fred import aggregate_dxy_monthly
from geoprice.data.alignment import align_datasets, validate_aligned_data, REQUIRED_COLUMNS

def test_gpr_loading():
    """Verify GPR loader returns valid DataFrame with required variables GPR, GPRT, GPRA."""
    df_gpr = load_gpr()
    assert isinstance(df_gpr, pd.DataFrame)
    assert isinstance(df_gpr.index, pd.PeriodIndex)
    for var in ['GPR', 'GPRT', 'GPRA']:
        assert var in df_gpr.columns
        assert pd.api.types.is_numeric_dtype(df_gpr[var])
        assert df_gpr[var].isna().sum() == 0, f"Unexpected nulls found in GPR variable {var}"
    assert df_gpr.index.min() <= pd.Period('1985-01', freq='M')

def test_world_bank_gold_inspection():
    """Verify Gold workbook inspection extracts correct metadata."""
    gold_info = inspect_world_bank_gold()
    assert gold_info['sheet_name'] == 'Monthly Prices'
    assert 'Gold' in gold_info['gold_col_name'] or 'gold' in str(gold_info['gold_unit']).lower()
    assert gold_info['valid_gold_obs'] > 500
    assert gold_info['first_valid_date'] <= '1985-01'

def test_world_bank_commodities_loading():
    """Verify World Bank commodity loader returns all 5 required commodity series."""
    df_comm = load_world_bank_commodities()
    assert isinstance(df_comm, pd.DataFrame)
    assert isinstance(df_comm.index, pd.PeriodIndex)
    expected_cols = ['Brent', 'Natural_Gas', 'Gold', 'Copper', 'Wheat']
    for c in expected_cols:
        assert c in df_comm.columns
        assert pd.api.types.is_numeric_dtype(df_comm[c])
        assert df_comm[c].notna().sum() > 400

def test_dxy_monthly_arithmetic_mean_aggregation():
    """Verify DXY daily to monthly conversion computes exact arithmetic mean, not last/first."""
    dates = pd.date_range(start='2026-01-01', end='2026-01-31', freq='D')
    values = np.linspace(100.0, 110.0, len(dates))
    df_daily = pd.DataFrame({'Date': dates, 'DTWEXBGS': values})
    
    df_monthly = aggregate_dxy_monthly(df_daily)
    assert len(df_monthly) == 1
    assert df_monthly.index[0] == pd.Period('2026-01', freq='M')
    expected_mean = values.mean()
    actual_mean = df_monthly.iloc[0]['DXY']
    assert np.isclose(actual_mean, expected_mean), f"Expected mean {expected_mean}, got {actual_mean}"
    assert not np.isclose(actual_mean, values[-1]), "DXY used last() instead of arithmetic mean!"

def test_alignment_and_validation():
    """Verify full Stage 1 alignment, output column names, and validation checks."""
    aligned_df, metrics = align_datasets()
    assert isinstance(aligned_df, pd.DataFrame)
    assert list(aligned_df.columns) == REQUIRED_COLUMNS
    assert len(aligned_df) > 400
    
    val_results = validate_aligned_data(aligned_df)
    assert bool(val_results['no_duplicate_dates']) is True
    assert bool(val_results['chronological_order']) is True
    assert bool(val_results['all_required_columns_exist']) is True
    assert bool(val_results['all_series_numeric']) is True
    assert bool(val_results['no_date_gaps']) is True
    assert bool(val_results['overall_pass']) is True

def test_dxy_missingness_preserves_pre_2006_history():
    """Verify pre-2006 rows are preserved even when DXY is missing (not dropped)."""
    aligned_df, _ = align_datasets()
    pre_2001_rows = aligned_df[aligned_df['Date'] < '2001-01']
    assert len(pre_2001_rows) > 0, "Pre-2001 rows were mistakenly dropped!"
    assert pre_2001_rows['DXY'].isna().all(), "DXY before start date should be NaN!"
    
    # GPR starts in 1985-01, check active GPR window pre-2001
    gpr_active_pre_2001 = aligned_df[(aligned_df['Date'] >= '1985-01') & (aligned_df['Date'] < '2001-01')]
    assert gpr_active_pre_2001['GPR'].notna().all(), "GPR should be valid during 1985-2001!"
    assert gpr_active_pre_2001['Brent'].notna().all(), "Brent should be valid pre-2001!"
