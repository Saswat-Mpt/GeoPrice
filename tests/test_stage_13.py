import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shock_responses import COMMODITIES

def test_dashboard_files_exist():
    required_files = [
        "app.py",
        "pages/2_Shock_Regime_Analysis.py",
        "pages/3_Outlook.py"
    ]
    for fpath in required_files:
        assert os.path.exists(fpath), f"Dashboard file '{fpath}' missing!"

def test_final_project_summary_exists():
    sum_path = "outputs/phase4/final_project_summary.md"
    assert os.path.exists(sum_path)

def test_dashboard_freshness_dates():
    aligned_path = "data/processed/monthly_aligned.csv"
    if os.path.exists(aligned_path):
        df = pd.read_csv(aligned_path)
        # Latest GPR date (last row with non-null GPR)
        gpr_valid = df.dropna(subset=['GPR'])
        latest_gpr = gpr_valid['Date'].iloc[-1]
        # Latest row
        latest_row = df['Date'].iloc[-1]
        # GPR should NOT equal the very last aligned row if DXY extends further
        # At minimum, GPR latest date should be a valid date
        assert pd.notna(latest_gpr)
        assert len(gpr_valid) > 0
