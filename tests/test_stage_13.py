import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.shock_responses import COMMODITIES

def test_dashboard_files_exist():
    required_files = [
        "app.py",
        "pages/1_Market_Overview.py",
        "pages/2_Shock_Regime_Analysis.py",
        "pages/3_Outlook.py"
    ]
    for fpath in required_files:
        assert os.path.exists(fpath), f"Dashboard file '{fpath}' missing!"

def test_final_project_summary_exists():
    sum_path = "outputs/phase4/final_project_summary.md"
    assert os.path.exists(sum_path)
