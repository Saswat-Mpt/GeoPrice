import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.analysis.conflicts import map_conflict_reference_cases

def test_conflict_mapping_anchored_to_systematic_shocks():
    episodes_path = "data/processed/shock_episodes.csv"
    responses_path = "data/processed/shock_responses.csv"
    
    if os.path.exists(episodes_path) and os.path.exists(responses_path):
        episodes_df = pd.read_csv(episodes_path)
        responses_df = pd.read_csv(responses_path)
        
        cases_df, summary_df = map_conflict_reference_cases(episodes_df, responses_df)
        assert len(cases_df) > 0
        
        # Verify every mapped conflict date exists in systematic shock episodes
        valid_dates = set(episodes_df['representative_shock_date'].tolist())
        for _, row in cases_df.iterrows():
            assert row['representative_shock_date'] in valid_dates
            assert pd.notna(row['source'])
            assert 'WWI' not in row['conflict_name'] and 'WWII' not in row['conflict_name']
