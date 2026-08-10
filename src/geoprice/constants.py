"""
Centralized constants for the GeoPrice package.
Defines canonical commodity lists, feature groups, and model hyperparameter defaults.
"""

COMMODITIES = (
    "Brent",
    "Natural_Gas",
    "Gold",
    "Copper",
    "Wheat",
)

COMMODITY_HISTORY_FEATURES = (
    "return_1m",
    "return_3m",
    "return_6m",
    "vol_3m",
)

GEOPOLITICAL_FEATURES = (
    "GPR",
    "GPR_change",
    "GPR_lag1",
    "GPR_lag3",
    "GPRT",
    "GPRA",
)

GEOPOLITICAL_Z12_FEATURE = "GPR_z12"

MACRO_FEATURES = (
    "DXY",
)

FORECAST_HORIZON = 1
MIN_TRAIN_MONTHS = 48
GPR_SHOCK_QUANTILE = 0.90

# Hyperparameter Tuning Candidate Grids
ALPHA_GRID = (0.0005, 0.001, 0.003, 0.01, 0.03, 0.1)
L1_RATIO_GRID = (0.1, 0.5, 0.9)
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
