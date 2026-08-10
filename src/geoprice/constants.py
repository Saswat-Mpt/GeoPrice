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

MACRO_FEATURES = (
    "DXY",
)

FORECAST_HORIZON = 1
MIN_TRAIN_MONTHS = 48
GPR_SHOCK_QUANTILE = 0.90
