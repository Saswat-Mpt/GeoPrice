# GeoPrice — Phase 3 Summary & Final Model Evaluation

## 1. Executive Summary
Phase 3 evaluated out-of-sample monthly commodity return predictions using **expanding-window time-series cross-validation** (2006–2026 dataset; first OOS evaluation beginning in 2010 after the 48-month minimum training window). Hyperparameters were recalibrated annually via inner `TimeSeriesSplit` cross-validation, while model refitting and return forecasting were executed monthly. The price-history ElasticNet Baseline (4 features) was compared against the canonical **GeoPrice Model** (11 features: 4 commodity history + 6 GPR + 1 DXY macro control), as well as a HistGradientBoosting (HGB) nonlinear candidate.

## 2. Model Performance & Incremental Value

### Out-of-Sample Metrics Comparison

| Commodity | Baseline MAE | GeoPrice MAE | MAE Improvement | Baseline DA | GeoPrice DA | DA Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brent** | 6.24% | 6.30% | -0.98% | 55.3% | 54.3% | -1.0 pts |
| **Natural_Gas** | 9.34% | 9.63% | -3.15% | 49.7% | 50.3% | +0.5 pts |
| **Gold** | 2.86% | 2.84% | +0.61% | 53.8% | 53.8% | +0.0 pts |
| **Copper** | 3.59% | 3.67% | -2.25% | 54.8% | 55.3% | +0.5 pts |
| **Wheat** | 5.30% | 5.30% | +0.04% | 55.8% | 56.3% | +0.5 pts |

## 3. Key Data-Driven & Methodological Conclusions
1. **Gold**: Gold showed a marginal reduction in out-of-sample MAE relative to the baseline (2.84% vs 2.86%), but the difference is minor (0.02 percentage points) and should not be interpreted as strong evidence of incremental predictive power.
2. **Wheat**: Wheat produced nearly identical MAE for Baseline (5.30%) and GeoPrice (5.30%), with a marginal directional accuracy difference (+0.5 pts).
3. **Commodity-Dependent Sensitivity**: Price history dominates short-term return error magnitudes for Brent Oil, Natural Gas, and Copper. Geopolitical features add variance without consistently reducing MAE across energy and industrial metals.
4. **Final Production Model Selection**: ElasticNet was retained as the final forecasting framework. HistGradientBoosting (HGB) was evaluated as a nonlinear alternative but did not outperform tuned ElasticNet under walk-forward validation.
5. **Regime Robustness**: Forecast error levels were naturally higher during elevated GPR regimes across commodities due to heightened market volatility.

## 4. Phase 3 Status
**PHASE 3 COMPLETE — ALL STAGES VALIDATED.**
