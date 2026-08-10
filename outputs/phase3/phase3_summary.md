# GeoPrice — Phase 3 Summary & Final Model Evaluation

## 1. Executive Summary
Phase 3 evaluated out-of-sample monthly commodity return predictions using **expanding-window time-series cross-validation** (2006–2026 dataset; first OOS evaluation beginning in 2010 after the 48-month minimum training window). Hyperparameters were recalibrated annually via inner `TimeSeriesSplit` cross-validation, while model refitting and return forecasting were executed monthly. The price-history ElasticNet Baseline (4 features) was compared against the canonical **GeoPrice Model** (11 features: 4 commodity history + 6 GPR + 1 DXY macro control), as well as a HistGradientBoosting (HGB) nonlinear candidate.

## 2. Model Performance & Incremental Value

### Out-of-Sample Metrics Comparison

| Commodity | Baseline MAE | GeoPrice MAE | MAE Improvement | Baseline DA | GeoPrice DA | DA Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brent** | 6.34% | 6.29% | +0.89% | 53.8% | 55.8% | +2.0 pts |
| **Natural_Gas** | 9.30% | 9.31% | -0.05% | 52.8% | 51.3% | -1.5 pts |
| **Gold** | 2.85% | 2.85% | +0.08% | 53.8% | 53.8% | +0.0 pts |
| **Copper** | 3.58% | 3.66% | -2.11% | 55.8% | 53.8% | -2.0 pts |
| **Wheat** | 5.27% | 5.37% | -1.88% | 56.3% | 49.2% | -7.1 pts |

## 3. Key Data-Driven & Methodological Conclusions
1. **Gold**: GeoPrice produced a marginally lower OOS MAE than the tuned baseline (2.852% vs 2.854%), but the difference was extremely small (0.002 percentage points) and not statistically significant under paired bootstrap error analysis.
2. **Brent Oil**: Brent showed the clearest directional improvement, with GeoPrice increasing OOS directional accuracy by 2.0 percentage points (55.8% vs 53.8%) over the tuned baseline, though the error difference is within sampling noise bounds.
3. **Commodity-Dependent Sensitivity**: Price history dominates short-term return error magnitudes for Natural Gas, Copper, and Wheat. Geopolitical risk features add variance without reducing error magnitudes for agricultural and regional gas commodities.
4. **Final Production Model Selection**: ElasticNet was retained as the final production forecasting framework (`models/*.joblib`). HistGradientBoosting (HGB) was evaluated as a nonlinear alternative but did not outperform tuned ElasticNet under walk-forward validation.
5. **Regime Robustness**: Forecast error levels were naturally higher during elevated GPR regimes across commodities due to heightened market volatility.

## 4. Phase 3 Status
**PHASE 3 COMPLETE — ALL STAGES VALIDATED.**
