# GeoPrice — Geopolitical Risk-Aware Commodity Outlook System

GeoPrice is a classical machine-learning and empirical event-study system designed to estimate potential monthly commodity price movements (**Brent Oil, Natural Gas, Gold, Copper, Wheat**) under varying geopolitical risk regimes. The project combines historical analogue analysis, systematic shock detection, threat vs. act differentiation, and regularized time-series ML forecasting — presented as separate pieces of evidence rather than a single collapsed prediction.

---

## 🏗️ Master 13-Stage Build Architecture

```text
PHASE 1: Data Collection, Alignment & Signature Feature Engineering
├── Stage 1: Data Collection & Alignment (Caldara-Iacoviello GPR, FRED commodity series, World Bank Pink Sheet Gold, FRED DXY)
└── Stage 2: Signature Feature Engineering (11 features per commodity; release-aware availability rule using the published monthly series; full historical vintage reconstruction is not performed.)

PHASE 2: Descriptive Geopolitical Analysis & Historical Analogue
├── Stage 3: Systematic GPR Shock Analysis (Top-decile positive ΔGPR; overlapping shocks collapsed into independent episodes)
├── Stage 4: Threats vs. Acts Analysis (Independent GPRT and GPRA shock-response analysis)
├── Stage 5: Current GPR Regime & Historical Analogue (LOW/MODERATE/HIGH/EXTREME percentile regimes)
└── Stage 6: Major Conflict Reference Cases (Systematic shock mapping: 9/11, 2003 Iraq, 2014 Crimea, 2022 Ukraine)

PHASE 3: Classical Machine-Learning Forecasting & Validation
├── Stage 7: Baseline Commodity Return Forecasting (Price-history-only ElasticNet, expanding-window OOS CV)
├── Stage 8: GeoPrice Model Forecasting (11 features: 4 commodity history + 6 GPR + 1 DXY control)
└── Stage 9: Final Model Evaluation & Validation (MAE/RMSE/DA comparison, GPR-only ablation, regime robustness)

PHASE 4: Production Pipeline, Interpretability & Streamlit Application
├── Stage 10: Production Pipeline & Inference Stack (Exported .joblib artifacts & model_metadata.json)
├── Stage 11: Scenario Explorer — Manual Mode (Strict non-ML historical regime & conflict lookup)
├── Stage 12: Coefficient Interpretation & Current Forecast Explanation (Exact β × z contribution breakdown)
└── Stage 13: Final Three-Page Streamlit Dashboard (Market Overview, Shock & Regime Analysis, Outlook & Scenarios)
```

---

## ⚡ Quick Start & Run Commands

### 1. Launch Interactive Streamlit Dashboard (3-Page Application)
```bash
streamlit run app.py
```

### 2. Update Data & Retrain Models
```bash
# Full pipeline refresh:
python scripts/update_data.py        # Stages 1-9
python scripts/retrain_models.py      # Retrain .joblib artifacts
python scripts/run_stage_10.py        # Production inference
python scripts/run_stage_11.py        # Scenario lookup tables
python scripts/run_stage_12.py        # Coefficient interpretation
python scripts/run_stage_13.py        # Dashboard validation & final report
```

### 3. Run Stage Runner Scripts
```bash
python scripts/run_stage_01_data.py
python scripts/run_stage_02_features.py
python scripts/run_stage_03_shocks.py
python scripts/run_stage_04_threats_acts.py
python scripts/run_stage_05_regimes.py
python scripts/run_stage_06_conflicts.py
python scripts/run_stage_07_baseline.py
python scripts/run_stage_08_geoprice.py
python scripts/run_stage_09_evaluation.py
python scripts/run_stage_10.py
python scripts/run_stage_11.py
python scripts/run_stage_12.py
python scripts/run_stage_13.py
```

### 4. Execute Full Automated Unit Test Suite
```bash
python -m pytest tests/ -v
```

---

## 📈 Key Out-of-Sample Performance Summary (Training: 2006+, OOS evaluation: ~2010-2026, N=197-198 per commodity)

| Commodity | Naive MAE | Baseline MAE | GeoPrice MAE | Naive DA | Baseline DA | GeoPrice DA | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Brent** | 6.27% | **6.24%** | 6.30% | N/A | **55.3%** | 54.3% | Price history dominates short-term energy returns. |
| **Natural Gas** | **9.29%** | 9.34% | 9.63% | N/A | 49.7% | **50.3%** | High volatility; GPR features add variance without error reduction. |
| **Gold** | 2.84% | 2.86% | **2.84%** | N/A | **53.8%** | **53.8%** | **Full feature set reduced Gold's out-of-sample MAE by 0.61% vs baseline.** |
| **Copper** | 3.65% | **3.59%** | 3.67% | N/A | 54.8% | **55.3%** | Industrial demand dynamics dominate short-term returns. |
| **Wheat** | **5.28%** | 5.30% | 5.30% | N/A | 55.8% | **56.3%** | GeoPrice improves Directional Accuracy (+0.5 pts) and slightly improves MAE. |

*Note: Naive zero-return model predicts no direction, so Naive Directional Accuracy is marked as N/A.*

### Conclusion on Forecasting Utility
Geopolitical features did not provide consistent incremental forecasting value across all commodities. GeoPrice produced small MAE improvements for Gold and Wheat, while the price-history baseline remained superior for Brent, Natural Gas, and Copper.

---

## 📚 Data Source Citations & Disclaimers

1. **Geopolitical Risk Index (GPR/GPRT/GPRA)**: Caldara, Dario, and Matteo Iacoviello (2022), "Measuring Geopolitical Risk," *American Economic Review*, 112(4), 1194–1225.
2. **Commodity Prices**: FRED (Federal Reserve Bank of St. Louis) for Brent (`POILBREUSDM`), Natural Gas (`PNGASUSUSDM`), Copper (`PCOPPUSDM`), Wheat (`PWHEAMTUSDM`); World Bank Pink Sheet for Gold.
3. **Trade-Weighted U.S. Dollar Index (DXY)**: Federal Reserve Bank of St. Louis (FRED) `DTWEXBGS`.

| Asset | Source | Series ID |
| :--- | :--- | :--- |
| Brent | FRED | POILBREUSDM |
| Natural Gas | FRED | PNGASUSUSDM |
| Gold | World Bank Pink Sheet | Monthly Prices (Col 69) |
| Copper | FRED | PCOPPUSDM |
| Wheat | FRED | PWHEAMTUSDM |
| DXY | FRED | DTWEXBGS |

*Note on Raw Data Licensing: Raw source files are included in the repository for reproducibility; users should respect the licensing/usage terms of each upstream provider.*

*Disclaimer: GeoPrice provides model-based forecasts and historical context for analytical purposes only. It is not investment advice and does not predict geopolitical events.*
