# GeoPrice — Geopolitical Risk-Aware Commodity Outlook System

GeoPrice is a classical machine-learning and empirical event-study system designed to estimate potential monthly commodity price movements (**Brent Oil, Natural Gas, Gold, Copper, Wheat**) under varying geopolitical risk regimes. The project combines historical analogue analysis, systematic shock detection, threat vs. act differentiation, and regularized time-series ML forecasting — presented as separate pieces of evidence rather than a single collapsed prediction.

---

## 🏗️ Master 13-Stage Build Architecture

```text
PHASE 1: Data Collection, Alignment & Signature Feature Engineering
├── Stage 1: Data Collection & Alignment (Caldara-Iacoviello GPR, Pink Sheet Commodities, FRED DTWEXBGS DXY)
└── Stage 2: Signature Feature Engineering (11 features per commodity; release-aware availability rules documented and validated)

PHASE 2: Descriptive Geopolitical Analysis & Historical Analogue
├── Stage 3: Systematic GPR Shock Analysis (Top-decile ΔGPR ≥ 37.49, 21 non-overlapping episodes)
├── Stage 4: Threats vs. Acts Analysis (GPRT ≥ 46.42 vs. GPRA ≥ 37.20 independent shock responses)
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
# Refresh data pipeline & run full validation
python scripts/update_data.py

# Retrain and export production .joblib model artifacts
python scripts/retrain_models.py
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

### 4. Execute Full Automated Unit Test Suite (61 Tests)
```bash
python -m pytest tests/ -v
```

---

## 📈 Key Out-of-Sample Performance Summary (2010–2026, N=198 per commodity)

| Commodity | Naive MAE | Baseline MAE | GeoPrice MAE | Naive DA | Baseline DA | GeoPrice DA | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Brent** | 6.74% | **6.60%** | 6.69% | N/A | **54.8%** | 53.8% | Price history dominates short-term energy returns. |
| **Natural Gas** | 11.34% | **11.42%** | 11.75% | N/A | 47.7% | 47.7% | High volatility; GPR features add variance without error reduction. |
| **Gold** | 2.84% | 2.86% | **2.84%** | N/A | 53.8% | **53.8%** | **Full feature set reduced Gold's out-of-sample MAE by 0.61%.** |
| **Copper** | 3.64% | **3.59%** | 3.68% | N/A | **54.3%** | 53.8% | Industrial demand dynamics dominate short-term returns. |
| **Wheat** | 4.99% | **5.11%** | 5.29% | N/A | 50.5% | **51.0%** | GeoPrice improves Directional Accuracy (+0.5 pts). |

*Note: Naive zero-return model predicts no direction, so Naive Directional Accuracy is marked as N/A.*

---

## 📚 Data Source Citations & Disclaimers

1. **Geopolitical Risk Index (GPR/GPRT/GPRA)**: Caldara, Dario, and Matteo Iacoviello (2022), "Measuring Geopolitical Risk," *American Economic Review*, 112(4), 1194–1225.
2. **Commodity Prices**: World Bank Commodity Price Data ("Pink Sheet") for Gold (`Col 69`); FRED / World Bank Primary Commodity Price series for Brent, Natural Gas, Copper, Wheat (`1960`–`2026`).
3. **Trade-Weighted U.S. Dollar Index (DXY)**: Federal Reserve Bank of St. Louis (FRED) `DTWEXBGS` daily series resampled via monthly arithmetic mean.

*Disclaimer: GeoPrice provides model-based forecasts and historical context for analytical purposes only. It is not investment advice and does not predict geopolitical events.*
