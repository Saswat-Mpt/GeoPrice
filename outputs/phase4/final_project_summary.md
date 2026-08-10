# GeoPrice — Final Project Summary & Master Architecture Report

## 1. Executive Summary
**GeoPrice** is a classical machine-learning and empirical event-study system designed to evaluate next-month commodity returns under varying geopolitical risk regimes. The project covers five primary commodity channels (**Brent Oil, Natural Gas, Gold, Copper, Wheat**) across four core phases spanning **13 distinct stages**.

---

## 2. System Architecture & Methodology

### Phase 1 — Data Collection, Alignment & Feature Engineering (Stages 1-2)
- **Data Ingestion**: Official Caldara-Iacoviello GPR index and subindices (GPRT, GPRA), FRED commodity series (Brent `POILBREUSDM`, Natural Gas `PNGASUSUSDM`, Copper `PCOPPUSDM`, Wheat `PWHEAMTUSDM`), World Bank Pink Sheet (Gold), and daily FRED DXY (`DTWEXBGS`) aggregated via monthly arithmetic mean.
- **Canonical Timeline**: 800 monthly observations (`1960-01` to `2026-08`).
- **Feature Set (11 features per commodity)**:
  - **Commodity History (4)**: `return_1m`, `return_3m`, `return_6m`, `vol_3m`
  - **Geopolitical Risk (6)**: `GPR`, `GPR_change`, `GPR_lag1`, `GPR_lag3`, `GPRT`, `GPRA`
  - **Macro Control (1)**: `DXY`
- **Temporal Leakage Controls**: Release-aware availability rules documented and validated; lagged predictor variables ($t$) ensure no target variable leakage ($t+1$).

### Phase 2 — Descriptive Geopolitical Analysis & Historical Analogue (Stages 3-6)
- **Stage 3 (GPR Shocks)**: Top-decile positive GPR change ($\Delta GPR \ge 36.14$) with 3-month overlap collapsing. Forward returns (+1M, +2M, +3M) calculated.
- **Stage 4 (Threats vs Acts)**: Threat shocks ($GPRT \ge 46.59$) vs Act shocks ($GPRA \ge 36.33$). Realized acts exhibited distinct post-shock price response dynamics across commodity classes.
- **Stage 5 (GPR Regimes & Analogue)**: Empirical level boundaries ($P_{50} = 91.2$, $P_{75} = 113.4$, $P_{90} = 146.1$). Identified current common commodity cutoff state (`2026-06` GPR 179.72, 96.1th percentile -> EXTREME regime) and representative historical analogue set.
- **Stage 6 (Major Conflict Reference Cases)**: 4 documented major historical conflict/crisis cases (9/11 Attacks, 2003 Iraq Invasion, 2014 Crimea Crisis, 2022 Russia-Ukraine Invasion) anchored strictly to systematic Stage 3 shock dates.

### Phase 3 — Classical ML Forecasting & Validation (Stages 7-9)
- **Stage 7 (Baseline Model)**: Price-history ElasticNet Baseline (4 features) using expanding-window time-series CV (`2006-2026`, N_OOS = 197-198 per commodity).
- **Stage 8 (GeoPrice Model)**: Full GeoPrice Model (11 features) under identical pipeline (StandardScaler -> ElasticNet), dates, targets (next-month return), and evaluation metrics.
- **Stage 9 (Final Evaluation & Validation)**: Geopolitical features did not provide consistent incremental forecasting value across all commodities. GeoPrice produced small MAE improvements for Gold (2.838% vs 2.856% Baseline) and Wheat (5.300% vs 5.302% Baseline), while the price-history baseline remained superior for Brent, Natural Gas, and Copper.

### Phase 4 — Production Inference, Interpretability & Dashboard (Stages 10-13)
- **Stage 10 (Production Pipeline & Inference)**: Production `.joblib` model artifacts exported to `models/`. Fast inference pipeline.
- **Stage 11 (Scenario Explorer)**: Manual historical lookup mode for GPR regimes and major conflict references. Guaranteed zero ML model calls.
- **Stage 12 (Coefficient Interpretation)**: Exact prediction explanation via standardized feature contributions (Contribution = $\beta \times z$). Passed 100% exact prediction reconstruction checks (Prediction == $\beta_0 + \sum \beta \times z$).
- **Stage 13 (Final Three-Page Streamlit Dashboard)**: Interactive Streamlit web application (`app.py` + `pages/` 3-page navigation).

---

## 3. Final Master Validation Checkpoint

- **Phase 1 Status**: `VALIDATED` (Stages 1-2)
- **Phase 2 Status**: `VALIDATED` (Stages 3-6)
- **Phase 3 Status**: `VALIDATED` (Stages 7-9)
- **Phase 4 Status**: `VALIDATED` (Stages 10-13)
- **Automated Unit Test Suite**: `77/77 PASSED` (100% pass rate)

---

## 4. Final System Status

**PHASE 4: COMPLETE**

**GEOPRICE PROJECT: COMPLETE**
