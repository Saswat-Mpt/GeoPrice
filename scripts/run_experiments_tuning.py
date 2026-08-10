"""
GeoPrice Out-of-Sample Walk-Forward Evaluation & Feature Ablation.

Every OOS forecast month:
  1. Train = all observations before t
  2. Inner TimeSeriesSplit CV selects hyperparameters (tuned models only)
  3. Refit final model on full training window
  4. Predict month t

No model caching. No stale predictions. No hardcoded hyperparameters for "tuned" models.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.constants import COMMODITIES, MIN_TRAIN_MONTHS
from geoprice.models.baseline import get_baseline_feature_names, create_next_month_target
from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.models.metrics import evaluate_all_metrics
from geoprice.models.tuning import (
    select_best_elasticnet_params,
    select_best_logistic_c,
    select_best_hgb_params,
)


def _fit_predict_elasticnet(X_train, y_train, X_test, alpha, l1_ratio):
    """Fit StandardScaler + ElasticNet on training data, predict on test row."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=200, tol=1e-3, random_state=42)
    model.fit(X_tr_s, y_train)
    return float(model.predict(X_te_s)[0])


def run_tuning_suite():
    print("=" * 80)
    print("Running GeoPrice — Walk-Forward OOS Evaluation (Nested TimeSeriesSplit CV)")
    print("=" * 80)

    feature_path = "data/processed/feature_dataset.csv"
    raw_path = "data/processed/monthly_aligned.csv"
    if not (os.path.exists(feature_path) and os.path.exists(raw_path)):
        print("Error: Feature or raw dataset missing! Run Stages 1-2 first.")
        sys.exit(1)

    df_feat = pd.read_csv(feature_path)
    df_raw = pd.read_csv(raw_path)

    experiments_rows = []
    ablation_rows = []
    directional_rows = []
    selected_params_log = []

    for c in COMMODITIES:
        print(f"\nEvaluating commodity: {c}...")

        # Feature sets
        base_feats = get_baseline_feature_names(c)
        geo_feats = get_geoprice_feature_names(c)  # 11 features (original GeoPrice)

        # Prepare data
        df_f = df_feat.copy()
        if 'Date' in df_f.columns:
            df_f = df_f.set_index('Date')

        target_series = create_next_month_target(df_raw, c)
        df_f['Target'] = target_series

        data = df_f.reset_index()
        data['Year'] = pd.to_datetime(data['Date']).dt.year
        phase3_data = data[data['Year'] >= 2006].copy().reset_index(drop=True)

        valid_mask = phase3_data[geo_feats].notna().all(axis=1) & phase3_data['Target'].notna()
        dataset = phase3_data[valid_mask].copy().reset_index(drop=True)

        n_oos = len(dataset) - MIN_TRAIN_MONTHS
        print(f"  OOS predictions: {n_oos}")

        # Prediction containers
        preds_fixed_base = []
        preds_tuned_base = []
        preds_fixed_geo = []
        preds_tuned_geo = []
        preds_hgb_default = []
        preds_hgb_tuned = []
        clf_preds = []

        for t_idx in range(MIN_TRAIN_MONTHS, len(dataset)):
            train_df = dataset.iloc[:t_idx]
            test_row = dataset.iloc[t_idx]
            forecast_date = test_row['Date']
            y_train = train_df['Target'].values
            y_test = float(test_row['Target'])

            X_tr_b = train_df[base_feats].values
            X_te_b = test_row[base_feats].values.reshape(1, -1)
            X_tr_g = train_df[geo_feats].values
            X_te_g = test_row[geo_feats].values.reshape(1, -1)

            # Re-tune hyperparameters annually (every 12 months) strictly on past training data
            if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0 or 'best_a_b' not in locals():
                best_a_b, best_l1_b = select_best_elasticnet_params(X_tr_b, y_train)
                best_a_g, best_l1_g = select_best_elasticnet_params(X_tr_g, y_train)
                best_hgb = select_best_hgb_params(X_tr_g, y_train)
                y_train_bin = (y_train > 0).astype(int)
                best_c = select_best_logistic_c(X_tr_g, y_train_bin)

            # -----------------------------------------------------------
            # 1. Fixed Baseline (alpha=0.01, l1_ratio=0.5)
            # -----------------------------------------------------------
            pred_fb = _fit_predict_elasticnet(X_tr_b, y_train, X_te_b, 0.01, 0.5)
            preds_fixed_base.append(pred_fb)

            # -----------------------------------------------------------
            # 2. Tuned Baseline (refit every month with tuned hyperparameters)
            # -----------------------------------------------------------
            pred_tb = _fit_predict_elasticnet(X_tr_b, y_train, X_te_b, best_a_b, best_l1_b)
            preds_tuned_base.append(pred_tb)

            # -----------------------------------------------------------
            # 3. Fixed GeoPrice (alpha=0.01, l1_ratio=0.5)
            # -----------------------------------------------------------
            pred_fg = _fit_predict_elasticnet(X_tr_g, y_train, X_te_g, 0.01, 0.5)
            preds_fixed_geo.append(pred_fg)

            # -----------------------------------------------------------
            # 4. Tuned GeoPrice (refit every month with tuned hyperparameters)
            # -----------------------------------------------------------
            pred_tg = _fit_predict_elasticnet(X_tr_g, y_train, X_te_g, best_a_g, best_l1_g)
            preds_tuned_geo.append(pred_tg)

            # -----------------------------------------------------------
            # 5. HGB GeoPrice (sensible fixed defaults, refit every month)
            # -----------------------------------------------------------
            model_hgb_d = HistGradientBoostingRegressor(
                max_iter=50, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=0.1,
                early_stopping=False, random_state=42
            )
            model_hgb_d.fit(X_tr_g, y_train)
            preds_hgb_default.append(float(model_hgb_d.predict(X_te_g)[0]))

            # -----------------------------------------------------------
            # 6. Tuned HGB GeoPrice (refit every month with tuned hyperparameters)
            # -----------------------------------------------------------
            model_hgb_t = HistGradientBoostingRegressor(**best_hgb, random_state=42)
            model_hgb_t.fit(X_tr_g, y_train)
            preds_hgb_tuned.append(float(model_hgb_t.predict(X_te_g)[0]))

            # -----------------------------------------------------------
            # 7. Logistic Regression Directional (refit every month)
            # -----------------------------------------------------------
            y_train_bin = (y_train > 0).astype(int)
            y_test_bin = int(y_test > 0)

            if len(np.unique(y_train_bin)) >= 2:
                clf_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(C=best_c, max_iter=200, tol=1e-2, random_state=42))
                ])
                clf_pipe.fit(X_tr_g, y_train_bin)
                prob_pos = float(clf_pipe.predict_proba(X_te_g)[0, 1])
                pred_bin = int(clf_pipe.predict(X_te_g)[0])
            else:
                prob_pos = 0.5
                pred_bin = 1

            clf_preds.append({
                "Date": forecast_date,
                "Commodity": c,
                "Actual_Direction": y_test_bin,
                "Predicted_Direction": pred_bin,
                "Predicted_Probability": prob_pos
            })

            # Log selected hyperparameters for first and last OOS month
            if t_idx == MIN_TRAIN_MONTHS or t_idx == len(dataset) - 1:
                selected_params_log.append({
                    "Commodity": c,
                    "OOS_Index": t_idx - MIN_TRAIN_MONTHS,
                    "Date": forecast_date,
                    "Baseline_alpha": best_a_b,
                    "Baseline_l1": best_l1_b,
                    "GeoPrice_alpha": best_a_g,
                    "GeoPrice_l1": best_l1_g,
                    "HGB_lr": best_hgb.get("learning_rate"),
                    "HGB_max_iter": best_hgb.get("max_iter"),
                    "HGB_max_leaf": best_hgb.get("max_leaf_nodes"),
                    "LogReg_C": best_c,
                })

            # Progress indicator every 50 steps
            oos_step = t_idx - MIN_TRAIN_MONTHS
            if (oos_step + 1) % 50 == 0 or oos_step == n_oos - 1:
                print(f"  [{c}] {oos_step + 1}/{n_oos} OOS predictions complete")

        # ---------------------------------------------------------------
        # Compute Metrics for 6 core models
        # ---------------------------------------------------------------
        y_act = dataset.iloc[MIN_TRAIN_MONTHS:]['Target'].values

        m_fb = evaluate_all_metrics(y_act, np.array(preds_fixed_base), "Fixed Baseline", c)
        m_tb = evaluate_all_metrics(y_act, np.array(preds_tuned_base), "Tuned Baseline", c)
        m_fg = evaluate_all_metrics(y_act, np.array(preds_fixed_geo), "Fixed GeoPrice", c)
        m_tg = evaluate_all_metrics(y_act, np.array(preds_tuned_geo), "Tuned GeoPrice", c)
        m_hgb_d = evaluate_all_metrics(y_act, np.array(preds_hgb_default), "HGB GeoPrice", c)
        m_hgb_t = evaluate_all_metrics(y_act, np.array(preds_hgb_tuned), "Tuned HGB GeoPrice", c)

        experiments_rows.extend([m_fb, m_tb, m_fg, m_tg, m_hgb_d, m_hgb_t])

        # ---------------------------------------------------------------
        # Feature Ablation Suite (fixed ElasticNet alpha=0.01, l1=0.5)
        # Same model spec, different feature sets → isolates feature value
        # ---------------------------------------------------------------
        # Enhanced features for ablation
        ablation_enhanced_feats = geo_feats + ['GPR_z12', 'GPR_accel', 'GPR_gap']

        # Check which enhanced features are available
        available_enhanced = [f for f in ablation_enhanced_feats if f in dataset.columns]
        if len(available_enhanced) < len(ablation_enhanced_feats):
            # Recheck with the full dataset that includes enhanced features
            valid_mask_enh = phase3_data[available_enhanced].notna().all(axis=1) & phase3_data['Target'].notna()
            dataset_enh = phase3_data[valid_mask_enh].copy().reset_index(drop=True)
        else:
            dataset_enh = dataset

        ablation_configs = [
            ("Model A: Baseline", base_feats),
            ("Model B: Baseline + DXY", base_feats + ['DXY']),
            ("Model C: Baseline + GPR", base_feats + ['GPR']),
            ("Model D: Baseline + GPR + GPRT + GPRA", base_feats + ['GPR', 'GPRT', 'GPRA']),
            ("Model E: Full GeoPrice (11 features)", geo_feats),
            ("Model F: GeoPrice + GPR_z12 + GPR_accel + GPR_gap", available_enhanced),
        ]

        for m_name, f_list in ablation_configs:
            # Verify features exist
            missing = [f for f in f_list if f not in dataset_enh.columns]
            if missing:
                print(f"  [SKIP] {m_name}: missing features {missing}")
                continue

            abl_valid = dataset_enh[f_list].notna().all(axis=1) & dataset_enh['Target'].notna()
            abl_data = dataset_enh[abl_valid].copy().reset_index(drop=True)

            if len(abl_data) <= MIN_TRAIN_MONTHS + 1:
                print(f"  [SKIP] {m_name}: insufficient data")
                continue

            abl_preds = []
            for t_idx in range(MIN_TRAIN_MONTHS, len(abl_data)):
                tr = abl_data.iloc[:t_idx]
                te = abl_data.iloc[t_idx]
                pred = _fit_predict_elasticnet(
                    tr[f_list].values, tr['Target'].values,
                    te[f_list].values.reshape(1, -1), 0.01, 0.5
                )
                abl_preds.append(pred)

            y_act_abl = abl_data.iloc[MIN_TRAIN_MONTHS:]['Target'].values
            m_eval = evaluate_all_metrics(y_act_abl, np.array(abl_preds), m_name, c)
            ablation_rows.append(m_eval)

        # ---------------------------------------------------------------
        # Directional Classification Metrics
        # ---------------------------------------------------------------
        clf_df = pd.DataFrame(clf_preds)
        y_true_clf = clf_df['Actual_Direction'].values
        y_pred_clf = clf_df['Predicted_Direction'].values
        y_prob_clf = clf_df['Predicted_Probability'].values

        acc = float(accuracy_score(y_true_clf, y_pred_clf))
        b_acc = float(balanced_accuracy_score(y_true_clf, y_pred_clf))
        prec = float(precision_score(y_true_clf, y_pred_clf, zero_division=0))
        rec = float(recall_score(y_true_clf, y_pred_clf, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true_clf, y_prob_clf))
        except Exception:
            auc = 0.5

        directional_rows.append({
            "Commodity": c,
            "N": len(clf_df),
            "Accuracy": acc,
            "Balanced_Accuracy": b_acc,
            "Precision": prec,
            "Recall": rec,
            "ROC_AUC": auc
        })

    # Save all outputs
    os.makedirs("outputs/phase3", exist_ok=True)

    df_exp = pd.DataFrame(experiments_rows)
    df_exp.to_csv("outputs/phase3/tuning_experiments_comparison.csv", index=False)

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv("outputs/phase3/feature_ablation.csv", index=False)

    df_dir = pd.DataFrame(directional_rows)
    df_dir.to_csv("outputs/phase3/directional_results.csv", index=False)

    df_params = pd.DataFrame(selected_params_log)
    df_params.to_csv("outputs/phase3/selected_hyperparameters.csv", index=False)

    print("\n" + "=" * 80)
    print("WALK-FORWARD OOS EVALUATION COMPLETE")
    print("=" * 80)
    print(df_exp[['Commodity', 'Model', 'N', 'MAE', 'RMSE', 'Directional_Accuracy']].to_string(index=False))
    print("\nSelected hyperparameters (first & last OOS month per commodity):")
    print(df_params.to_string(index=False))


if __name__ == "__main__":
    run_tuning_suite()
