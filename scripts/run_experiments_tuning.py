"""
GeoPrice Out-of-Sample Hyperparameter Tuning, Feature Ablation, and Directional Experiments.
Executes Step 2 to Step 10 of the GeoPrice Improvement Specification.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.constants import COMMODITIES, ALPHA_GRID, L1_RATIO_GRID, LOGISTIC_C_GRID, MIN_TRAIN_MONTHS
from geoprice.models.baseline import get_baseline_feature_names, create_next_month_target, build_baseline_pipeline
from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.models.metrics import evaluate_all_metrics
from geoprice.models.tuning import select_best_elasticnet_params, select_best_logistic_c

def run_tuning_suite():
    print("=" * 80)
    print("Running GeoPrice — Out-of-Sample Hyperparameter Tuning & Model Experiments")
    print("=" * 80)

    feature_path = "data/processed/feature_dataset.csv"
    raw_path = "data/processed/monthly_aligned.csv"
    if not (os.path.exists(feature_path) and os.path.exists(raw_path)):
        print("Error: Feature or raw dataset missing! Run Stages 1-2 first.")
        sys.exit(1)

    df_feat = pd.read_csv(feature_path)
    df_raw = pd.read_csv(raw_path)

    # Output storage
    experiments_rows = []
    ablation_rows = []
    directional_rows = []

    # Iterate over all 5 commodities
    for c in COMMODITIES:
        print(f"\nEvaluating commodity: {c}...")
        
        # Prepare commodity data
        base_feats = get_baseline_feature_names(c)
        geo_feats = get_geoprice_feature_names(c)  # 11 features
        all_feats_z12 = geo_feats + ['GPR_z12']

        # Ensure Date set
        df_f = df_feat.copy()
        if 'Date' in df_f.columns:
            df_f = df_f.set_index('Date')

        target_series = create_next_month_target(df_raw, c)
        df_f['Target'] = target_series

        data = df_f.reset_index()
        data['Year'] = pd.to_datetime(data['Date']).dt.year
        phase3_data = data[data['Year'] >= 2006].copy().reset_index(drop=True)

        valid_mask = phase3_data[all_feats_z12].notna().all(axis=1) & phase3_data['Target'].notna()
        dataset = phase3_data[valid_mask].copy().reset_index(drop=True)

        n_oos = len(dataset) - MIN_TRAIN_MONTHS

        # Containers for predictions
        preds_fixed_base = []
        preds_tuned_base = []
        preds_fixed_geo = []
        preds_tuned_geo = []
        preds_tuned_geo_z12 = []

        # Classification container
        clf_preds = []

        # Selected hyperparameter log
        selected_base_hyperparams = []
        selected_geo_hyperparams = []

        for t_idx in range(MIN_TRAIN_MONTHS, len(dataset)):
            train_df = dataset.iloc[:t_idx]
            test_row = dataset.iloc[t_idx]
            forecast_date = test_row['Date']
            y_train = train_df['Target'].values
            y_test = float(test_row['Target'])

            # -------------------------------------------------------------
            # A. Fixed Baseline (alpha=0.01, l1_ratio=0.5)
            # -------------------------------------------------------------
            X_tr_b = train_df[base_feats].values
            X_te_b = test_row[base_feats].values.reshape(1, -1)
            p_fb = build_baseline_pipeline(alpha=0.01, l1_ratio=0.5)
            p_fb.fit(X_tr_b, y_train)
            pred_fb = float(p_fb.predict(X_te_b)[0])
            preds_fixed_base.append(pred_fb)

            # -------------------------------------------------------------
            # B. Tuned Baseline (Inner CV tuning - annual recalibration)
            # -------------------------------------------------------------
            if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0 or len(selected_base_hyperparams) == 0:
                best_a_b, best_l1_b = select_best_elasticnet_params(X_tr_b, y_train)
                curr_base_params = (best_a_b, best_l1_b)
            else:
                best_a_b, best_l1_b = curr_base_params

            selected_base_hyperparams.append((best_a_b, best_l1_b))
            p_tb = build_baseline_pipeline(alpha=best_a_b, l1_ratio=best_l1_b)
            p_tb.fit(X_tr_b, y_train)
            pred_tb = float(p_tb.predict(X_te_b)[0])
            preds_tuned_base.append(pred_tb)

            # -------------------------------------------------------------
            # C. Fixed GeoPrice (11 features, alpha=0.01, l1_ratio=0.5)
            # -------------------------------------------------------------
            X_tr_g = train_df[geo_feats].values
            X_te_g = test_row[geo_feats].values.reshape(1, -1)
            p_fg = build_baseline_pipeline(alpha=0.01, l1_ratio=0.5)
            p_fg.fit(X_tr_g, y_train)
            pred_fg = float(p_fg.predict(X_te_g)[0])
            preds_fixed_geo.append(pred_fg)

            # -------------------------------------------------------------
            # D. Tuned GeoPrice (11 features, Inner CV tuning - annual recalibration)
            # -------------------------------------------------------------
            if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0 or len(selected_geo_hyperparams) == 0:
                best_a_g, best_l1_g = select_best_elasticnet_params(X_tr_g, y_train)
                curr_geo_params = (best_a_g, best_l1_g)
            else:
                best_a_g, best_l1_g = curr_geo_params

            selected_geo_hyperparams.append((best_a_g, best_l1_g))
            p_tg = build_baseline_pipeline(alpha=best_a_g, l1_ratio=best_l1_g)
            p_tg.fit(X_tr_g, y_train)
            pred_tg = float(p_tg.predict(X_te_g)[0])
            preds_tuned_geo.append(pred_tg)

            # -------------------------------------------------------------
            # E. Tuned GeoPrice + GPR_z12 (12 features)
            # -------------------------------------------------------------
            X_tr_z = train_df[all_feats_z12].values
            X_te_z = test_row[all_feats_z12].values.reshape(1, -1)
            if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0 or 'curr_z12_params' not in locals():
                best_a_z, best_l1_z = select_best_elasticnet_params(X_tr_z, y_train)
                curr_z12_params = (best_a_z, best_l1_z)
            else:
                best_a_z, best_l1_z = curr_z12_params

            p_tz = build_baseline_pipeline(alpha=best_a_z, l1_ratio=best_l1_z)
            p_tz.fit(X_tr_z, y_train)
            pred_tz = float(p_tz.predict(X_te_z)[0])
            preds_tuned_geo_z12.append(pred_tz)

            # -------------------------------------------------------------
            # F. Logistic Regression Directional Classification (Target > 0)
            # -------------------------------------------------------------
            y_train_bin = (y_train > 0).astype(int)
            y_test_bin = int(y_test > 0)
            if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0 or 'curr_c_param' not in locals():
                best_c = select_best_logistic_c(X_tr_g, y_train_bin)
                curr_c_param = best_c
            else:
                best_c = curr_c_param

            if len(np.unique(y_train_bin)) >= 2:
                clf_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(C=best_c, max_iter=10000, random_state=42))
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

        # Calculate metrics for 4 core experiments + GPR_z12
        y_act = dataset.iloc[MIN_TRAIN_MONTHS:]['Target'].values

        m_fb = evaluate_all_metrics(y_act, np.array(preds_fixed_base), "Fixed Baseline", c)
        m_tb = evaluate_all_metrics(y_act, np.array(preds_tuned_base), "Tuned Baseline", c)
        m_fg = evaluate_all_metrics(y_act, np.array(preds_fixed_geo), "Fixed GeoPrice", c)
        m_tg = evaluate_all_metrics(y_act, np.array(preds_tuned_geo), "Tuned GeoPrice", c)
        m_tz = evaluate_all_metrics(y_act, np.array(preds_tuned_geo_z12), "Tuned GeoPrice + GPR_z12", c)

        experiments_rows.extend([m_fb, m_tb, m_fg, m_tg, m_tz])

        # Feature Ablation Suite (Step 6)
        # Models 1 to 6
        ablation_configs = [
            ("Model 1: Baseline", base_feats),
            ("Model 2: Baseline + DXY", base_feats + ['DXY']),
            ("Model 3: Baseline + GPR", base_feats + ['GPR']),
            ("Model 4: Baseline + GPR + GPRT + GPRA", base_feats + ['GPR', 'GPRT', 'GPRA']),
            ("Model 5: Full GeoPrice", geo_feats),
            ("Model 6: Full GeoPrice + GPR_z12", all_feats_z12),
        ]

        for m_name, f_list in ablation_configs:
            m_preds = []
            curr_ab_params = (0.01, 0.5)
            for t_idx in range(MIN_TRAIN_MONTHS, len(dataset)):
                tr = dataset.iloc[:t_idx]
                te = dataset.iloc[t_idx]
                X_tr = tr[f_list].values
                X_te = te[f_list].values.reshape(1, -1)
                if (t_idx - MIN_TRAIN_MONTHS) % 12 == 0:
                    curr_ab_params = select_best_elasticnet_params(X_tr, tr['Target'].values)
                best_a, best_l1 = curr_ab_params
                pipe = build_baseline_pipeline(alpha=best_a, l1_ratio=best_l1)
                pipe.fit(X_tr, tr['Target'].values)
                m_preds.append(float(pipe.predict(X_te)[0]))

            m_eval = evaluate_all_metrics(y_act, np.array(m_preds), m_name, c)
            ablation_rows.append(m_eval)

        # Directional Metrics (Step 7)
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

    # Save outputs
    os.makedirs("outputs/phase3", exist_ok=True)
    df_exp = pd.DataFrame(experiments_rows)
    df_exp.to_csv("outputs/phase3/tuning_experiments_comparison.csv", index=False)

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv("outputs/phase3/feature_ablation.csv", index=False)

    df_dir = pd.DataFrame(directional_rows)
    df_dir.to_csv("outputs/phase3/directional_results.csv", index=False)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING & ABLATION EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(df_exp[['Commodity', 'Model', 'MAE', 'RMSE', 'Directional_Accuracy']])

if __name__ == "__main__":
    run_tuning_suite()
