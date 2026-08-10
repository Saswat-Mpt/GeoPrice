import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import sklearn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.models.geoprice import get_geoprice_feature_names
from geoprice.models.baseline import build_baseline_pipeline, create_next_month_target
from geoprice.models.tuning import select_best_elasticnet_params
from geoprice.analysis.shock_responses import COMMODITIES

def main():
    print("=" * 80)
    print("GeoPrice Production Model Retraining Script (Tuned Hyperparameter Selection)")
    print("=" * 80)

    feat_path = "data/processed/feature_dataset.csv"
    raw_path = "data/processed/monthly_aligned.csv"
    output_dir = "models"

    if not os.path.exists(feat_path) or not os.path.exists(raw_path):
        print("Error: Feature dataset or monthly aligned dataset missing!")
        sys.exit(1)

    df_features = pd.read_csv(feat_path)
    df_raw = pd.read_csv(raw_path)

    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "project": "GeoPrice",
        "version": "1.0.0",
        "pipeline": "StandardScaler -> ElasticNet",
        "hyperparameter_tuning_method": "TimeSeriesSplit Inner Cross-Validation",
        "sklearn_version": sklearn.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "commodities": {}
    }

    print("\nTuning and exporting production models for 5 commodities...")

    for c in COMMODITIES:
        full_feats = get_geoprice_feature_names(c)
        target_series = create_next_month_target(df_raw, c)

        df_feat = df_features.set_index('Date') if 'Date' in df_features.columns else df_features.copy()
        data = df_feat[full_feats].copy()
        data['Target'] = target_series

        data = data.reset_index()
        data['Year'] = pd.to_datetime(data['Date']).dt.year
        phase3_data = data[data['Year'] >= 2006].copy().reset_index(drop=True)

        valid_mask = phase3_data[full_feats].notna().all(axis=1) & phase3_data['Target'].notna()
        dataset = phase3_data[valid_mask].copy().reset_index(drop=True)

        X_full = dataset[full_feats].values
        y_full = dataset['Target'].values

        # Perform inner TimeSeriesSplit hyperparameter tuning on all historical training data
        best_alpha, best_l1_ratio = select_best_elasticnet_params(X_full, y_full)

        # Build pipeline with selected tuned hyperparameters and fit on full dataset
        pipeline = build_baseline_pipeline(alpha=best_alpha, l1_ratio=best_l1_ratio)
        pipeline.fit(X_full, y_full)

        model_filename = f"{c.lower()}_model.joblib"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(pipeline, model_path)

        train_start = dataset['Date'].iloc[0]
        train_end = dataset['Date'].iloc[-1]

        metadata["commodities"][c] = {
            "model_file": model_filename,
            "feature_columns": full_feats,
            "selected_alpha": best_alpha,
            "selected_l1_ratio": best_l1_ratio,
            "training_samples": len(dataset),
            "training_start": str(train_start),
            "training_end": str(train_end)
        }

        print(f"  -> {c:15s} | Saved to '{model_path}' (alpha={best_alpha}, l1={best_l1_ratio}, {len(dataset)} samples, {train_start} to {train_end})")

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nMetadata saved to '{meta_path}'.")
    print("=" * 80)
    print("Production models tuned, trained, and exported successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
