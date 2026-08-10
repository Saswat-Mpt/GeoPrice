"""
Freezes existing pre-tuning Phase 3 out-of-sample prediction benchmark
into outputs/phase3/baseline_before_tuning.csv.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

def freeze_benchmark():
    os.makedirs("outputs/phase3", exist_ok=True)
    
    base_preds_path = "data/processed/baseline_predictions.csv"
    geo_preds_path = "data/processed/geoprice_predictions.csv"
    base_metrics_path = "data/processed/baseline_metrics.csv"
    geo_metrics_path = "data/processed/geoprice_metrics.csv"

    if not all(os.path.exists(p) for p in [base_preds_path, geo_preds_path, base_metrics_path, geo_metrics_path]):
        print("Error: Missing baseline/geoprice output files. Run stages 7-8 first.")
        sys.exit(1)

    base_preds = pd.read_csv(base_preds_path)
    geo_preds = pd.read_csv(geo_preds_path)
    base_metrics_df = pd.read_csv(base_metrics_path)
    geo_metrics_df = pd.read_csv(geo_metrics_path)
    base_metrics = base_metrics_df[base_metrics_df['Model'] == 'ElasticNet Baseline'].set_index("Commodity")
    geo_metrics = geo_metrics_df[geo_metrics_df['Model'] == 'GeoPrice'].set_index("Commodity")

    merged = pd.merge(
        base_preds[['Commodity', 'Date', 'Actual_Return', 'Predicted_Return']],
        geo_preds[['Commodity', 'Date', 'Predicted_Return']],
        on=['Commodity', 'Date'],
        suffixes=('_Baseline', '_GeoPrice')
    )
    merged.rename(columns={
        'Predicted_Return_Baseline': 'Baseline_Prediction',
        'Predicted_Return_GeoPrice': 'GeoPrice_Prediction'
    }, inplace=True)

    merged['Baseline_MAE'] = merged['Commodity'].map(base_metrics['MAE'])
    merged['GeoPrice_MAE'] = merged['Commodity'].map(geo_metrics['MAE'])
    merged['Baseline_R2'] = merged['Commodity'].map(base_metrics.get('R2', pd.Series(dtype=float)))
    merged['GeoPrice_R2'] = merged['Commodity'].map(geo_metrics.get('R2', pd.Series(dtype=float)))
    merged['Baseline_Directional_Accuracy'] = merged['Commodity'].map(base_metrics['Directional_Accuracy'])
    merged['GeoPrice_Directional_Accuracy'] = merged['Commodity'].map(geo_metrics['Directional_Accuracy'])

    out_file = "outputs/phase3/baseline_before_tuning.csv"
    merged.to_csv(out_file, index=False)
    print(f"Benchmark frozen to '{out_file}' ({len(merged)} rows).")

if __name__ == "__main__":
    freeze_benchmark()
