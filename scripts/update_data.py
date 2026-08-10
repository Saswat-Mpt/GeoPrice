import sys
import os
import subprocess

def main():
    print("=" * 80)
    print("GeoPrice Data Refresh & Pipeline Update Script")
    print("=" * 80)

    stages = [
        ("Phase 1, Stage 1 (Data Collection & Alignment)", "scripts/run_stage_01_data.py"),
        ("Phase 1, Stage 2 (Feature Engineering)", "scripts/run_stage_02_features.py"),
        ("Phase 2, Stage 3 (Systematic Shock Analysis)", "scripts/run_stage_03_shocks.py"),
        ("Phase 2, Stage 4 (Threats vs Acts Analysis)", "scripts/run_stage_04_threats_acts.py"),
        ("Phase 2, Stage 5 (GPR Regimes & Analogue)", "scripts/run_stage_05_regimes.py"),
        ("Phase 2, Stage 6 (Major Conflict Cases)", "scripts/run_stage_06_conflicts.py"),
        ("Phase 3, Stage 7 (Baseline Forecasting Benchmark)", "scripts/run_stage_07_baseline.py"),
        ("Phase 3, Stage 8 (GeoPrice Model Benchmark)", "scripts/run_stage_08_geoprice.py"),
        ("Phase 3, Stage 9 (Tuned Out-of-Sample Experiments)", "scripts/run_experiments_tuning.py"),
        ("Phase 3, Stage 9 (Final Evaluation & Checkpoint)", "scripts/run_stage_09_evaluation.py")
    ]

    for name, script_path in stages:
        print(f"\n---> Running {name}...")
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error running {script_path}:\n{res.stderr}")
            sys.exit(1)
        else:
            print(f"     [PASS] {name} completed successfully.")

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES REFRESHED & VALIDATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
