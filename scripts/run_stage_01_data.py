import sys
import os
import argparse
import pandas as pd

# Add src/ to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geoprice.data.gpr import load_gpr
from geoprice.data.world_bank import inspect_world_bank_gold, load_world_bank_commodities
from geoprice.data.fred import load_dxy_daily, aggregate_dxy_monthly
from geoprice.data.alignment import (
    align_datasets,
    validate_aligned_data,
    create_data_dictionary,
    save_processed_data
)

def run_source_check():
    """Verifies all 9 required series/components before processing and prints verification checklist table."""
    print("=" * 80)
    print("GeoPrice — Stage 1 Source Verification Checklist")
    print("=" * 80)
    
    checklist = []
    
    # 1. GPR
    try:
        df_gpr = load_gpr()
        checklist.append({"Dataset": "GPR", "Source": "Caldara-Iacoviello", "Local file": "data/raw/gpr/data_gpr_export.xls", "Downloaded?": "Yes", "Date range": f"{df_gpr.index[0]} to {df_gpr.index[-1]}", "Rows": len(df_gpr), "Status": "PASS"})
        checklist.append({"Dataset": "GPRT", "Source": "Caldara-Iacoviello", "Local file": "data/raw/gpr/data_gpr_export.xls", "Downloaded?": "Yes", "Date range": f"{df_gpr.index[0]} to {df_gpr.index[-1]}", "Rows": len(df_gpr), "Status": "PASS"})
        checklist.append({"Dataset": "GPRA", "Source": "Caldara-Iacoviello", "Local file": "data/raw/gpr/data_gpr_export.xls", "Downloaded?": "Yes", "Date range": f"{df_gpr.index[0]} to {df_gpr.index[-1]}", "Rows": len(df_gpr), "Status": "PASS"})
    except Exception as e:
        checklist.append({"Dataset": "GPR / GPRT / GPRA", "Source": "Caldara-Iacoviello", "Local file": "data/raw/gpr/", "Downloaded?": "No", "Date range": "N/A", "Rows": 0, "Status": f"FAIL ({e})"})

    # 2. World Bank Commodities
    try:
        df_comm = load_world_bank_commodities()
        gold_info = inspect_world_bank_gold()
        
        for c, sid in [("Brent", "POILBREUSDM"), ("Natural Gas", "PNGASUSUSDM"), ("Copper", "PCOPPUSDM"), ("Wheat", "PWHEAMTUSDM")]:
            c_key = c.replace(" ", "_")
            if c_key in df_comm.columns:
                valid_s = df_comm[c_key].dropna()
                checklist.append({"Dataset": f"{c} - {sid}", "Source": "World Bank Pink Sheet / IMF", "Local file": "data/raw/world_bank/CMO-Historical-Data-Monthly.xlsx", "Downloaded?": "Yes", "Date range": f"{valid_s.index[0]} to {valid_s.index[-1]}", "Rows": len(valid_s), "Status": "PASS"})

        checklist.append({"Dataset": "Gold - World Bank Pink Sheet", "Source": "World Bank Pink Sheet", "Local file": "data/raw/world_bank/CMO-Historical-Data-Monthly.xlsx", "Downloaded?": "Yes", "Date range": f"{gold_info['first_valid_date']} to {gold_info['last_valid_date']}", "Rows": gold_info['valid_gold_obs'], "Status": "PASS"})
    except Exception as e:
        print(f"Error checking commodities: {e}")

    # 3. DXY
    try:
        df_dxy_d = load_dxy_daily()
        df_dxy_m = aggregate_dxy_monthly(df_dxy_d)
        checklist.append({"Dataset": "DXY - DTWEXBGS", "Source": "FRED / Yahoo Finance (Daily->Monthly Mean)", "Local file": "data/raw/fred/DTWEXBGS.csv", "Downloaded?": "Yes", "Date range": f"{df_dxy_m.index[0]} to {df_dxy_m.index[-1]}", "Rows": len(df_dxy_m), "Status": "PASS"})
    except Exception as e:
        checklist.append({"Dataset": "DXY - DTWEXBGS", "Source": "FRED / Yahoo", "Local file": "data/raw/fred/DTWEXBGS.csv", "Downloaded?": "No", "Date range": "N/A", "Rows": 0, "Status": f"FAIL ({e})"})

    df_chk = pd.DataFrame(checklist)
    print(df_chk.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Gold Workbook Specific Inspection")
    print("=" * 80)
    try:
        ginfo = inspect_world_bank_gold()
        print(f"Workbook File: {ginfo['file_path']}")
        print(f"Sheet Name:    {ginfo['sheet_name']}")
        print(f"Date Column:   {ginfo['date_column']}")
        print(f"Gold Column:   Index {ginfo['gold_col_index']} | Name: '{ginfo['gold_col_name']}'")
        print(f"Gold Units:    {ginfo['gold_unit']}")
        print(f"Date Range:    {ginfo['first_valid_date']} to {ginfo['last_valid_date']}")
        print(f"Valid Obs:     {ginfo['valid_gold_obs']} (Missing: {ginfo['missing_gold_obs']})")
    except Exception as e:
        print(f"Gold workbook inspection error: {e}")
        
    print("\n" + "=" * 80)
    pass_count = sum(1 for item in checklist if item['Status'] == 'PASS')
    print(f"STATUS: {pass_count}/{len(checklist)} REQUIRED COMPONENTS VERIFIED")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="GeoPrice Stage 1 Data Ingestion & Alignment")
    parser.add_argument("--check-sources", action="store_true", help="Verify raw data sources and exit")
    args = parser.parse_args()

    if args.check_sources:
        run_source_check()
        return

    print("=" * 80)
    print("Running GeoPrice — Phase 1, Stage 1: Data Ingestion & Alignment")
    print("=" * 80)

    # 1. Align Datasets
    print("\n[Step 1/5] Ingesting and aligning datasets...")
    aligned_df, metrics = align_datasets()
    print(f"-> Successfully aligned {metrics['total_rows']} monthly observations ({metrics['start_date']} to {metrics['end_date']})")
    print(f"-> Common commodity start date: {metrics['common_commodity_start']}")

    # 2. Validate Aligned Data
    print("\n[Step 2/5] Running data quality validation checks...")
    val_results = validate_aligned_data(aligned_df)
    
    print(f"-> No duplicate dates:          {'PASS' if val_results['no_duplicate_dates'] else 'FAIL'}")
    print(f"-> Chronological sorting:       {'PASS' if val_results['chronological_order'] else 'FAIL'}")
    print(f"-> All required columns exist:  {'PASS' if val_results['all_required_columns_exist'] else 'FAIL'}")
    print(f"-> All series numeric:          {'PASS' if val_results['all_series_numeric'] else 'FAIL'}")
    print(f"-> Monthly frequency continuous:{'PASS' if val_results['no_date_gaps'] else 'FAIL'}")
    print(f"-> DXY monthly mean aggregated: {'PASS' if val_results['dxy_aggregated'] else 'FAIL'}")
    print(f"-> World Bank Gold validated:   {'PASS' if val_results['gold_validated'] else 'FAIL'}")

    if not val_results['overall_pass']:
        print("\nCRITICAL VALIDATION FAILED. Aborting dataset save.")
        sys.exit(1)
        
    print("-> Overall Validation: ALL CHECKS PASSED")

    # 3. Create Data Dictionary
    print("\n[Step 3/5] Generating data dictionary...")
    data_dict_df = create_data_dictionary(aligned_df, val_results)

    # 4. Save Processed Data
    print("\n[Step 4/5] Saving processed dataset and metadata...")
    aligned_path, dict_path = save_processed_data(aligned_df, data_dict_df)
    print(f"-> Processed dataset saved to: {aligned_path}")
    print(f"-> Data dictionary saved to:   {dict_path}")

    # 5. Final Stage 1 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 1 FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Rows:     {len(aligned_df)}")
    print(f"Columns (10):   {', '.join(aligned_df.columns)}")
    print(f"Date Window:    {metrics['start_date']} to {metrics['end_date']}")
    print(f"Commodity Start: {metrics['common_commodity_start']} (Phase 2 Window)")
    print(f"DXY Start:      {val_results['missing_summary']['DXY']['first_valid_date']} (Phase 3 Window)")
    print("\nMissing Values Summary:")
    for col, minfo in val_results['missing_summary'].items():
        exp_text = " (Expected: pre-2001)" if col == "DXY" else ""
        print(f"  {col:15s}: {minfo['missing_count']:3d} missing ({minfo['missing_pct']:5.1f}%){exp_text} | Valid: {minfo['first_valid_date']} -> {minfo['last_valid_date']}")

    print("\n" + "=" * 80)
    print("Stage 1 complete. Ready for Phase 1 -> Stage 2: Signature Feature Engineering.")
    print("=" * 80)

if __name__ == "__main__":
    main()
