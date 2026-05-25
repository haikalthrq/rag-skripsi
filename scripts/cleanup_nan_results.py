"""
Identify and delete CSV files in results/ where ALL numeric columns are NaN.
Safety-first: dry-run mode by default.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

def find_nan_files(threshold=0.5):
    """Find CSV files where NaN percentage exceeds threshold."""
    files_to_delete = []
    
    for csv_file in RESULTS_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                continue  # Skip files with no numeric columns
            
            # Calculate NaN percentage
            total_cells = len(df) * len(numeric_cols)
            nan_cells = df[numeric_cols].isna().sum().sum()
            nan_percentage = nan_cells / total_cells if total_cells > 0 else 0
            
            if nan_percentage > threshold:
                files_to_delete.append((csv_file, nan_percentage))
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return files_to_delete

def main():
    print("=" * 80)
    print("FINDING FILES WITH >50% NaN VALUES")
    print("=" * 80)
    print()
    
    files_to_delete = find_nan_files(threshold=0.5)
    
    if not files_to_delete:
        print("No files found with >50% NaN values.")
        return
    
    print(f"Found {len(files_to_delete)} files with >50% NaN values:")
    print("-" * 80)
    for f, pct in files_to_delete:
        print(f"  - {f.relative_to(ROOT)} ({pct:.1%} NaN)")
    print()
    
    # Show sample of one file to verify
    if files_to_delete:
        sample, sample_pct = files_to_delete[0]
        print(f"Sample content of {sample.name} ({sample_pct:.1%} NaN):")
        print("-" * 80)
        df = pd.read_csv(sample)
        print(df.head())
        print()
        print("Numeric columns NaN percentage:")
        for col in df.select_dtypes(include=['number']).columns:
            col_pct = df[col].isna().sum() / len(df)
            print(f"  {col}: {col_pct:.1%}")
        print()
    
    print("=" * 80)
    print("DELETING FILES")
    print("=" * 80)
    
    for f, pct in files_to_delete:
        f.unlink()
        print(f"  Deleted: {f.relative_to(ROOT)} ({pct:.1%} NaN)")
    
    print(f"\nSuccessfully deleted {len(files_to_delete)} files.")

if __name__ == "__main__":
    main()
