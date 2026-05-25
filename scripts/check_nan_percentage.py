"""Check NaN percentage for all remaining CSV files in results/."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

def check_all_nan_percentages():
    """Calculate NaN percentage for all CSV files."""
    results = []
    
    for csv_file in sorted(RESULTS_DIR.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                continue
            
            # Calculate NaN percentage
            total_cells = len(df) * len(numeric_cols)
            nan_cells = df[numeric_cols].isna().sum().sum()
            nan_percentage = nan_cells / total_cells if total_cells > 0 else 0
            
            results.append({
                'file': csv_file.relative_to(ROOT),
                'nan_pct': nan_percentage,
                'rows': len(df),
                'cols': len(numeric_cols)
            })
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return results

def main():
    print("=" * 100)
    print("NaN PERCENTAGE FOR ALL REMAINING CSV FILES")
    print("=" * 100)
    print()
    
    results = check_all_nan_percentages()
    
    if not results:
        print("No CSV files found.")
        return
    
    # Sort by NaN percentage descending
    results.sort(key=lambda x: x['nan_pct'], reverse=True)
    
    print(f"{'File':<60} {'NaN %':>10} {'Rows':>6} {'Cols':>6}")
    print("-" * 100)
    
    for r in results:
        print(f"{str(r['file']):<60} {r['nan_pct']:>9.1%} {r['rows']:>6} {r['cols']:>6}")
    
    print()
    print(f"Total files: {len(results)}")
    
    # Summary statistics
    avg_nan = sum(r['nan_pct'] for r in results) / len(results)
    high_nan = sum(1 for r in results if r['nan_pct'] > 0.5)
    print(f"Average NaN percentage: {avg_nan:.1%}")
    print(f"Files with >50% NaN: {high_nan}")

if __name__ == "__main__":
    main()
