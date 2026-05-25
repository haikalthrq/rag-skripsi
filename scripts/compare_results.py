"""Compare old vs new evaluation results (excluding RTX 4050 6GB)."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

def categorize_files():
    """Categorize files into old vs new."""
    retrieval = {'old': [], 'new': []}
    generation = {'old': [], 'new': []}
    
    for csv_file in RESULTS_DIR.rglob("*.csv"):
        if 'streamlit' in str(csv_file):
            continue
        if 'RTX 4050 6GB' in str(csv_file):
            continue
        
        if 'retrieval' in csv_file.name:
            if '_new' in csv_file.name:
                retrieval['new'].append(csv_file)
            else:
                retrieval['old'].append(csv_file)
        elif 'generation' in csv_file.name:
            if 'per_query' in csv_file.name or 'report' in csv_file.name:
                continue
            if 'summary' in csv_file.name:
                if '_new' in csv_file.parent.name or 'new' in csv_file.name:
                    generation['new'].append(csv_file)
                else:
                    generation['old'].append(csv_file)
    
    return retrieval, generation

def load_retrieval_summary(files):
    """Load and aggregate retrieval summary data."""
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except:
            pass
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def load_generation_summary(files):
    """Load and aggregate generation summary data."""
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except:
            pass
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def main():
    print("=" * 80)
    print("COMPARING OLD VS NEW RESULTS (excluding RTX 4050 6GB)")
    print("=" * 80)
    print()
    
    retrieval, generation = categorize_files()
    
    # Retrieval comparison
    print("RETRIEVAL FILES")
    print("-" * 80)
    print(f"Old: {len(retrieval['old'])} files")
    for f in retrieval['old']:
        print(f"  - {f.relative_to(ROOT)}")
    print()
    print(f"New: {len(retrieval['new'])} files")
    for f in retrieval['new']:
        print(f"  - {f.relative_to(ROOT)}")
    print()
    
    # Load retrieval summaries
    retrieval_old = load_retrieval_summary([f for f in retrieval['old'] if 'summary' in f.name])
    retrieval_new = load_retrieval_summary([f for f in retrieval['new'] if 'summary' in f.name])
    
    print("RETRIEVAL METRICS COMPARISON")
    print("-" * 80)
    if not retrieval_old.empty:
        print("OLD RESULTS:")
        print(retrieval_old[['method', 'precision_at_5', 'recall_at_5', 'mrr']].to_string(index=False))
    print()
    if not retrieval_new.empty:
        print("NEW RESULTS:")
        print(retrieval_new[['method', 'precision_at_5', 'recall_at_5', 'mrr']].to_string(index=False))
    print()
    
    # Generation comparison
    print()
    print("GENERATION FILES")
    print("-" * 80)
    print(f"Old: {len(generation['old'])} files")
    for f in generation['old']:
        print(f"  - {f.relative_to(ROOT)}")
    print()
    print(f"New: {len(generation['new'])} files")
    for f in generation['new']:
        print(f"  - {f.relative_to(ROOT)}")
    print()
    
    # Load generation summaries
    generation_old = load_generation_summary(generation['old'])
    generation_new = load_generation_summary(generation['new'])
    
    print("GENERATION METRICS COMPARISON")
    print("-" * 80)
    if not generation_old.empty:
        print("OLD RESULTS:")
        print(generation_old[['method', 'mean_bleu', 'mean_rouge_l']].to_string(index=False))
    print()
    if not generation_new.empty:
        print("NEW RESULTS:")
        print(generation_new[['method', 'mean_bleu', 'mean_rouge_l']].to_string(index=False))
    print()
    
    # Summary comparison
    print()
    print("=" * 80)
    print("SUMMARY: WHICH IS BETTER?")
    print("=" * 80)
    
    if not retrieval_old.empty and not retrieval_new.empty:
        print("RETRIEVAL:")
        old_mrr = retrieval_old['mrr'].mean()
        new_mrr = retrieval_new['mrr'].mean()
        print(f"  Old avg MRR: {old_mrr:.4f}")
        print(f"  New avg MRR: {new_mrr:.4f}")
        if new_mrr > old_mrr:
            print(f"  → NEW is better by {(new_mrr - old_mrr):.4f}")
        else:
            print(f"  → OLD is better by {(old_mrr - new_mrr):.4f}")
    
    if not generation_old.empty and not generation_new.empty:
        print()
        print("GENERATION:")
        old_bleu = generation_old['mean_bleu'].mean()
        new_bleu = generation_new['mean_bleu'].mean()
        print(f"  Old avg BLEU: {old_bleu:.4f}")
        print(f"  New avg BLEU: {new_bleu:.4f}")
        if new_bleu > old_bleu:
            print(f"  → NEW is better by {(new_bleu - old_bleu):.4f}")
        else:
            print(f"  → OLD is better by {(old_bleu - new_bleu):.4f}")

if __name__ == "__main__":
    main()
