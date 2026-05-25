"""
Rank all evaluation results from results/ folder and generate a ranking report.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_FILE = ROOT / "docs" / "Results Ranking.txt"

def collect_retrieval_results():
    """Collect all retrieval summary CSV files."""
    results = []
    
    # No GPU Info folder
    retrieval_dir = RESULTS_DIR / "No GPU Info"
    if retrieval_dir.exists():
        for csv_file in retrieval_dir.glob("*_summary.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                results.append({
                    'file': csv_file.name,
                    'method': row['method'],
                    'precision_at_5': row.get('precision_at_5', 0),
                    'recall_at_5': row.get('recall_at_5', 0),
                    'mrr': row.get('mrr', 0),
                    'n_queries': row.get('n_queries_evaluated', 0),
                    'environment': 'No GPU Info'
                })
    
    return results

def collect_generation_results():
    """Collect all generation summary CSV files."""
    results = []
    
    # Scan all GPU folders
    for gpu_folder in RESULTS_DIR.iterdir():
        if gpu_folder.is_dir() and gpu_folder.name != "No GPU Info":
            gen_dir = gpu_folder / "generation_eval"
            if gen_dir.exists():
                for csv_file in gen_dir.glob("summary_*.csv"):
                    df = pd.read_csv(csv_file)
                    for _, row in df.iterrows():
                        results.append({
                            'file': f"{gpu_folder.name}/generation_eval/{csv_file.name}",
                            'method': row['method'],
                            'mean_bleu': row.get('mean_bleu', 0),
                            'mean_rouge_l': row.get('mean_rouge_l', 0),
                            'n_queries': row.get('n_queries', 0),
                            'n_success': row.get('n_success', 0),
                            'environment': gpu_folder.name
                        })
            
            # Also check generation_eval_bf16 folder
            gen_bf16_dir = gpu_folder / "generation_eval_bf16"
            if gen_bf16_dir.exists():
                for csv_file in gen_bf16_dir.glob("summary_*.csv"):
                    df = pd.read_csv(csv_file)
                    for _, row in df.iterrows():
                        results.append({
                            'file': f"{gpu_folder.name}/generation_eval_bf16/{csv_file.name}",
                            'method': row['method'],
                            'mean_bleu': row.get('mean_bleu', 0),
                            'mean_rouge_l': row.get('mean_rouge_l', 0),
                            'n_queries': row.get('n_queries', 0),
                            'n_success': row.get('n_success', 0),
                            'environment': f"{gpu_folder.name} (BF16)"
                        })
    
    return results

def generate_report():
    """Generate the ranking report."""
    retrieval_results = collect_retrieval_results()
    generation_results = collect_generation_results()
    
    # Sort retrieval by MRR (primary), then P@5 (secondary)
    retrieval_sorted = sorted(retrieval_results, key=lambda x: (-x['mrr'], -x['precision_at_5']))
    
    # Sort generation by BLEU (primary), then ROUGE-L (secondary)
    generation_sorted = sorted(generation_results, key=lambda x: (-x['mean_bleu'], -x['mean_rouge_l']))
    
    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RESULTS RANKING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Retrieval Ranking
    report_lines.append("RETRIEVAL EVALUATION RANKING")
    report_lines.append("-" * 80)
    report_lines.append("Rank | File                      | Method          | P@5     | R@5     | MRR     | Queries | Environment")
    report_lines.append("-" * 80)
    
    for i, r in enumerate(retrieval_sorted, 1):
        report_lines.append(
            f"{i:4d} | {r['file']:25s} | {r['method']:14s} | "
            f"{r['precision_at_5']:7.4f} | {r['recall_at_5']:7.4f} | "
            f"{r['mrr']:7.4f} | {r['n_queries']:7d} | {r['environment']}"
        )
    
    report_lines.append("")
    report_lines.append("")
    
    # Generation Ranking
    report_lines.append("GENERATION EVALUATION RANKING")
    report_lines.append("-" * 80)
    report_lines.append("Rank | File                                          | Method          | BLEU    | ROUGE-L | Queries | Success | Environment")
    report_lines.append("-" * 80)
    
    for i, r in enumerate(generation_sorted, 1):
        report_lines.append(
            f"{i:4d} | {r['file']:44s} | {r['method']:14s} | "
            f"{r['mean_bleu']:7.4f} | {r['mean_rouge_l']:7.4f} | "
            f"{r['n_queries']:7d} | {r['n_success']:7d} | {r['environment']}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("NOTES")
    report_lines.append("-" * 80)
    report_lines.append("- Retrieval: Sorted by MRR (primary), then P@5 (secondary)")
    report_lines.append("- Generation: Sorted by BLEU (primary), then ROUGE-L (secondary)")
    report_lines.append("- Environment indicates GPU used for evaluation")
    report_lines.append("- BF16 indicates bfloat16 precision model was used")
    report_lines.append("=" * 80)
    
    # Write to file
    OUTPUT_FILE.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"Report generated: {OUTPUT_FILE}")
    print(f"Retrieval results: {len(retrieval_sorted)} entries")
    print(f"Generation results: {len(generation_sorted)} entries")

if __name__ == "__main__":
    generate_report()
