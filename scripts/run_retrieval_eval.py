"""
Evaluasi retrieval RAG pipeline: Precision@k, Recall@k, MRR.

Pipeline:
  1. Load ground truth JSON (dari convert_ground_truth_to_json.py)
  2. Load GGUF embedder + ChromaDB
  3. Untuk setiap method (element_based, maxmin_semantic, recursive):
       - Embed setiap query
       - Retrieve top-k dari ChromaDB
       - Hitung P@k, R@k, MRR per query, lalu rata-rata
  4. Simpan hasil ke CSV + cetak tabel ringkasan

Penggunaan:
  # Strict ground truth (hanya label 2)
  python scripts/run_retrieval_eval.py \\
      --gt data/ground_truth/qa_pairs_strict.json \\
      --output results/retrieval_eval_strict.csv \\
      --top_k 5

  # Lenient ground truth (label 1 dan 2)
  python scripts/run_retrieval_eval.py \\
      --gt data/ground_truth/qa_pairs_lenient.json \\
      --output results/retrieval_eval_lenient.csv \\
      --top_k 5
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Tambahkan root project ke sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


COLLECTION_NAMES: Dict[str, str] = {
    "element_based":   "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive":       "collection_recursive",
}

DEFAULT_GT          = "data/ground_truth/qa_pairs_binary.json"
DEFAULT_OUTPUT      = "results/retrieval_eval.csv"
DEFAULT_TOP_K       = 8
DEFAULT_MODE        = "huggingface"
DEFAULT_HF_MODEL    = "/workspace/models/Qwen3-Embedding-4B"
DEFAULT_EMBEDDER    = "models/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_CHROMA_PATH = "data/chroma"


# ── Metrics ────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for r in retrieved[:k] if r in rel_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for r in retrieved[:k] if r in rel_set)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    rel_set = set(relevant)
    for rank, r in enumerate(retrieved, start=1):
        if r in rel_set:
            return 1.0 / rank
    return 0.0


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_method(
    method: str,
    ground_truth: List[Dict[str, Any]],
    embedder: Any,
    chroma_client: Any,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k untuk setiap query dan hitung metrik per query.
    Return list of per-query result dicts.
    """
    from src.chroma.client import get_or_create_collection
    from src.chroma.query import similarity_search

    collection_name = COLLECTION_NAMES[method]
    collection = get_or_create_collection(chroma_client, collection_name)
    if collection is None:
        logger.error(f"Collection '{collection_name}' tidak ditemukan.")
        return []

    doc_count = collection.count()
    logger.info(f"  [{method}] {collection_name} — {doc_count} dokumen")

    rows = []
    for item in ground_truth:
        q_id      = item["id"]
        question  = item["question"]
        rel_all   = item.get("relevant_chunk_ids", {})
        rel_ids   = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all

        # Embed query
        emb = embedder.embed(question)
        query_vec = emb[0]

        # Retrieve
        results      = similarity_search(collection, query_vec, k=top_k)
        retrieved_ids = [r["id"] for r in results]

        if rel_ids:
            p = precision_at_k(retrieved_ids, rel_ids, top_k)
            r = recall_at_k(retrieved_ids, rel_ids, top_k)
            m = mrr(retrieved_ids, rel_ids)
        else:
            p = r = m = None  # query dieksklusi dari agregasi metrik

        rows.append({
            "method":         method,
            "query_id":       q_id,
            "question":       question[:80],
            "n_relevant":     len(rel_ids),
            "n_retrieved":    len(retrieved_ids),
            f"precision_at_{top_k}": round(p, 6) if p is not None else "",
            f"recall_at_{top_k}":    round(r, 6) if r is not None else "",
            "mrr":            round(m, 6) if m is not None else "",
            "retrieved_ids":  "|".join(retrieved_ids),
            "relevant_ids":   "|".join(rel_ids),
        })

        status = f"P={p:.3f} R={r:.3f} MRR={m:.3f}" if rel_ids else "no-relevant (skipped)"
        logger.info(f"    {q_id}: {status}")

    return rows


def aggregate(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, float]:
    """Hitung rata-rata metrik (hanya query yang punya relevant_ids)."""
    p_col = f"precision_at_{top_k}"
    r_col = f"recall_at_{top_k}"

    ps = [float(r[p_col]) for r in rows if r[p_col] != ""]
    rs = [float(r[r_col]) for r in rows if r[r_col] != ""]
    ms = [float(r["mrr"])  for r in rows if r["mrr"] != ""]

    return {
        "n_queries_evaluated": len(ps),
        p_col: round(sum(ps) / len(ps), 6) if ps else 0.0,
        r_col: round(sum(rs) / len(rs), 6) if rs else 0.0,
        "mrr": round(sum(ms) / len(ms), 6) if ms else 0.0,
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def print_summary(summary: List[Dict[str, Any]], top_k: int) -> None:
    p_col = f"precision_at_{top_k}"
    r_col = f"recall_at_{top_k}"
    header = f"{'Method':<20} {'N_eval':>6} {p_col:>14} {r_col:>12} {'MRR':>8}"
    print("\n" + "=" * len(header))
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summary:
        print(
            f"{s['method']:<20} "
            f"{s['n_queries_evaluated']:>6} "
            f"{s[p_col]:>14.4f} "
            f"{s[r_col]:>12.4f} "
            f"{s['mrr']:>8.4f}"
        )
    print("=" * len(header))


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        logger.warning("Tidak ada baris untuk disimpan.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Per-query CSV disimpan ke: {path}")


def save_summary_csv(summary: List[Dict[str, Any]], path: Path) -> None:
    if not summary:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"Summary CSV disimpan ke: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluasi retrieval RAG: Precision@k, Recall@k, MRR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gt", type=str, default=DEFAULT_GT,
        help=f"Path ke ground truth JSON (default: {DEFAULT_GT})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Path output CSV per-query (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Jumlah chunk yang di-retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated methods (default: semua). Contoh: element_based,recursive",
    )
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE,
        choices=["gguf", "huggingface"],
        help=f"Mode embedder: gguf atau huggingface (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--hf-model", type=str, default=DEFAULT_HF_MODEL,
        help=f"Path ke HuggingFace embedding model (default: {DEFAULT_HF_MODEL})",
    )
    parser.add_argument(
        "--embedder", type=str, default=DEFAULT_EMBEDDER,
        help=f"Path ke GGUF embedding model (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--chroma_path", type=str, default=DEFAULT_CHROMA_PATH,
        help=f"Path ke ChromaDB (default: {DEFAULT_CHROMA_PATH})",
    )
    args = parser.parse_args()

    # Validate gt file
    gt_path = Path(args.gt)
    if not gt_path.exists():
        logger.error(f"Ground truth file tidak ditemukan: {gt_path}")
        sys.exit(1)

    # Parse methods
    methods = (
        [m.strip() for m in args.methods.split(",")]
        if args.methods
        else list(COLLECTION_NAMES.keys())
    )
    invalid = [m for m in methods if m not in COLLECTION_NAMES]
    if invalid:
        logger.error(f"Method tidak dikenal: {invalid}. Pilih dari: {list(COLLECTION_NAMES.keys())}")
        sys.exit(1)

    # Load ground truth
    logger.info(f"Loading ground truth: {gt_path}")
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    logger.info(f"  {len(ground_truth)} QA pairs loaded")

    # Load embedder
    if args.mode == "huggingface":
        logger.info(f"Loading HF embedder: {args.hf_model}")
        from src.embedding.embedder import initialize_hf_embedder
        embedder = initialize_hf_embedder(
            model_name=args.hf_model,
            device='cuda',
            normalize=True,
        )
    else:
        logger.info(f"Loading GGUF embedder: {args.embedder}")
        from src.embedding.embedder import initialize_gguf_embedder
        embedder = initialize_gguf_embedder(
            model_path=args.embedder,
            n_gpu_layers=-1,
            verbose=False,
        )
    if embedder is None:
        logger.error("Gagal load embedder.")
        sys.exit(1)

    # Load ChromaDB
    logger.info(f"Connecting to ChromaDB: {args.chroma_path}")
    from src.chroma.client import initialize_chroma_client
    chroma_client = initialize_chroma_client(persist_directory=args.chroma_path)
    if chroma_client is None:
        logger.error("Gagal koneksi ChromaDB.")
        sys.exit(1)

    # Evaluate
    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    t0 = time.time()

    for method in methods:
        logger.info(f"\nEvaluating: {method} (top_k={args.top_k})")
        rows = evaluate_method(
            method=method,
            ground_truth=ground_truth,
            embedder=embedder,
            chroma_client=chroma_client,
            top_k=args.top_k,
        )
        all_rows.extend(rows)

        agg = aggregate(rows, args.top_k)
        summary.append({"method": method, **agg})

    elapsed = time.time() - t0
    logger.info(f"\nTotal waktu: {elapsed:.1f}s")

    # Print & save
    print_summary(summary, args.top_k)

    out_path     = Path(args.output)
    summary_path = out_path.parent / out_path.name.replace(".csv", "_summary.csv")

    save_csv(all_rows, out_path)
    save_summary_csv(summary, summary_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
