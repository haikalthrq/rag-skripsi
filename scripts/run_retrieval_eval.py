"""
Standalone retrieval evaluation aligned with src/streamlit/rag_chat.py.

This script uses the active binary ground truth:
  data/ground_truth/qa_pairs_binary.json

For each QA and chunking method it:
  1. embeds each query once,
  2. retrieves top-k chunks from ChromaDB,
  3. computes Precision@k, Recall@k, and MRR with src.evaluation.metrics,
  4. writes per-query CSV plus summary CSV.

The output uses the same metric semantics as the Streamlit batch evaluation:
queries without relevant chunks for a method are reported as N/A and excluded
from aggregate retrieval means.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in requirements, this is defensive.
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is in requirements, this is defensive.
    torch = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_GT = ROOT / "data/ground_truth/qa_pairs_binary.json"
DEFAULT_OUTPUT = ROOT / "results/retrieval_eval.csv"
DEFAULT_TOP_K = 8
# Catatan: mode default HuggingFace membutuhkan direktori model lengkap di
# DEFAULT_HF_MODEL. Downloader project hanya menyiapkan file GGUF, sehingga
# instalasi tersebut perlu menjalankan script ini dengan --mode gguf.
DEFAULT_MODE = "huggingface"
DEFAULT_HF_MODEL = ROOT / "models/Qwen3-Embedding-4B"
DEFAULT_GGUF_EMBEDDER = ROOT / "models/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_CHROMA_PATH = ROOT / "data/chroma"

METHOD_LABELS = {
    "element_based": "Element-Based",
    "maxmin_semantic": "MaxMin Semantic",
    "recursive": "Recursive",
}

COLLECTION_NAMES = {
    "element_based": "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive": "collection_recursive",
}

PER_QUERY_COLUMNS = [
    "query_id",
    "method",
    "question",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "n_relevant",
    "n_retrieved",
    "retrieved_ids",
    "relevant_ids",
    "error",
    "hardware_info",
]


def get_hardware_info() -> dict[str, Any]:
    """Match the hardware logging shape used by rag_chat.py."""
    info: dict[str, Any] = {
        "cpu": platform.processor(),
    }
    if psutil is not None:
        info.update({
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        })
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.memory_reserved(0)
        info.update({
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_total_gb": round(props.total_memory / (1024**3), 2),
            "gpu_vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "gpu_vram_reserved_gb": round(reserved / (1024**3), 2),
            "gpu_vram_free_gb": round((props.total_memory - reserved) / (1024**3), 2),
        })
    else:
        info["gpu_available"] = False
    return info


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Ground truth must be a list: {path}")
    return data


def load_embedder(args: argparse.Namespace) -> Any:
    if args.mode == "huggingface":
        from src.embedding.embedder import initialize_hf_embedder

        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        return initialize_hf_embedder(
            model_name=str(args.hf_model),
            device=device,
            normalize=True,
        )
    from src.embedding.embedder import initialize_gguf_embedder

    return initialize_gguf_embedder(
        model_path=str(args.embedder),
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )


def precompute_query_embeddings(
    ground_truth: list[dict[str, Any]],
    embedder: Any,
) -> dict[str, tuple[Any, str]]:
    query_vectors: dict[str, tuple[Any, str]] = {}
    for i, item in enumerate(ground_truth, 1):
        q_id = str(item["id"])
        question = str(item["question"])
        logger.info("Embedding query %s/%s: %s", i, len(ground_truth), q_id)
        try:
            query_vectors[q_id] = (embedder.embed(question)[0], "")
        except Exception as exc:
            query_vectors[q_id] = (None, str(exc))
    return query_vectors


def evaluate_method(
    method: str,
    ground_truth: list[dict[str, Any]],
    query_vectors: dict[str, tuple[Any, str]],
    chroma_client: Any,
    top_k: int,
    hardware_info: str,
) -> list[dict[str, Any]]:
    from src.chroma.client import get_or_create_collection
    from src.chroma.query import similarity_search

    collection = get_or_create_collection(chroma_client, COLLECTION_NAMES[method])
    if collection is None:
        raise RuntimeError(f"Collection not found: {COLLECTION_NAMES[method]}")

    logger.info("[%s] %s docs", method, collection.count())
    rows: list[dict[str, Any]] = []

    for item in ground_truth:
        q_id = str(item["id"])
        question = str(item["question"])
        rel_all = item.get("relevant_chunk_ids", {})
        rel_ids = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all
        query_vec, embed_error = query_vectors.get(q_id, (None, "missing query embedding"))

        row: dict[str, Any] = {
            "query_id": q_id,
            "method": METHOD_LABELS[method],
            "question": question,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "n_relevant": len(rel_ids),
            "n_retrieved": 0,
            "retrieved_ids": "",
            "relevant_ids": "|".join(rel_ids),
            "error": "",
            "hardware_info": hardware_info,
        }

        try:
            if embed_error:
                raise RuntimeError(embed_error)
            retrieved = similarity_search(collection, query_vec, k=top_k)
            retrieved_ids = [doc.get("id", "") for doc in retrieved]
            row["n_retrieved"] = len(retrieved_ids)
            row["retrieved_ids"] = "|".join(retrieved_ids)

            if rel_ids:
                row["precision_at_k"] = round(compute_precision_at_k(retrieved_ids, rel_ids, top_k), 4)
                row["recall_at_k"] = round(compute_recall_at_k(retrieved_ids, rel_ids, top_k), 4)
                row["mrr"] = round(compute_mrr(retrieved_ids, rel_ids), 4)
            else:
                row["precision_at_k"] = "N/A"
                row["recall_at_k"] = "N/A"
                row["mrr"] = "N/A"
        except Exception as exc:
            row["error"] = str(exc)

        rows.append(row)

    return rows


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method_key, method_label in METHOD_LABELS.items():
        method_rows = [row for row in rows if row["method"] == method_label]
        if not method_rows:
            continue
        precision = [_as_float(row["precision_at_k"]) for row in method_rows]
        recall = [_as_float(row["recall_at_k"]) for row in method_rows]
        mrr = [_as_float(row["mrr"]) for row in method_rows]
        precision_vals = [v for v in precision if v is not None]
        recall_vals = [v for v in recall if v is not None]
        mrr_vals = [v for v in mrr if v is not None]
        summary.append({
            "method": method_label,
            "method_key": method_key,
            "n_queries": len(method_rows),
            "n_queries_evaluated": len(precision_vals),
            "missing_retrieval_gt": sum(1 for row in method_rows if row["precision_at_k"] == "N/A"),
            "error_count": sum(1 for row in method_rows if row.get("error")),
            "mean_precision_at_k": round(sum(precision_vals) / len(precision_vals), 6) if precision_vals else None,
            "mean_recall_at_k": round(sum(recall_vals) / len(recall_vals), 6) if recall_vals else None,
            "mean_mrr": round(sum(mrr_vals) / len(mrr_vals), 6) if mrr_vals else None,
        })
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nRetrieval summary")
    print("-" * 86)
    print(f"{'Method':<22} {'N':>4} {'Eval':>5} {'N/A':>5} {'P@k':>10} {'R@k':>10} {'MRR':>10}")
    print("-" * 86)
    for row in summary:
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{float(value):.4f}"
        print(
            f"{row['method']:<22} {row['n_queries']:>4} {row['n_queries_evaluated']:>5} "
            f"{row['missing_retrieval_gt']:>5} {fmt(row['mean_precision_at_k']):>10} "
            f"{fmt(row['mean_recall_at_k']):>10} {fmt(row['mean_mrr']):>10}"
        )
    print("-" * 86)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval evaluation aligned with rag_chat.py")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="Binary ground truth JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Per-query output CSV")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help=f"Top-k retrieval (default: {DEFAULT_TOP_K})")
    parser.add_argument("--methods", nargs="+", choices=list(COLLECTION_NAMES.keys()), default=None)
    parser.add_argument("--mode", choices=["gguf", "huggingface"], default=DEFAULT_MODE)
    parser.add_argument("--hf-model", dest="hf_model", default=str(DEFAULT_HF_MODEL))
    parser.add_argument("--embedder", default=str(DEFAULT_GGUF_EMBEDDER))
    parser.add_argument("--chroma_path", default=str(DEFAULT_CHROMA_PATH))
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.methods or list(COLLECTION_NAMES.keys())

    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    logger.info("Loading binary ground truth: %s", gt_path)
    ground_truth = load_ground_truth(gt_path)
    logger.info("Loaded %s QA pairs", len(ground_truth))

    logger.info("Loading embedder (%s)", args.mode)
    embedder = load_embedder(args)
    if embedder is None:
        raise RuntimeError("Failed to load embedder")

    logger.info("Connecting ChromaDB: %s", args.chroma_path)
    from src.chroma.client import initialize_chroma_client

    chroma_client = initialize_chroma_client(persist_directory=args.chroma_path)
    if chroma_client is None:
        raise RuntimeError("Failed to connect ChromaDB")

    hardware_info = json.dumps(get_hardware_info(), ensure_ascii=False)
    query_vectors = precompute_query_embeddings(ground_truth, embedder)

    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for method in methods:
        logger.info("Evaluating %s at top-%s", method, args.top_k)
        all_rows.extend(evaluate_method(
            method=method,
            ground_truth=ground_truth,
            query_vectors=query_vectors,
            chroma_client=chroma_client,
            top_k=args.top_k,
            hardware_info=hardware_info,
        ))

    summary = build_summary(all_rows)
    out_path = Path(args.output)
    summary_path = out_path.parent / f"{out_path.stem}_summary.csv"

    write_csv(all_rows, out_path, PER_QUERY_COLUMNS)
    write_csv(summary, summary_path)
    print_summary(summary)
    logger.info("Saved per-query: %s", out_path)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Done in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
