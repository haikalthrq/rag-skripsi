#!/usr/bin/env python
"""
CLI untuk RAG Evaluation Pipeline.

Mengevaluasi 3 chunking methods dan menghasilkan tabel perbandingan.

Contoh penggunaan:

  # Non-LLM only (Context Precision/Recall via reference_contexts di ground truth)
  python evaluate.py --ground_truth data/ground_truth/qa_pairs.json

  # Dengan generator untuk BLEU/ROUGE (HF)
  python evaluate.py \\
      --ground_truth data/ground_truth/qa_pairs.json \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8

  # Dengan generator GGUF
  python evaluate.py \\
      --ground_truth data/ground_truth/qa_pairs.json \\
      --generator_type gguf \\
      --generator_path models/Qwen3-4B-Instruct-Q8_0.gguf

  # LLM-based Precision/Recall via OpenAI
  python evaluate.py \\
      --ground_truth data/ground_truth/qa_pairs.json \\
      --eval_llm_type openai \\
      --eval_llm_model gpt-4o-mini \\
      --eval_llm_api_key sk-...

  # LLM-based Precision/Recall via local vLLM/SGLang
  python evaluate.py \\
      --ground_truth data/ground_truth/qa_pairs.json \\
      --eval_llm_type local \\
      --eval_llm_model Qwen3-4B-Thinking-2507 \\
      --eval_llm_base_url http://localhost:8000/v1

  # Evaluasi method tertentu + simpan hasil
  python evaluate.py \\
      --ground_truth data/ground_truth/qa_pairs.json \\
      --methods element_based recursive \\
      --output_json results/evaluation/eval_results.json \\
      --output_csv  results/evaluation/eval_results.csv

Format file ground truth (JSON):
  [
    {
      "id": "q_001",
      "question": "...",
      "reference_answer": "...",
      "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],  // untuk MRR
      "reference_contexts": ["teks konteks relevan 1", "..."]  // untuk NonLLM Precision/Recall
    },
    ...
  ]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.evaluator import (
    build_evaluator,
    load_ground_truth,
    COLLECTION_NAMES,
    DEFAULT_EMBEDDER_PATH,
    DEFAULT_CHROMA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_comparison_table(results: list) -> None:
    """Print tabel perbandingan metrics per method ke stdout."""
    all_keys = set()
    for r in results:
        all_keys.update(r.metrics.keys())

    metric_order = ["context_precision", "context_recall", "mrr", "bleu", "rouge1", "rougeL"]
    metric_cols = [k for k in metric_order if k in all_keys]
    metric_cols += [k for k in sorted(all_keys) if k not in metric_cols]

    col_w  = 16
    name_w = 20
    total_w = name_w + col_w * len(metric_cols)

    print("\n" + "=" * total_w)
    print("  HASIL EVALUASI — PERBANDINGAN CHUNKING METHODS")
    print("=" * total_w)
    print(f"{'Method':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in metric_cols))
    print("-" * total_w)

    for r in results:
        row = f"{r.method:<{name_w}}"
        for k in metric_cols:
            val = r.metrics.get(k)
            row += f"{val:>{col_w}.4f}" if isinstance(val, float) else f"{'—':>{col_w}}"
        print(row)

    print("=" * total_w + "\n")


def save_json(results: list, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"JSON disimpan ke: {output}")


def save_csv(results: list, path: str) -> None:
    all_keys: set = set()
    for r in results:
        all_keys.update(r.metrics.keys())

    metric_order = ["context_precision", "context_recall", "mrr", "bleu", "rouge1", "rougeL"]
    metric_cols = [k for k in metric_order if k in all_keys]
    metric_cols += [k for k in sorted(all_keys) if k not in metric_cols]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + metric_cols)
        for r in results:
            row = [r.method] + [
                round(r.metrics[k], 6) if k in r.metrics else ""
                for k in metric_cols
            ]
            writer.writerow(row)

    logger.info(f"CSV disimpan ke: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path ke file JSON ground truth",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--methods", nargs="+", default=None,
        choices=list(COLLECTION_NAMES.keys()),
        help="Chunking methods (default: semua 3 method)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Jumlah chunk per query (default: 5)",
    )
    parser.add_argument(
        "--mrr_at_k", type=int, default=None,
        help="MRR@K threshold (default: tanpa threshold)",
    )
    parser.add_argument(
        "--embedder_path", type=str, default=DEFAULT_EMBEDDER_PATH,
        help=f"Path ke GGUF embedding model (default: {DEFAULT_EMBEDDER_PATH})",
    )
    parser.add_argument(
        "--chroma_path", type=str, default=DEFAULT_CHROMA_PATH,
        help=f"Path ke ChromaDB storage (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--n_gpu_layers", type=int, default=-1,
        help="GPU layers untuk embedder (-1 = semua, 0 = CPU only)",
    )

    # ── Generator (opsional, untuk BLEU/ROUGE) ────────────────────────────────
    parser.add_argument(
        "--generator_type", type=str, default=None,
        choices=["gguf", "hf"],
        help="Backend generator untuk BLEU/ROUGE",
    )
    parser.add_argument(
        "--generator_path", type=str, default=None,
        help="GGUF: path ke .gguf file | HF: model name (contoh: Qwen/Qwen3-4B-Thinking-2507-FP8)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max output tokens generator (GGUF default: 512, HF default: 32768)",
    )

    # ── Evaluator LLM (opsional, untuk LLM-based Precision/Recall) ───────────
    parser.add_argument(
        "--eval_llm_type", type=str, default=None,
        choices=["openai", "local"],
        help="Evaluator LLM untuk LLM-based Context Precision/Recall",
    )
    parser.add_argument(
        "--eval_llm_model", type=str, default="gpt-4o-mini",
        help="Nama model evaluator LLM",
    )
    parser.add_argument(
        "--eval_llm_base_url", type=str, default=None,
        help="Base URL untuk local LLM server (contoh: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--eval_llm_api_key", type=str, default="EMPTY",
        help="API key evaluator LLM",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Simpan hasil lengkap ke JSON file",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Simpan tabel perbandingan ke CSV file",
    )

    args = parser.parse_args()

    # ── Load ground truth ─────────────────────────────────────────────────────
    ground_truth = load_ground_truth(args.ground_truth)

    # ── Build evaluator ───────────────────────────────────────────────────────
    try:
        evaluator = build_evaluator(
            embedder_path=args.embedder_path,
            chroma_path=args.chroma_path,
            n_gpu_layers=args.n_gpu_layers,
            eval_llm_type=args.eval_llm_type,
            eval_llm_model=args.eval_llm_model,
            eval_llm_base_url=args.eval_llm_base_url,
            eval_llm_api_key=args.eval_llm_api_key,
        )
    except RuntimeError as e:
        logger.error(f"Gagal membuat evaluator: {e}")
        sys.exit(1)

    # ── Build generator (opsional) ────────────────────────────────────────────
    generator = None
    if args.generator_type and args.generator_path:
        try:
            if args.generator_type == "gguf":
                from src.rag.generator import initialize_gguf_generator
                max_t = args.max_tokens or 512
                generator = initialize_gguf_generator(
                    model_path=args.generator_path,
                    max_tokens=max_t,
                    n_gpu_layers=args.n_gpu_layers,
                )
            else:
                from src.rag.generator import initialize_hf_generator
                max_t = args.max_tokens or 32768
                generator = initialize_hf_generator(
                    model_name=args.generator_path,
                    max_new_tokens=max_t,
                )

            if generator is None:
                logger.warning(
                    "Generator gagal di-load. BLEU/ROUGE tidak akan dihitung."
                )
        except Exception as e:
            logger.warning(f"Generator load error: {e}. BLEU/ROUGE dilewati.")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = evaluator.evaluate_all(
        ground_truth=ground_truth,
        methods=args.methods,
        top_k=args.top_k,
        generator=generator,
        mrr_at_k=args.mrr_at_k,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if args.output_json:
        save_json(results, args.output_json)

    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
