#!/usr/bin/env python
"""
CLI untuk RAG Pipeline: query → embed → retrieve → generate.

Contoh penggunaan:

  # GGUF generator
  python rag_query.py \\
      --query "Apa itu Retrieval-Augmented Generation?" \\
      --generator_type gguf \\
      --generator_path models/Qwen3-4B-Instruct-Q8_0.gguf

  # HuggingFace generator (Qwen3-4B-Thinking-2507-FP8)
  python rag_query.py \\
      --query "Jelaskan metode chunking element-based" \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8 \\
      --max_tokens 32768

  # HF dengan thinking content terlihat
  python rag_query.py \\
      --query "..." \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8 \\
      --return_thinking

  # Mode interaktif
  python rag_query.py \\
      --interactive \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8

  # Simpan hasil ke JSON
  python rag_query.py \\
      --query "..." \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8 \\
      --output_json results/rag_result.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.pipeline import build_pipeline, COLLECTION_NAMES
from src.rag.pipeline import DEFAULT_EMBEDDER_PATH, DEFAULT_CHROMA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_result(result: dict, show_chunks: bool = True) -> None:
    """Print hasil RAG pipeline ke stdout."""
    sep = "=" * 70
    sep_thin = "-" * 70

    print(f"\n{sep}")
    print(f"  Query   : {result['query']}")
    print(f"  Method  : {result['chunking_method']}")
    print(f"  Chunks  : {result['num_chunks']}")
    print(f"  Time    : {result['elapsed_seconds']}s")
    print(sep)

    if result.get("thinking"):
        print("\n[Thinking]")
        print(result["thinking"])
        print(sep_thin)

    print("\nJawaban:\n")
    print(result["answer"])

    if show_chunks and result["retrieved_chunks"]:
        print(f"\n{sep_thin}")
        print(f"Retrieved Chunks ({result['num_chunks']}):")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source_file", "unknown")
            pages = meta.get("page_numbers", "-")
            dist = chunk.get("distance")
            dist_str = f"{dist:.4f}" if dist is not None else "-"
            preview = chunk["document"][:200].replace("\n", " ")
            print(f"\n  [{i}] {src} | hal: {pages} | dist: {dist_str}")
            print(f"      {preview}...")

    print(f"{sep}\n")


def result_to_serializable(result: dict) -> dict:
    """Konversi result dict ke format JSON-serializable."""
    return {
        "query": result["query"],
        "answer": result["answer"],
        "thinking": result.get("thinking", ""),
        "chunking_method": result["chunking_method"],
        "num_chunks": result["num_chunks"],
        "elapsed_seconds": result["elapsed_seconds"],
        "retrieved_chunks": [
            {
                "id": c.get("id"),
                "document": c.get("document"),
                "metadata": c.get("metadata", {}),
                "distance": (
                    float(c["distance"])
                    if c.get("distance") is not None
                    else None
                ),
            }
            for c in result["retrieved_chunks"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Query CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query", type=str, default=None,
        help="Pertanyaan yang akan dijawab oleh RAG pipeline",
    )
    parser.add_argument(
        "--method", type=str, default="element_based",
        choices=list(COLLECTION_NAMES.keys()),
        help="Chunking method / ChromaDB collection (default: element_based)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Jumlah chunk yang di-retrieve (default: 5)",
    )
    parser.add_argument(
        "--embedder_path", type=str, default=DEFAULT_EMBEDDER_PATH,
        help=f"Path ke GGUF embedding model (default: {DEFAULT_EMBEDDER_PATH})",
    )
    parser.add_argument(
        "--generator_type", type=str, default="gguf",
        choices=["gguf", "hf"],
        help="Backend generator: 'gguf' (llama-cpp) atau 'hf' (HuggingFace transformers)",
    )
    parser.add_argument(
        "--generator_path", type=str, required=True,
        help="GGUF: path ke file .gguf | HF: model name (contoh: Qwen/Qwen3-4B-Thinking-2507-FP8)",
    )
    parser.add_argument(
        "--chroma_path", type=str, default=DEFAULT_CHROMA_PATH,
        help=f"Path ke ChromaDB storage (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--n_gpu_layers", type=int, default=-1,
        help="Jumlah GPU layers (-1 = semua, 0 = CPU only)",
    )
    parser.add_argument(
        "--n_ctx", type=int, default=4096,
        help="Context length untuk generator (default: 4096)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max output tokens (GGUF default: 512, HF default: 32768)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (GGUF default: 0.7, HF default: 0.6)",
    )
    parser.add_argument(
        "--top_p", type=float, default=None,
        help="Nucleus sampling (GGUF default: 0.9, HF default: 0.95)",
    )
    parser.add_argument(
        "--top_k_gen", type=int, default=20,
        help="Top-K sampling untuk HF generator (default: 20)",
    )
    parser.add_argument(
        "--return_thinking", action="store_true",
        help="Tampilkan thinking content dari Qwen3-Thinking model (HF only)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Mode interaktif: loop input query dari terminal",
    )
    parser.add_argument(
        "--no_show_chunks", action="store_true",
        help="Sembunyikan detail retrieved chunks dari output",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Simpan semua hasil ke file JSON",
    )

    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("Harus memberikan --query atau menggunakan --interactive")

    # Defaults per generator type
    is_hf = args.generator_type == "hf"
    max_tokens  = args.max_tokens  if args.max_tokens  is not None else (32768 if is_hf else 512)
    temperature = args.temperature if args.temperature is not None else (0.6   if is_hf else 0.7)
    top_p       = args.top_p       if args.top_p       is not None else (0.95  if is_hf else 0.9)

    # Build pipeline
    try:
        pipeline = build_pipeline(
            chunking_method=args.method,
            embedder_path=args.embedder_path,
            generator_path=args.generator_path,
            generator_type=args.generator_type,
            chroma_path=args.chroma_path,
            top_k=args.top_k,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k_gen=args.top_k_gen,
            return_thinking=args.return_thinking,
        )
    except RuntimeError as e:
        logger.error(f"Gagal membuat pipeline: {e}")
        sys.exit(1)

    results = []
    show_chunks = not args.no_show_chunks

    if args.interactive:
        print(f"\n  RAG Pipeline — Mode Interaktif")
        print(f"  Method: {args.method} | Top-K: {args.top_k}")
        print("  Ketik 'exit' atau 'quit' untuk keluar.\n")

        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar.")
                break

            if not query or query.lower() in ("exit", "quit"):
                break

            result = pipeline.run(query)
            print_result(result, show_chunks=show_chunks)
            results.append(result)

    else:
        result = pipeline.run(args.query)
        print_result(result, show_chunks=show_chunks)
        results.append(result)

    # Save to JSON
    if args.output_json and results:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = [result_to_serializable(r) for r in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        logger.info(f"Hasil disimpan ke: {output_path}")


if __name__ == "__main__":
    main()
