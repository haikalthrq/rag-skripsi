"""
Generation-only evaluation: BLEU + ROUGE-L Recall per chunking method.

Tidak butuh retrieval ground truth (anotasi manual) — cukup QA gold xlsx.

Alur:
  Load QA gold (xlsx)
    → load embedder + ChromaDB (sekali)
    → load generator (sekali)
    → untuk setiap chunking method:
        → retrieve top-k chunks per query
        → generate jawaban
        → hitung BLEU + ROUGE-L Recall
    → print tabel perbandingan + simpan hasil

Output di results/generation_eval/:
  per_query_<timestamp>.csv   — semua 90 baris (30 QA × 3 method)
  summary_<timestamp>.csv     — tabel ringkasan per method
  report_<timestamp>.txt      — laporan question-first siap kutip untuk skripsi
  run_<timestamp>.log         — log lengkap eksekusi

Usage:
  # HuggingFace — Instruct (non-thinking, direkomendasikan):
  python scripts/run_generation_eval.py \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Instruct-2507-FP8

  # HuggingFace — Thinking:
  python scripts/run_generation_eval.py \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Thinking-2507-FP8 \\
      --temperature 0.6 --top_p 0.95

  # GGUF lokal:
  python scripts/run_generation_eval.py \\
      --generator_type gguf \\
      --generator_path models/Qwen3-4B-Instruct-Q8_0.gguf

  # Hanya 1 method, resume dari run sebelumnya:
  python scripts/run_generation_eval.py \\
      --generator_type hf \\
      --generator_path Qwen/Qwen3-4B-Instruct-2507-FP8 \\
      --methods element_based \\
      --resume
"""

import os
# Fix: kernels package memerlukan trust_remote_code untuk load FP8 CUDA kernel
# dari kernels-community/finegrained-fp8. Tanpa ini, setiap generate() gagal.
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
# Fix: CUDA memory fragmentation menyebabkan OOM di method ke-3 (recursive).
# expandable_segments memungkinkan PyTorch memakai memory yang tidak kontinu.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.evaluator import build_evaluator, COLLECTION_NAMES
from src.evaluation.metrics import compute_bleu, compute_rouge

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    """Setup dual logging: console (INFO) + file (DEBUG)."""
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(fh)

# ── Paths default ──────────────────────────────────────────────────────────────

QA_GOLD_XLSX  = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"
RESULTS_DIR   = ROOT / "results/final/generation"
EMBEDDER_PATH = ROOT / "models/Qwen3-Embedding-4B"
CHROMA_PATH   = ROOT / "data/chroma"

# ── Default: Qwen3-4B-Instruct-2507-FP8 (sesuai dokumentasi model) ──────────────────
# Prioritas: local models/ folder → HF Hub (butuh internet)
# Download lokal: from huggingface_hub import snapshot_download
#   snapshot_download("Qwen/Qwen3-4B-Instruct-2507-FP8",
#                     local_dir="models/Qwen3-4B-Instruct-2507-FP8")
DEFAULT_GENERATOR_TYPE = "hf"
_LOCAL_GENERATOR        = ROOT / "models/Qwen3-4B-Instruct-2507-FP8"
DEFAULT_GENERATOR_PATH  = str(_LOCAL_GENERATOR) if _LOCAL_GENERATOR.exists() else "Qwen/Qwen3-4B-Instruct-2507-FP8"
DEFAULT_TEMPERATURE    = 0.7    # Instruct: 0.7 (Thinking: 0.6)
DEFAULT_TOP_P          = 0.8    # Instruct: 0.8 (Thinking: 0.95)
DEFAULT_TOP_K_GEN      = 20
DEFAULT_MAX_TOKENS     = 1024   # Eval: kompromi antara dok (16384) dan batas VRAM 6GB; cukup untuk jawaban faktual BPS
DEFAULT_TOP_K          = 8


# ── QA Gold loader ─────────────────────────────────────────────────────────────

def load_qa_gold(path: Path) -> list:
    """Load QA gold dari xlsx → list of {id, question, reference_answer}."""
    df = pd.read_excel(str(path), sheet_name="qa_gold", dtype=str).fillna("")
    rows = [
        {
            "id":               str(r["query_id"]).strip(),
            "question":         str(r["question"]).strip(),
            "reference_answer": str(r["gold_answer"]).strip(),
            "relevant_chunk_ids": {},   # kosong — skip retrieval metrics
        }
        for _, r in df.iterrows()
        if str(r.get("query_id", "")).strip()
    ]
    logger.info(f"[OK] Loaded {len(rows)} QA items dari {path.name}")
    return rows


# ── Output helpers ─────────────────────────────────────────────────────────────

_PER_QUERY_COLS = [
    "method", "q_id", "question", "reference",
    "answer", "bleu", "rouge_l", "elapsed_s", "error",
]

def _build_config(args, methods: list, ts: str) -> dict:
    """Bangun dict konfigurasi run untuk header report."""
    return {
        "timestamp":       ts,
        "generator_type":  args.generator_type,
        "generator_path":  args.generator_path,
        "embedder_mode":   args.embedder_mode,
        "hf_model":        args.hf_model,
        "embedder_path":   args.embedder_path,
        "chroma_path":     args.chroma_path,
        "qa_xlsx":         args.qa_xlsx,
        "methods":         methods,
        "top_k":           args.top_k,
        "max_tokens":      args.max_tokens,
        "temperature":     args.temperature,
        "top_p":           args.top_p,
        "top_k_gen":       args.top_k_gen,
        "n_gpu_layers":    args.n_gpu_layers,
        "return_thinking": args.return_thinking,
        "resume":          args.resume,
    }


def _format_question_block(q_id: str, question: str, reference: str,
                           method_rows: dict) -> list:
    """
    Format satu blok query dengan 3 method berdampingan.
    Digunakan bersama oleh generate_report() dan print_question_comparison().
    """
    sep80  = "=" * 80
    sep80d = "-" * 80
    block  = [
        sep80,
        f"{q_id} | {question}",
        sep80d,
        f"Referensi : {reference}",
        "",
    ]
    for method in COLLECTION_NAMES:
        if method not in method_rows:
            block += [f"[{method}]  — tidak ada data —", ""]
            continue
        r      = method_rows[method]
        bleu   = f"{float(r['bleu']):.4f}"   if r.get("bleu")   not in (None, "", "—") else "—"
        rouge  = f"{float(r['rouge_l']):.4f}" if r.get("rouge_l") not in (None, "", "—") else "—"
        answer = r.get("answer") or "[kosong]"
        if r.get("error"):
            answer = f"[ERROR] {r['error']}"
        block += [
            f"[{method}]  BLEU={bleu}  ROUGE-L={rouge}",
            "Jawaban:",
            answer,
            "",
        ]
    return block


def generate_report(
    per_query: list,
    summary: list,
    config: dict,
    path: Path,
) -> None:
    """
    Generate laporan evaluasi terformat (.txt) untuk dokumentasi skripsi.
    Format question-first: tiap query menampilkan jawaban + skor 3 method sekaligus.
    Konfigurasi run disertakan di header (tidak ada config.json terpisah).
    """
    sep80  = "=" * 80
    sep80d = "-" * 80

    lines = [
        sep80,
        "  LAPORAN EVALUASI GENERASI RAG",
        "  Perbandingan Metode Chunking: element_based | maxmin_semantic | recursive",
        sep80,
        "",
        "KONFIGURASI RUN",
        sep80d,
        f"  Tanggal/Waktu    : {config.get('timestamp', '-')}",
        f"  Model Generator  : {config.get('generator_path', '-')}",
        f"  Tipe Generator   : {config.get('generator_type', '-')}",
        f"  Embedding Model  : {Path(config.get('embedder_path', '-')).name}",
        f"  ChromaDB Path    : {config.get('chroma_path', '-')}",
        f"  QA Gold File     : {Path(config.get('qa_xlsx', '-')).name}",
        f"  Top-K Retrieval  : {config.get('top_k', '-')}",
        f"  Max Tokens Out   : {config.get('max_tokens', '-')}",
        f"  Temperature      : {config.get('temperature', '-')}",
        f"  Top-P            : {config.get('top_p', '-')}",
        f"  Top-K Sampling   : {config.get('top_k_gen', '-')}",
        "",
    ]

    # ── Tabel ringkasan ──────────────────────────────────────────────────────
    lines += [
        "RINGKASAN METRIK — PERBANDINGAN CHUNKING METHOD",
        sep80d,
    ]
    col_w  = 16
    name_w = 22
    cols   = ["mean_bleu", "mean_rouge_l", "n_success", "n_queries"]
    header = f"  {'Method':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    lines.append(header)
    lines.append("  " + "-" * (name_w + col_w * len(cols)))
    for s in summary:
        row = f"  {s['method']:<{name_w}}"
        for c in cols:
            val = s.get(c)
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            elif isinstance(val, int):
                row += f"{val:>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        lines.append(row)
    lines += ["", sep80d, ""]

    # ── Detail per query (question-first) ────────────────────────────────────
    lines += [
        "DETAIL PER QUERY — FORMAT: PERTANYAAN => 3 METODE",
        sep80,
        "",
    ]

    from collections import defaultdict
    by_qid: dict = defaultdict(dict)
    for row in per_query:
        by_qid[row["q_id"]][row["method"]] = row

    for q_id in sorted(by_qid.keys()):
        method_rows = by_qid[q_id]
        sample      = next(iter(method_rows.values()))
        lines.extend(_format_question_block(
            q_id, sample.get("question", ""),
            sample.get("reference", ""), method_rows,
        ))

    lines += [sep80, "  END OF REPORT", sep80]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_per_query_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_QUERY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary: list, path: Path) -> None:
    cols = ["method", "n_queries", "n_success", "mean_bleu", "mean_rouge_l"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def print_question_comparison(per_query: list) -> None:
    """Print perbandingan hasil per query (question-first) ke terminal setelah semua selesai."""
    from collections import defaultdict
    by_qid: dict = defaultdict(dict)
    for row in per_query:
        by_qid[row["q_id"]][row["method"]] = row

    sep = "=" * 80
    print(f"\n{sep}")
    print("  HASIL PER QUERY — PERBANDINGAN 3 METODE")
    print(f"{sep}\n")

    for q_id in sorted(by_qid.keys()):
        method_rows = by_qid[q_id]
        sample      = next(iter(method_rows.values()))
        for line in _format_question_block(
            q_id, sample.get("question", ""),
            sample.get("reference", ""), method_rows,
        ):
            print(line)


# ── Summary builder ────────────────────────────────────────────────────────────

def build_summary(per_query: list) -> list:
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in per_query:
        buckets[row["method"]].append(row)

    summary = []
    for method in COLLECTION_NAMES:                             # tetap urutan konsisten
        rows = buckets.get(method, [])
        if not rows:
            continue
        bleu_vals  = [float(r["bleu"])    for r in rows if r.get("bleu")    not in (None, "", "—")]
        rouge_vals = [float(r["rouge_l"]) for r in rows if r.get("rouge_l") not in (None, "", "—")]
        n_success  = sum(1 for r in rows if r.get("answer"))
        summary.append({
            "method":       method,
            "n_queries":    len(rows),
            "n_success":    n_success,
            "mean_bleu":    round(sum(bleu_vals)  / len(bleu_vals),  6) if bleu_vals  else None,
            "mean_rouge_l": round(sum(rouge_vals) / len(rouge_vals), 6) if rouge_vals else None,
        })
    return summary


def print_summary(summary: list) -> None:
    col_w  = 16
    name_w = 22
    cols   = ["mean_bleu", "mean_rouge_l", "n_success", "n_queries"]
    total_w = name_w + col_w * len(cols)

    print("\n" + "=" * total_w)
    print("  HASIL EVALUASI GENERASI — BLEU + ROUGE-L Recall")
    print("=" * total_w)
    print(f"{'Method':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols))
    print("-" * total_w)
    for s in summary:
        row = f"{s['method']:<{name_w}}"
        for c in cols:
            val = s.get(c)
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            elif isinstance(val, int):
                row += f"{val:>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)
    print("=" * total_w + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation-only evaluation: BLEU + ROUGE-L Recall",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generator — default sudah diset ke Instruct model
    parser.add_argument("--generator_type", default=DEFAULT_GENERATOR_TYPE,
                        choices=["gguf", "hf"],
                        help=f"Backend generator (default: {DEFAULT_GENERATOR_TYPE})")
    parser.add_argument("--generator_path", default=DEFAULT_GENERATOR_PATH,
                        help=f"HF model name atau path .gguf (default: {DEFAULT_GENERATOR_PATH})")

    # QA gold + paths
    parser.add_argument("--qa_xlsx", default=str(QA_GOLD_XLSX),
                        help=f"Path ke QA gold xlsx (default: {QA_GOLD_XLSX.name})")
    parser.add_argument("--embedder_mode", default="huggingface",
                        choices=["gguf", "huggingface"],
                        help="Mode embedder (default: huggingface)")
    parser.add_argument("--hf_model", default="/workspace/models/Qwen3-Embedding-4B",
                        help="Path ke HF embedding model")
    parser.add_argument("--embedder_path", default=str(EMBEDDER_PATH),
                        help="Path ke GGUF embedding model")
    parser.add_argument("--chroma_path", default=str(CHROMA_PATH),
                        help="Path ke ChromaDB storage")

    # Eval config
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(COLLECTION_NAMES.keys()),
                        help="Methods (default: semua 3)")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Jumlah chunk per query (default: {DEFAULT_TOP_K})")

    # Generator sampling — default sesuai Instruct model docs
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                        help=f"Nucleus sampling (default: {DEFAULT_TOP_P})")
    parser.add_argument("--top_k_gen", type=int, default=DEFAULT_TOP_K_GEN,
                        help=f"Top-K sampling (default: {DEFAULT_TOP_K_GEN})")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="GPU layers GGUF (-1 = semua, 0 = CPU only)")
    parser.add_argument("--return_thinking", action="store_true",
                        help="Simpan thinking content (HF Thinking model only)")

    # Output
    parser.add_argument("--output_dir", default=str(RESULTS_DIR),
                        help="Folder output (default: results/generation_eval/)")
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari per_query CSV terakhir (skip yang sudah selesai)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_query_path = out_dir / f"per_query_{ts}.csv"
    summary_path   = out_dir / f"summary_{ts}.csv"
    report_path    = out_dir / f"report_{ts}.txt"
    log_path       = out_dir / f"run_{ts}.log"

    _setup_logging(log_path)

    # ── Load QA gold ──────────────────────────────────────────────────────────
    qa_items = load_qa_gold(Path(args.qa_xlsx))

    methods = args.methods or list(COLLECTION_NAMES.keys())
    config  = _build_config(args, methods, ts)

    # ── Resume: load existing results ─────────────────────────────────────────
    done: set  = set()
    all_rows: list = []
    if args.resume:
        existing = sorted(out_dir.glob("per_query_*.csv"), reverse=True)
        if existing:
            old_df   = pd.read_csv(str(existing[0]), dtype=str).fillna("")
            loaded   = old_df.to_dict("records")
            # Hanya anggap "done" jika ada jawaban dan tidak ada error
            done     = {(r["method"], r["q_id"]) for r in loaded
                        if r.get("answer") and not r.get("error")}
            # Simpan SEMUA rows (termasuk error) agar n_queries tetap 30
            all_rows = loaded
            per_query_path = existing[0]
            n_retry  = sum(1 for r in loaded if r.get("error") or not r.get("answer"))
            logger.info(f"[RESUME] {len(loaded)} rows dari {existing[0].name}")
            if n_retry:
                logger.info(f"[RESUME] {n_retry} rows error akan di-retry")
        else:
            logger.info("[RESUME] Tidak ada file sebelumnya — mulai dari awal")

    # ── Load embedder + ChromaDB (sekali) ─────────────────────────────────────
    logger.info("\n[INIT] Memuat embedder + ChromaDB...")
    try:
        evaluator = build_evaluator(
            embedder_path=args.embedder_path,
            chroma_path=args.chroma_path,
            n_gpu_layers=args.n_gpu_layers,
            embedder_mode=args.embedder_mode,
            hf_model_name=args.hf_model,
        )
    except RuntimeError as e:
        logger.error(f"[FATAL] Gagal memuat evaluator: {e}")
        sys.exit(1)

    # ── Load generator (sekali, shared semua methods) ─────────────────────────
    logger.info(f"[INIT] Memuat generator ({args.generator_type}): {args.generator_path}")
    generator = None
    try:
        if args.generator_type == "hf":
            from src.rag.generator import initialize_hf_generator
            generator = initialize_hf_generator(
                model_name      = args.generator_path,
                max_new_tokens  = args.max_tokens,
                temperature     = args.temperature,
                top_p           = args.top_p,
                top_k           = args.top_k_gen,
                return_thinking = args.return_thinking,
            )
        else:
            from src.rag.generator import initialize_gguf_generator
            generator = initialize_gguf_generator(
                model_path   = args.generator_path,
                max_tokens   = args.max_tokens,
                temperature  = args.temperature,
                top_p        = args.top_p,
                n_gpu_layers = args.n_gpu_layers,
            )
    except Exception as e:
        logger.error(f"[FATAL] Gagal memuat generator: {e}")
        sys.exit(1)

    if generator is None:
        logger.error("[FATAL] Generator None — periksa path/model name")
        sys.exit(1)

    # ── Evaluasi per method ───────────────────────────────────────────────────
    logger.info(f"\n[START] {len(methods)} methods × {len(qa_items)} queries")
    logger.info(f"        top_k={args.top_k}  max_tokens={args.max_tokens}")
    logger.info(f"        temperature={args.temperature}  top_p={args.top_p}\n")

    for method in methods:
        skip_count = sum(1 for r in all_rows if r.get("method") == method)
        todo_items = [
            item for item in qa_items
            if (method, item["id"]) not in done
        ]
        if not todo_items:
            logger.info(f"[SKIP] {method} — semua query sudah selesai")
            continue

        logger.info(f"{'='*64}")
        logger.info(f"  Method: {method}  ({skip_count} done, {len(todo_items)} remaining)")
        logger.info(f"{'='*64}")

        from src.chroma.client import get_or_create_collection
        collection_name = COLLECTION_NAMES[method]
        collection = get_or_create_collection(evaluator.chroma_client, collection_name)
        if collection is None:
            logger.error(f"[ERROR] Collection '{collection_name}' tidak ditemukan — skip")
            continue
        logger.info(f"  Collection: {collection_name} ({collection.count()} docs)")

        for i, item in enumerate(todo_items, 1):
            q_id      = item["id"]
            question  = item["question"]
            reference = item["reference_answer"]

            logger.info(f"  [{i:02d}/{len(todo_items)}] {q_id}: {question[:65]}...")

            row = {
                "method":    method,
                "q_id":      q_id,
                "question":  question,
                "reference": reference,
                "answer":    None,
                "bleu":      None,
                "rouge_l":   None,
                "elapsed_s": None,
                "error":     None,
            }

            try:
                t0 = time.time()

                # Retrieve
                retrieved = evaluator._retrieve(collection, question, top_k=args.top_k)
                contexts  = [r["document"] for r in retrieved]

                # Generate
                raw    = generator.generate(question, contexts)
                answer = raw[0] if isinstance(raw, tuple) else raw
                row["elapsed_s"] = round(time.time() - t0, 2)

                # Bebaskan GPU memory setelah generate agar tidak OOM pada query berikutnya
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                row["answer"]    = answer

                # Metrics
                if answer:
                    row["bleu"]    = round(compute_bleu(answer, reference),                          6)
                    row["rouge_l"] = round(compute_rouge(answer, reference, "rougeL", "recall"),     6)
                    logger.info(
                        f"           BLEU={row['bleu']:.4f}  "
                        f"ROUGE-L={row['rouge_l']:.4f}  "
                        f"({row['elapsed_s']}s)"
                    )
                else:
                    logger.warning("           [WARN] answer kosong")

            except Exception as e:
                logger.error(f"           [ERROR] {e}")
                row["error"] = str(e)

            # Upsert: replace existing row jika ini retry, append jika baru
            idx = next((i for i, r in enumerate(all_rows)
                        if r.get("method") == method and r.get("q_id") == q_id), None)
            if idx is not None:
                all_rows[idx] = row
            else:
                all_rows.append(row)
            done.add((method, q_id))

            # Auto-save setelah setiap query (aman untuk resume)
            save_per_query_csv(all_rows, per_query_path)

    # ── Summary + simpan ─────────────────────────────────────────────────────
    summary = build_summary(all_rows)
    print_summary(summary)
    print_question_comparison(all_rows)

    save_per_query_csv(all_rows, per_query_path)
    save_summary_csv(summary, summary_path)
    generate_report(all_rows, summary, config, report_path)

    logger.info(f"[OK] per_query → {per_query_path}")
    logger.info(f"[OK] summary   → {summary_path}")
    logger.info(f"[OK] report    → {report_path}")
    logger.info(f"[OK] log file  → {log_path}")


if __name__ == "__main__":
    main()
