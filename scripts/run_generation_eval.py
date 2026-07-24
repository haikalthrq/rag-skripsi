"""
Batch RAG evaluation aligned with src/streamlit/rag_chat.py.

The script mirrors the Streamlit "Evaluasi Batch" behavior:
  - QA source: qa_gold_standard_rag_bps_30qa_question_newest.xlsx
  - retrieval GT: qa_pairs_binary.json
  - methods: element_based, maxmin_semantic, recursive
  - metrics: Precision@k, Recall@k, MRR, F1@k, BLEU, ROUGE-L recall
  - timing: retrieval, generation, and total response latency
  - output schema: rag_chat.py batch columns plus benchmark timing
  - output files: one CSV per top-k, eval_<timestamp>_<mode>_top{k}.csv

Default run evaluates full 30 QA for Top-1 through Top-10.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import statistics
import sys
import time
import zipfile
from datetime import datetime, timedelta
from xml.etree import ElementTree
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in requirements, this is defensive.
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is in requirements, this is defensive.
    torch = None

from src.evaluation.metrics import (
    compute_bleu,
    compute_f1_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_rouge,
)
logger = logging.getLogger(__name__)

QA_GOLD_XLSX = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"
GT_BINARY_JSON = ROOT / "data/ground_truth/qa_pairs_binary.json"
RESULTS_DIR = ROOT / "results/final/generation"
CHROMA_PATH = ROOT / "data/chroma"
DEFAULT_CHROMA_PATH = "data/chroma"

COLLECTION_NAMES = {
    "element_based": "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive": "collection_recursive",
}

LOCAL_GEN_BF16 = ROOT / "models/Qwen3-4B-Instruct-2507"
LOCAL_GEN_FP8 = ROOT / "models/Qwen3-4B-Instruct-2507-FP8"
LOCAL_EMBED_HF = ROOT / "models/Qwen3-Embedding-4B"
LOCAL_EMBED_GGUF = ROOT / "models/Qwen3-Embedding-4B-Q8_0.gguf"

DEFAULT_GENERATOR_TYPE = "hf"
DEFAULT_GENERATOR_PATH = (
    str(LOCAL_GEN_BF16)
    if LOCAL_GEN_BF16.exists()
    else str(LOCAL_GEN_FP8)
    if LOCAL_GEN_FP8.exists()
    else "Qwen/Qwen3-4B-Instruct-2507"
)
DEFAULT_EMBEDDER_MODE = "huggingface" if LOCAL_EMBED_HF.exists() else "gguf"
DEFAULT_EMBEDDER_PATH = str(LOCAL_EMBED_HF if LOCAL_EMBED_HF.exists() else LOCAL_EMBED_GGUF)
DEFAULT_HF_MODEL = str(LOCAL_EMBED_HF if LOCAL_EMBED_HF.exists() else ROOT / "models/Qwen3-Embedding-4B")

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K_GEN = 20
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TOP_K_MIN = 1
DEFAULT_TOP_K_MAX = 10

QUICK_EVAL_IDS = ["Q005", "Q010", "Q011", "Q013", "Q020"]

METHODS = list(COLLECTION_NAMES.keys())
METHOD_LABELS = {
    "element_based": "Element-Based",
    "maxmin_semantic": "MaxMin Semantic",
    "recursive": "Recursive",
}

OUTPUT_COLUMNS = [
    "query_id",
    "method",
    "top_k",
    "question",
    "gold_answer",
    "generated_answer",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "f1_at_k",
    "bleu",
    "rouge_l_recall",
    "retrieval_seconds",
    "generation_seconds",
    "total_response_seconds",
    "error",
    "hardware_info",
]


def _patch_hf_user_agent() -> None:
    """Keep parity with rag_chat.py for HF Hub user-agent edge cases."""
    try:
        import huggingface_hub.utils._headers as headers

        original = headers._deduplicate_user_agent

        def fixed(user_agent: str) -> str:
            return original(user_agent).rstrip("; ").rstrip(";")

        headers._deduplicate_user_agent = fixed
    except Exception:
        pass


_patch_hf_user_agent()


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)


def get_hardware_info() -> dict[str, Any]:
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


def _synchronize_cuda() -> None:
    """Wait for queued CUDA work so latency measurements are accurate."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_qa_gold(path: Path) -> list[dict[str, Any]]:
    """Load QA gold and keep the legacy fields expected by tests."""
    try:
        import pandas as pd

        df = pd.read_excel(str(path), sheet_name="qa_gold", dtype=str).fillna("")
        records = df.to_dict("records")
    except ImportError:
        records = _load_qa_gold_xlsx_stdlib(path)

    rows = []
    for row in records:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        rows.append({
            "id": query_id,
            "query_id": query_id,
            "question": str(row.get("question", "")).strip(),
            "reference_answer": str(row.get("gold_answer", "")).strip(),
            "gold_answer": str(row.get("gold_answer", "")).strip(),
            "relevant_chunk_ids": {},
        })
    logger.info("Loaded %s QA items from %s", len(rows), path.name)
    return rows


def _load_qa_gold_xlsx_stdlib(path: Path) -> list[dict[str, str]]:
    """Read the qa_gold sheet without pandas/openpyxl for lightweight tests."""
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                texts = [node.text or "" for node in item.findall(".//main:t", ns)]
                shared_strings.append("".join(texts))

        workbook = ElementTree.fromstring(zf.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkg:Relationship", ns)
        }
        sheet_target = None
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            if sheet.attrib.get("name") == "qa_gold":
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                sheet_target = rel_targets.get(rel_id)
                break
        if not sheet_target:
            raise ValueError("Sheet qa_gold not found")
        sheet_path = sheet_target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = "xl/" + sheet_path
        sheet_xml = ElementTree.fromstring(zf.read(sheet_path))

    def cell_value(cell) -> str:
        if cell.attrib.get("t") == "inlineStr":
            texts = [node.text or "" for node in cell.findall(".//main:t", ns)]
            return "".join(texts)
        value = cell.find("main:v", ns)
        if value is None or value.text is None:
            return ""
        text = value.text
        if cell.attrib.get("t") == "s":
            return shared_strings[int(text)]
        return text

    def column_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return max(idx - 1, 0)

    table: list[list[str]] = []
    for row in sheet_xml.findall("main:sheetData/main:row", ns):
        values: list[str] = []
        for cell in row.findall("main:c", ns):
            col_idx = column_index(cell.attrib.get("r", "A1"))
            while len(values) <= col_idx:
                values.append("")
            values[col_idx] = cell_value(cell)
        table.append(values)
    if not table:
        return []
    headers = [str(value).strip() for value in table[0]]
    return [
        {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        for row in table[1:]
    ]


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(item.get("id")): item for item in data}


def build_summary(per_query: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows; keeps mean_bleu/mean_rouge_l compatibility for tests."""
    def normalize_top_k(value: Any) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    summary = []
    for method in METHODS:
        labels = {method, METHOD_LABELS[method]}
        method_rows = [row for row in per_query if row.get("method") in labels]
        top_k_values = sorted(
            {normalize_top_k(row.get("top_k")) for row in method_rows},
            key=lambda value: (value is None, value or 0),
        )
        for top_k in top_k_values:
            rows = [row for row in method_rows if normalize_top_k(row.get("top_k")) == top_k]

            def numeric_values(key: str, fallback: str | None = None) -> list[float]:
                values = []
                for row in rows:
                    value = row.get(key, row.get(fallback) if fallback else None)
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        pass
                return values

            def latency_stats(
                values: list[float],
            ) -> tuple[float | None, float | None, float | None]:
                if not values:
                    return None, None, None
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                return (
                    round(statistics.mean(values), 6),
                    round(statistics.median(values), 6),
                    round(std, 6),
                )

            bleu_vals = numeric_values("bleu")
            rouge_vals = numeric_values("rouge_l_recall", "rouge_l")
            precision_vals = numeric_values("precision_at_k")
            recall_vals = numeric_values("recall_at_k")
            mrr_vals = numeric_values("mrr")
            f1_vals = numeric_values("f1_at_k")
            retrieval_vals = numeric_values("retrieval_seconds")
            generation_vals = numeric_values("generation_seconds")
            total_vals = numeric_values("total_response_seconds")
            mean_retrieval, median_retrieval, std_retrieval = latency_stats(retrieval_vals)
            mean_generation, median_generation, std_generation = latency_stats(generation_vals)
            mean_total, median_total, std_total = latency_stats(total_vals)

            summary.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "top_k": top_k,
                "n_queries": len(rows),
                "n_success": sum(1 for row in rows if row.get("generated_answer") or row.get("answer")),
                "n_retrieval_evaluated": len(precision_vals),
                "n_timed": len(total_vals),
                "mean_precision_at_k": round(statistics.mean(precision_vals), 6) if precision_vals else None,
                "mean_recall_at_k": round(statistics.mean(recall_vals), 6) if recall_vals else None,
                "mean_mrr": round(statistics.mean(mrr_vals), 6) if mrr_vals else None,
                "mean_f1_at_k": round(statistics.mean(f1_vals), 6) if f1_vals else None,
                "mean_bleu": round(statistics.mean(bleu_vals), 6) if bleu_vals else None,
                "mean_rouge_l": round(statistics.mean(rouge_vals), 6) if rouge_vals else None,
                "mean_retrieval_seconds": mean_retrieval,
                "median_retrieval_seconds": median_retrieval,
                "std_retrieval_seconds": std_retrieval,
                "mean_generation_seconds": mean_generation,
                "median_generation_seconds": median_generation,
                "std_generation_seconds": std_generation,
                "mean_total_response_seconds": mean_total,
                "median_total_response_seconds": median_total,
                "std_total_response_seconds": std_total,
            })
    return summary


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_done(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            {key: (value if value is not None else "") for key, value in row.items()}
            for row in csv.DictReader(f)
        ]
    done = {
        (str(row["query_id"]), str(row["method"]))
        for row in rows
        if row.get("generated_answer")
        and not row.get("error")
        and row.get("total_response_seconds") not in (None, "")
    }
    return rows, done


def upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["query_id"], row["method"])
    for idx, existing in enumerate(rows):
        if (existing.get("query_id"), existing.get("method")) == key:
            rows[idx] = row
            return
    rows.append(row)


def precompute_query_embeddings(pipeline: Any, qa_items: list[dict[str, Any]]) -> dict[str, tuple[Any, bool]]:
    query_embeddings: dict[str, tuple[Any, bool]] = {}
    for i, item in enumerate(qa_items, 1):
        q_id = item["query_id"]
        logger.info("Pre-computing embedding %s/%s: %s", i, len(qa_items), q_id)
        try:
            query_embeddings[q_id] = (pipeline.embedder.embed(item["question"])[0], True)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", q_id, exc)
            query_embeddings[q_id] = (None, False)
    return query_embeddings


def build_pipeline_from_args(args: argparse.Namespace) -> Any:
    from src.rag.pipeline import build_pipeline

    embedder_path = args.hf_model if args.embedder_mode == "huggingface" else args.embedder_path
    return build_pipeline(
        chunking_method="element_based",
        embedder_path=embedder_path,
        generator_path=args.generator_path,
        generator_type=args.generator_type,
        embedder_mode=args.embedder_mode,
        chroma_path=args.chroma_path,
        top_k=args.top_k_max,
        n_gpu_layers=args.n_gpu_layers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k_gen=args.top_k_gen,
        return_thinking=args.return_thinking,
    )


def evaluate_top_k(
    pipeline: Any,
    qa_items: list[dict[str, Any]],
    gt_lookup: dict[str, dict[str, Any]],
    query_embeddings: dict[str, tuple[Any, bool]],
    methods: list[str],
    current_k: int,
    existing_rows: list[dict[str, Any]],
    done: set[tuple[str, str]],
    output_path: Path,
    hardware_info: str,
) -> list[dict[str, Any]]:
    rows = existing_rows

    for item in qa_items:
        q_id = item["query_id"]
        question = item["question"]
        gold_answer = item["gold_answer"]
        gt_item = gt_lookup.get(q_id)
        q_vec, embed_ok = query_embeddings.get(q_id, (None, False))

        for method in methods:
            method_label = METHOD_LABELS[method]
            if (q_id, method_label) in done:
                continue

            row: dict[str, Any] = {
                "query_id": q_id,
                "method": method_label,
                "top_k": current_k,
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": None,
                "precision_at_k": None,
                "recall_at_k": None,
                "mrr": None,
                "f1_at_k": None,
                "bleu": None,
                "rouge_l_recall": None,
                "retrieval_seconds": None,
                "generation_seconds": None,
                "total_response_seconds": None,
                "error": "",
                "hardware_info": hardware_info,
            }

            logger.info("[%s] %s top-%s", q_id, method_label, current_k)
            try:
                from src.rag.pipeline import RAGPipeline

                p = RAGPipeline(
                    embedder=pipeline.embedder,
                    generator=pipeline.generator,
                    chroma_client=pipeline.chroma_client,
                    chunking_method=method,
                    top_k=current_k,
                )

                _synchronize_cuda()
                response_started = time.perf_counter()
                retrieved = (
                    p.retrieve_by_vector(q_vec, k=current_k)
                    if embed_ok
                    else p.retrieve(question, k=current_k)
                )
                _synchronize_cuda()
                row["retrieval_seconds"] = round(time.perf_counter() - response_started, 6)
                retrieved_ids = [doc.get("id", "") for doc in retrieved]

                contexts = [p._format_context(doc) for doc in retrieved]
                _synchronize_cuda()
                generation_started = time.perf_counter()
                raw = pipeline.generator.generate(question, contexts)
                _synchronize_cuda()
                row["generation_seconds"] = round(time.perf_counter() - generation_started, 6)
                row["total_response_seconds"] = round(time.perf_counter() - response_started, 6)
                answer = raw[0] if isinstance(raw, tuple) else raw
                row["generated_answer"] = answer

                rel_ids: list[str] = []
                if gt_item:
                    rel_all = gt_item.get("relevant_chunk_ids", {})
                    rel_ids = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all
                if rel_ids:
                    precision = compute_precision_at_k(retrieved_ids, rel_ids, current_k)
                    recall = compute_recall_at_k(retrieved_ids, rel_ids, current_k)
                    row["precision_at_k"] = round(precision, 4)
                    row["recall_at_k"] = round(recall, 4)
                    row["mrr"] = round(compute_mrr(retrieved_ids, rel_ids), 4)
                    row["f1_at_k"] = round(
                        compute_f1_at_k(precision, recall),
                        4,
                    )
                else:
                    row["precision_at_k"] = "N/A"
                    row["recall_at_k"] = "N/A"
                    row["mrr"] = "N/A"
                    row["f1_at_k"] = "N/A"
                row["bleu"] = round(compute_bleu(answer, gold_answer), 4)
                row["rouge_l_recall"] = round(
                    compute_rouge(
                        answer,
                        gold_answer,
                        rouge_type="rougeL",
                        mode="recall",
                    ),
                    4,
                )

                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                row["generated_answer"] = f"[ERROR] {exc}"
                row["error"] = str(exc)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            upsert_row(rows, row)
            save_rows(rows, output_path)

    return rows


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nBatch evaluation summary")
    print("-" * 143)
    print(
        f"{'Method':<22} {'K':>3} {'N':>4} {'Eval':>5} {'P@k':>9} "
        f"{'R@k':>9} {'F1@k':>9} {'MRR':>9} {'BLEU':>9} {'ROUGE-L':>9} "
        f"{'Retrieval(s)':>13} {'Generation(s)':>14} {'Total(s)':>10}"
    )
    print("-" * 143)
    for row in summary:
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{float(value):.4f}"
        print(
            f"{row['method_label']:<22} {str(row.get('top_k') or '-'):>3} "
            f"{row['n_queries']:>4} {row['n_retrieval_evaluated']:>5} "
            f"{fmt(row['mean_precision_at_k']):>9} {fmt(row['mean_recall_at_k']):>9} "
            f"{fmt(row['mean_f1_at_k']):>9} {fmt(row['mean_mrr']):>9} {fmt(row['mean_bleu']):>9} "
            f"{fmt(row['mean_rouge_l']):>9} {fmt(row['mean_retrieval_seconds']):>13} "
            f"{fmt(row['mean_generation_seconds']):>14} {fmt(row['mean_total_response_seconds']):>10}"
        )
    print("-" * 143)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RAG evaluation aligned with rag_chat.py")
    parser.add_argument("--qa_xlsx", default=str(QA_GOLD_XLSX))
    parser.add_argument("--gt", default=str(GT_BINARY_JSON), help="Binary retrieval ground truth JSON")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR))
    parser.add_argument("--mode_tag", choices=["full", "quick"], default="full")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--top_k_min", type=int, default=DEFAULT_TOP_K_MIN)
    parser.add_argument("--top_k_max", type=int, default=DEFAULT_TOP_K_MAX)
    parser.add_argument("--top_k", type=int, default=None, help="Shortcut: evaluate one top-k only")

    parser.add_argument("--generator_type", choices=["gguf", "hf"], default=DEFAULT_GENERATOR_TYPE)
    parser.add_argument("--generator_path", default=DEFAULT_GENERATOR_PATH)
    parser.add_argument("--embedder_mode", choices=["gguf", "huggingface"], default=DEFAULT_EMBEDDER_MODE)
    parser.add_argument("--hf_model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--embedder_path", default=DEFAULT_EMBEDDER_PATH)
    parser.add_argument("--chroma_path", default=str(CHROMA_PATH if CHROMA_PATH.exists() else ROOT / DEFAULT_CHROMA_PATH))

    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k_gen", type=int, default=DEFAULT_TOP_K_GEN)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--return_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k is not None:
        args.top_k_min = args.top_k
        args.top_k_max = args.top_k
    # Catatan: entry point standalone ini hanya mendukung Top-1 sampai Top-10.
    # Artefak Top-11 sampai Top-20 yang ada dibuat melalui workflow Streamlit.
    if not 1 <= args.top_k_min <= args.top_k_max <= 10:
        raise ValueError("Top-k range must satisfy 1 <= min <= max <= 10")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_wib = (datetime.now() + timedelta(hours=7)).strftime("%Y%m%d_%H%M%S")
    _setup_logging(out_dir / f"run_{ts_wib}_{args.mode_tag}_top{args.top_k_min}-{args.top_k_max}.log")

    qa_items = load_qa_gold(Path(args.qa_xlsx))
    if args.mode_tag == "quick":
        qa_items = [item for item in qa_items if item["query_id"] in QUICK_EVAL_IDS]
    if not qa_items:
        raise RuntimeError("No QA items selected")

    gt_lookup = load_ground_truth(Path(args.gt))
    methods = args.methods or METHODS
    hardware_info = json.dumps(get_hardware_info(), ensure_ascii=False)

    logger.info("Loading pipeline")
    pipeline = build_pipeline_from_args(args)
    query_embeddings = precompute_query_embeddings(pipeline, qa_items)

    all_rows_for_summary: list[dict[str, Any]] = []
    for current_k in range(args.top_k_min, args.top_k_max + 1):
        output_path = out_dir / f"eval_{ts_wib}_{args.mode_tag}_top{current_k}.csv"
        existing_rows: list[dict[str, Any]] = []
        done: set[tuple[str, str]] = set()
        if args.resume:
            existing = sorted(out_dir.glob(f"eval_*_{args.mode_tag}_top{current_k}.csv"), reverse=True)
            if existing:
                output_path = existing[0]
                existing_rows, done = load_existing_done(output_path)
                logger.info("Resume top-%s from %s (%s done)", current_k, output_path.name, len(done))

        rows = evaluate_top_k(
            pipeline=pipeline,
            qa_items=qa_items,
            gt_lookup=gt_lookup,
            query_embeddings=query_embeddings,
            methods=methods,
            current_k=current_k,
            existing_rows=existing_rows,
            done=done,
            output_path=output_path,
            hardware_info=hardware_info,
        )
        all_rows_for_summary.extend(rows)
        logger.info("Saved top-%s CSV: %s", current_k, output_path)

    summary = build_summary(all_rows_for_summary)
    summary_path = out_dir / f"summary_{ts_wib}_{args.mode_tag}_top{args.top_k_min}-{args.top_k_max}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print_summary(summary)
    logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
