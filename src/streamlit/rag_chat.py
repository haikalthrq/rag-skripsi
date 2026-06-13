"""
RAG Chat — Streamlit demo app untuk interaksi dengan RAG pipeline.

Mendukung 2 mode:
  - Single method : pilih satu metode chunking, tampil 1 jawaban
  - Bandingkan 3  : tampil 3 kolom berdampingan, cocok untuk demo sidang

Model di-load sekali via @st.cache_resource — tidak reload tiap query.

Jalankan:
  streamlit run src/streamlit/rag_chat.py
"""

# ── Fix: huggingface_hub user-agent header bug ────────────────────────────────
# huggingface_hub 1.14.0 + kernels 0.14.0 produce a user-agent string with
# trailing "; " which h11 rejects as "Illegal header value".
# Monkey-patch _deduplicate_user_agent BEFORE any HF import to strip it.
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # reduce noise

def _patch_hf_user_agent() -> None:
    """Patch huggingface_hub user-agent to strip trailing semicolons."""
    try:
        import huggingface_hub.utils._headers as _hdr
        _orig = _hdr._deduplicate_user_agent

        def _fixed(ua: str) -> str:
            return _orig(ua).rstrip("; ").rstrip(";")

        _hdr._deduplicate_user_agent = _fixed
    except Exception:
        pass  # jika HF hub belum terinstall atau berubah, skip

_patch_hf_user_agent()

import html
import io
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psutil
import torch

import streamlit as st

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Eval persistence & quick-eval subset ─────────────────────────────────────
EVAL_RESULTS_DIR = ROOT / "results" / "final" / "generation"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 3 table-lookup + 2 narrative/definition — subset stabil untuk demo cepat
QUICK_EVAL_IDS = ["Q005", "Q010", "Q011", "Q013", "Q020"]

from src.rag.pipeline import (
    RAGPipeline,
    COLLECTION_NAMES,
    build_pipeline,
    DEFAULT_EMBEDDER_PATH,
    DEFAULT_CHROMA_PATH,
)
from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_f1_at_k,
)

# ── Chat history persistence (untuk dokumentasi sidang) ──────────────────────
# Disimpan ke disk (JSONL) agar TIDAK hilang saat Streamlit restart/mati.
CHAT_HISTORY_DIR  = ROOT / "results" / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "chat_history.jsonl"

# Batas panjang teks chunk yang disimpan per entri (jaga ukuran file tetap wajar)
_CHUNK_TEXT_CAP = 5000


def _extract_chunk_records(retrieved: list) -> list:
    """Ubah hasil retrieve menjadi record ringkas untuk disimpan (adaptif top-k)."""
    chunks = []
    for i, chunk in enumerate(retrieved, 1):
        meta = chunk.get("metadata", {}) or {}
        dist = chunk.get("distance")
        chunks.append({
            "rank":     i,
            "chunk_id": chunk.get("id", "-"),
            "source":   Path(meta.get("source_file", "?")).name,
            "pages":    meta.get("page_numbers", "-"),
            "distance": round(dist, 4) if isinstance(dist, (int, float)) else None,
            "text":     (chunk.get("document", "") or "")[:_CHUNK_TEXT_CAP],
        })
    return chunks


def _build_chat_record(query: str, mode: str, top_k: int,
                       gold: str, results: list) -> dict:
    """Bangun satu record turn chat untuk disimpan."""
    return {
        "id":          datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "timestamp":   (datetime.now() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"),
        "query":       query,
        "mode":        mode,
        "top_k":       top_k,
        "gold_answer": gold,
        "results":     results,
    }


def _save_chat_turn(record: dict) -> None:
    """Append satu turn chat ke JSONL (persisten, tahan restart)."""
    try:
        with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Gagal menyimpan chat history: {e}")


def _load_chat_history() -> list:
    """Load semua turn chat dari JSONL (terbaru di urutan depan)."""
    if not CHAT_HISTORY_FILE.exists():
        return []
    records = []
    try:
        with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Gagal memuat chat history: {e}")
    return list(reversed(records))

# ── Load QA Gold Standard (for BLEU/ROUGE scoring) ───────────────────────────────

_QA_GOLD_DF = None
_GT_BINARY  = None   # label >= 1 dianggap relevan (skema binary 0/1)

def _load_qa_gold():
    """Load QA gold standard for reference answers."""
    global _QA_GOLD_DF
    if _QA_GOLD_DF is None:
        try:
            import pandas as pd
            qa_path = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"
            if qa_path.exists():
                _QA_GOLD_DF = pd.read_excel(qa_path, sheet_name="qa_gold")
                logger.info(f"Loaded QA gold: {len(_QA_GOLD_DF)} queries")
            else:
                _QA_GOLD_DF = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not load QA gold: {e}")
            _QA_GOLD_DF = pd.DataFrame()
    return _QA_GOLD_DF


def _load_ground_truth():
    """Load ground truth JSON binary (label >= 1 = relevan) untuk evaluasi retrieval.

    Menggunakan skema label binary:
      1 = relevan (mencakup label anotasi 1 dan 2 yang digabung)
      0 = tidak relevan
    File: data/ground_truth/qa_pairs_binary.json

    Returns:
        List of QA pairs with relevant_chunk_ids
    """
    global _GT_BINARY
    if _GT_BINARY is None:
        try:
            gt_path = ROOT / "data/ground_truth/qa_pairs_binary.json"
            if gt_path.exists():
                with open(gt_path, encoding="utf-8") as f:
                    _GT_BINARY = json.load(f)
                logger.info(f"Loaded binary ground truth: {len(_GT_BINARY)} QA pairs")
            else:
                _GT_BINARY = []
                logger.warning("qa_pairs_binary.json tidak ditemukan")
        except Exception as e:
            logger.warning(f"Could not load binary ground truth: {e}")
            _GT_BINARY = []
    return _GT_BINARY

def _compute_chat_retrieval_metrics(
    q_id: str | None,
    method: str,
    retrieved: list,
    top_k: int,
) -> dict | None:
    """Compute retrieval metrics (binary ground truth: label >= 1 = relevan)."""
    if not q_id:
        return None
    gt_items = _load_ground_truth()
    gt_item = next((item for item in gt_items if str(item.get("id")) == q_id), None)
    if not gt_item:
        return None
    rel_all = gt_item.get("relevant_chunk_ids", {})
    rel_ids = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all
    if not rel_ids:
        return None
    retrieved_ids = [doc.get("id", "") for doc in retrieved]
    precision_at_k = compute_precision_at_k(retrieved_ids, rel_ids, top_k)
    recall_at_k = compute_recall_at_k(retrieved_ids, rel_ids, top_k)
    return {
        "top_k": top_k,
        "n_relevant": len(rel_ids),
        "precision_at_k": precision_at_k,
        "recall_at_k":    recall_at_k,
        "mrr":            compute_mrr(retrieved_ids, rel_ids),
        "f1_at_k":        compute_f1_at_k(precision_at_k, recall_at_k),
    }


def _section_header(text: str) -> None:
    """Render section header konsisten (lebih besar dari isi, dark-mode safe)."""
    st.markdown(f"<div class='rag-section'>{html.escape(text)}</div>",
                unsafe_allow_html=True)


def _render_retrieval_metrics(metrics: dict | None, bleu: float | None = None, rouge: float | None = None) -> None:
    """Render Metrik Evaluasi — HTML table, center-aligned, font konsisten."""
    has_gen = isinstance(bleu, float) or isinstance(rouge, float)
    has_ret = metrics is not None

    if not has_gen and not has_ret:
        return

    row = {}
    if has_ret:
        k = metrics.get("top_k", "-")
        precision = metrics.get("precision_at_k")
        recall = metrics.get("recall_at_k")
        mrr = metrics.get("mrr")
        f1 = metrics.get("f1_at_k")

        if f1 is None and isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            f1 = compute_f1_at_k(precision, recall)

        row[f"Precision@{k}"] = f"{precision:.4f}" if isinstance(precision, (int, float)) else "—"
        row[f"Recall@{k}"]    = f"{recall:.4f}"    if isinstance(recall, (int, float))    else "—"
        row["MRR"]             = f"{mrr:.4f}"       if isinstance(mrr, (int, float))       else "—"
        row[f"F1@{k}"]         = f"{f1:.4f}"        if isinstance(f1, (int, float))        else "—"
    if has_gen:
        row["BLEU"]    = f"{bleu:.4f}"  if isinstance(bleu,  float) else "—"
        row["ROUGE-L"] = f"{rouge:.4f}" if isinstance(rouge, float) else "—"

    headers = "".join(f"<th>{k}</th>" for k in row)
    values  = "".join(f"<td>{v}</td>" for v in row.values())

    _section_header("Metrik Evaluasi")
    st.markdown(f"""
<table class="rag-metrics" style="width:100%;border-collapse:collapse;font-size:0.9rem;text-align:center;">
  <thead>
    <tr style="border-bottom:1px solid #888;">
      {headers}
    </tr>
  </thead>
  <tbody>
    <tr>{values}</tr>
  </tbody>
</table>
""", unsafe_allow_html=True)


def get_hardware_info() -> dict:
    """Collect hardware information for logging.
    
    Returns:
        Dictionary with GPU/CPU/VRAM info
    """
    hw_info = {
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }
    
    if torch.cuda.is_available():
        hw_info["gpu_available"] = True
        hw_info["gpu_count"] = torch.cuda.device_count()
        hw_info["gpu_name"] = torch.cuda.get_device_name(0)
        
        props = torch.cuda.get_device_properties(0)
        hw_info["gpu_vram_total_gb"] = round(props.total_memory / (1024**3), 2)
        hw_info["gpu_vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        hw_info["gpu_vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        hw_info["gpu_vram_free_gb"] = hw_info["gpu_vram_total_gb"] - hw_info["gpu_vram_reserved_gb"]
    else:
        hw_info["gpu_available"] = False
    
    return hw_info

# ── Defaults & auto-detect environment ───────────────────────────────────────
# Deteksi otomatis: Vast AI (HF BF16) vs Laptop (GGUF FP8 + CPU embedder)

_LOCAL_GEN_BF16   = ROOT / "models/Qwen3-4B-Instruct-2507"         # Vast AI BF16
_LOCAL_GEN_FP8    = ROOT / "models/Qwen3-4B-Instruct-2507-FP8"     # Laptop FP8
_LOCAL_EMBED_HF   = ROOT / "models/Qwen3-Embedding-4B"             # Vast AI HF safetensors
_LOCAL_EMBED_GGUF = ROOT / "models/Qwen3-Embedding-4B-Q8_0.gguf"   # Laptop GGUF

# Tentukan mode embedder
if _LOCAL_EMBED_HF.exists():
    _EMBEDDER_MODE = "huggingface"
    _EMBEDDER_PATH = str(_LOCAL_EMBED_HF)
    _EMBEDDER_DEVICE_NOTE = "GPU (HF safetensors)"
else:
    _EMBEDDER_MODE = "gguf"
    _EMBEDDER_PATH = str(_LOCAL_EMBED_GGUF)
    _EMBEDDER_DEVICE_NOTE = "CPU (GGUF, hemat VRAM untuk generator)"

# Tentukan generator
if _LOCAL_GEN_BF16.exists():
    DEFAULT_GEN_TYPE = "hf"
    DEFAULT_GEN_PATH = str(_LOCAL_GEN_BF16)
elif _LOCAL_GEN_FP8.exists():
    DEFAULT_GEN_TYPE = "hf"
    DEFAULT_GEN_PATH = str(_LOCAL_GEN_FP8)
else:
    DEFAULT_GEN_TYPE = "hf"
    DEFAULT_GEN_PATH = "Qwen/Qwen3-4B-Instruct-2507"  # HF Hub fallback

DEFAULT_TEMP      = 0.7
DEFAULT_TOP_P     = 0.8
DEFAULT_TOP_K_GEN = 20
DEFAULT_MAX_TOK   = 16384  # Sesuai rekomendasi output length resmi model card Qwen3-4B-Instruct-2507.
                           # Batas atas (bukan alokasi muka); KV cache tumbuh mengikuti token yang benar2 di-generate.
DEFAULT_TOP_K     = 8
METHODS           = list(COLLECTION_NAMES.keys())
METHOD_LABELS     = {
    "element_based":   "Element-Based",
    "maxmin_semantic": "MaxMin Semantic",
    "recursive":       "Recursive",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Evaluasi RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — minimal, dark-mode-safe ────────────────────────────────────────────
st.markdown("""
<style>
.chunk-meta {
    font-size: 0.82rem;
    font-weight: 600;
    opacity: 0.75;
    margin-bottom: 2px;
    font-family: monospace;
}
/* Section header — konsisten di semua section hasil generasi.
   Lebih besar dari isi (1.25rem vs ~1rem) agar jelas terpisah, dark-mode safe. */
.rag-section {
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.3;
    margin: 1.1rem 0 0.35rem 0;
    padding: 0;
}
/* Metrik Evaluasi table — padding sel agar rapi saat di-screenshot */
.rag-metrics th, .rag-metrics td {
    padding: 6px 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Model loader (cache) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat model (hanya sekali)...")
def load_pipeline() -> RAGPipeline:
    """
    Load embedder + generator + ChromaDB. Di-cache — tidak reload tiap query.

    Auto-detect environment:
      - Vast AI  : HF safetensors embedder (GPU) + BF16 generator
      - Laptop   : GGUF embedder (CPU, hemat VRAM) + FP8 generator (GPU RTX 4050)
    ChromaDB: data/chroma/ (lokal di kedua environment)
    """
    return build_pipeline(
        chunking_method="element_based",
        embedder_path=_EMBEDDER_PATH,
        generator_path=DEFAULT_GEN_PATH,
        generator_type=DEFAULT_GEN_TYPE,
        embedder_mode=_EMBEDDER_MODE,
        chroma_path=str(ROOT / DEFAULT_CHROMA_PATH),
        top_k=DEFAULT_TOP_K,
        temperature=DEFAULT_TEMP,
        top_p=DEFAULT_TOP_P,
        top_k_gen=DEFAULT_TOP_K_GEN,
        max_tokens=DEFAULT_MAX_TOK,
        return_thinking=False,
    )


def run_method(base_pipeline: RAGPipeline, method: str,
               query: str, top_k: int) -> dict:
    """Reuse embedder + generator, ganti collection sesuai method."""
    p = RAGPipeline(
        embedder=base_pipeline.embedder,
        generator=base_pipeline.generator,
        chroma_client=base_pipeline.chroma_client,
        chunking_method=method,
        top_k=top_k,
    )
    return p.run(query)


def render_answer_box(answer: str) -> None:
    """Render generated answer safely (non-streaming fallback)."""
    st.write(answer)


def stream_answer(generator, query: str, contexts: list,
                  timer_placeholder=None, t0: float = None) -> str:
    """
    Stream jawaban ke UI via st.write_stream, return full answer string.
    Jika timer_placeholder dan t0 diberikan, update timer live selama streaming.
    Fallback ke generate() biasa jika generator bukan HFRAGGenerator.
    """
    from src.rag.generator import HFRAGGenerator

    if isinstance(generator, HFRAGGenerator):
        def _timed_stream():
            last_tick = time.time()
            for token in generator.generate_stream(query, contexts):
                if timer_placeholder is not None and t0 is not None:
                    now = time.time()
                    if now - last_tick >= 0.5:  # update tiap 0.5s
                        timer_placeholder.caption(f"⏳ {now - t0:.1f}s...")
                        last_tick = now
                yield token

        full = st.write_stream(_timed_stream())
        return full if isinstance(full, str) else "".join(full)
    else:
        raw = generator.generate(query, contexts)
        answer = raw[0] if isinstance(raw, tuple) else raw
        render_answer_box(answer)
        return answer


def render_generation_error(error: Exception) -> None:
    """Show generation errors instead of silently rendering an empty dash."""
    st.error(f"Generation gagal: {error}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc(), language="python")


def _render_chunks(retrieved: list, show: bool = True) -> None:
    """Tampilkan Retrieved Chunks — tiap chunk sebagai expander collapsed (ringkas untuk screenshot)."""
    if not show or not retrieved:
        return
    _section_header("Retrieved Chunks")
    for i, chunk in enumerate(retrieved, 1):
        meta      = chunk.get("metadata", {}) or {}
        chunk_id  = chunk.get("id", "-")
        src       = Path(meta.get("source_file", "?")).name
        pages     = meta.get("page_numbers", "-")
        dist      = chunk.get("distance")
        dist_str  = f"{dist:.4f}" if dist is not None else "-"
        full_text = chunk.get("document", "")
        label     = f"[{i}] {chunk_id}  ·  {src}  ·  hal {pages}  ·  dist {dist_str}"
        with st.expander(label, expanded=False):
            st.code(full_text, language=None)


def _render_history_turn(record: dict) -> None:
    """Render satu turn chat tersimpan — dark-mode safe, Streamlit native."""
    ts      = record.get("timestamp", "-")
    query   = record.get("query", "")
    mode    = record.get("mode", "-")
    top_k   = record.get("top_k", "-")
    gold    = record.get("gold_answer")
    results = record.get("results", [])

    st.caption(f"🕒 {ts} · Mode: {mode} · Top-K: {top_k}")
    _section_header("User Question")
    st.write(query)

    if gold:
        _section_header("Ground Truth Answer")
        st.markdown(gold)

    for res in results:
        method_label = res.get("method", "-")
        answer       = res.get("answer", "") or "[kosong]"
        bleu         = res.get("bleu")
        rouge        = res.get("rouge_l")
        elapsed      = res.get("elapsed_s")
        chunks       = res.get("chunks", [])

        st.markdown(f"**{method_label}**")
        _section_header("Generated Answer")
        st.write(answer)

        ret = res.get("retrieval") or res.get("retrieval_strict") or res.get("retrieval_lenient")
        _render_retrieval_metrics(
            ret,
            bleu if isinstance(bleu, float) else None,
            rouge if isinstance(rouge, float) else None,
        )
        if elapsed is not None:
            st.caption(f"⏱ {elapsed}s")

        if chunks:
            _section_header("Retrieved Chunks")
            for ch in chunks:
                rank     = ch.get("rank", "?")
                chunk_id = ch.get("chunk_id", ch.get("id", "-"))
                src      = ch.get("source", "?")
                pages    = ch.get("pages", "-")
                dist     = ch.get("distance")
                dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "-"
                label    = f"[{rank}] {chunk_id}  ·  {src}  ·  hal {pages}  ·  dist {dist_str}"
                with st.expander(label, expanded=False):
                    st.code(ch.get("text", ""), language=None)
    st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Konfigurasi")

    compare_mode = st.toggle("Bandingkan 3 Metode", value=False,
                             help="Tampilkan jawaban dari ketiga metode chunking sekaligus")

    if not compare_mode:
        selected_method = st.selectbox(
            "Metode Chunking",
            options=METHODS,
            format_func=lambda m: METHOD_LABELS[m],
        )
    else:
        selected_method = None

    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    show_chunks = st.checkbox("Tampilkan retrieved chunks", value=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔍 Evaluasi RAG")

# Load pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    with st.expander("🔍 Detail error (untuk debugging)"):
        st.code(traceback.format_exc(), language="python")
    st.stop()

tab_chat, tab_eval, tab_history = st.tabs(["💬 Chat", "📊 Evaluasi Batch", "🕒 Riwayat Chat"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat interaktif
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    qa_df_chat = _load_qa_gold()
    selected_qa_idx = None
    query = ""
    q_id = None
    gold = None

    if qa_df_chat.empty:
        st.warning("QA gold standard tidak ditemukan, chat belum bisa dijalankan.")
        submitted = False
    else:
        qa_options = list(qa_df_chat.index)

        def _format_qa_option(idx):
            row = qa_df_chat.loc[idx]
            question = str(row.get("question", "")).strip()
            preview = question[:140] + ("..." if len(question) > 140 else "")
            return f"{str(row.get('query_id', '')).strip()} - {preview}"

        with st.form("query_form", clear_on_submit=False):
            selected_qa_idx = st.selectbox(
                "Pertanyaan QA Gold",
                options=qa_options,
                format_func=_format_qa_option,
            )
            submitted = st.form_submit_button("🚀 Kirim", width="stretch")

    if submitted and selected_qa_idx is not None:
        selected_qa = qa_df_chat.loc[selected_qa_idx]
        q_id = str(selected_qa.get("query_id", "")).strip()
        query = str(selected_qa.get("question", "")).strip()
        gold = str(selected_qa.get("gold_answer", "")).strip()

        # ── Header: User Question only ────────────────────────────────────
        _section_header("User Question")
        st.write(query)

        if compare_mode:
            # ── Mode bandingkan 3 kolom ───────────────────────────────────
            cols = st.columns(3)
            turn_results = []

            try:
                with st.spinner("🔍 Meng-embed query..."):
                    query_vec = pipeline.embedder.embed(query)[0]
            except Exception as e:
                render_generation_error(e)
                st.stop()

            for col, method in zip(cols, METHODS):
                with col:
                    st.markdown(f"**{METHOD_LABELS[method]}**")
                    st.divider()
                    status = st.empty()
                    try:
                        t0 = time.time()
                        p = RAGPipeline(
                            embedder=pipeline.embedder,
                            generator=pipeline.generator,
                            chroma_client=pipeline.chroma_client,
                            chunking_method=method,
                            top_k=top_k,
                        )
                        status.caption("📚 Retrieve chunks...")
                        retrieved = p.retrieve_by_vector(query_vec, k=top_k)
                        status.empty()
                        contexts = [p._format_context(doc) for doc in retrieved]

                        # ── Generated Answer ──────────────────────────────
                        _section_header("Generated Answer")
                        answer = stream_answer(pipeline.generator, query, contexts,
                                              timer_placeholder=status, t0=t0)
                        elapsed = round(time.time() - t0, 1)
                        status.empty()
                    except Exception as e:
                        status.empty()
                        render_generation_error(e)
                        continue

                    # ── Ground Truth ──────────────────────────────────────
                    if gold:
                        _section_header("Ground Truth Answer")
                        st.markdown(gold)

                    # ── Metrik ────────────────────────────────────────────
                    bleu = rouge = None
                    retrieval_metrics = _compute_chat_retrieval_metrics(
                        q_id=q_id, method=method, retrieved=retrieved, top_k=top_k,
                    )
                    if gold:
                        bleu  = compute_bleu(answer, gold)
                        rouge = compute_rouge(answer, gold, rouge_type="rougeL", mode="recall")
                    _render_retrieval_metrics(retrieval_metrics, bleu, rouge)
                    st.caption(f"⏱ {elapsed}s · {len(retrieved)} chunks")

                    # ── Retrieved Chunks ──────────────────────────────────
                    _render_chunks(retrieved, show=show_chunks)

                    turn_results.append({
                        "method":    METHOD_LABELS[method],
                        "answer":    answer,
                        "bleu":      bleu,
                        "rouge_l":   rouge,
                        "retrieval": retrieval_metrics,
                        "elapsed_s": elapsed,
                        "chunks":    _extract_chunk_records(retrieved),
                    })

            # Simpan turn ke disk (persisten, tahan restart)
            if turn_results:
                _save_chat_turn(_build_chat_record(
                    query, "Bandingkan 3 Metode", top_k, gold, turn_results,
                ))
        else:
            # ── Mode single method ────────────────────────────────────────
            try:
                t0 = time.time()
                with st.spinner("🔍 Retrieve chunks..."):
                    p = RAGPipeline(
                        embedder=pipeline.embedder,
                        generator=pipeline.generator,
                        chroma_client=pipeline.chroma_client,
                        chunking_method=selected_method,
                        top_k=top_k,
                    )
                    retrieved = p.retrieve(query, k=top_k)
                timer_ph = st.empty()
                contexts = [p._format_context(doc) for doc in retrieved]

                # ── Generated Answer ──────────────────────────────────────
                _section_header("Generated Answer")
                answer = stream_answer(pipeline.generator, query, contexts,
                                      timer_placeholder=timer_ph, t0=t0)
                elapsed = round(time.time() - t0, 1)
                timer_ph.empty()
            except Exception as e:
                render_generation_error(e)
                st.stop()

            # ── Ground Truth Answer ───────────────────────────────────────
            if gold:
                _section_header("Ground Truth Answer")
                st.markdown(gold)

            # ── Metrik ────────────────────────────────────────────────────
            bleu = rouge = None
            retrieval_metrics = _compute_chat_retrieval_metrics(
                q_id=q_id, method=selected_method, retrieved=retrieved, top_k=top_k,
            )
            if gold:
                bleu  = compute_bleu(answer, gold)
                rouge = compute_rouge(answer, gold, rouge_type="rougeL", mode="recall")
            _render_retrieval_metrics(retrieval_metrics, bleu, rouge)
            st.caption(f"⏱ {elapsed}s · {len(retrieved)} chunks")

            # ── Retrieved Chunks ──────────────────────────────────────────
            _render_chunks(retrieved, show=show_chunks)

            # Simpan turn ke disk (persisten, tahan restart)
            _save_chat_turn(_build_chat_record(
                query, METHOD_LABELS[selected_method], top_k, gold,
                [{
                    "method":    METHOD_LABELS[selected_method],
                    "answer":    answer,
                    "bleu":      bleu,
                    "rouge_l":   rouge,
                    "retrieval": retrieval_metrics,
                    "elapsed_s": elapsed,
                    "chunks":    _extract_chunk_records(retrieved),
                }],
            ))

    elif submitted:
        st.warning("Pertanyaan QA Gold belum dipilih.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluasi Batch (Persistent + History)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Evaluasi Batch — Retrieval (P@k, R@k, MRR, F1@k) + Generation (BLEU, ROUGE-L)")
    st.caption(
        "Hasil disimpan permanen ke disk dan tidak hilang saat app restart. "
        "Menggunakan ground truth binary (label 0/1). "
        "Mendukung rentang top-k=1 s/d 20 dan otomatis skip top-k yang sudah punya CSV valid."
    )

    # ── Konfigurasi run ───────────────────────────────────────────────────
    col_qa_mode, col_topk, col_btn = st.columns([2, 1, 1])
    
    with col_qa_mode:
        eval_mode = st.radio(
            "Mode QA",
            ["Full — 30 QA", "Quick — 5 QA"],
            horizontal=True,
            key="eval_mode",
        )
    
    with col_topk:
        top_k_min = st.number_input("Min Top-K", min_value=1, max_value=20, value=1, key="top_k_min")
        top_k_max = st.number_input("Max Top-K", min_value=1, max_value=20, value=10, key="top_k_max")
    
    with col_btn:
        st.write("")
        run_btn = st.button("▶ Jalankan Evaluasi", width="stretch", type="primary")

    if eval_mode.startswith("Quick"):
        st.caption(
            f"Quick subset: **{', '.join(QUICK_EVAL_IDS)}** "
            "(Q005 Nilai Konstruksi · Q010 Pengeluaran Bahan · "
            "Q011 Mismatch Lama Bekerja · Q013 Hazard Ratio · Q020 Sampel HS 8-digit)"
        )
    
    # Validate top-k range
    if top_k_min > top_k_max:
        st.error("❌ Min Top-K tidak boleh lebih besar dari Max Top-K")
        run_btn = False

    def _get_existing_eval_file(mode_tag: str, top_k_value: int) -> tuple[pd.DataFrame, Path] | None:
        """Return CSV evaluasi yang sudah ada untuk mode/top-k jika valid."""
        matches = sorted(
            EVAL_RESULTS_DIR.glob(f"eval_*_{mode_tag}_top{top_k_value}.csv"),
            reverse=True,
        )
        for path in matches:
            try:
                df_existing = pd.read_csv(path)
                if not df_existing.empty:
                    return df_existing, path
            except Exception as exc:
                logger.warning(f"Existing eval file ignored because it cannot be read: {path} ({exc})")
        return None

    # ── Helper: jalankan satu run dan simpan ke disk ──────────────────────
    def _run_eval_and_save(qa_subset: pd.DataFrame, mode_tag: str,
                          top_k_range: tuple) -> list:
        """Evaluasi semua query × 3 metode × top-k range, simpan CSV ke disk.

        Menggunakan ground truth binary (qa_pairs_binary.json, label >= 1 = relevan).

        Args:
            qa_subset: DataFrame dengan QA pairs
            mode_tag: 'quick' atau 'full'
            top_k_range: tuple (min_k, max_k) untuk top-k evaluation

        Returns:
            List of (df, path, status) tuples untuk setiap top-k.
            status = "created" untuk file baru, "skipped" untuk CSV valid yang sudah ada.
        """
        # Load ground truth binary
        gt_data = _load_ground_truth()
        if not gt_data:
            st.error("❌ Ground truth binary (qa_pairs_binary.json) tidak ditemukan.")
            return []
        
        # Create lookup dict for ground truth
        gt_lookup = {item["id"]: item for item in gt_data}
        
        # Get hardware info
        hw_info = get_hardware_info()
        hw_info_str = json.dumps(hw_info, ensure_ascii=False)
        
        min_k, max_k = top_k_range
        requested_k_values = list(range(min_k, max_k + 1))
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        all_results = []
        pending_k_values = []
        skipped_paths = []

        for current_k in requested_k_values:
            existing = _get_existing_eval_file(mode_tag, current_k)
            if existing is None:
                pending_k_values.append(current_k)
            else:
                df_existing, existing_path = existing
                all_results.append((df_existing, existing_path, "skipped"))
                skipped_paths.append(existing_path)

        if skipped_paths:
            st.info(
                "Skip top-k yang sudah punya CSV valid: "
                + ", ".join(f"top-{path.stem.rsplit('top', 1)[-1]}" for path in skipped_paths)
            )

        if not pending_k_values:
            st.success("Semua top-k pada rentang ini sudah punya file CSV valid. Tidak ada evaluasi baru yang dijalankan.")
            return all_results

        total_files = len(pending_k_values)
        total_steps = len(qa_subset) * len(METHODS) * total_files
        prog = st.progress(0, text="Memulai evaluasi...")
        status_txt = st.empty()
        step = 0
        
        # Pre-compute query embeddings once per query (reused for all top-k & methods)
        query_embeddings: dict[str, tuple] = {}
        embed_status = st.empty()
        for i, (_, qa_row) in enumerate(qa_subset.iterrows(), 1):
            q_id = str(qa_row["query_id"])
            question = str(qa_row["question"])
            embed_status.caption(f"⏳ Pre-computing embeddings... {i}/{len(qa_subset)} ({q_id})")
            try:
                q_vec = pipeline.embedder.embed(question)[0]
                query_embeddings[q_id] = (q_vec, True)
            except Exception:
                query_embeddings[q_id] = (None, False)
        embed_status.empty()
        
        # Loop through each top-k
        for current_k in pending_k_values:
            rows = []
            
            for _, qa_row in qa_subset.iterrows():
                question = str(qa_row["question"])
                gold_ans = str(qa_row["gold_answer"])
                q_id     = str(qa_row["query_id"])
                
                # Get ground truth item for this query
                gt_item = gt_lookup.get(q_id)
                
                # Reuse cached embedding
                q_vec, embed_ok = query_embeddings.get(q_id, (None, False))
                
                for method in METHODS:
                    step += 1
                    prog.progress(step / total_steps,
                                  text=f"[{q_id}] {METHOD_LABELS[method]} top-{current_k} ({step}/{total_steps})")
                    status_txt.caption(f"⏳ {question[:80]}...")
                    
                    # Get relevant chunk IDs for this method
                    if gt_item:
                        rel_all = gt_item.get("relevant_chunk_ids", {})
                        rel_ids = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all
                    else:
                        rel_ids = []
                    
                    # Initialize metrics
                    precision_val = recall_val = mrr_val = f1_val = None
                    gen_answer = bleu_val = rouge_val = None
                    error_msg = ""
                    is_oom = False
                    
                    try:
                        # Retrieve with OOM handling
                        p = RAGPipeline(
                            embedder=pipeline.embedder,
                            generator=pipeline.generator,
                            chroma_client=pipeline.chroma_client,
                            chunking_method=method,
                            top_k=current_k,
                        )
                        
                        if embed_ok:
                            retrieved = p.retrieve_by_vector(q_vec, k=current_k)
                        else:
                            retrieved = p.retrieve(question, k=current_k)
                        
                        # Get retrieved chunk IDs
                        retrieved_ids = [doc.get("id", "") for doc in retrieved]
                        
                        # Compute retrieval metrics if relevant chunks exist
                        if rel_ids:
                            precision_val = compute_precision_at_k(retrieved_ids, rel_ids, current_k)
                            recall_val = compute_recall_at_k(retrieved_ids, rel_ids, current_k)
                            mrr_val = compute_mrr(retrieved_ids, rel_ids)
                            f1_val = compute_f1_at_k(precision_val, recall_val)
                        else:
                            # No relevant chunks for this query
                            precision_val = recall_val = mrr_val = f1_val = "N/A"
                        
                        # Generate answer
                        contexts = [p._format_context(doc) for doc in retrieved]
                        raw = pipeline.generator.generate(question, contexts)
                        gen_answer = raw[0] if isinstance(raw, tuple) else raw
                        bleu_val = compute_bleu(gen_answer, gold_ans)
                        rouge_val = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")
                        
                    except torch.cuda.OutOfMemoryError as oom_exc:
                        # OOM handling
                        is_oom = True
                        gen_answer = "[OOM - Out of Memory]"
                        precision_val = recall_val = mrr_val = f1_val = "OOM"
                        bleu_val = rouge_val = "OOM"
                        error_msg = f"OOM at top-{current_k}: {str(oom_exc)}"
                        logger.error(f"OOM error for {q_id} {method} top-{current_k}: {oom_exc}")
                        
                        # Clear GPU cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    except Exception as exc:
                        gen_answer = f"[ERROR] {exc}"
                        precision_val = recall_val = mrr_val = f1_val = None
                        bleu_val = rouge_val = None
                        error_msg = str(exc)
                    
                    rows.append({
                        "query_id"         : q_id,
                        "method"           : METHOD_LABELS[method],
                        "question"         : question,
                        "gold_answer"      : gold_ans,
                        "generated_answer" : gen_answer,
                        "precision_at_k"   : round(precision_val, 4) if isinstance(precision_val, (int, float)) else precision_val,
                        "recall_at_k"      : round(recall_val, 4) if isinstance(recall_val, (int, float)) else recall_val,
                        "mrr"              : round(mrr_val, 4) if isinstance(mrr_val, (int, float)) else mrr_val,
                        "f1_at_k"          : round(f1_val, 4) if isinstance(f1_val, (int, float)) else f1_val,
                        "bleu"             : round(bleu_val, 4) if isinstance(bleu_val, (int, float)) else bleu_val,
                        "rouge_l_recall"   : round(rouge_val, 4) if isinstance(rouge_val, (int, float)) else rouge_val,
                        "error"            : error_msg,
                        "hardware_info"    : hw_info_str,
                    })
            
            # Save file for this top-k — satu subfolder langsung di generation/
            df_result = pd.DataFrame(rows)
            # WIB timestamp (UTC+7)
            ts_wib = (datetime.now() + timedelta(hours=7)).strftime("%Y%m%d_%H%M%S")
            save_path = EVAL_RESULTS_DIR / f"eval_{ts_wib}_{mode_tag}_top{current_k}.csv"
            df_result.to_csv(save_path, index=False)
            all_results.append((df_result, save_path, "created"))
        
        prog.empty()
        status_txt.empty()
        
        return all_results

    # ── Jalankan evaluasi ─────────────────────────────────────────────────
    if run_btn:
        qa_df = _load_qa_gold()
        if qa_df.empty:
            st.error("❌ QA gold standard tidak ditemukan.")
        else:
            is_quick  = eval_mode.startswith("Quick")
            mode_tag  = "quick" if is_quick else "full"
            qa_subset = qa_df[qa_df["query_id"].isin(QUICK_EVAL_IDS)] if is_quick else qa_df

            st.info("🔄 Menjalankan evaluasi (binary ground truth)...")
            all_results = _run_eval_and_save(qa_subset, mode_tag, (top_k_min, top_k_max))

            if all_results:
                n_q = len(qa_subset)
                n_files = len(all_results)
                created_results = [(df, path) for df, path, status in all_results if status == "created"]
                skipped_results = [(df, path) for df, path, status in all_results if status == "skipped"]
                total_rows = sum(len(df) for df, _, _ in all_results)

                st.success(
                    f"✅ Selesai: {len(created_results)} file baru, {len(skipped_results)} file di-skip "
                    f"({n_files} file, {total_rows} baris total, "
                    f"{n_q} pertanyaan × {len(METHODS)} metode × {top_k_max - top_k_min + 1} top-k diminta)"
                )

                # Show list of generated files
                st.markdown("**File hasil evaluasi:**")
                for df, path, status in all_results:
                    oom_count = len(df[df["precision_at_k"] == "OOM"])
                    oom_note = f" ({oom_count} OOM)" if oom_count > 0 else ""
                    status_label = "baru" if status == "created" else "skip"
                    st.markdown(f"- `{path.name}` — {status_label}{oom_note}")
            else:
                st.error("❌ Evaluasi gagal. Cek log untuk detail.")

    st.divider()

    # ── Helper: render DataFrame hasil ───────────────────────────────────
    def _render_results(df_res: pd.DataFrame) -> None:
        """Tampilkan ringkasan + detail + tombol export untuk DataFrame hasil eval."""
        # Ringkasan per metode (termasuk retrieval metrics)
        st.markdown("**Ringkasan per Metode**")
        
        # Filter out OOM and N/A for numeric aggregation
        valid_metrics = df_res[
            (df_res["precision_at_k"] != "OOM") & 
            (df_res["precision_at_k"] != "N/A") &
            (df_res["error"].fillna("") == "")
        ]
        
        if not valid_metrics.empty:
            # Convert to numeric for aggregation
            numeric_cols = ["precision_at_k", "recall_at_k", "mrr", "f1_at_k", "bleu", "rouge_l_recall"]
            for col in numeric_cols:
                valid_metrics[col] = pd.to_numeric(valid_metrics[col], errors="coerce")
            
            summary = (
                valid_metrics.groupby("method")[numeric_cols]
                .agg(n=("bleu", "count"), 
                     mean_precision=("precision_at_k", "mean"),
                     mean_recall=("recall_at_k", "mean"),
                     mean_mrr=("mrr", "mean"),
                     mean_f1=("f1_at_k", "mean"),
                     mean_bleu=("bleu", "mean"),
                     mean_rouge_l=("rouge_l_recall", "mean"))
                .round(4)
            )
            st.dataframe(summary, width="stretch")
        else:
            st.warning("Tidak ada data valid untuk ringkasan.")
            summary = pd.DataFrame()

        # Detail per query
        st.markdown("**Detail Per Query**")
        display_cols = ["query_id", "method", "question", "gold_answer",
                        "generated_answer", "precision_at_k", "recall_at_k",
                        "mrr", "f1_at_k", "bleu", "rouge_l_recall", "error"]
        st.dataframe(
            df_res[display_cols],
            width="stretch",
            height=400,
            column_config={
                "question"         : st.column_config.TextColumn(width="medium"),
                "gold_answer"      : st.column_config.TextColumn(width="medium"),
                "generated_answer" : st.column_config.TextColumn(width="large"),
                "precision_at_k"   : st.column_config.TextColumn(width="small"),
                "recall_at_k"      : st.column_config.TextColumn(width="small"),
                "mrr"              : st.column_config.TextColumn(width="small"),
                "f1_at_k"          : st.column_config.TextColumn(width="small"),
                "bleu"             : st.column_config.NumberColumn(format="%.4f"),
                "rouge_l_recall"   : st.column_config.NumberColumn(format="%.4f"),
            },
        )

        # Export
        st.markdown("**Ekspor**")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download CSV",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="eval_results.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_dl2:
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                df_res.to_excel(writer, index=False, sheet_name="per_query")
                if not summary.empty:
                    summary.reset_index().to_excel(writer, index=False, sheet_name="summary")
            st.download_button(
                "⬇ Download XLSX",
                data=xlsx_buf.getvalue(),
                file_name="eval_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    # ── Riwayat evaluasi dari disk ────────────────────────────────────────
    st.subheader("📂 Riwayat Evaluasi")
    saved_files = sorted(EVAL_RESULTS_DIR.glob("eval_*.csv"), reverse=True)

    if not saved_files:
        st.info("Belum ada hasil evaluasi tersimpan. Jalankan evaluasi terlebih dahulu.")
    else:
        file_labels = {p: p.name for p in saved_files}
        selected_file = st.selectbox(
            f"Pilih run ({len(saved_files)} tersedia)",
            options=saved_files,
            format_func=lambda p: p.name,
            key="eval_history_select",
        )
        if selected_file is not None:
            try:
                df_view = pd.read_csv(selected_file)
                _render_results(df_view)
            except Exception as exc:
                st.error(f"Gagal membaca file: {exc}")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Riwayat Chat (Persistent)
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("🕒 Riwayat Chat")
    st.caption(
        "Riwayat percakapan tersimpan permanen ke disk — tidak hilang saat app restart. "
        "Untuk dokumentasi sidang."
    )

    chat_history = _load_chat_history()

    if not chat_history:
        st.info("Belum ada riwayat chat. Ajukan pertanyaan di tab 💬 Chat terlebih dahulu.")
    else:
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.markdown(f"**{len(chat_history)} percakapan tersimpan** (terbaru di atas)")
        with col_clear:
            with st.popover("🗑 Hapus Riwayat", width="stretch"):
                st.warning("Menghapus SEMUA riwayat chat. Tindakan ini tidak dapat dibatalkan.")
                if st.button("Ya, hapus semua", type="primary", width="stretch"):
                    try:
                        CHAT_HISTORY_FILE.unlink(missing_ok=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal menghapus: {e}")

        # Pencarian
        search = st.text_input("🔎 Cari pertanyaan", placeholder="kata kunci...")
        if search:
            s = search.strip().lower()
            chat_history = [r for r in chat_history if s in str(r.get("query", "")).lower()]
            st.caption(f"{len(chat_history)} hasil cocok")

        # Download
        if CHAT_HISTORY_FILE.exists():
            st.download_button(
                "⬇ Download Riwayat (JSONL)",
                data=CHAT_HISTORY_FILE.read_bytes(),
                file_name="chat_history.jsonl",
                mime="application/jsonl",
            )

        st.divider()

        for idx, record in enumerate(chat_history, 1):
            label = f"#{len(chat_history) - idx + 1} · {record.get('timestamp', '-')} · {str(record.get('query',''))[:70]}"
            with st.expander(label, expanded=(idx == 1)):
                _render_history_turn(record)








