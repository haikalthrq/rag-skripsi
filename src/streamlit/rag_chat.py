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
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

import streamlit as st

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Eval persistence & quick-eval subset ─────────────────────────────────────
EVAL_RESULTS_DIR = ROOT / "results" / "generation_eval" / "streamlit"
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
from src.evaluation.metrics import compute_bleu, compute_rouge

# ── Load QA Gold Standard (for BLEU/ROUGE scoring) ───────────────────────────────

_QA_GOLD_DF = None

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

def get_gold_answer(query: str):
    """Find gold answer for a query from QA gold standard.

    Menggunakan fuzzy match (SequenceMatcher) agar toleran terhadap perbedaan
    kecil antara teks yang diketik user vs teks tersimpan di Excel.
    """
    from difflib import SequenceMatcher
    df = _load_qa_gold()
    if df.empty:
        return None
    query_norm = query.strip().lower()
    best_ratio, best_answer = 0.0, None
    for _, row in df.iterrows():
        stored = str(row["question"]).strip().lower()
        ratio = SequenceMatcher(None, query_norm, stored).ratio()
        if ratio > best_ratio:
            best_ratio, best_answer = ratio, row["gold_answer"]
    # Threshold 0.75: cukup fleksibel tapi hindari false positive
    return best_answer if best_ratio >= 0.75 else None

# ── Defaults ──────────────────────────────────────────────────────────────────

_LOCAL_GEN        = ROOT / "models/Qwen3-4B-Instruct-2507-FP8"
DEFAULT_GEN_TYPE  = "hf"
DEFAULT_GEN_PATH  = str(_LOCAL_GEN) if _LOCAL_GEN.exists() else "Qwen/Qwen3-4B-Instruct-2507-FP8"
DEFAULT_TEMP      = 0.7
DEFAULT_TOP_P     = 0.8
DEFAULT_TOP_K_GEN = 20
DEFAULT_MAX_TOK   = 1024   # Chat: 1024 sweet spot — cepat tapi tidak terpotong. Eval tetap 16384.
DEFAULT_TOP_K     = 8
METHODS           = list(COLLECTION_NAMES.keys())
METHOD_LABELS     = {
    "element_based":   "Element-Based",
    "maxmin_semantic": "MaxMin Semantic",
    "recursive":       "Recursive",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chat — BPS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.answer-box {
    background: #f0f4f8;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 0.95rem;
    line-height: 1.65;
    white-space: pre-wrap;
}
.chunk-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #475569;
}
.method-header {
    font-weight: 700;
    font-size: 1.05rem;
    color: #1e40af;
    margin-bottom: 4px;
}
.query-display {
    background: #eff6ff;
    border-radius: 8px;
    padding: 10px 14px;
    font-weight: 600;
    color: #1e3a8a;
    margin-bottom: 12px;
}
.hist-item {
    font-size: 0.85rem;
    color: #64748b;
    padding: 2px 0;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ── Model loader (cache) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Memuat model (hanya sekali)...")
def load_pipeline() -> RAGPipeline:
    """
    Load embedder + generator + ChromaDB. Di-cache — tidak reload tiap query.
    top_k tidak masuk cache key agar model tidak reload saat slider berubah.

    Embedder dijalankan di CPU (n_gpu_layers=0) agar VRAM penuh untuk
    generator HF FP8 (~5 GB pada RTX 4050 6 GB).
    """
    return build_pipeline(
        chunking_method="element_based",
        embedder_path=str(ROOT / DEFAULT_EMBEDDER_PATH),
        generator_path=DEFAULT_GEN_PATH,
        generator_type=DEFAULT_GEN_TYPE,
        chroma_path=str(ROOT / DEFAULT_CHROMA_PATH),
        top_k=DEFAULT_TOP_K,
        embedder_n_gpu_layers=0,   # CPU — hemat VRAM untuk generator
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
    st.markdown(
        f'<div class="answer-box">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )


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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Konfigurasi")

    compare_mode = st.toggle("Bandingkan 3 Metode", value=True,
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

    with st.expander("🔧 Parameter Generator"):
        st.caption("🔒 **Parameter dikunci sesuai dokumentasi resmi Qwen3-4B-Instruct-2507**")
        st.caption(
            f"Temperature: **{DEFAULT_TEMP}** · Top-P: **{DEFAULT_TOP_P}** "
            f"· Top-K: **{DEFAULT_TOP_K_GEN}** · MinP: **0** "
            f"· Max tokens: **{DEFAULT_MAX_TOK}**"
        )

    with st.expander("🤖 Model"):
        st.caption("Model generator dikunci untuk menjaga konsistensi evaluasi.")
        st.caption(f"Generator type: **{DEFAULT_GEN_TYPE}**")
        st.caption(f"Generator path: `{DEFAULT_GEN_PATH}`")
        show_chunks = st.checkbox("Tampilkan retrieved chunks", value=True)

    st.divider()
    st.caption("📋 **Riwayat Query (sesi ini)**")
    if "history" not in st.session_state:
        st.session_state.history = []
    for h in reversed(st.session_state.history[-10:]):
        st.markdown(f'<div class="hist-item">↳ {h[:60]}{"..." if len(h)>60 else ""}</div>',
                    unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔍 RAG Chat — Sistem Informasi BPS")
st.caption("Tanyakan apa saja tentang dokumen BPS. Model akan mencari chunk yang relevan dan menghasilkan jawaban.")

# Load pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    with st.expander("🔍 Detail error (untuk debugging)"):
        st.code(traceback.format_exc(), language="python")
    st.stop()

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluasi Batch"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat interaktif
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    with st.form("query_form", clear_on_submit=False):
        query = st.text_area("Pertanyaan", placeholder="Contoh: Berapa nilai impor Indonesia pada Agustus 2025?",
                             height=80)
        submitted = st.form_submit_button("🚀 Kirim", use_container_width=True)

    if submitted and query.strip():
        query = query.strip()
        st.session_state.history.append(query)

        st.markdown(f'<div class="query-display">❓ {query}</div>', unsafe_allow_html=True)

        # ── Tampilkan gold answer di bawah pertanyaan (jika tersedia) ─────
        gold = get_gold_answer(query)
        if gold:
            st.markdown(
                f'<div style="background:#f0fdf4; border-left:4px solid #16a34a; padding:8px 12px; '
                f'border-radius:4px; font-size:0.88rem; color:#15803d; margin-bottom:8px;">'
                f'📖 <b>Jawaban Referensi:</b> {gold}</div>',
                unsafe_allow_html=True,
            )

        if compare_mode:
            # ── Mode bandingkan 3 kolom ───────────────────────────────────
            cols = st.columns(3)

            try:
                with st.spinner("🔍 Meng-embed query..."):
                    query_vec = pipeline.embedder.embed(query)[0]
            except Exception as e:
                render_generation_error(e)
                st.stop()

            for col, method in zip(cols, METHODS):
                with col:
                    st.markdown(f'<div class="method-header">📦 {METHOD_LABELS[method]}</div>',
                                unsafe_allow_html=True)
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
                        answer = stream_answer(pipeline.generator, query, contexts,
                                              timer_placeholder=status, t0=t0)
                        elapsed = round(time.time() - t0, 1)
                        status.empty()
                    except Exception as e:
                        status.empty()
                        render_generation_error(e)
                        continue

                    st.caption(f"⏱ Selesai dalam {elapsed}s | {len(retrieved)} chunks")
                    if gold:
                        bleu = compute_bleu(answer, gold)
                        rouge = compute_rouge(answer, gold, rouge_type="rougeL", mode="recall")
                        st.markdown(
                            f'<div style="margin-top:6px; font-size:0.8rem; color:#374151;">'
                            f'✅ QA Match: BLEU <b>{bleu:.4f}</b> · ROUGE-L <b>{rouge:.4f}</b></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("💬 Scoring QA tidak tersedia untuk pertanyaan ini")

                    if show_chunks and retrieved:
                        with st.expander(f"📄 Chunks ({len(retrieved)})"):
                            for i, chunk in enumerate(retrieved, 1):
                                meta     = chunk.get("metadata", {})
                                src      = Path(meta.get("source_file", "?")).name
                                pages    = meta.get("page_numbers", "-")
                                dist     = chunk.get("distance")
                                dist_str = f"{dist:.4f}" if dist is not None else "-"
                                preview  = chunk["document"][:300].replace("\n", " ")
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<b>[{i}]</b> {src} · hal {pages} · dist {dist_str}<br>'
                                    f'{preview}...</div>',
                                    unsafe_allow_html=True,
                                )
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
                answer = stream_answer(pipeline.generator, query, contexts,
                                      timer_placeholder=timer_ph, t0=t0)
                elapsed = round(time.time() - t0, 1)
                timer_ph.empty()
            except Exception as e:
                render_generation_error(e)
                st.stop()

            st.caption(f"⏱ Selesai dalam {elapsed}s | Metode: {METHOD_LABELS[selected_method]} | {len(retrieved)} chunks")
            if gold:
                bleu = compute_bleu(answer, gold)
                rouge = compute_rouge(answer, gold, rouge_type="rougeL", mode="recall")
                st.markdown(
                    f'<div style="margin-top:6px; font-size:0.8rem; color:#374151;">'
                    f'✅ QA Match: BLEU <b>{bleu:.4f}</b> · ROUGE-L <b>{rouge:.4f}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("💬 Scoring QA tidak tersedia untuk pertanyaan ini")

            if show_chunks and retrieved:
                with st.expander(f"📄 Retrieved Chunks ({len(retrieved)})"):
                    for i, chunk in enumerate(retrieved, 1):
                        meta     = chunk.get("metadata", {})
                        src      = Path(meta.get("source_file", "?")).name
                        pages    = meta.get("page_numbers", "-")
                        dist     = chunk.get("distance")
                        dist_str = f"{dist:.4f}" if dist is not None else "-"
                        preview  = chunk["document"][:400].replace("\n", " ")
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>[{i}]</b> {src} · hal {pages} · dist {dist_str}<br>'
                            f'{preview}...</div>',
                            unsafe_allow_html=True,
                        )

    elif submitted:
        st.warning("Pertanyaan tidak boleh kosong.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluasi Batch (Persistent + History)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Evaluasi Batch — BLEU & ROUGE-L")
    st.caption(
        "Hasil disimpan permanen ke disk dan tidak hilang saat app restart. "
        "Precision@k / Recall@k / MRR = N/A (belum ada retrieval label)."
    )

    # ── Konfigurasi run ───────────────────────────────────────────────────
    col_mode, col_topk, col_btn = st.columns([2, 1, 2])
    with col_mode:
        eval_mode = st.radio(
            "Mode Evaluasi",
            ["Full — 30 QA", "Quick — 5 QA (3 tabel + 2 narasi)"],
            horizontal=True,
            key="eval_mode",
        )
    with col_topk:
        top_k_eval = st.slider("Top-K Retrieval", 1, 10, DEFAULT_TOP_K, key="eval_topk")
    with col_btn:
        st.write("")
        run_btn = st.button("▶ Jalankan Evaluasi", use_container_width=True, type="primary")

    if eval_mode.startswith("Quick"):
        st.caption(
            f"Quick subset: **{', '.join(QUICK_EVAL_IDS)}** "
            "(Q005 Nilai Konstruksi · Q010 Pengeluaran Bahan · "
            "Q011 Mismatch Lama Bekerja · Q013 Hazard Ratio · Q020 Sampel HS 8-digit)"
        )

    # ── Helper: jalankan satu run dan simpan ke disk ──────────────────────
    def _run_eval_and_save(qa_subset: pd.DataFrame, mode_tag: str, top_k: int) -> tuple:
        """Evaluasi semua query × 3 metode, simpan CSV ke disk, return (df, path)."""
        total_steps = len(qa_subset) * len(METHODS)
        prog = st.progress(0, text="Memulai evaluasi...")
        status_txt = st.empty()
        rows, step = [], 0

        for _, qa_row in qa_subset.iterrows():
            question = str(qa_row["question"])
            gold_ans = str(qa_row["gold_answer"])
            q_id     = str(qa_row["query_id"])

            try:
                q_vec    = pipeline.embedder.embed(question)[0]
                embed_ok = True
            except Exception:
                embed_ok = False

            for method in METHODS:
                step += 1
                prog.progress(step / total_steps,
                              text=f"[{q_id}] {METHOD_LABELS[method]} ({step}/{total_steps})")
                status_txt.caption(f"⏳ {question[:80]}...")

                try:
                    p = RAGPipeline(
                        embedder=pipeline.embedder,
                        generator=pipeline.generator,
                        chroma_client=pipeline.chroma_client,
                        chunking_method=method,
                        top_k=top_k,
                    )
                    retrieved  = (p.retrieve_by_vector(q_vec, k=top_k) if embed_ok
                                  else p.retrieve(question, k=top_k))
                    contexts   = [p._format_context(doc) for doc in retrieved]
                    raw        = pipeline.generator.generate(question, contexts)
                    gen_answer = raw[0] if isinstance(raw, tuple) else raw
                    bleu_val   = compute_bleu(gen_answer, gold_ans)
                    rouge_val  = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")
                    error_msg  = ""
                except Exception as exc:
                    gen_answer = f"[ERROR] {exc}"
                    bleu_val   = None
                    rouge_val  = None
                    error_msg  = str(exc)

                rows.append({
                    "query_id"         : q_id,
                    "method"           : METHOD_LABELS[method],
                    "question"         : question,
                    "gold_answer"      : gold_ans,
                    "generated_answer" : gen_answer,
                    "precision_at_k"   : "N/A",
                    "recall_at_k"      : "N/A",
                    "mrr"              : "N/A",
                    "bleu"             : round(bleu_val,  4) if bleu_val  is not None else None,
                    "rouge_l_recall"   : round(rouge_val, 4) if rouge_val is not None else None,
                    "error"            : error_msg,
                })

        prog.empty()
        status_txt.empty()

        df_result = pd.DataFrame(rows)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = EVAL_RESULTS_DIR / f"eval_{ts}_{mode_tag}_top{top_k}.csv"
        df_result.to_csv(save_path, index=False)
        return df_result, save_path

    # ── Jalankan evaluasi ─────────────────────────────────────────────────
    if run_btn:
        qa_df = _load_qa_gold()
        if qa_df.empty:
            st.error("❌ QA gold standard tidak ditemukan.")
        else:
            is_quick  = eval_mode.startswith("Quick")
            mode_tag  = "quick" if is_quick else "full"
            qa_subset = qa_df[qa_df["query_id"].isin(QUICK_EVAL_IDS)] if is_quick else qa_df

            df_new, saved_path = _run_eval_and_save(qa_subset, mode_tag, top_k_eval)
            n_q = len(qa_subset)
            st.success(
                f"✅ Selesai: {len(df_new)} baris ({n_q} pertanyaan × {len(METHODS)} metode) "
                f"— disimpan sebagai **{saved_path.name}**"
            )

    st.divider()

    # ── Helper: render DataFrame hasil ───────────────────────────────────
    def _render_results(df_res: pd.DataFrame) -> None:
        """Tampilkan ringkasan + detail + tombol export untuk DataFrame hasil eval."""
        # Ringkasan per metode
        st.markdown("**Ringkasan per Metode**")
        ok = df_res[df_res["error"].fillna("") == ""]
        if not ok.empty:
            summary = (
                ok.groupby("method")[["bleu", "rouge_l_recall"]]
                .agg(n=("bleu", "count"), mean_bleu=("bleu", "mean"),
                     mean_rouge_l=("rouge_l_recall", "mean"))
                .round(4)
            )
            st.dataframe(summary, use_container_width=True)
        else:
            st.warning("Semua query mengalami error.")
            summary = pd.DataFrame()

        # Detail per query
        st.markdown("**Detail Per Query**")
        display_cols = ["query_id", "method", "question", "gold_answer",
                        "generated_answer", "precision_at_k", "recall_at_k",
                        "mrr", "bleu", "rouge_l_recall"]
        st.dataframe(
            df_res[display_cols],
            use_container_width=True,
            height=400,
            column_config={
                "question"         : st.column_config.TextColumn(width="medium"),
                "gold_answer"      : st.column_config.TextColumn(width="medium"),
                "generated_answer" : st.column_config.TextColumn(width="large"),
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
                use_container_width=True,
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
                use_container_width=True,
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
