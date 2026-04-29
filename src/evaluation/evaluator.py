"""
Orkestrasi evaluasi RAG Pipeline.

Alur:
  Load ground truth
    → untuk setiap chunking method:
        → retrieve top-k chunks per query
        → (opsional) generate jawaban
        → hitung semua metrik
    → output tabel perbandingan

Metrik yang dihitung:
  Retrieval  : Context Precision, Context Recall (LLM atau NonLLM), MRR (rank_eval)
  Generation : BLEU, ROUGE-1 F1, ROUGE-L F1
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    compute_bleu,
    compute_rouge,
    compute_context_precision_llm,
    compute_context_precision_nonllm,
    compute_context_recall_llm,
    compute_context_recall_nonllm,
    compute_mrr,
)
from ..chroma.client import initialize_chroma_client, get_or_create_collection
from ..chroma.query import similarity_search
from ..embedding.embedder import QwenEmbedder, initialize_gguf_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAMES: Dict[str, str] = {
    "element_based":   "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive":       "collection_recursive",
}

DEFAULT_EMBEDDER_PATH = "models/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_CHROMA_PATH   = "data/chroma"


# ─── Ground Truth ──────────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth dari JSON file.

    Format yang diharapkan:
    [
      {
        "id": "q_001",
        "question": "...",
        "reference_answer": "...",
        "relevant_chunk_ids": ["chunk_id_1"],  // opsional, untuk MRR
        "reference_contexts": ["teks konteks"] // opsional, untuk NonLLM Precision/Recall
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} ground truth items dari: {path}")
    return data


# ─── Evaluator LLM ─────────────────────────────────────────────────────────────

def build_evaluator_llm(
    llm_type: str = "openai",
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
) -> Optional[Any]:
    """
    Buat evaluator LLM untuk LLM-based Ragas metrics.

    Args:
        llm_type : 'openai' atau 'local' (OpenAI-compatible endpoint)
        model    : Nama model
        base_url : Untuk 'local' → base URL endpoint (contoh: http://localhost:8000/v1)
        api_key  : API key

    Returns:
        LangchainLLMWrapper instance atau None jika gagal
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        kwargs: Dict[str, Any] = {"model": model, "temperature": 0.0, "api_key": api_key}
        if llm_type == "local" and base_url:
            kwargs["base_url"] = base_url

        lc_llm = ChatOpenAI(**kwargs)
        logger.info(f"Evaluator LLM: {llm_type} / {model}")
        return LangchainLLMWrapper(lc_llm)

    except ImportError:
        logger.error(
            "langchain-openai tidak tersedia. "
            "Install: pip install langchain-openai"
        )
        return None
    except Exception as e:
        logger.error(f"build_evaluator_llm error: {e}")
        return None


# ─── Result Container ──────────────────────────────────────────────────────────

class MethodResult:
    """Hasil evaluasi untuk satu chunking method."""

    def __init__(self, method: str):
        self.method = method
        self.per_query: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "metrics": self.metrics,
            "per_query": self.per_query,
        }


# ─── RAGEvaluator ──────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluator untuk RAG pipeline.

    Penggunaan:
        evaluator = build_evaluator(...)
        ground_truth = load_ground_truth("data/ground_truth/qa_pairs.json")
        results = evaluator.evaluate_all(ground_truth)

    Prioritas metrik:
      Context Precision/Recall:
        1. LLM-based  → jika eval_llm tersedia
        2. Non-LLM    → jika reference_contexts ada di ground truth
        3. Di-skip    → jika keduanya tidak tersedia

      MRR:
        → Dihitung jika relevant_chunk_ids ada di ground truth

      BLEU / ROUGE:
        → Dihitung jika generator tersedia
    """

    def __init__(
        self,
        embedder: QwenEmbedder,
        chroma_client: Any,
        eval_llm: Optional[Any] = None,
    ):
        self.embedder = embedder
        self.chroma_client = chroma_client
        self.eval_llm = eval_llm
        self.use_llm_metrics = eval_llm is not None

        logger.info("RAGEvaluator initialized")
        logger.info(
            f"  - LLM metrics  : {'enabled' if self.use_llm_metrics else 'disabled (NonLLM/skip)'}"
        )

    def _retrieve(
        self,
        collection: Any,
        question: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Embed query dan retrieve dari ChromaDB."""
        embedding = self.embedder.embed(question)
        query_vec = embedding[0]
        return similarity_search(collection, query_vec, k=top_k)

    def evaluate_method(
        self,
        chunking_method: str,
        ground_truth: List[Dict[str, Any]],
        top_k: int = 5,
        generator: Optional[Any] = None,
        mrr_at_k: Optional[int] = None,
    ) -> MethodResult:
        """
        Evaluasi satu chunking method terhadap ground truth.

        Args:
            chunking_method : 'element_based', 'maxmin_semantic', atau 'recursive'
            ground_truth    : List item ground truth (dari load_ground_truth)
            top_k           : Jumlah chunk per query
            generator       : RAGGenerator/HFRAGGenerator (opsional, untuk BLEU/ROUGE)
            mrr_at_k        : MRR@K threshold (None = tanpa threshold)

        Returns:
            MethodResult dengan per_query scores dan aggregated metrics
        """
        if chunking_method not in COLLECTION_NAMES:
            raise ValueError(
                f"Unknown chunking_method: '{chunking_method}'. "
                f"Pilih dari: {list(COLLECTION_NAMES.keys())}"
            )

        collection_name = COLLECTION_NAMES[chunking_method]
        collection = get_or_create_collection(self.chroma_client, collection_name)

        if collection is None:
            raise RuntimeError(
                f"Collection '{collection_name}' tidak ditemukan. "
                "Jalankan load_to_chroma.py terlebih dahulu."
            )

        doc_count = collection.count()
        logger.info(
            f"\nEvaluating: {chunking_method} | "
            f"{collection_name} ({doc_count} docs) | top_k={top_k}"
        )

        result = MethodResult(chunking_method)

        # Akumulasi untuk MRR
        qrels_data: Dict[str, List[str]] = {}
        run_data: Dict[str, Dict[str, float]] = {}

        for item in ground_truth:
            q_id          = item["id"]
            question      = item["question"]
            reference     = item["reference_answer"]
            rel_ids       = item.get("relevant_chunk_ids", [])
            ref_contexts  = item.get("reference_contexts", [])

            logger.info(f"  [{q_id}] {question[:60]}...")

            # ── Retrieve ──────────────────────────────────────────────────────
            retrieved      = self._retrieve(collection, question, top_k=top_k)
            retrieved_texts = [r["document"] for r in retrieved]
            retrieved_ids   = [r["id"]       for r in retrieved]

            # ── Generate (opsional) ──────────────────────────────────────────
            answer = None
            if generator is not None:
                try:
                    raw = generator.generate(question, retrieved_texts)
                    answer = raw[0] if isinstance(raw, tuple) else raw
                    logger.info(f"    answer length: {len(answer)} chars")
                except Exception as e:
                    logger.warning(f"    generation error: {e}")

            # ── Per-query result ──────────────────────────────────────────────
            q_result: Dict[str, Any] = {
                "q_id"          : q_id,
                "question"      : question,
                "retrieved_ids" : retrieved_ids,
                "answer"        : answer,
            }

            # ── BLEU + ROUGE ──────────────────────────────────────────────────
            if answer:
                q_result["bleu"]   = compute_bleu(answer, reference)
                q_result["rouge1"] = compute_rouge(answer, reference, "rouge1", "fmeasure")
                q_result["rougeL"] = compute_rouge(answer, reference, "rougeL", "fmeasure")
                logger.info(
                    f"    BLEU={q_result['bleu']:.4f} | "
                    f"R1={q_result['rouge1']:.4f} | "
                    f"RL={q_result['rougeL']:.4f}"
                )

            # ── Context Precision & Recall ────────────────────────────────────
            if self.use_llm_metrics:
                q_result["context_precision"] = compute_context_precision_llm(
                    question, retrieved_texts, reference, self.eval_llm
                )
                q_result["context_recall"] = compute_context_recall_llm(
                    question, retrieved_texts, reference, self.eval_llm
                )
                logger.info(
                    f"    P={q_result['context_precision']:.4f} | "
                    f"R={q_result['context_recall']:.4f} (LLM)"
                )
            elif ref_contexts:
                q_result["context_precision"] = compute_context_precision_nonllm(
                    retrieved_texts, ref_contexts
                )
                q_result["context_recall"] = compute_context_recall_nonllm(
                    retrieved_texts, ref_contexts
                )
                logger.info(
                    f"    P={q_result['context_precision']:.4f} | "
                    f"R={q_result['context_recall']:.4f} (NonLLM)"
                )

            # ── Akumulasi data untuk MRR ──────────────────────────────────────
            if rel_ids:
                qrels_data[q_id] = rel_ids
                run_data[q_id] = {
                    r["id"]: float(1.0 - r["distance"])
                    if r.get("distance") is not None
                    else 1.0
                    for r in retrieved
                }

            result.per_query.append(q_result)

        # ── Aggregate (mean) ──────────────────────────────────────────────────
        for key in ["bleu", "rouge1", "rougeL", "context_precision", "context_recall"]:
            values = [q[key] for q in result.per_query if key in q]
            if values:
                result.metrics[key] = round(sum(values) / len(values), 6)

        # ── MRR ───────────────────────────────────────────────────────────────
        if qrels_data:
            result.metrics["mrr"] = compute_mrr(qrels_data, run_data, mrr_at_k)
            logger.info(f"  MRR = {result.metrics['mrr']:.4f}")

        logger.info(f"✓ {chunking_method}: {result.metrics}")
        return result

    def evaluate_all(
        self,
        ground_truth: List[Dict[str, Any]],
        methods: Optional[List[str]] = None,
        top_k: int = 5,
        generator: Optional[Any] = None,
        mrr_at_k: Optional[int] = None,
    ) -> List[MethodResult]:
        """
        Evaluasi semua (atau sebagian) chunking methods.

        Args:
            ground_truth : List ground truth items
            methods      : List method names (default: semua 3 method)
            top_k        : Jumlah chunk per query
            generator    : Generator model (opsional, untuk BLEU/ROUGE)
            mrr_at_k     : MRR@K threshold

        Returns:
            List of MethodResult, satu per method
        """
        methods = methods or list(COLLECTION_NAMES.keys())
        results = []

        for method in methods:
            r = self.evaluate_method(
                chunking_method=method,
                ground_truth=ground_truth,
                top_k=top_k,
                generator=generator,
                mrr_at_k=mrr_at_k,
            )
            results.append(r)

        return results


# ─── Factory ────────────────────────────────────────────────────────────────────

def build_evaluator(
    embedder_path: str = DEFAULT_EMBEDDER_PATH,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    n_gpu_layers: int = -1,
    eval_llm_type: Optional[str] = None,
    eval_llm_model: str = "gpt-4o-mini",
    eval_llm_base_url: Optional[str] = None,
    eval_llm_api_key: str = "EMPTY",
) -> RAGEvaluator:
    """
    Factory function: load komponen dan return RAGEvaluator.

    Args:
        embedder_path     : Path ke GGUF embedding model
        chroma_path       : Path ke ChromaDB persistent storage
        n_gpu_layers      : GPU layers untuk embedder (-1 = semua)
        eval_llm_type     : 'openai' atau 'local' (None = non-LLM only)
        eval_llm_model    : Nama model evaluator LLM
        eval_llm_base_url : Base URL untuk local LLM server
        eval_llm_api_key  : API key

    Returns:
        RAGEvaluator yang siap digunakan

    Raises:
        RuntimeError jika embedder atau ChromaDB gagal di-load
    """
    logger.info("Building RAGEvaluator...")

    embedder = initialize_gguf_embedder(
        model_path=embedder_path,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    if embedder is None:
        raise RuntimeError(f"Gagal memuat embedder: {embedder_path}")

    chroma_client = initialize_chroma_client(persist_directory=chroma_path)
    if chroma_client is None:
        raise RuntimeError(f"Gagal koneksi ChromaDB: {chroma_path}")

    eval_llm = None
    if eval_llm_type:
        eval_llm = build_evaluator_llm(
            llm_type=eval_llm_type,
            model=eval_llm_model,
            base_url=eval_llm_base_url,
            api_key=eval_llm_api_key,
        )

    return RAGEvaluator(
        embedder=embedder,
        chroma_client=chroma_client,
        eval_llm=eval_llm,
    )
