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
  Retrieval  : Precision@k, Recall@k, MRR, F1@k (pure Python, berbasis chunk ID)
  Generation : BLEU, ROUGE-L Recall
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    compute_bleu,
    compute_rouge,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_f1_at_k,
)
from ..chroma.client import initialize_chroma_client, get_or_create_collection
from ..chroma.query import similarity_search
from ..embedding.embedder import QwenEmbedder, initialize_gguf_embedder, initialize_hf_embedder

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
        "relevant_chunk_ids": {
          "element_based": ["chunk_id_1"],
          "recursive": ["chunk_id_2"],
          "maxmin_semantic": ["chunk_id_3"]
        }
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} ground truth items dari: {path}")
    return data


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

    Metrik:
      Precision@k, Recall@k, MRR → berbasis chunk ID matching (relevant_chunk_ids per method)
      BLEU / ROUGE-L Recall       → Dihitung jika generator tersedia
    """

    def __init__(
        self,
        embedder: QwenEmbedder,
        chroma_client: Any,
    ):
        self.embedder = embedder
        self.chroma_client = chroma_client

        logger.info("RAGEvaluator initialized")

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
    ) -> MethodResult:
        """
        Evaluasi satu chunking method terhadap ground truth.

        Args:
            chunking_method : 'element_based', 'maxmin_semantic', atau 'recursive'
            ground_truth    : List item ground truth (dari load_ground_truth)
            top_k           : Jumlah chunk per query
            generator       : RAGGenerator/HFRAGGenerator (opsional, untuk BLEU/ROUGE)

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

        for item in ground_truth:
            q_id          = item["id"]
            question      = item["question"]
            reference     = item["reference_answer"]
            rel_ids_all   = item.get("relevant_chunk_ids", {})
            rel_ids       = rel_ids_all.get(chunking_method, []) if isinstance(rel_ids_all, dict) else rel_ids_all

            logger.info(f"  [{q_id}] {question[:60]}...")

            # ── Retrieve ────────────────────────────────────────────────
            retrieved       = self._retrieve(collection, question, top_k=top_k)
            retrieved_texts = [r["document"] for r in retrieved]
            retrieved_ids   = [r["id"]       for r in retrieved]

            # ── Generate (opsional) ───────────────────────────────────────
            answer = None
            if generator is not None:
                try:
                    raw = generator.generate(question, retrieved_texts)
                    answer = raw[0] if isinstance(raw, tuple) else raw
                    logger.info(f"    answer length: {len(answer)} chars")
                except Exception as e:
                    logger.warning(f"    generation error: {e}")

            # ── Per-query result ───────────────────────────────────────────
            q_result: Dict[str, Any] = {
                "q_id"          : q_id,
                "question"      : question,
                "retrieved_ids" : retrieved_ids,
                "answer"        : answer,
            }

            # ── BLEU + ROUGE-L ───────────────────────────────────────────────
            if answer:
                q_result["bleu"]   = compute_bleu(answer, reference)
                q_result["rouge_l"] = compute_rouge(answer, reference, "rougeL", "recall")
                logger.info(
                    f"    BLEU={q_result['bleu']:.4f} | "
                    f"RL={q_result['rouge_l']:.4f}"
                )

            # ── Precision@k, Recall@k, MRR ──────────────────────────────────────
            if rel_ids:
                q_result["precision_at_k"] = compute_precision_at_k(retrieved_ids, rel_ids, top_k)
                q_result["recall_at_k"]    = compute_recall_at_k(retrieved_ids, rel_ids, top_k)
                q_result["mrr"]            = compute_mrr(retrieved_ids, rel_ids)
                q_result["f1_at_k"]        = compute_f1_at_k(
                    q_result["precision_at_k"],
                    q_result["recall_at_k"],
                )
                logger.info(
                    f"    P@{top_k}={q_result['precision_at_k']:.4f} | "
                    f"R@{top_k}={q_result['recall_at_k']:.4f} | "
                    f"MRR={q_result['mrr']:.4f} | "
                    f"F1@{top_k}={q_result['f1_at_k']:.4f}"
                )
            else:
                logger.warning(
                    f"    [{q_id}] no relevant_chunk_ids for method '{chunking_method}' — skipping retrieval metrics"
                )

            result.per_query.append(q_result)

        # ── Aggregate (mean) ─────────────────────────────────────────────────
        for key in ["precision_at_k", "recall_at_k", "mrr", "f1_at_k", "bleu", "rouge_l"]:
            values = [q[key] for q in result.per_query if key in q]
            if values:
                result.metrics[key] = round(sum(values) / len(values), 6)

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
            )
            results.append(r)

        return results


# ─── Factory ────────────────────────────────────────────────────────────────────

def build_evaluator(
    embedder_path: str = DEFAULT_EMBEDDER_PATH,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    n_gpu_layers: int = -1,
    embedder_mode: str = "gguf",
    hf_model_name: str = "/workspace/models/Qwen3-Embedding-4B",
) -> RAGEvaluator:
    """
    Factory function: load komponen dan return RAGEvaluator.

    Args:
        embedder_path : Path ke GGUF embedding model
        chroma_path   : Path ke ChromaDB persistent storage
        n_gpu_layers  : GPU layers untuk embedder (-1 = semua)
        embedder_mode : "gguf" atau "huggingface"
        hf_model_name : Path ke HF embedding model (jika mode=huggingface)

    Returns:
        RAGEvaluator yang siap digunakan

    Raises:
        RuntimeError jika embedder atau ChromaDB gagal di-load
    """
    logger.info("Building RAGEvaluator...")

    if embedder_mode == "huggingface":
        embedder = initialize_hf_embedder(
            model_name=hf_model_name,
            device='cuda',
            normalize=True,
        )
    else:
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

    return RAGEvaluator(
        embedder=embedder,
        chroma_client=chroma_client,
    )
