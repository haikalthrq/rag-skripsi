"""
Fungsi komputasi metrik evaluasi RAG.

Libraries:
- ragas (collections-based API) : BLEU, ROUGE-L Recall
  BleuScore   → sacrebleu.corpus_bleu internally
  RougeScore  → rouge_type='rougeL', mode='recall'
- pure Python: Precision@k, Recall@k, MRR

Dependencies:
  pip install ragas sacrebleu rouge-score
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# ─── Precision@k ──────────────────────────────────────────────────────────────

def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Precision@k berbasis chunk ID.

    Args:
        retrieved_ids : List chunk ID yang di-retrieve (ordered, top-k)
        relevant_ids  : List chunk ID yang relevan (dari ground truth)
        k             : Cutoff — hanya pertimbangkan top-k retrieved

    Returns:
        Precision@k (0.0 – 1.0)
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r_id in top_k if r_id in relevant_set)
    return hits / k


# ─── Recall@k ─────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Recall@k berbasis chunk ID.

    Args:
        retrieved_ids : List chunk ID yang di-retrieve (ordered)
        relevant_ids  : List chunk ID yang relevan (dari ground truth)
        k             : Cutoff — hanya pertimbangkan top-k retrieved

    Returns:
        Recall@k (0.0 – 1.0), atau 0.0 jika relevant_ids kosong
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r_id in top_k if r_id in relevant_set)
    return hits / len(relevant_ids)


# ─── MRR — pure Python ────────────────────────────────────────────────────────

def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    Reciprocal Rank untuk satu query.

    Args:
        retrieved_ids : List chunk ID yang di-retrieve (ordered, best first)
        relevant_ids  : List chunk ID yang relevan (dari ground truth)

    Returns:
        Reciprocal Rank (0.0 – 1.0) — 0.0 jika tidak ada chunk relevan
    """
    relevant_set = set(relevant_ids)
    for rank, r_id in enumerate(retrieved_ids, start=1):
        if r_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def compute_bleu(response: str, reference: str) -> float:
    """
    BLEU score via ragas BleuScore (collections-based API, non-LLM).
    Internally menggunakan sacrebleu.corpus_bleu.

    Args:
        response : Generated answer
        reference: Reference answer (gold)

    Returns:
        BLEU score (0.0 – 1.0)
    """
    try:
        from ragas.metrics.collections import BleuScore
        scorer = BleuScore()
        result = scorer.score(response=response, reference=reference)
        return float(result.value)
    except Exception as e:
        logger.error(f"compute_bleu error: {e}")
        return 0.0


# ─── ROUGE ────────────────────────────────────────────────────────────────────

def compute_rouge(
    response: str,
    reference: str,
    rouge_type: str = "rougeL",
    mode: str = "recall",
) -> float:
    """
    ROUGE score via ragas RougeScore (collections-based API, non-LLM).
    Default: rouge_type='rougeL', mode='recall' → ROUGE-L Recall.

    Args:
        response  : Generated answer
        reference : Reference answer (gold)
        rouge_type: 'rouge1' atau 'rougeL' (default: 'rougeL')
        mode      : 'precision', 'recall', atau 'fmeasure' (default: 'recall')

    Returns:
        ROUGE score (0.0 – 1.0)
    """
    try:
        from ragas.metrics.collections import RougeScore
        scorer = RougeScore(rouge_type=rouge_type, mode=mode)
        result = scorer.score(response=response, reference=reference)
        return float(result.value)
    except Exception as e:
        logger.error(f"compute_rouge ({rouge_type}, {mode}) error: {e}")
        return 0.0


