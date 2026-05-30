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
    BLEU score via sacrebleu.corpus_bleu (non-LLM).
    Menggunakan library sacrebleu langsung untuk menghindari dependency issue dengan ragas.

    Args:
        response : Generated answer
        reference: Reference answer (gold)

    Returns:
        BLEU score (0.0 – 1.0)
    """
    try:
        from sacrebleu import corpus_bleu
        # sacrebleu signature: corpus_bleu(hypotheses, references)
        #   - hypotheses : Sequence[str]            → [response]
        #   - references : Sequence[Sequence[str]]  → [[reference]]
        # Urutan/nesting argumen WAJIB seperti ini; jika tertukar, skor selalu 0.0.
        result = corpus_bleu([response], [[reference]])
        return result.score / 100.0  # sacrebleu returns 0-100, convert to 0-1
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
    ROUGE score via rouge-score (non-LLM).
    Menggunakan library rouge-score langsung untuk menghindari dependency issue dengan ragas.
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
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
        scores = scorer.score(reference, response)
        # Extract the appropriate metric based on mode
        rouge_score = scores[rouge_type]
        if mode == "precision":
            return rouge_score.precision
        elif mode == "recall":
            return rouge_score.recall
        elif mode == "fmeasure":
            return rouge_score.fmeasure
        else:
            logger.warning(f"Unknown mode: {mode}, defaulting to recall")
            return rouge_score.recall
    except Exception as e:
        logger.error(f"compute_rouge ({rouge_type}, {mode}) error: {e}")
        return 0.0


