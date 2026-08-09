"""Metrik per-query untuk evaluasi retrieval dan generation RAG.

Precision@k, Recall@k, MRR, dan F1@k dihitung langsung dengan Python. BLEU
memakai ``sacrebleu`` dan ROUGE memakai ``rouge-score`` jika tersedia; keduanya
memiliki fallback Python sederhana yang tidak ekuivalen dengan backend utama.
"""

import logging
import math
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
    hits = len(set(top_k) & relevant_set)
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
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = len(set(top_k) & relevant_set)
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


def compute_f1_at_k(precision, recall):
    """
    F1@k dari nilai Precision@k dan Recall@k.

    Args:
        precision: Nilai Precision@k numerik, atau nilai non-numerik seperti "N/A".
        recall   : Nilai Recall@k numerik, atau nilai non-numerik seperti "N/A".

    Returns:
        Float F1@k jika input valid, 0.0 jika precision + recall = 0,
        atau "N/A" jika input tidak valid.
    """
    try:
        if precision is None or recall is None:
            return "N/A"
        if isinstance(precision, str) and precision.strip() == "":
            return "N/A"
        if isinstance(recall, str) and recall.strip() == "":
            return "N/A"

        precision_val = float(precision)
        recall_val = float(recall)

        if math.isnan(precision_val) or math.isnan(recall_val):
            return "N/A"

        denominator = precision_val + recall_val
        if denominator == 0:
            return 0.0
        return 2 * precision_val * recall_val / denominator
    except (TypeError, ValueError):
        return "N/A"


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
    except ImportError:
        # Catatan: fallback ini adalah overlap unigram sederhana dengan brevity
        # factor, bukan BLEU SacreBLEU yang ekuivalen. Hasil antar-environment
        # tidak boleh dibandingkan tanpa mencatat dependency yang tersedia.
        response_tokens = response.split()
        reference_tokens = reference.split()
        if not response_tokens or not reference_tokens:
            return 0.0
        if response == reference:
            return 1.0
        ref_counts = {}
        for token in reference_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        hits = 0
        for token in response_tokens:
            if ref_counts.get(token, 0) > 0:
                hits += 1
                ref_counts[token] -= 1
        precision = hits / len(response_tokens)
        brevity = min(1.0, len(response_tokens) / len(reference_tokens))
        return precision * brevity
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
    except ImportError:
        # Catatan: fallback ini menghitung LCS berbasis whitespace dan bukan
        # implementasi rouge-score yang ekuivalen. Catat backend metrik saat
        # memakai hasil evaluasi untuk perbandingan eksperimen.
        response_tokens = response.split()
        reference_tokens = reference.split()
        if not response_tokens or not reference_tokens:
            return 0.0
        m, n = len(response_tokens), len(reference_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if response_tokens[i - 1] == reference_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / m
        recall = lcs / n
        if mode == "precision":
            return precision
        if mode == "fmeasure":
            return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return recall
    except Exception as e:
        logger.error(f"compute_rouge ({rouge_type}, {mode}) error: {e}")
        return 0.0
