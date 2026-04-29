"""
Fungsi komputasi metrik evaluasi RAG.

Libraries:
- Ragas  : BLEU, ROUGE, Context Precision (LLM/NonLLM), Context Recall (LLM/NonLLM)
- rank_eval : MRR
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Jalankan coroutine secara sinkron, kompatibel dengan event loop yang sudah ada."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def compute_bleu(response: str, reference: str) -> float:
    """
    BLEU score via ragas BleuScore (non-LLM).

    Args:
        response : Generated answer
        reference: Reference answer

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
    mode: str = "fmeasure",
) -> float:
    """
    ROUGE score via ragas RougeScore (non-LLM).

    Args:
        response  : Generated answer
        reference : Reference answer
        rouge_type: 'rouge1' atau 'rougeL'
        mode      : 'precision', 'recall', atau 'fmeasure'

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


# ─── Context Precision — LLM-based ────────────────────────────────────────────

def compute_context_precision_llm(
    user_input: str,
    retrieved_contexts: List[str],
    reference: str,
    eval_llm: Any,
) -> float:
    """
    LLMContextPrecisionWithReference via ragas (membutuhkan evaluator LLM).

    Menggunakan legacy SingleTurnSample API karena lebih mudah diintegrasikan.

    Args:
        user_input         : Query / pertanyaan
        retrieved_contexts : List teks chunk yang di-retrieve
        reference          : Reference answer
        eval_llm           : Evaluator LLM (LangchainLLMWrapper)

    Returns:
        Context Precision score (0.0 – 1.0)
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import LLMContextPrecisionWithReference

        scorer = LLMContextPrecisionWithReference(llm=eval_llm)
        sample = SingleTurnSample(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        return float(_run_async(scorer.single_turn_ascore(sample)))
    except Exception as e:
        logger.error(f"compute_context_precision_llm error: {e}")
        return 0.0


# ─── Context Precision — Non-LLM ──────────────────────────────────────────────

def compute_context_precision_nonllm(
    retrieved_contexts: List[str],
    reference_contexts: List[str],
) -> float:
    """
    NonLLMContextPrecisionWithReference via ragas.

    Args:
        retrieved_contexts : List teks chunk yang di-retrieve
        reference_contexts : List teks konteks referensi dari ground truth

    Returns:
        Context Precision score (0.0 – 1.0)
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import NonLLMContextPrecisionWithReference

        scorer = NonLLMContextPrecisionWithReference()
        sample = SingleTurnSample(
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )
        return float(_run_async(scorer.single_turn_ascore(sample)))
    except Exception as e:
        logger.error(f"compute_context_precision_nonllm error: {e}")
        return 0.0


# ─── Context Recall — LLM-based ───────────────────────────────────────────────

def compute_context_recall_llm(
    user_input: str,
    retrieved_contexts: List[str],
    reference: str,
    eval_llm: Any,
) -> float:
    """
    LLMContextRecall via ragas (membutuhkan evaluator LLM).

    Args:
        user_input         : Query / pertanyaan
        retrieved_contexts : List teks chunk yang di-retrieve
        reference          : Reference answer
        eval_llm           : Evaluator LLM (LangchainLLMWrapper)

    Returns:
        Context Recall score (0.0 – 1.0)
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import LLMContextRecall

        scorer = LLMContextRecall(llm=eval_llm)
        sample = SingleTurnSample(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        return float(_run_async(scorer.single_turn_ascore(sample)))
    except Exception as e:
        logger.error(f"compute_context_recall_llm error: {e}")
        return 0.0


# ─── Context Recall — Non-LLM ─────────────────────────────────────────────────

def compute_context_recall_nonllm(
    retrieved_contexts: List[str],
    reference_contexts: List[str],
) -> float:
    """
    NonLLMContextRecall via ragas.

    Args:
        retrieved_contexts : List teks chunk yang di-retrieve
        reference_contexts : List teks konteks referensi dari ground truth

    Returns:
        Context Recall score (0.0 – 1.0)
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import NonLLMContextRecall

        scorer = NonLLMContextRecall()
        sample = SingleTurnSample(
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )
        return float(_run_async(scorer.single_turn_ascore(sample)))
    except Exception as e:
        logger.error(f"compute_context_recall_nonllm error: {e}")
        return 0.0


# ─── MRR — rank_eval ──────────────────────────────────────────────────────────

def compute_mrr(
    qrels_data: Dict[str, List[str]],
    run_data: Dict[str, Dict[str, float]],
    mrr_at_k: Optional[int] = None,
) -> float:
    """
    Mean Reciprocal Rank via rank_eval.

    Args:
        qrels_data : {query_id: [relevant_chunk_id, ...]}
                     ID chunk yang dianggap relevan per query.
        run_data   : {query_id: {chunk_id: similarity_score}}
                     Hasil retrieval dengan skor per chunk (lebih tinggi = lebih relevan).
        mrr_at_k   : Jika diberikan, hitung MRR@K. Jika None, hitung MRR tanpa threshold.

    Returns:
        MRR score (0.0 – 1.0)
    """
    try:
        from rank_eval import Qrels, Run, evaluate

        q_ids = list(qrels_data.keys())

        qrels = Qrels()
        qrels.add_multi(
            q_ids=q_ids,
            doc_ids=[qrels_data[q] for q in q_ids],
            scores=[[1] * len(qrels_data[q]) for q in q_ids],
        )

        run_q_ids = list(run_data.keys())
        run = Run()
        run.add_multi(
            q_ids=run_q_ids,
            doc_ids=[list(run_data[q].keys()) for q in run_q_ids],
            scores=[list(run_data[q].values()) for q in run_q_ids],
        )

        metric_key = f"mrr@{mrr_at_k}" if mrr_at_k else "mrr"
        result = evaluate(qrels, run, metric_key)
        return float(result)
    except Exception as e:
        logger.error(f"compute_mrr error: {e}")
        return 0.0
