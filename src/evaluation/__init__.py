"""
Evaluation module untuk RAG pipeline.

Components:
- metrics.py   : Fungsi komputasi per-metrik (BLEU, ROUGE, Precision, Recall, MRR)
- evaluator.py : Orkestrasi evaluasi + perbandingan 3 chunking methods
"""

__all__ = [
    "RAGEvaluator",
    "build_evaluator",
    "load_ground_truth",
    "COLLECTION_NAMES",
]


def __getattr__(name):
    """Lazy-load evaluator so importing metrics does not require ChromaDB/numpy."""
    if name in __all__:
        from . import evaluator

        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
