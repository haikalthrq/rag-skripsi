"""
Evaluation module untuk RAG pipeline.

Components:
- metrics.py   : Fungsi komputasi per-metrik (BLEU, ROUGE, Precision, Recall, MRR)
- evaluator.py : Orkestrasi evaluasi + perbandingan 3 chunking methods
"""

from .evaluator import RAGEvaluator, build_evaluator, load_ground_truth, COLLECTION_NAMES

__all__ = [
    "RAGEvaluator",
    "build_evaluator",
    "load_ground_truth",
    "COLLECTION_NAMES",
]
