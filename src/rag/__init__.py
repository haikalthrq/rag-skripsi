"""
RAG (Retrieval-Augmented Generation) module.

Components:
- generator.py : LLM generation via GGUF (llama-cpp-python)
- pipeline.py  : End-to-end pipeline (embed → retrieve → generate)
"""

from .generator import (
    RAGGenerator,
    HFRAGGenerator,
    initialize_gguf_generator,
    initialize_hf_generator,
)
from .pipeline import RAGPipeline, COLLECTION_NAMES, build_pipeline

__all__ = [
    "RAGGenerator",
    "HFRAGGenerator",
    "initialize_gguf_generator",
    "initialize_hf_generator",
    "RAGPipeline",
    "COLLECTION_NAMES",
    "build_pipeline",
]
