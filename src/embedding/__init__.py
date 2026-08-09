"""Embedding chunk dengan Qwen3 melalui backend GGUF atau HuggingFace.

Pipeline batch memperkaya teks tabel, membuang chunk kosong, dan menambahkan
context prefix sementara untuk metode MaxMin/recursive sebelum embedding.
Lihat ``embed_chunks`` untuk aturan transformasi dan provenance teks.
"""

from .embedder import QwenEmbedder, initialize_gguf_embedder, initialize_hf_embedder
from .io import load_chunks_from_json, save_embeddings, load_embeddings
from .embed_chunks import embed_all_chunks, embed_single_file

__all__ = [
    'QwenEmbedder',
    'initialize_gguf_embedder',
    'initialize_hf_embedder',
    'load_chunks_from_json',
    'save_embeddings',
    'load_embeddings',
    'embed_all_chunks',
    'embed_single_file',
]

__version__ = '1.0.0'
