"""
Modul embedding untuk generate vector embeddings dari chunks.

Mendukung:
- Qwen3-Embedding-4B (GGUF format via llama-cpp-python - RECOMMENDED)
- Qwen3-Embedding-4B (HuggingFace transformers - fallback)

Pipeline:
1. Load chunks dari JSON (element_based, maxmin_semantic, recursive)
2. Clean whitespace dan filter chunk kosong
3. Generate embeddings menggunakan model
4. Save embeddings ke file
"""

# Catatan: pipeline aktual juga memperkaya teks tabel dan menambahkan context
# prefix untuk MaxMin/recursive sebelum embedding. Lihat embed_chunks.py untuk
# membedakan teks chunk tersimpan dari teks persis yang masuk ke model.

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
