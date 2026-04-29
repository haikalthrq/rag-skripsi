"""
End-to-end RAG Pipeline.

Alur:
  query (str)
    → embed query  (QwenEmbedder)
    → retrieve     (ChromaDB similarity search)
    → generate     (RAGGenerator / GGUF LLM)
    → result dict

Mendukung 3 chunking method:
  - element_based   → collection_element_based
  - maxmin_semantic → collection_maxmin_semantic
  - recursive       → collection_recursive
"""

import logging
import time
from typing import Optional, List, Dict, Any

from .generator import (
    RAGGenerator,
    HFRAGGenerator,
    initialize_gguf_generator,
    initialize_hf_generator,
)
from ..chroma.client import initialize_chroma_client, get_or_create_collection
from ..chroma.query import similarity_search
from ..embedding.embedder import QwenEmbedder, initialize_gguf_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAMES: Dict[str, str] = {
    "element_based":   "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive":       "collection_recursive",
}

DEFAULT_EMBEDDER_PATH = "models/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_CHROMA_PATH   = "data/chroma"


class RAGPipeline:
    """
    End-to-end RAG Pipeline.

    Attributes:
        embedder        : QwenEmbedder untuk embed query
        generator       : RAGGenerator untuk generate jawaban
        chroma_client   : ChromaDB client
        collection      : ChromaDB collection yang aktif
        chunking_method : Nama metode chunking
        top_k           : Default jumlah chunk yang di-retrieve
    """

    def __init__(
        self,
        embedder: QwenEmbedder,
        generator: RAGGenerator,
        chroma_client: Any,
        chunking_method: str = "element_based",
        top_k: int = 5,
    ):
        """
        Args:
            embedder        : Instance QwenEmbedder yang sudah di-load
            generator       : Instance RAGGenerator yang sudah di-load
            chroma_client   : ChromaDB persistent client
            chunking_method : Salah satu dari COLLECTION_NAMES.keys()
            top_k           : Jumlah chunk yang di-retrieve per query
        """
        if chunking_method not in COLLECTION_NAMES:
            raise ValueError(
                f"Unknown chunking_method: '{chunking_method}'. "
                f"Pilih dari: {list(COLLECTION_NAMES.keys())}"
            )

        self.embedder = embedder
        self.generator = generator
        self.chroma_client = chroma_client
        self.chunking_method = chunking_method
        self.top_k = top_k

        collection_name = COLLECTION_NAMES[chunking_method]
        self.collection = get_or_create_collection(chroma_client, collection_name)

        if self.collection is None:
            raise RuntimeError(
                f"Gagal memuat collection '{collection_name}'. "
                "Pastikan data sudah di-load ke ChromaDB via load_to_chroma.py"
            )

        doc_count = self.collection.count()
        logger.info("RAGPipeline initialized")
        logger.info(f"  - Chunking method : {chunking_method}")
        logger.info(f"  - Collection      : {collection_name} ({doc_count} docs)")
        logger.info(f"  - Top-k           : {top_k}")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query lalu retrieve top-k chunks dari ChromaDB.

        Args:
            query : String pertanyaan user
            k     : Jumlah chunk (default: self.top_k)

        Returns:
            List of dicts: {id, document, metadata, distance}
        """
        k = k if k is not None else self.top_k

        query_embedding = self.embedder.embed(query)
        query_vec = query_embedding[0]

        results = similarity_search(self.collection, query_vec, k=k)
        logger.info(f"Retrieved {len(results)} chunks untuk query: '{query[:60]}...'")

        return results

    def run(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Jalankan full RAG pipeline untuk satu query.

        Args:
            query : Pertanyaan user
            k     : Override jumlah chunk yang di-retrieve

        Returns:
            Dictionary dengan keys:
              - query            : str
              - answer           : str
              - retrieved_chunks : List[Dict]
              - chunking_method  : str
              - num_chunks       : int
              - elapsed_seconds  : float
        """
        start = time.time()

        # --- Step 1: Retrieve ---
        retrieved = self.retrieve(query, k=k)

        if not retrieved:
            logger.warning("Tidak ada chunk yang berhasil di-retrieve")
            return {
                "query": query,
                "answer": "Tidak dapat menemukan informasi yang relevan dalam dokumen.",
                "retrieved_chunks": [],
                "chunking_method": self.chunking_method,
                "num_chunks": 0,
                "elapsed_seconds": round(time.time() - start, 3),
            }

        # --- Step 2: Extract context texts ---
        contexts = [doc["document"] for doc in retrieved]

        # --- Step 3: Generate ---
        logger.info(f"Generating dari {len(contexts)} konteks...")
        raw = self.generator.generate(query, contexts)

        # HFRAGGenerator dengan return_thinking=True mengembalikan (answer, thinking)
        if isinstance(raw, tuple):
            answer, thinking = raw
        else:
            answer, thinking = raw, ""

        elapsed = round(time.time() - start, 3)
        logger.info(f"✓ Pipeline selesai dalam {elapsed}s")

        return {
            "query": query,
            "answer": answer,
            "thinking": thinking,
            "retrieved_chunks": retrieved,
            "chunking_method": self.chunking_method,
            "num_chunks": len(retrieved),
            "elapsed_seconds": elapsed,
        }


def build_pipeline(
    chunking_method: str = "element_based",
    embedder_path: str = DEFAULT_EMBEDDER_PATH,
    generator_path: str = "",
    generator_type: str = "gguf",
    chroma_path: str = DEFAULT_CHROMA_PATH,
    top_k: int = 5,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k_gen: int = 20,
    return_thinking: bool = False,
    verbose: bool = False,
) -> RAGPipeline:
    """
    Factory function: load semua komponen dan return RAGPipeline.

    Args:
        chunking_method : Metode chunking ('element_based', 'maxmin_semantic', 'recursive')
        embedder_path   : Path ke GGUF embedding model
        generator_path  : Path/nama model generator (GGUF path atau HF model name)
        generator_type  : 'gguf' atau 'hf'
        chroma_path     : Path ke ChromaDB persistent storage
        top_k           : Jumlah chunk per query
        n_gpu_layers    : GPU layers untuk GGUF (-1 = semua)
        n_ctx           : Context length untuk GGUF generator
        max_tokens      : Max output tokens
        temperature     : Sampling temperature
        top_p           : Nucleus sampling
        top_k_gen       : Top-K sampling untuk HF generator (default: 20)
        return_thinking : Kembalikan thinking content (HF only)
        verbose         : Verbose llama.cpp output (GGUF only)

    Returns:
        RAGPipeline instance yang siap digunakan

    Raises:
        RuntimeError jika salah satu komponen gagal di-load
    """
    if generator_type not in ("gguf", "hf"):
        raise ValueError(f"generator_type harus 'gguf' atau 'hf', bukan '{generator_type}'")

    # Load embedder
    logger.info("Memuat embedder...")
    embedder = initialize_gguf_embedder(
        model_path=embedder_path,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    if embedder is None:
        raise RuntimeError(f"Gagal memuat embedder: {embedder_path}")

    # Load generator
    if generator_type == "gguf":
        logger.info("Memuat generator (GGUF)...")
        generator = initialize_gguf_generator(
            model_path=generator_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
    else:
        logger.info("Memuat generator (HuggingFace)...")
        generator = initialize_hf_generator(
            model_name=generator_path,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_gen,
            return_thinking=return_thinking,
        )

    if generator is None:
        raise RuntimeError(f"Gagal memuat generator ({generator_type}): {generator_path}")

    # ChromaDB client
    chroma_client = initialize_chroma_client(persist_directory=chroma_path)
    if chroma_client is None:
        raise RuntimeError(f"Gagal koneksi ke ChromaDB: {chroma_path}")

    return RAGPipeline(
        embedder=embedder,
        generator=generator,
        chroma_client=chroma_client,
        chunking_method=chunking_method,
        top_k=top_k,
    )
