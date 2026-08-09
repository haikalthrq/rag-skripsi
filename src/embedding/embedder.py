"""Wrapper Qwen3-Embedding untuk llama-cpp-python dan sentence-transformers."""

import logging
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

# GGUF Support via llama-cpp-python
try:
    from llama_cpp import Llama  # type: ignore[import-not-found, import-untyped]
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[misc]
    _LLAMA_CPP_AVAILABLE = False

# HuggingFace Support via sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found, import-untyped]
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc]
    _SENTENCE_TRANSFORMER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default model paths
DEFAULT_GGUF_MODEL_PATH = "models/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_HF_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


class QwenEmbedder:
    """
    Wrapper class untuk Qwen3 embedding model.
    
    Supports both GGUF (via llama-cpp) and HuggingFace (via sentence-transformers).
    """
    
    def __init__(
        self,
        model: Union[Llama, SentenceTransformer],
        mode: str,
        normalize: bool = True
    ):
        """
        Initialize embedder dengan model yang sudah di-load.
        
        Args:
            model: Instance dari Llama (GGUF) atau SentenceTransformer (HF)
            mode: 'gguf' atau 'huggingface'
            normalize: Normalize embeddings ke unit vectors
        """
        self.model = model
        self.mode = mode
        self.normalize = normalize
        
        logger.info(f"QwenEmbedder initialized (mode: {mode}, normalize: {normalize})")
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings untuk satu atau multiple texts.
        
        Args:
            texts: Single text atau list of texts
            batch_size: Saat ini diabaikan. GGUF memproses satu per satu dan
                HuggingFace menggunakan batch size 1.
            
        Returns:
            np.ndarray: Embeddings dengan shape (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) == 0:
            return np.array([])
        
        # Generate embeddings based on mode
        if self.mode == 'gguf':
            embeddings = self._embed_gguf(texts)
        elif self.mode == 'huggingface':
            embeddings = self._embed_hf(texts)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Normalize if requested
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings
    
    def _embed_gguf(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using GGUF model via llama-cpp."""
        embeddings = []
        
        for text in texts:
            # llama-cpp embed() returns a list of floats
            embedding = self.model.embed(text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    # Catatan: pemotongan berikut mengubah konten yang di-embed dan dilakukan
    # berdasarkan jumlah karakter, bukan token. Bagian setelah batas dibuang.
    # Maksimum karakter per teks sebelum encode. Qwen3-Embedding memiliki
    # max_seq_length 8192 token; 4096 chars ≈ 2048 token.
    # Diturunkan dari 8192 → 4096 agar aman di RTX 3090 24GB tanpa OOM.
    _MAX_CHARS_HF = 4096

    def _embed_hf(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace model."""
        # Truncate teks yang terlalu panjang sebelum encode untuk mencegah OOM.
        truncated = [t[:self._MAX_CHARS_HF] if len(t) > self._MAX_CHARS_HF else t for t in texts]
        n_truncated = sum(1 for t in texts if len(t) > self._MAX_CHARS_HF)
        if n_truncated:
            logger.warning(f"_embed_hf: {n_truncated} texts truncated to {self._MAX_CHARS_HF} chars (mencegah OOM)")
        # batch_size=1 untuk keamanan memori maksimal pada model 4B param.
        embeddings = self.model.encode(
            truncated,
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=len(truncated) > 100
        )
        
        return embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors (L2 norm = 1)."""
        if embeddings.size == 0:
            return embeddings
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.mode == 'gguf':
            # Test with dummy text
            test_emb = self.model.embed("test")
            return len(test_emb)
        elif self.mode == 'huggingface':
            return self.model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def initialize_gguf_embedder(
    model_path: str = DEFAULT_GGUF_MODEL_PATH,
    n_gpu_layers: int = -1,
    n_ctx: int = 8192,  # Harus cukup untuk chunk terpanjang (~3500 token dari kalimat 13753 chars)
    n_batch: int = 64,  # Kecilkan dari 512 → 64 untuk hindari OOM
    normalize: bool = True,
    verbose: bool = False
) -> Optional[QwenEmbedder]:
    """
    Initialize GGUF embedder menggunakan llama-cpp-python.
    
    Args:
        model_path: Path ke GGUF model file
        n_gpu_layers: Jumlah layer di GPU (-1 = all)
        n_ctx: Context length
        n_batch: Batch size
        normalize: Normalize embeddings
        verbose: Enable verbose logging
        
    Returns:
        QwenEmbedder instance atau None jika gagal
    """
    if not _LLAMA_CPP_AVAILABLE:
        logger.error("llama-cpp-python tidak tersedia. Install dengan: pip install llama-cpp-python")
        return None
    
    try:
        model_file = Path(model_path)
        
        if not model_file.exists():
            logger.error(f"Model GGUF tidak ditemukan: {model_path}")
            return None
        
        logger.info(f"Loading GGUF model: {model_path}")
        logger.info(f"  - GPU Layers: {n_gpu_layers} (-1 = all)")
        logger.info(f"  - Context Length: {n_ctx}")
        logger.info(f"  - Batch Size: {n_batch}")
        
        model = Llama(
            model_path=str(model_file),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            embedding=True,  # Enable embedding mode
            verbose=verbose
        )
        
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info("✓ GGUF model loaded successfully")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        
        embedder = QwenEmbedder(model=model, mode='gguf', normalize=normalize)
        
        # Test embedding dimension
        dim = embedder.get_embedding_dim()
        logger.info(f"  - Embedding dimension: {dim}")
        
        return embedder
        
    except Exception as e:
        logger.error(f"Error loading GGUF model: {str(e)}")
        return None


def initialize_hf_embedder(
    model_name: str = DEFAULT_HF_MODEL_NAME,
    device: str = 'cuda',
    normalize: bool = True
) -> Optional[QwenEmbedder]:
    """
    Initialize HuggingFace embedder menggunakan sentence-transformers.
    
    Args:
        model_name: HuggingFace model name
        device: Device untuk inference ('cpu' atau 'cuda')
        normalize: Normalize embeddings
        
    Returns:
        QwenEmbedder instance atau None jika gagal
    """
    if not _SENTENCE_TRANSFORMER_AVAILABLE:
        logger.error("sentence-transformers tidak tersedia. Install dengan: pip install sentence-transformers")
        return None
    
    try:
        logger.info(f"Loading HuggingFace model: {model_name}")
        logger.info(f"  - Device: {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
        logger.info("✓ HuggingFace model loaded successfully")
        
        embedder = QwenEmbedder(model=model, mode='huggingface', normalize=normalize)
        
        # Test embedding dimension
        dim = embedder.get_embedding_dim()
        logger.info(f"  - Embedding dimension: {dim}")
        
        return embedder
        
    except Exception as e:
        logger.error(f"Error loading HuggingFace model: {str(e)}")
        return None
