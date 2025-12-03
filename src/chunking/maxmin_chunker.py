"""
Modul MaxMin Semantic Chunking.

Modul ini memanggil library maxmin_chunker untuk melakukan semantic chunking
berdasarkan similarity threshold dinamis. Menggunakan Qwen3-Embedding model
dalam format GGUF untuk generate sentence embeddings.

Supported Models:
- Qwen3-Embedding-4B-GGUF (q4_K_M, q5_0, q5_K_M, q6_K, q8_0, f16)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import nltk  # type: ignore[import-untyped]
    from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
except ImportError:
    nltk = None  # type: ignore[assignment]
    sent_tokenize = None  # type: ignore[assignment]

# Cosine similarity dari sklearn
try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]
except ImportError:
    cosine_similarity = None  # type: ignore[assignment]

# GGUF Support via llama-cpp-python
try:
    from llama_cpp import Llama  # type: ignore[import-not-found, import-untyped]
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[misc]
    _LLAMA_CPP_AVAILABLE = False

# Fallback: SentenceTransformer (optional)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found, import-untyped]
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc]
    _SENTENCE_TRANSFORMER_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MAXMIN CHUNKING ALGORITHM IMPLEMENTATION
# Berdasarkan paper: "Max-Min Semantic Chunking of Documents for RAG application"
# ==============================================================================

def sigmoid(x: float) -> float:
    """
    Fungsi sigmoid untuk adaptive threshold.
    
    Karakteristik:
    - Input: cluster_size - 1 (jumlah sentence di cluster minus 1)
    - Output: nilai 0 hingga 1
    - Cluster size kecil → sigmoid kecil → threshold rendah → mudah gabung
    - Cluster size besar → sigmoid besar → threshold tinggi → sulit gabung
    
    Args:
        x (float): Input value.
        
    Returns:
        float: Sigmoid output (0-1).
    """
    return 1 / (1 + np.exp(-x))


def process_sentences(
    sentences: List[str],
    embeddings: np.ndarray,
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5
) -> List[List[str]]:
    """
    Implementasi algoritma MaxMin Semantic Chunking.
    
    Mengelompokkan kalimat berdasarkan semantic similarity dengan threshold
    yang adaptif. Kalimat yang semantically similar digabung dalam satu chunk.
    
    Algoritma:
    1. Mulai dengan sentence pertama sebagai cluster awal
    2. Untuk setiap sentence berikutnya:
       - Hitung similarity dengan sentences di cluster saat ini
       - Hitung adaptive threshold berdasarkan cluster size dan pairwise_min
       - Jika similarity > threshold: gabung ke cluster
       - Jika tidak: buat cluster baru
    
    Args:
        sentences (List[str]): List kalimat yang sudah di-tokenize.
        embeddings (np.ndarray): Sentence embeddings shape (n_sentences, embedding_dim).
        fixed_threshold (float): Threshold minimum untuk gabung (default: 0.6).
        c (float): Coefficient untuk adaptive threshold (default: 0.9).
        init_constant (float): Boost factor untuk sentence ke-2 (default: 1.5).
        
    Returns:
        List[List[str]]: List of paragraphs, setiap paragraph adalah list of sentences.
    """
    if cosine_similarity is None:
        raise ImportError("sklearn tidak terinstall. Install dengan: pip install scikit-learn")
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) == 1:
        return [sentences]
    
    # Validasi embeddings
    if embeddings.shape[0] != len(sentences):
        raise ValueError(f"Mismatch: {len(sentences)} sentences tapi {embeddings.shape[0]} embeddings")
    
    # Initialization (STEP 0 dari dokumentasi)
    paragraphs: List[List[str]] = []
    current_paragraph: List[str] = [sentences[0]]  # Start dengan sentence pertama
    cluster_start: int = 0
    cluster_end: int = 1
    pairwise_min: float = float('-inf')  # Tracking minimum pairwise similarity
    
    # Iterasi untuk setiap sentence (STEP 1)
    for i in range(1, len(sentences)):
        # Ambil embeddings dari current cluster
        cluster_embeddings = embeddings[cluster_start:cluster_end]
        cluster_size = cluster_end - cluster_start
        
        # Hitung similarity antara sentence baru dengan cluster
        new_sentence_embedding = embeddings[i].reshape(1, -1)
        new_sentence_similarities = cosine_similarity(
            new_sentence_embedding, 
            cluster_embeddings
        )[0]  # Result shape: (cluster_size,)
        
        if cluster_size > 1:
            # CASE 1: Cluster memiliki lebih dari 1 sentence
            # Hitung adjusted threshold menggunakan formula dari paper
            adjusted_threshold = pairwise_min * c * sigmoid(cluster_size - 1)
            
            # Ambil maximum similarity (sentence baru paling similar dengan siapa)
            new_sentence_similarity = float(np.max(new_sentence_similarities))
            
            # Update pairwise_min (tracking MIN dari semua similarities)
            pairwise_min = min(float(np.min(new_sentence_similarities)), pairwise_min)
            
        else:
            # CASE 2: Cluster hanya memiliki 1 sentence
            # Set adjusted_threshold = 0 (threshold tidak ada untuk cluster size 1)
            adjusted_threshold = 0.0
            
            # Similarity dengan satu sentence (scalar value)
            pairwise_min = float(new_sentence_similarities[0])
            
            # Apply initial constant (boost untuk sentence ke-2)
            # Ini membuat sentence ke-2 lebih mudah bergabung
            new_sentence_similarity = init_constant * pairwise_min
        
        # DECISION: Gabung atau Pisah?
        final_threshold = max(adjusted_threshold, fixed_threshold)
        
        if new_sentence_similarity > final_threshold:
            # GABUNG: Tambahkan ke current paragraph
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            # PISAH: Start new paragraph
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start = i
            cluster_end = i + 1
            pairwise_min = float('-inf')  # Reset untuk cluster baru
    
    # Finalization (STEP 2): Append paragraph terakhir
    paragraphs.append(current_paragraph)
    
    return paragraphs


# Default GGUF model paths - bisa diubah sesuai lokasi model
DEFAULT_GGUF_MODEL_PATH = "models/Qwen3-Embedding-4B-Q8_0.gguf"


def initialize_embedding_model_gguf(
    model_path: str = DEFAULT_GGUF_MODEL_PATH,
    n_gpu_layers: int = -1,
    n_ctx: int = 8192,
    n_batch: int = 512,
    verbose: bool = False,
    suppress_output: bool = True
) -> Optional[Llama]:
    """
    Inisialisasi model embedding GGUF menggunakan llama-cpp-python.
    
    Model GGUF lebih hemat memory karena sudah di-quantize.
    Quantization options: q4_K_M (~2GB), q5_K_M (~2.5GB), q8_0 (~4GB), f16 (~8GB)
    
    Args:
        model_path (str): Path ke file GGUF model.
        n_gpu_layers (int): Jumlah layer di GPU (-1 = semua layer).
        n_ctx (int): Context length (default: 8192).
        n_batch (int): Batch size untuk processing (default: 512).
        verbose (bool): Tampilkan log detail dari llama.cpp.
        suppress_output (bool): Suppress stderr warnings dari llama.cpp (default: True).
        
    Returns:
        Optional[Any]: Model Llama atau None jika gagal.
    """
    if not _LLAMA_CPP_AVAILABLE:
        logger.error("Library 'llama-cpp-python' tidak terinstall.")
        logger.error("Install dengan: pip install llama-cpp-python")
        logger.error("Untuk GPU (CUDA): pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122")
        return None
    
    try:
        import sys
        import os
        
        model_file = Path(model_path)
        
        if not model_file.exists():
            logger.error(f"File model GGUF tidak ditemukan: {model_path}")
            logger.error("Download model dari: https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF")
            logger.error("Contoh: hf download Qwen/Qwen3-Embedding-4B-GGUF Qwen3-Embedding-4B-Q8_0.gguf --local-dir models/")
            return None
        
        logger.info(f"Inisialisasi GGUF embedding model: {model_path}")
        logger.info(f"  - GPU Layers: {n_gpu_layers} (-1 = all)")
        logger.info(f"  - Context Length: {n_ctx}")
        logger.info(f"  - Batch Size: {n_batch}")
        
        # Load model GGUF dengan llama-cpp-python
        model = Llama(
            model_path=str(model_path),
            embedding=True,  # Enable embedding mode
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            pooling_type=2  # LLAMA_POOLING_TYPE_LAST untuk Qwen3 embedding
        )
        
        logger.info("✓ Model GGUF berhasil diinisialisasi")
        
        # Get model info
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi model GGUF: {str(e)}")
        logger.error("Troubleshooting:")
        logger.error("  1. Pastikan file GGUF valid dan tidak corrupt")
        logger.error("  2. Pastikan llama-cpp-python terinstall dengan benar")
        logger.error("  3. Untuk GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall")
        return None


def initialize_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "cuda",
    low_memory: bool = False
) -> Optional[Any]:
    """
    Inisialisasi model embedding Qwen3 menggunakan SentenceTransformer.
    
    DEPRECATED: Gunakan initialize_embedding_model_gguf() untuk model GGUF.
    Fungsi ini tetap tersedia untuk backward compatibility.
    
    Mengikuti best practices dari dokumentasi Qwen3:
    - Menggunakan SentenceTransformer API (lebih mudah dan stabil)
    - Normalisasi embeddings untuk cosine similarity
    
    Args:
        model_name (str): Nama model HuggingFace. Default: Qwen3-Embedding-4B
        device (str): Device untuk inference ('cpu' atau 'cuda')
        low_memory (bool): Gunakan half precision (float16) untuk hemat VRAM. Default: False
        
    Returns:
        Optional[Any]: Model SentenceTransformer atau None jika gagal
    """
    if not _SENTENCE_TRANSFORMER_AVAILABLE:
        logger.error("Library 'sentence-transformers' tidak terinstall.")
        logger.error("Install dengan: pip install sentence-transformers>=2.7.0")
        logger.error("")
        logger.error("REKOMENDASI: Gunakan model GGUF untuk hemat memory!")
        logger.error("  Gunakan: initialize_embedding_model_gguf()")
        return None
    
    try:
        logger.info(f"Inisialisasi embedding model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Low memory mode: {low_memory}")
        logger.info("Menggunakan SentenceTransformer API (recommended by Qwen3)")
        
        # Load model dengan SentenceTransformer
        # Sesuai dokumentasi Qwen3, ini adalah cara paling simple dan reliable
        if low_memory:
            # Mode hemat VRAM: gunakan float16 dan 4-bit quantization
            import torch
            logger.info("  - Loading dengan half precision (float16) untuk hemat VRAM")
            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }
            )
        else:
            model = SentenceTransformer(model_name, device=device)
        
        # Opsional: Untuk performa lebih baik dengan flash attention
        # Uncomment jika sudah install flash-attention-2:
        # model = SentenceTransformer(
        #     model_name,
        #     model_kwargs={
        #         "attn_implementation": "flash_attention_2",
        #         "device_map": "auto"
        #     },
        #     tokenizer_kwargs={"padding_side": "left"}
        # )
        
        logger.info("✓ Model embedding berhasil diinisialisasi")
        logger.info("  - Embeddings akan dinormalisasi (L2 norm) untuk cosine similarity")
        
        return model
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi model: {str(e)}")
        logger.error("Troubleshooting:")
        logger.error("  1. Pastikan sudah install: pip install sentence-transformers>=2.7.0")
        logger.error("  2. Pastikan sudah install: pip install transformers>=4.51.0")
        logger.error("  3. Jika Keras error: pip install tf-keras ATAU pip uninstall keras && pip install keras==2.15.0")
        logger.error("")
        logger.error("REKOMENDASI: Gunakan model GGUF untuk hemat memory!")
        logger.error("  Gunakan: initialize_embedding_model_gguf()")
        return None


def load_text(text_path: str) -> Optional[str]:
    """
    Memuat file teks dari data/cleaned_text/.
    
    Args:
        text_path (str): Path ke file teks.
        
    Returns:
        Optional[str]: Isi file teks atau None jika gagal.
    """
    try:
        text_file = Path(text_path)
        
        if not text_file.exists():
            logger.error(f"File tidak ditemukan: {text_path}")
            return None
        
        if not text_file.suffix.lower() == '.txt':
            logger.error(f"File bukan .txt: {text_path}")
            return None
        
        logger.info(f"Memuat teks: {text_file.name}")
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"✓ Berhasil memuat {len(text)} karakter")
        return text
        
    except Exception as e:
        logger.error(f"Error saat memuat teks {text_path}: {str(e)}")
        return None


def split_sentences(text: str) -> Optional[List[str]]:
    """
    Split teks menjadi kalimat menggunakan NLTK sent_tokenize.
    
    Args:
        text (str): Teks yang akan di-split.
        
    Returns:
        Optional[List[str]]: List kalimat atau None jika gagal.
    """
    if nltk is None or sent_tokenize is None:
        logger.error("Library 'nltk' tidak terinstall.")
        logger.error("Install dengan: pip install nltk")
        return None
    
    try:
        # Download punkt tokenizer jika belum ada
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        logger.info("Splitting teks menjadi kalimat...")
        sentences = sent_tokenize(text)
        
        # Filter kalimat kosong
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.info(f"✓ Berhasil split menjadi {len(sentences)} kalimat")
        
        # Log statistik
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentences) if sentences else 0
        logger.info(f"  - Rata-rata panjang kalimat: {avg_length:.2f} karakter")
        logger.info(f"  - Kalimat terpendek: {min(sentence_lengths) if sentences else 0}")
        logger.info(f"  - Kalimat terpanjang: {max(sentence_lengths) if sentences else 0}")
        
        return sentences
        
    except Exception as e:
        logger.error(f"Error saat split sentences: {str(e)}")
        return None


def embed_sentences(
    sentences: List[str],
    embedding_model: Any,
    normalize: bool = True,
    show_progress: bool = True,
    batch_size: int = 8,
    use_gguf: bool = False
) -> Optional[np.ndarray]:
    """
    Generate embeddings untuk setiap kalimat.
    
    Mendukung dua mode:
    1. GGUF mode (use_gguf=True): Menggunakan llama-cpp-python
    2. SentenceTransformer mode (use_gguf=False): Menggunakan sentence-transformers
    
    Args:
        sentences (List[str]): List kalimat.
        embedding_model (Any): Model embedding (Llama atau SentenceTransformer).
        normalize (bool): Normalize embeddings dengan L2 norm (default: True).
        show_progress (bool): Show progress bar (default: True).
        batch_size (int): Batch size untuk encoding (default: 8).
        use_gguf (bool): True jika menggunakan model GGUF (default: False).
        
    Returns:
        Optional[np.ndarray]: Array embeddings dengan shape (n_sentences, embedding_dim)
                             atau None jika gagal.
    """
    try:
        logger.info(f"Generating embeddings untuk {len(sentences)} kalimat...")
        logger.info(f"  - Mode: {'GGUF (llama-cpp)' if use_gguf else 'SentenceTransformer'}")
        logger.info(f"  - Normalization: {normalize}")
        
        if use_gguf:
            # Mode GGUF: gunakan llama-cpp-python
            import sys
            import os
            
            embeddings_list = []
            
            # Progress tracking
            total = len(sentences)
            for i, sentence in enumerate(sentences):
                if show_progress and (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
                
                # Suppress stderr warnings dari llama.cpp (init: embeddings required...)
                # Warning ini normal dan tidak mempengaruhi hasil
                stderr_fd = sys.stderr.fileno()
                with open(os.devnull, 'w') as devnull:
                    old_stderr = os.dup(stderr_fd)
                    os.dup2(devnull.fileno(), stderr_fd)
                    try:
                        emb = embedding_model.embed(sentence)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                
                embeddings_list.append(emb)
            
            embeddings = np.array(embeddings_list)
            
            # Normalize jika diminta
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                embeddings = embeddings / norms
        
        else:
            # Mode SentenceTransformer
            logger.info(f"  - Batch size: {batch_size}")
            
            # Generate embeddings menggunakan SentenceTransformer.encode()
            # NOTE: Untuk documents, JANGAN gunakan prompt/instruction
            # Sesuai dokumentasi Qwen3: instructions hanya untuk queries
            embeddings = embedding_model.encode(
                sentences,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                batch_size=batch_size
            )
        
        logger.info(f"✓ Embeddings berhasil digenerate")
        logger.info(f"  - Shape: {embeddings.shape}")
        logger.info(f"  - Dtype: {embeddings.dtype}")
        
        # Verifikasi normalization
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            logger.info(f"  - L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
            logger.info(f"    (Expected ~1.0 jika normalized)")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error saat generate embeddings: {str(e)}")
        return None


def apply_maxmin_chunking(
    sentences: List[str],
    embeddings: np.ndarray,
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5
) -> Optional[List[List[str]]]:
    """
    Apply algoritma MaxMin Semantic Chunking.
    
    Implementasi langsung dari paper "Max-Min Semantic Chunking of Documents 
    for RAG application" (Kiss, Nagy, Szilágyi, 2025).
    
    Args:
        sentences (List[str]): List kalimat.
        embeddings (np.ndarray): Array embeddings dengan shape (n_sentences, embedding_dim).
        fixed_threshold (float): Fixed threshold untuk similarity (default: 0.6).
        c (float): Parameter untuk adaptive threshold (default: 0.9).
        init_constant (float): Initial constant untuk threshold (default: 1.5).
        
    Returns:
        Optional[List[List[str]]]: List of chunks, dimana setiap chunk adalah list of sentences,
                                   atau None jika gagal.
    """
    try:
        logger.info("Menjalankan MaxMin chunking...")
        logger.info(f"  - fixed_threshold: {fixed_threshold}")
        logger.info(f"  - c: {c}")
        logger.info(f"  - init_constant: {init_constant}")
        
        # Panggil implementasi process_sentences
        paragraphs = process_sentences(
            sentences,
            embeddings,
            fixed_threshold=fixed_threshold,
            c=c,
            init_constant=init_constant
        )
        
        logger.info(f"✓ MaxMin chunking selesai")
        logger.info(f"  - Total chunks: {len(paragraphs)}")
        
        # Log statistik chunks
        chunk_sizes = [len(p) for p in paragraphs]
        logger.info(f"  - Rata-rata kalimat per chunk: {np.mean(chunk_sizes):.2f}")
        logger.info(f"  - Min kalimat per chunk: {np.min(chunk_sizes)}")
        logger.info(f"  - Max kalimat per chunk: {np.max(chunk_sizes)}")
        
        return paragraphs
        
    except Exception as e:
        logger.error(f"Error saat MaxMin chunking: {str(e)}")
        return None


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_path: str,
    pretty_print: bool = True
) -> bool:
    """
    Menyimpan chunks dalam format JSON.
    
    Args:
        chunks (List[Dict[str, Any]]): List chunks untuk disimpan.
        output_path (str): Path file output JSON.
        pretty_print (bool): Jika True, format JSON dengan indentasi.
        
    Returns:
        bool: True jika berhasil, False jika gagal.
    """
    try:
        output_file = Path(output_path)
        
        # Buat direktori jika belum ada
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan ke JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            else:
                json.dump(chunks, f, ensure_ascii=False)
        
        logger.info(f"✓ Berhasil menyimpan {len(chunks)} chunks ke: {output_file}")
        logger.info(f"  - Ukuran file: {output_file.stat().st_size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saat menyimpan chunks ke {output_path}: {str(e)}")
        return False


def convert_paragraphs_to_chunks(
    paragraphs: List[List[str]],
    source_filename: str,
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Konversi paragraphs (list of list of sentences) menjadi format chunks dengan metadata.
    
    Args:
        paragraphs (List[List[str]]): List of chunks dari MaxMin.
        source_filename (str): Nama file sumber.
        include_metadata (bool): Jika True, sertakan metadata.
        
    Returns:
        List[Dict[str, Any]]: List chunks dengan metadata.
    """
    chunks = []
    
    try:
        for chunk_id, paragraph in enumerate(paragraphs):
            # Gabungkan kalimat menjadi text
            chunk_text = ' '.join(paragraph)
            
            chunk: Dict[str, Any] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'num_sentences': len(paragraph)
            }
            
            if include_metadata:
                chunk['metadata'] = {
                    'source_file': source_filename,
                    'chunking_method': 'maxmin_semantic',
                    'sentences': paragraph,
                    'num_characters': len(chunk_text)
                }
            
            chunks.append(chunk)
        
        logger.info(f"✓ Konversi {len(chunks)} paragraphs menjadi chunks")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error saat konversi paragraphs: {str(e)}")
        return []


def process_single_text(
    text_path: str,
    output_dir: str,
    embedding_model: Any,
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5,
    include_metadata: bool = True,
    batch_size: int = 8,
    use_gguf: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """
    Memproses satu file teks: load, split, embed, chunk, dan save.
    
    Args:
        text_path (str): Path ke file teks.
        output_dir (str): Direktori output untuk hasil chunking.
        embedding_model (Any): Model embedding yang sudah diinisialisasi.
        fixed_threshold (float): Fixed threshold untuk MaxMin.
        c (float): Parameter c untuk MaxMin.
        init_constant (float): Parameter init_constant untuk MaxMin.
        include_metadata (bool): Sertakan metadata dalam chunks.
        batch_size (int): Batch size untuk embedding (default: 8).
        use_gguf (bool): True jika menggunakan model GGUF.
        
    Returns:
        Optional[List[Dict[str, Any]]]: List chunks jika berhasil, None jika gagal.
    """
    try:
        logger.info(f"Memproses teks: {Path(text_path).name}")
        
        # 1. Load text
        text = load_text(text_path)
        if not text:
            return None
        
        # 2. Split into sentences
        sentences = split_sentences(text)
        if not sentences or len(sentences) == 0:
            logger.warning(f"Tidak ada kalimat yang dihasilkan dari {Path(text_path).name}")
            return None
        
        # 3. Generate embeddings
        embeddings = embed_sentences(
            sentences, 
            embedding_model, 
            batch_size=batch_size,
            use_gguf=use_gguf
        )
        if embeddings is None:
            return None
        
        # 4. Apply MaxMin chunking
        paragraphs = apply_maxmin_chunking(
            sentences,
            embeddings,
            fixed_threshold=fixed_threshold,
            c=c,
            init_constant=init_constant
        )
        if not paragraphs:
            logger.warning(f"Tidak ada chunks yang dihasilkan dari {Path(text_path).name}")
            return None
        
        # 5. Convert to chunks format
        chunks = convert_paragraphs_to_chunks(
            paragraphs,
            Path(text_path).name,
            include_metadata=include_metadata
        )
        if not chunks:
            return None
        
        # 6. Save chunks
        output_filename = Path(text_path).stem + "_chunks.json"
        output_path = Path(output_dir) / output_filename
        
        success = save_chunks(chunks, str(output_path), pretty_print=True)
        
        if success:
            return chunks
        else:
            return None
        
    except Exception as e:
        logger.error(f"Error saat memproses teks {text_path}: {str(e)}")
        return None


def get_text_files(input_dir: str) -> List[Path]:
    """
    Mendapatkan daftar semua file .txt dalam direktori.
    
    Args:
        input_dir (str): Path ke direktori input.
        
    Returns:
        List[Path]: List path file .txt.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Direktori input tidak ditemukan: {input_dir}")
        return []
    
    text_files = list(input_path.glob("*.txt"))
    logger.info(f"Ditemukan {len(text_files)} file .txt di {input_dir}")
    
    return text_files


def run_maxmin_chunking(
    input_dir: str = "data/cleaned",
    output_dir: str = "data/chunked/maxmin_semantic",
    model_path: str = DEFAULT_GGUF_MODEL_PATH,
    use_gguf: bool = True,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "cuda",
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5,
    include_metadata: bool = True,
    skip_existing: bool = True,
    low_memory: bool = False,
    batch_size: int = 8,
    n_gpu_layers: int = -1
) -> Dict[str, Any]:
    """
    Menjalankan MaxMin semantic chunking untuk semua file teks di direktori input.
    
    Args:
        input_dir (str): Direktori berisi file teks input (default: data/cleaned).
        output_dir (str): Direktori output untuk hasil chunking (default: data/chunked/maxmin_semantic).
        model_path (str): Path ke file GGUF model (untuk use_gguf=True).
        use_gguf (bool): Gunakan model GGUF (default: True, RECOMMENDED).
        model_name (str): Nama model HuggingFace (untuk use_gguf=False).
        device (str): Device untuk inference (default: 'cuda').
        fixed_threshold (float): Fixed threshold untuk MaxMin (default: 0.6).
        c (float): Parameter c untuk MaxMin (default: 0.9).
        init_constant (float): Parameter init_constant untuk MaxMin (default: 1.5).
        include_metadata (bool): Sertakan metadata (default: True).
        skip_existing (bool): Skip file yang sudah diproses (default: True).
        low_memory (bool): Gunakan mode hemat VRAM dengan float16 (default: False).
        batch_size (int): Batch size untuk embedding (default: 8).
        n_gpu_layers (int): Jumlah layer di GPU untuk GGUF (-1 = semua).
        
    Returns:
        Dict[str, Any]: Statistik hasil chunking.
    """
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Memulai MaxMin Semantic Chunking Pipeline")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {'GGUF (llama-cpp)' if use_gguf else 'SentenceTransformer'}")
    if use_gguf:
        logger.info(f"Model GGUF: {model_path}")
        logger.info(f"GPU Layers: {n_gpu_layers}")
    else:
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
    logger.info(f"Parameters: threshold={fixed_threshold}, c={c}, init={init_constant}")
    
    # Initialize embedding model
    logger.info("\nInisialisasi embedding model...")
    
    if use_gguf:
        embedding_model = initialize_embedding_model_gguf(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers
        )
    else:
        embedding_model = initialize_embedding_model(
            model_name=model_name, 
            device=device, 
            low_memory=low_memory
        )
    
    if embedding_model is None:
        logger.error("Gagal inisialisasi embedding model")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0,
            'duration': 0
        }
    
    # Buat direktori output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dapatkan daftar file teks
    text_files = get_text_files(input_dir)
    
    if not text_files:
        logger.warning("Tidak ada file teks yang ditemukan untuk diproses")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0,
            'duration': 0
        }
    
    # Proses setiap file teks
    stats: Dict[str, Any] = {
        'total_files': len(text_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_chunks': 0,
        'output_files': []
    }
    
    for i, text_path in enumerate(text_files, 1):
        logger.info(f"\n[{i}/{len(text_files)}] Processing: {text_path.name}")
        
        # Cek apakah file output sudah ada
        output_filename = text_path.stem + "_chunks.json"
        output_file = Path(output_dir) / output_filename
        
        if skip_existing and output_file.exists():
            logger.info(f"⊙ File output sudah ada, skip: {output_filename}")
            stats['skipped'] += 1
            continue
        
        # Proses file teks
        chunks = process_single_text(
            str(text_path),
            output_dir,
            embedding_model,
            fixed_threshold=fixed_threshold,
            c=c,
            init_constant=init_constant,
            include_metadata=include_metadata,
            batch_size=batch_size,
            use_gguf=use_gguf
        )
        
        if chunks:
            stats['processed'] += 1
            stats['total_chunks'] += len(chunks)
            stats['output_files'].append(str(output_file))
        else:
            stats['failed'] += 1
    
    # Hitung durasi
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats['duration'] = duration
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info("MaxMin Semantic Chunking Selesai")
    logger.info("="*70)
    logger.info(f"Total file teks: {stats['total_files']}")
    logger.info(f"Berhasil diproses: {stats['processed']}")
    logger.info(f"Di-skip (sudah ada): {stats['skipped']}")
    logger.info(f"Gagal: {stats['failed']}")
    logger.info(f"Total chunks dihasilkan: {stats['total_chunks']}")
    logger.info(f"Durasi: {duration:.2f} detik")
    
    if stats['processed'] > 0:
        avg_chunks = stats['total_chunks'] / stats['processed']
        logger.info(f"Rata-rata chunks per dokumen: {avg_chunks:.2f}")
    
    logger.info("="*70)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MaxMin semantic chunking untuk dokumen teks dengan dukungan GGUF"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/cleaned',
        help='Direktori input yang berisi file .txt (default: data/cleaned)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/chunked/maxmin_semantic',
        help='Direktori output untuk hasil chunking (default: data/chunked/maxmin_semantic)'
    )
    
    # GGUF arguments (RECOMMENDED)
    parser.add_argument(
        '--gguf',
        type=str,
        default=DEFAULT_GGUF_MODEL_PATH,
        help=f'Path ke file GGUF model (default: {DEFAULT_GGUF_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--no-gguf',
        action='store_true',
        help='Gunakan SentenceTransformer bukan GGUF (tidak direkomendasikan karena butuh lebih banyak VRAM)'
    )
    
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=-1,
        help='Jumlah layer di GPU untuk GGUF (-1 = semua layer, default: -1)'
    )
    
    # SentenceTransformer arguments (fallback)
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Qwen/Qwen3-Embedding-4B',
        help='Nama model HuggingFace (hanya jika --no-gguf, default: Qwen3-Embedding-4B)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device untuk inference (default: cuda)'
    )
    
    # MaxMin parameters
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.6,
        help='Fixed threshold untuk MaxMin (default: 0.6)'
    )
    
    parser.add_argument(
        '--c',
        type=float,
        default=0.9,
        help='Parameter c untuk MaxMin (default: 0.9)'
    )
    
    parser.add_argument(
        '--init',
        type=float,
        default=1.5,
        help='Parameter init_constant untuk MaxMin (default: 1.5)'
    )
    
    # Other arguments
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Jangan sertakan metadata dalam chunks'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Proses ulang file yang sudah ada'
    )
    
    parser.add_argument(
        '--single',
        type=str,
        help='Proses satu file .txt saja (berikan path ke file)'
    )
    
    parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Mode hemat VRAM: gunakan float16 (hanya untuk SentenceTransformer)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size untuk embedding (default: 8, turunkan ke 2-4 jika OOM)'
    )
    
    args = parser.parse_args()
    
    # Determine mode: GGUF or SentenceTransformer
    use_gguf = not args.no_gguf
    
    # Jalankan chunking
    if args.single:
        # Mode single file
        if use_gguf:
            embedding_model = initialize_embedding_model_gguf(
                model_path=args.gguf,
                n_gpu_layers=args.n_gpu_layers
            )
        else:
            embedding_model = initialize_embedding_model(
                args.model, 
                args.device, 
                low_memory=args.low_memory
            )
        
        if embedding_model:
            chunks = process_single_text(
                args.single,
                args.output,
                embedding_model,
                fixed_threshold=args.threshold,
                c=args.c,
                init_constant=args.init,
                include_metadata=not args.no_metadata,
                batch_size=args.batch_size,
                use_gguf=use_gguf
            )
            exit(0 if chunks else 1)
        else:
            exit(1)
    else:
        # Mode batch
        stats = run_maxmin_chunking(
            input_dir=args.input,
            output_dir=args.output,
            model_path=args.gguf,
            use_gguf=use_gguf,
            model_name=args.model,
            device=args.device,
            fixed_threshold=args.threshold,
            c=args.c,
            init_constant=args.init,
            include_metadata=not args.no_metadata,
            skip_existing=not args.no_skip,
            low_memory=args.low_memory,
            batch_size=args.batch_size,
            n_gpu_layers=args.n_gpu_layers
        )
        
        exit(0 if stats['processed'] > 0 else 1)
