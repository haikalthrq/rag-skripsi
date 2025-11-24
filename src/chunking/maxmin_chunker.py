"""
Modul MaxMin Semantic Chunking.

Modul ini memanggil library maxmin_chunker untuk melakukan semantic chunking
berdasarkan similarity threshold dinamis. Menggunakan Qwen3-Embedding model
untuk generate sentence embeddings.
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

try:
    from maxmin_chunker import process_sentences  # type: ignore[import-not-found, import-untyped]
except ImportError:
    process_sentences = None  # type: ignore[assignment]

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


def initialize_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    device: str = "cuda"
) -> Optional[Any]:
    """
    Inisialisasi model embedding Qwen3 menggunakan SentenceTransformer.
    
    Mengikuti best practices dari dokumentasi Qwen3:
    - Menggunakan SentenceTransformer API (lebih mudah dan stabil)
    - Normalisasi embeddings untuk cosine similarity
    
    Args:
        model_name (str): Nama model HuggingFace. Default: Qwen3-Embedding-8B
        device (str): Device untuk inference ('cpu' atau 'cuda')
        
    Returns:
        Optional[Any]: Model SentenceTransformer atau None jika gagal
    """
    if not _SENTENCE_TRANSFORMER_AVAILABLE:
        logger.error("Library 'sentence-transformers' tidak terinstall.")
        logger.error("Install dengan: pip install sentence-transformers>=2.7.0")
        return None
    
    try:
        logger.info(f"Inisialisasi embedding model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info("Menggunakan SentenceTransformer API (recommended by Qwen3)")
        
        # Load model dengan SentenceTransformer
        # Sesuai dokumentasi Qwen3, ini adalah cara paling simple dan reliable
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
    show_progress: bool = True
) -> Optional[np.ndarray]:
    """
    Generate embeddings untuk setiap kalimat menggunakan SentenceTransformer.
    
    Mengikuti best practices Qwen3:
    - Tidak menggunakan instruction untuk documents (hanya untuk queries)
    - Normalisasi embeddings dengan L2 norm untuk cosine similarity
    - Batch processing untuk efisiensi
    
    Args:
        sentences (List[str]): List kalimat.
        embedding_model (Any): Model SentenceTransformer.
        normalize (bool): Normalize embeddings dengan L2 norm (default: True).
        show_progress (bool): Show progress bar (default: True).
        
    Returns:
        Optional[np.ndarray]: Array embeddings dengan shape (n_sentences, embedding_dim)
                             atau None jika gagal.
    """
    try:
        logger.info(f"Generating embeddings untuk {len(sentences)} kalimat...")
        logger.info(f"  - Normalization: {normalize} (recommended: True untuk cosine similarity)")
        
        # Generate embeddings menggunakan SentenceTransformer.encode()
        # NOTE: Untuk documents, JANGAN gunakan prompt/instruction
        # Sesuai dokumentasi Qwen3: instructions hanya untuk queries
        embeddings = embedding_model.encode(
            sentences,
            normalize_embeddings=normalize,  # L2 normalization untuk cosine similarity
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=32  # Adjust sesuai VRAM
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
    Memanggil process_sentences() dari library maxmin_chunker.
    
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
    if process_sentences is None:
        logger.error("Library 'maxmin_chunker' tidak terinstall.")
        logger.error("Install dengan: pip install maxmin-chunker")
        return None
    
    try:
        logger.info("Menjalankan MaxMin chunking...")
        logger.info(f"  - fixed_threshold: {fixed_threshold}")
        logger.info(f"  - c: {c}")
        logger.info(f"  - init_constant: {init_constant}")
        
        # Panggil process_sentences dari library
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
    include_metadata: bool = True
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
        embeddings = embed_sentences(sentences, embedding_model)
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
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    device: str = "cuda",
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5,
    include_metadata: bool = True,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Menjalankan MaxMin semantic chunking untuk semua file teks di direktori input.
    
    Args:
        input_dir (str): Direktori berisi file teks input (default: data/cleaned).
        output_dir (str): Direktori output untuk hasil chunking (default: data/chunked/maxmin_semantic).
        model_name (str): Nama model embedding (default: Qwen3-Embedding).
        device (str): Device untuk inference (default: 'cpu').
        fixed_threshold (float): Fixed threshold untuk MaxMin (default: 0.6).
        c (float): Parameter c untuk MaxMin (default: 0.9).
        init_constant (float): Parameter init_constant untuk MaxMin (default: 1.5).
        include_metadata (bool): Sertakan metadata (default: True).
        skip_existing (bool): Skip file yang sudah diproses (default: True).
        
    Returns:
        Dict[str, Any]: Statistik hasil chunking.
    """
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Memulai MaxMin Semantic Chunking Pipeline")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Parameters: threshold={fixed_threshold}, c={c}, init={init_constant}")
    
    # Initialize embedding model
    logger.info("\nInisialisasi embedding model...")
    embedding_model = initialize_embedding_model(model_name, device)
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
            include_metadata=include_metadata
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
        description="MaxMin semantic chunking untuk dokumen teks"
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
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Qwen/Qwen3-Embedding-8B',
        help='Nama model embedding (default: Qwen3-Embedding-8B)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device untuk inference (default: cpu)'
    )
    
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
    
    args = parser.parse_args()
    
    # Jalankan chunking
    if args.single:
        # Mode single file
        embedding_model = initialize_embedding_model(args.model, args.device)
        if embedding_model:
            chunks = process_single_text(
                args.single,
                args.output,
                embedding_model,
                fixed_threshold=args.threshold,
                c=args.c,
                init_constant=args.init,
                include_metadata=not args.no_metadata
            )
            exit(0 if chunks else 1)
        else:
            exit(1)
    else:
        # Mode batch
        stats = run_maxmin_chunking(
            input_dir=args.input,
            output_dir=args.output,
            model_name=args.model,
            device=args.device,
            fixed_threshold=args.threshold,
            c=args.c,
            init_constant=args.init,
            include_metadata=not args.no_metadata,
            skip_existing=not args.no_skip
        )
        
        exit(0 if stats['processed'] > 0 else 1)
