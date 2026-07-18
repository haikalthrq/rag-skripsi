"""
Entrypoint untuk embedding chunks dari hasil chunking.

Pipeline:
1. Load chunks dari JSON file
2. Clean whitespace dan filter empty chunks
3. Generate embeddings menggunakan Qwen3 model
4. Save embeddings ke output directory

Supports 3 chunking methods:
- element_based
- maxmin_semantic
- recursive
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

import torch

from .embedder import QwenEmbedder, initialize_gguf_embedder, initialize_hf_embedder
from .io import (
    load_chunks_from_json,
    clean_and_filter_chunks,
    enrich_table_chunk_texts,
    save_embeddings,
    save_embeddings_numpy
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metode yang mendapat context prefix injection (bukan element_based yang sudah punya enrich)
_METHODS_WITH_CONTEXT_PREFIX = {"maxmin_semantic", "recursive"}

# Jumlah karakter dari chunk sebelumnya yang di-prepend sebagai prefix konteks
CONTEXT_PREFIX_CHARS = 200


def inject_context_prefix(chunks: List[Dict[str, Any]], context_chars: int = CONTEXT_PREFIX_CHARS) -> List[str]:
    """Buat daftar teks untuk embedding dengan context prefix dari chunk sebelumnya.

    Untuk setiap chunk ke-i (i > 0), prepend ``context_chars`` karakter terakhir
    dari teks chunk ke-(i-1) sebagai prefix. Ini membantu chunk tabel/pendek
    mendapatkan konteks semantik dari konten sebelumnya.

    Fungsi ini TIDAK memodifikasi chunks in-place — hanya mengembalikan daftar
    teks yang dipakai untuk embedding. File chunks JSON tidak berubah, sehingga
    chunk IDs di ChromaDB dan ground truth tetap valid.

    Args:
        chunks: List of chunk dicts dengan field ``text``.
        context_chars: Jumlah chars dari chunk sebelumnya yang di-prepend (default: 200).

    Returns:
        List of strings siap di-embed, satu per chunk.
    """
    texts = []
    for i, chunk in enumerate(chunks):
        text = (chunk.get("text") or "").strip()
        if i > 0 and context_chars > 0:
            prev_text = (chunks[i - 1].get("text") or "").strip()
            suffix = prev_text[-context_chars:].strip()
            if suffix:
                text = suffix + "\n\n" + text
        texts.append(text)
    injected = sum(1 for i in range(1, len(chunks)) if (chunks[i - 1].get("text") or "").strip())
    if injected:
        logger.info(f"inject_context_prefix: {injected} chunks mendapat context prefix ({context_chars} chars)")
    return texts


def embed_single_file(
    json_path: str,
    output_dir: str,
    embedder: QwenEmbedder,
    chunking_method: str,
    save_numpy: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Embed chunks dari single JSON file.
    
    Args:
        json_path: Path ke JSON file chunks
        output_dir: Directory untuk output embeddings
        embedder: QwenEmbedder instance
        chunking_method: Nama metode chunking ('element_based', 'maxmin_semantic', 'recursive')
        save_numpy: Juga save dalam format numpy (.npy)
        
    Returns:
        Dictionary dengan stats atau None jika error
    """
    try:
        json_file = Path(json_path)
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {json_file.name}")
        logger.info(f"{'='*70}")
        
        # 1. Load chunks
        chunks = load_chunks_from_json(json_path)
        if chunks is None:
            logger.error("Failed to load chunks")
            return None
        
        # Catatan provenance: teks tabel dapat dimutasi in-place sebelum
        # disimpan, sedangkan context prefix hanya memengaruhi teks embedding
        # dan tidak tersimpan pada chunk. JSON chunk saja tidak selalu cukup
        # untuk merekonstruksi teks persis yang masuk ke model embedding.
        # 2a. Enrich table chunks: ganti OCR text dengan HTML-parsed text
        #     (hanya berlaku untuk element_based yang punya text_as_html)
        n_enriched = enrich_table_chunk_texts(chunks)
        if n_enriched:
            logger.info(f"  {n_enriched} table chunks enriched from HTML before embedding")

        # 2b. Clean and filter
        logger.info("Cleaning and filtering chunks...")
        cleaned_texts, valid_indices = clean_and_filter_chunks(chunks)
        
        if len(cleaned_texts) == 0:
            logger.error("No valid chunks after filtering")
            return None
        
        logger.info(f"Valid chunks: {len(cleaned_texts)}/{len(chunks)}")

        # 2c. Context prefix injection untuk maxmin_semantic dan recursive.
        #     Prepend N chars dari chunk sebelumnya agar chunk tabel/pendek
        #     mendapat konteks semantik dari konten sekitarnya.
        #     TIDAK mengubah chunks JSON — hanya teks yang di-embed yang berbeda.
        valid_chunks = [chunks[i] for i in valid_indices]
        if chunking_method in _METHODS_WITH_CONTEXT_PREFIX:
            embed_texts = inject_context_prefix(valid_chunks, CONTEXT_PREFIX_CHARS)
        else:
            embed_texts = [c.get("text", "") for c in valid_chunks]

        # Gunakan embed_texts (bukan cleaned_texts) agar prefix sudah tercakup
        # namun tetap lakukan strip/whitespace-norm yang sama
        embed_texts = [' '.join(t.split()) for t in embed_texts]

        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(embed_texts)} chunks...")
        embeddings = embedder.embed(embed_texts)
        
        # Bersihkan cache GPU agar memori tidak terfragmentasi antar-file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings (dim: {embeddings.shape[1]})")
        
        # 4. Save embeddings
        output_path = Path(output_dir) / chunking_method / f"{json_file.stem}_embeddings.json"
        
        metadata = {
            "source_file": json_file.name,
            "source_path": str(json_file),
            "chunking_method": chunking_method,
            "embedding_model": embedder.mode,
            "normalized": embedder.normalize
        }
        
        logger.info(f"Saving embeddings to: {output_path}")
        success = save_embeddings(
            embeddings=embeddings,
            chunks=chunks,
            valid_indices=valid_indices,
            output_path=str(output_path),
            metadata=metadata
        )
        
        if not success:
            logger.error("Failed to save embeddings")
            return None
        
        # Optional: Save numpy format
        if save_numpy:
            numpy_path = output_path.with_suffix('.npy')
            logger.info(f"Saving numpy embeddings to: {numpy_path}")
            save_embeddings_numpy(embeddings, str(numpy_path))
        
        stats = {
            "file": json_file.name,
            "total_chunks": len(chunks),
            "valid_chunks": len(cleaned_texts),
            "embedding_dim": embeddings.shape[1],
            "output_path": str(output_path)
        }
        
        logger.info("✓ Processing completed successfully")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {str(e)}")
        return None


def embed_all_chunks(
    chunked_dir: str = "data/chunked",
    output_dir: str = "data/embeddings",
    mode: str = "gguf",
    gguf_model_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    device: str = "cuda",
    normalize: bool = True,
    save_numpy: bool = False,
    skip_existing: bool = True,
    methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Embed semua chunks dari 3 metode chunking.
    
    Args:
        chunked_dir: Directory berisi hasil chunking (dengan subdirs: element_based, maxmin_semantic, recursive)
        output_dir: Directory untuk output embeddings
        mode: 'gguf' atau 'huggingface'
        gguf_model_path: Path ke GGUF model (jika mode='gguf')
        hf_model_name: HuggingFace model name (jika mode='huggingface')
        device: Device untuk inference
        normalize: Normalize embeddings
        save_numpy: Juga save dalam format numpy
        skip_existing: Skip file yang sudah di-embed
        
    Returns:
        Dictionary dengan statistics
    """
    logger.info("="*70)
    logger.info("EMBEDDING CHUNKS PIPELINE")
    logger.info("="*70)
    logger.info(f"Input directory: {chunked_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Normalize: {normalize}")
    logger.info("")
    
    # Initialize embedder
    logger.info("Initializing embedding model...")
    
    if mode == 'gguf':
        if gguf_model_path is None:
            from .embedder import DEFAULT_GGUF_MODEL_PATH
            gguf_model_path = DEFAULT_GGUF_MODEL_PATH
        
        embedder = initialize_gguf_embedder(
            model_path=gguf_model_path,
            normalize=normalize
        )
    elif mode == 'huggingface':
        if hf_model_name is None:
            from .embedder import DEFAULT_HF_MODEL_NAME
            hf_model_name = DEFAULT_HF_MODEL_NAME
        
        embedder = initialize_hf_embedder(
            model_name=hf_model_name,
            device=device,
            normalize=normalize
        )
    else:
        logger.error(f"Unknown mode: {mode}")
        return {"error": "Invalid mode"}
    
    if embedder is None:
        logger.error("Failed to initialize embedder")
        return {"error": "Failed to initialize embedder"}
    
    logger.info("")
    
    # Process each chunking method
    all_methods = ['element_based', 'maxmin_semantic', 'recursive']
    chunking_methods = methods if methods is not None else all_methods
    
    stats = {
        'total_files': 0,
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'by_method': {}
    }
    
    for method in chunking_methods:
        method_dir = Path(chunked_dir) / method
        
        if not method_dir.exists():
            logger.warning(f"Directory not found: {method_dir}")
            continue
        
        # Find all JSON files
        json_files = list(method_dir.glob("*_chunks.json"))
        
        if len(json_files) == 0:
            logger.warning(f"No JSON files found in {method_dir}")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing method: {method.upper()}")
        logger.info(f"Found {len(json_files)} files")
        logger.info(f"{'='*70}")
        
        method_stats = {
            'total': len(json_files),
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"\n[{i}/{len(json_files)}] {json_file.name}")
            
            # Check if already processed
            output_path = Path(output_dir) / method / f"{json_file.stem}_embeddings.json"
            
            if skip_existing and output_path.exists():
                logger.info(f"⊘ Skipping (already exists): {output_path.name}")
                method_stats['skipped'] += 1
                stats['skipped'] += 1
                continue
            
            # Process file
            result = embed_single_file(
                json_path=str(json_file),
                output_dir=output_dir,
                embedder=embedder,
                chunking_method=method,
                save_numpy=save_numpy
            )
            
            if result is not None:
                method_stats['processed'] += 1
                stats['processed'] += 1
            else:
                method_stats['failed'] += 1
                stats['failed'] += 1
        
        stats['by_method'][method] = method_stats
        stats['total_files'] += method_stats['total']
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("EMBEDDING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    
    for method, method_stats in stats['by_method'].items():
        logger.info(f"{method}:")
        logger.info(f"  Processed: {method_stats['processed']}/{method_stats['total']}")
        if method_stats['skipped'] > 0:
            logger.info(f"  Skipped: {method_stats['skipped']}")
        if method_stats['failed'] > 0:
            logger.info(f"  Failed: {method_stats['failed']}")
    
    logger.info(f"{'='*70}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings untuk chunks dari hasil chunking"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/chunked',
        help='Directory berisi hasil chunking (default: data/chunked)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/embeddings',
        help='Directory untuk output embeddings (default: data/embeddings)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='gguf',
        choices=['gguf', 'huggingface'],
        help='Mode embedding: gguf (recommended) atau huggingface (default: gguf)'
    )
    
    parser.add_argument(
        '--gguf-model',
        type=str,
        help='Path ke GGUF model (default: models/Qwen3-Embedding-4B-Q8_0.gguf)'
    )
    
    parser.add_argument(
        '--hf-model',
        type=str,
        help='HuggingFace model name (default: Qwen/Qwen3-Embedding-4B)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device untuk inference (default: cuda)'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Jangan normalize embeddings'
    )
    
    parser.add_argument(
        '--save-numpy',
        action='store_true',
        help='Juga save dalam format numpy (.npy)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Proses ulang file yang sudah ada'
    )
    
    args = parser.parse_args()
    
    # Run embedding pipeline
    stats = embed_all_chunks(
        chunked_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        gguf_model_path=args.gguf_model,
        hf_model_name=args.hf_model,
        device=args.device,
        normalize=not args.no_normalize,
        save_numpy=args.save_numpy,
        skip_existing=not args.no_skip
    )
    
    # Exit code based on results
    if stats.get('error'):
        sys.exit(1)
    elif stats['processed'] == 0 and stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
