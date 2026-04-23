"""
Script wrapper untuk generate embeddings dari chunks.

Usage:
    # Default (GGUF mode, recommended)
    python embed_chunks.py

    # HuggingFace mode
    python embed_chunks.py --mode huggingface

    # Custom paths
    python embed_chunks.py --input data/chunked --output data/embeddings

    # Process ulang file existing
    python embed_chunks.py --no-skip
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from embedding.embed_chunks import embed_all_chunks  # type: ignore[import-not-found]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings untuk chunks dari hasil chunking (3 metode)"
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
