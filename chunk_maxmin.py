"""
Script wrapper untuk menjalankan MaxMin semantic chunking.
Dapat dipanggil langsung dari root directory project.

Mendukung 2 mode embedding:
1. GGUF (default, recommended): Menggunakan llama-cpp-python dengan Qwen3-Embedding-4B-GGUF
2. SentenceTransformer (fallback): Menggunakan sentence-transformers (lebih boros VRAM)
"""

import sys
from pathlib import Path

# Tambahkan src ke Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chunking.maxmin_chunker import (  # type: ignore
    run_maxmin_chunking, 
    process_single_text, 
    initialize_embedding_model,
    initialize_embedding_model_gguf,
    DEFAULT_GGUF_MODEL_PATH
)

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
