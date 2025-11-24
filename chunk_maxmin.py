"""
Script wrapper untuk menjalankan MaxMin semantic chunking.
Dapat dipanggil langsung dari root directory project.
"""

import sys
from pathlib import Path

# Tambahkan src ke Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chunking.maxmin_chunker import run_maxmin_chunking, process_single_text, initialize_embedding_model  # type: ignore

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MaxMin semantic chunking untuk dokumen teks"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/cleaned_text',
        help='Direktori input yang berisi file .txt (default: data/cleaned_text)'
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
        default='cpu',
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
