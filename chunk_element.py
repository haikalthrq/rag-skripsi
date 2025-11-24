"""
Script wrapper untuk menjalankan element-based chunking.
Dapat dipanggil langsung dari root directory project.
"""

import sys
from pathlib import Path

# Tambahkan src ke Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chunking.element_based import run_element_based_chunking, process_single_pdf  # type: ignore

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Element-based chunking untuk dokumen PDF menggunakan Unstructured"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw',
        help='Direktori input yang berisi file PDF (default: data/raw)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/chunked/element_based',
        help='Direktori output untuk hasil chunking (default: data/chunked/element_based)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='hi_res',
        choices=['auto', 'hi_res', 'fast', 'ocr_only'],
        help='Strategi partitioning (default: hi_res)'
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
        help='Proses satu file PDF saja (berikan path ke file)'
    )
    
    args = parser.parse_args()
    
    # Jalankan chunking
    if args.single:
        # Mode single file
        chunks = process_single_pdf(
            args.single,
            args.output,
            strategy=args.strategy,
            include_metadata=not args.no_metadata
        )
        exit(0 if chunks else 1)
    else:
        # Mode batch
        stats = run_element_based_chunking(
            input_dir=args.input,
            output_dir=args.output,
            strategy=args.strategy,
            include_metadata=not args.no_metadata,
            skip_existing=not args.no_skip
        )
        
        exit(0 if stats['processed'] > 0 else 1)
