"""
Script wrapper untuk menjalankan recursive chunking menggunakan RecursiveCharacterTextSplitter.

Script ini memanggil modul recursive_split.py dari src/chunking/.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))  # type: ignore[arg-type]

from chunking.recursive_split import run_recursive_chunking  # type: ignore[import-not-found]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recursive chunking untuk dokumen teks menggunakan LangChain RecursiveCharacterTextSplitter"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/cleaned_text',
        help='Direktori input berisi file .txt (default: data/cleaned_text)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/chunked/recursive',
        help='Direktori output hasil chunking (default: data/chunked/recursive)'
    )
    
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=1000,
        help='Maximum size per chunk dalam characters (default: 1000)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Overlap antara chunks dalam characters (default: 200)'
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
    
    args = parser.parse_args()
    
    # Jalankan recursive chunking
    stats = run_recursive_chunking(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        include_metadata=not args.no_metadata,
        skip_existing=not args.no_skip
    )
    
    # Exit dengan kode sesuai hasil
    exit(0 if stats['processed'] > 0 else 1)
