"""
Script wrapper untuk menjalankan preprocessing pipeline.
Dapat dipanggil langsung dari root directory project.
"""

import sys
from pathlib import Path

# Tambahkan src ke Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.pipeline import run_preprocessing, run_preprocessing_single  # type: ignore

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline preprocessing untuk ekstraksi dan pembersihan teks dari PDF"
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
        default='data/cleaned',
        help='Direktori output untuk hasil teks bersih (default: data/cleaned_text)'
    )
    
    parser.add_argument(
        '--metadata', '-m',
        action='store_true',
        help='Simpan metadata PDF dalam file terpisah'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Proses ulang file yang sudah ada (default: skip file yang sudah ada)'
    )
    
    parser.add_argument(
        '--single', '-s',
        type=str,
        help='Proses satu file PDF saja (berikan path ke file)'
    )
    
    args = parser.parse_args()
    
    # Jalankan preprocessing
    if args.single:
        # Mode single file
        success = run_preprocessing_single(
            args.single,
            args.output,
            save_metadata=args.metadata
        )
        exit(0 if success else 1)
    else:
        # Mode batch (semua file di direktori)
        stats = run_preprocessing(
            input_dir=args.input,
            output_dir=args.output,
            save_metadata=args.metadata,
            skip_existing=not args.no_skip
        )
        
        # Exit code: 0 jika ada file yang berhasil, 1 jika semua gagal
        exit(0 if stats['processed'] > 0 else 1)
