"""
Pipeline preprocessing lengkap untuk ekstraksi dan pembersihan teks dari PDF.
Modul ini dapat dijalankan sebagai script atau diimport oleh modul lain.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from .pdf_extractor import extract_text, extract_text_with_metadata
from .text_cleaner import clean_text, remove_headers_footers


# Setup logging dengan file handler
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging dengan file dan console handler."""
    # Buat direktori logs jika belum ada
    Path(log_dir).mkdir(exist_ok=True)
    
    # Format timestamp untuk nama file log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"preprocessing_{timestamp}.log"
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Hapus handler yang sudah ada (jika dipanggil ulang)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Tambahkan handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# Setup logger global untuk modul ini
logger = setup_logging()


def get_pdf_files(input_dir: str) -> List[Path]:
    """
    Mendapatkan daftar semua file PDF dalam direktori.
    
    Args:
        input_dir (str): Path ke direktori input.
        
    Returns:
        List[Path]: List path file PDF.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Direktori input tidak ditemukan: {input_dir}")
        return []
    
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Ditemukan {len(pdf_files)} file PDF di {input_dir}")
    
    return pdf_files


def process_single_pdf(
    pdf_path: Path,
    output_dir: str,
    save_metadata: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Memproses satu file PDF: ekstraksi dan pembersihan teks.
    
    Args:
        pdf_path (Path): Path ke file PDF.
        output_dir (str): Direktori output untuk menyimpan hasil.
        save_metadata (bool): Jika True, simpan metadata dalam file terpisah.
        
    Returns:
        Tuple[bool, Optional[str]]: (success status, output file path)
    """
    try:
        logger.info(f"Memproses: {pdf_path.name}")
        
        # 1. Ekstrak teks dari PDF
        if save_metadata:
            result = extract_text_with_metadata(str(pdf_path))
            if not result:
                return False, None
            raw_text = result['text']
            metadata = result['metadata']
        else:
            raw_text = extract_text(str(pdf_path))
            if not raw_text:
                return False, None
            metadata = None
        
        # 2. Bersihkan teks
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            logger.warning(f"Teks hasil pembersihan kosong untuk {pdf_path.name}")
            return False, None
        
        # 3. Siapkan output path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Nama file output (ganti ekstensi .pdf dengan .txt)
        output_filename = pdf_path.stem + ".txt"
        output_file = output_path / output_filename
        
        # 4. Simpan teks yang sudah dibersihkan
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"✓ Berhasil disimpan ke: {output_file}")
        logger.info(f"  - Karakter akhir: {len(cleaned_text)}")
        
        # 5. Simpan metadata jika diminta
        if save_metadata and metadata:
            metadata_file = output_path / (pdf_path.stem + "_metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            logger.info(f"  - Metadata disimpan ke: {metadata_file}")
        
        return True, str(output_file)
        
    except Exception as e:
        logger.error(f"✗ Error memproses {pdf_path.name}: {str(e)}")
        return False, None


def run_preprocessing(
    input_dir: str = "data/raw",
    output_dir: str = "data/cleaned_text",
    save_metadata: bool = False,
    skip_existing: bool = True
) -> dict:
    """
    Menjalankan pipeline preprocessing lengkap untuk semua PDF di input_dir.
    
    Args:
        input_dir (str): Direktori yang berisi file PDF input.
        output_dir (str): Direktori untuk menyimpan hasil teks yang dibersihkan.
        save_metadata (bool): Jika True, simpan metadata PDF dalam file terpisah.
        skip_existing (bool): Jika True, skip file yang sudah diproses.
        
    Returns:
        dict: Statistik hasil preprocessing.
    """
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Memulai Pipeline Preprocessing")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Save metadata: {save_metadata}")
    logger.info(f"Skip existing: {skip_existing}")
    
    # Buat direktori output jika belum ada
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dapatkan daftar PDF
    pdf_files = get_pdf_files(input_dir)
    
    if not pdf_files:
        logger.warning("Tidak ada file PDF yang ditemukan untuk diproses")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'duration': 0
        }
    
    # Proses setiap PDF
    stats: dict = {
        'total_files': len(pdf_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'output_files': []
    }
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        # Cek apakah file output sudah ada
        output_filename = pdf_path.stem + ".txt"
        output_file = Path(output_dir) / output_filename
        
        if skip_existing and output_file.exists():
            logger.info(f"⊙ File output sudah ada, skip: {output_filename}")
            stats['skipped'] += 1
            continue
        
        # Proses PDF
        success, output_path = process_single_pdf(
            pdf_path,
            output_dir,
            save_metadata=save_metadata
        )
        
        if success:
            stats['processed'] += 1
            stats['output_files'].append(output_path)
        else:
            stats['failed'] += 1
    
    # Hitung durasi
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats['duration'] = duration
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info("Preprocessing Selesai")
    logger.info("="*70)
    logger.info(f"Total file PDF: {stats['total_files']}")
    logger.info(f"Berhasil diproses: {stats['processed']}")
    logger.info(f"Di-skip (sudah ada): {stats['skipped']}")
    logger.info(f"Gagal: {stats['failed']}")
    logger.info(f"Durasi: {duration:.2f} detik")
    logger.info("="*70)
    
    return stats


def run_preprocessing_single(
    pdf_path: str,
    output_dir: str = "data/cleaned_text",
    save_metadata: bool = False
) -> bool:
    """
    Menjalankan preprocessing untuk satu file PDF.
    
    Args:
        pdf_path (str): Path ke file PDF.
        output_dir (str): Direktori output.
        save_metadata (bool): Simpan metadata atau tidak.
        
    Returns:
        bool: True jika berhasil, False jika gagal.
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        logger.error(f"File tidak ditemukan: {pdf_path}")
        return False
    
    if not pdf_file.suffix.lower() == '.pdf':
        logger.error(f"File bukan PDF: {pdf_path}")
        return False
    
    logger.info(f"Memproses file tunggal: {pdf_file.name}")
    success, _ = process_single_pdf(pdf_file, output_dir, save_metadata)
    
    return success


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
        default='data/cleaned_text',
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
