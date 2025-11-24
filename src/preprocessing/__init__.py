"""
Modul Preprocessing untuk RAG System
=====================================

Modul ini menyediakan pipeline lengkap untuk preprocessing dokumen PDF:
1. Ekstraksi teks dari PDF menggunakan PyMuPDF
2. Pembersihan teks menggunakan regex
3. Pipeline otomatis untuk batch processing

Struktur Modul:
--------------
- pdf_extractor.py: Ekstraksi teks dari file PDF
- text_cleaner.py: Pembersihan teks dengan regex
- pipeline.py: Pipeline preprocessing lengkap

Penggunaan:
----------
Sebagai script langsung:
    python -m src.preprocessing.pipeline --input data/raw --output data/cleaned_text

Atau import dalam kode Python:
    from src.preprocessing import run_preprocessing, extract_text, clean_text
    
    # Jalankan preprocessing batch
    stats = run_preprocessing(
        input_dir='data/raw',
        output_dir='data/cleaned_text'
    )
    
    # Atau proses individual
    text = extract_text('path/to/file.pdf')
    cleaned = clean_text(text)
"""

from .pdf_extractor import extract_text, extract_text_with_metadata
from .text_cleaner import clean_text, clean_text_advanced, remove_headers_footers
from .pipeline import (
    run_preprocessing,
    run_preprocessing_single,
    process_single_pdf,
    get_pdf_files
)

__version__ = '1.0.0'

__all__ = [
    # PDF Extraction
    'extract_text',
    'extract_text_with_metadata',
    
    # Text Cleaning
    'clean_text',
    'clean_text_advanced',
    'remove_headers_footers',
    
    # Pipeline
    'run_preprocessing',
    'run_preprocessing_single',
    'process_single_pdf',
    'get_pdf_files',
]
