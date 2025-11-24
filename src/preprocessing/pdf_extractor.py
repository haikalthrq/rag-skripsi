"""
Modul untuk ekstraksi teks dari file PDF menggunakan PyMuPDF (fitz).
"""

import fitz  # type: ignore[import-untyped]  # PyMuPDF
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> Optional[str]:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF.
    
    Args:
        pdf_path (str): Path ke file PDF yang akan diekstrak.
        
    Returns:
        Optional[str]: Teks yang berhasil diekstrak dari PDF, atau None jika gagal.
    """
    try:
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            logger.error(f"File tidak ditemukan: {pdf_path}")
            return None
            
        if not pdf_path_obj.suffix.lower() == '.pdf':
            logger.error(f"File bukan PDF: {pdf_path}")
            return None
        
        logger.info(f"Memulai ekstraksi teks dari: {pdf_path_obj.name}")
        
        # Buka dokumen PDF
        doc = fitz.open(str(pdf_path_obj))
        
        # Ekstrak teks dari setiap halaman
        extracted_text = []
        page_count = len(doc)
        
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():  # Hanya tambahkan jika ada teks
                extracted_text.append(text)
        
        # Gabungkan semua teks dengan newline
        full_text = "\n".join(extracted_text)
        
        # Tutup dokumen setelah semua selesai
        doc.close()
        
        logger.info(f"Berhasil mengekstrak {page_count} halaman dari {pdf_path_obj.name}")
        logger.info(f"Total karakter yang diekstrak: {len(full_text)}")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error saat mengekstrak PDF {pdf_path}: {str(e)}")
        return None


def extract_text_with_metadata(pdf_path: str) -> Optional[dict]:
    """
    Mengekstrak teks dari PDF beserta metadata dokumen.
    
    Args:
        pdf_path (str): Path ke file PDF yang akan diekstrak.
        
    Returns:
        Optional[dict]: Dictionary berisi teks dan metadata, atau None jika gagal.
    """
    try:
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            logger.error(f"File tidak ditemukan: {pdf_path}")
            return None
        
        # Buka dokumen PDF
        doc = fitz.open(str(pdf_path_obj))
        
        # Ekstrak metadata
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc)
        }
        
        # Ekstrak teks dari setiap halaman
        extracted_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                extracted_text.append(text)
        
        full_text = "\n".join(extracted_text)
        
        # Tutup dokumen
        doc.close()
        
        return {
            'text': full_text,
            'metadata': metadata,
            'filename': pdf_path_obj.name
        }
        
    except Exception as e:
        logger.error(f"Error saat mengekstrak PDF dengan metadata {pdf_path}: {str(e)}")
        return None


if __name__ == "__main__":
    # Testing
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        text = extract_text(pdf_file)
        if text:
            print(f"Berhasil mengekstrak {len(text)} karakter")
            print("\n--- Sample (200 karakter pertama) ---")
            print(text[:200])
    else:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
