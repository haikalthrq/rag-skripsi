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


def _extract_page_hybrid(page) -> str:
    """
    Ekstrak satu halaman PDF dengan pendekatan hybrid:
    - Tabel  : find_tables() → format baris|kolom agar struktur 2D terjaga
    - Narasi : get_text("blocks") dengan area tabel di-exclude → teks paragraf normal

    Kedua komponen digabungkan dalam urutan reading order (top-to-bottom berdasarkan y0).

    Returns:
        str: Teks gabungan narasi + tabel dalam reading order.
    """
    segments = []  # list of (y0, text_string)

    # ── 1. Ekstrak tabel ─────────────────────────────────────────────────────
    table_rects = []
    try:
        tabs = page.find_tables()
        for tab in tabs.tables:
            bbox = tab.bbox  # (x0, y0, x1, y1)
            table_rects.append(fitz.Rect(bbox))

            # Format tabel sebagai Markdown — LLM memahami format ini lebih baik.
            # fill_empty=True: isi sel kosong dari sel di atas/kiri (handle merged cells BPS).
            try:
                table_text = tab.to_markdown(fill_empty=True)
            except Exception:
                rows = []
                for row in tab.extract():
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    rows.append(" | ".join(cells))
                table_text = "\n".join(rows)
            if table_text.strip():
                segments.append((bbox[1], table_text))
    except Exception as e:
        logger.debug(f"find_tables() tidak menemukan tabel atau error: {e}")

    # ── 2. Ekstrak narasi di luar region tabel ───────────────────────────────
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    narrative_blocks = []  # (x0, x1, y0, text)
    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        if not text.strip():
            continue
        block_rect = fitz.Rect(x0, y0, x1, y1)
        # Skip blok yang overlap signifikan dengan region tabel
        is_inside_table = any(
            block_rect.intersects(tr) and (block_rect & tr).get_area() > 0.5 * block_rect.get_area()
            for tr in table_rects
        )
        if not is_inside_table:
            narrative_blocks.append((x0, x1, y0, text.strip()))

    # ── 2b. Deteksi layout dua kolom & filter kolom kanan (bilingual) ────────
    # Gunakan center blok (x0+x1)/2, bukan x0, agar kolom kanan yang dimulai
    # sebelum mid_x (misal x0≈299 < mid_x=327) tetap terdeteksi sebagai kanan.
    page_width = page.rect.width
    if page_width >= 400 and narrative_blocks:
        mid_x = page_width * 0.55
        left_col  = [b for b in narrative_blocks if (b[0] + b[1]) / 2 < mid_x]
        right_col = [b for b in narrative_blocks if (b[0] + b[1]) / 2 >= mid_x]
        if left_col and right_col:  # dua kolom terdeteksi → buang kolom kanan
            narrative_blocks = left_col

    for x0, x1, y0, text in narrative_blocks:
        segments.append((y0, text))

    # ── 3. Gabungkan dalam reading order (urut y0) ───────────────────────────
    segments.sort(key=lambda s: s[0])
    return "\n\n".join(text for _, text in segments)


def extract_text(pdf_path: str) -> Optional[str]:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF dengan pendekatan hybrid:
    - Tabel diekstrak via find_tables() agar struktur baris-kolom terjaga
    - Narasi diekstrak via get_text("blocks") di area non-tabel
    - Setiap halaman diberi penanda <<<PAGE_N>>> untuk metadata page_numbers di chunker

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

        logger.info(f"Memulai ekstraksi teks (hybrid) dari: {pdf_path_obj.name}")

        doc = fitz.open(str(pdf_path_obj))
        extracted_text = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            page_text = _extract_page_hybrid(page)

            if page_text.strip():
                extracted_text.append(f"<<<PAGE_{page_num + 1}>>>\n{page_text}")

        doc.close()

        full_text = "\n".join(extracted_text)
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
