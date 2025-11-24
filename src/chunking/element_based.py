"""
Modul Element-Based Chunking menggunakan library Unstructured.

Modul ini melakukan ekstraksi dan chunking dokumen PDF berdasarkan struktur elemen
seperti Title, Paragraph, ListItem, Table, dan lainnya menggunakan partition_pdf.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from unstructured.partition.pdf import partition_pdf  # type: ignore[import-not-found, import-untyped]
except ImportError as e:
    partition_pdf = None  # type: ignore[assignment]
    _import_error = str(e)
except Exception as e:
    # Tangkap error lain seperti ModuleNotFoundError untuk dependency
    partition_pdf = None  # type: ignore[assignment]
    _import_error = str(e)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> Optional[str]:
    """
    Memuat file PDF dan memverifikasi keberadaannya.
    
    Args:
        pdf_path (str): Path ke file PDF yang akan dimuat.
        
    Returns:
        Optional[str]: Path absolute ke PDF jika valid, None jika tidak valid.
    """
    try:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            logger.error(f"File tidak ditemukan: {pdf_path}")
            return None
        
        if not pdf_file.suffix.lower() == '.pdf':
            logger.error(f"File bukan PDF: {pdf_path}")
            return None
        
        logger.info(f"Memuat PDF: {pdf_file.name}")
        return str(pdf_file.resolve())
        
    except Exception as e:
        logger.error(f"Error saat memuat PDF {pdf_path}: {str(e)}")
        return None


def partition_document(pdf_path: str, strategy: str = "hi_res", languages: Optional[List[str]] = None) -> Optional[List[Any]]:
    """
    Melakukan partitioning dokumen PDF menggunakan unstructured.partition_pdf.
    
    Args:
        pdf_path (str): Path ke file PDF.
        strategy (str): Strategi partitioning ('auto', 'hi_res', 'fast', 'ocr_only').
                       Default: 'hi_res' untuk akurasi maksimal
        languages (Optional[List[str]]): List kode bahasa untuk OCR (contoh: ['ind'] untuk Indonesia).
                                        Default: None (akan diset ke ['ind'] jika tidak ditentukan)
        
    Returns:
        Optional[List[Any]]: List elemen dokumen (Title, Paragraph, Table, dll),
                            atau None jika gagal.
    """
    if partition_pdf is None:
        logger.error("Library 'unstructured' atau dependencies-nya tidak tersedia.")
        logger.error(f"Error: {_import_error}")
        logger.error("Install dengan: pip install unstructured[pdf]")
        logger.error("Atau lengkap: pip install unstructured[all-docs]")
        return None
    
    try:
        # Set default language ke Indonesia jika tidak ditentukan
        if languages is None:
            languages = ['ind']  # Kode bahasa Indonesia untuk Tesseract
        
        logger.info(f"Memulai partitioning dokumen: {Path(pdf_path).name}")
        logger.info(f"Strategi: {strategy}")
        logger.info(f"Bahasa: {languages}")
        
        # Partition PDF menggunakan unstructured dengan hi_res strategy
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True,        # Ekstrak struktur tabel
            extract_image_block_types=["table"],  # Ekstrak tabel dari gambar
            extract_images_in_pdf=False,       # Skip ekstraksi gambar untuk performa
            include_page_breaks=True,          # Sertakan informasi page breaks
            languages=languages,               # Bahasa untuk OCR (Indonesia: 'ind')
        )
        
        logger.info(f"Berhasil mempartisi dokumen: {len(elements)} elemen ditemukan")
        
        # Log distribusi tipe elemen
        element_types: Dict[str, int] = {}
        for elem in elements:
            elem_type = type(elem).__name__
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        logger.info("Distribusi tipe elemen:")
        for elem_type, count in sorted(element_types.items()):
            logger.info(f"  - {elem_type}: {count}")
        
        return elements
        
    except Exception as e:
        logger.error(f"Error saat partitioning dokumen {pdf_path}: {str(e)}")
        return None


def convert_elements_to_chunks(
    elements: List[Any],
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Konversi elemen dokumen menjadi list chunks dengan metadata.
    
    Args:
        elements (List[Any]): List elemen dari partition_pdf.
        include_metadata (bool): Jika True, sertakan metadata dalam chunk.
        
    Returns:
        List[Dict[str, Any]]: List chunks dengan text dan metadata.
    """
    chunks = []
    
    try:
        for idx, element in enumerate(elements):
            # Ekstrak text dari elemen
            text = str(element.text) if hasattr(element, 'text') else str(element)
            
            # Skip elemen kosong
            if not text or not text.strip():
                continue
            
            # Buat chunk dictionary
            chunk: Dict[str, Any] = {
                'chunk_id': idx,
                'text': text.strip()
            }
            
            # Tambahkan metadata jika diminta
            if include_metadata:
                metadata: Dict[str, Any] = {
                    'element_type': type(element).__name__,
                    'element_id': getattr(element, 'id', None),
                }
                
                # Tambahkan metadata tambahan jika ada
                if hasattr(element, 'metadata'):
                    elem_metadata = element.metadata
                    
                    # Page number
                    if hasattr(elem_metadata, 'page_number'):
                        metadata['page_number'] = elem_metadata.page_number
                    
                    # Coordinates
                    if hasattr(elem_metadata, 'coordinates'):
                        coords = elem_metadata.coordinates
                        if coords:
                            metadata['coordinates'] = str(coords)
                    
                    # Filename
                    if hasattr(elem_metadata, 'filename'):
                        metadata['filename'] = elem_metadata.filename
                    
                    # Text as HTML
                    if hasattr(elem_metadata, 'text_as_html'):
                        metadata['text_as_html'] = elem_metadata.text_as_html
                
                chunk['metadata'] = metadata
            
            chunks.append(chunk)
        
        logger.info(f"Berhasil konversi {len(chunks)} chunks dari {len(elements)} elemen")
        
        # Log statistik
        if include_metadata and chunks:
            total_chars = sum(len(c['text']) for c in chunks)
            avg_chars = total_chars / len(chunks)
            logger.info(f"Total karakter: {total_chars}")
            logger.info(f"Rata-rata karakter per chunk: {avg_chars:.2f}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error saat konversi elemen ke chunks: {str(e)}")
        return []


def convert_elements_to_text_list(elements: List[Any]) -> List[str]:
    """
    Konversi elemen dokumen menjadi list string sederhana (hanya text).
    
    Args:
        elements (List[Any]): List elemen dari partition_pdf.
        
    Returns:
        List[str]: List string dari setiap elemen.
    """
    text_chunks = []
    
    try:
        for element in elements:
            # Ekstrak text dari elemen
            text = str(element.text) if hasattr(element, 'text') else str(element)
            
            # Skip elemen kosong
            if text and text.strip():
                text_chunks.append(text.strip())
        
        logger.info(f"Berhasil ekstrak {len(text_chunks)} text chunks")
        return text_chunks
        
    except Exception as e:
        logger.error(f"Error saat konversi elemen ke text list: {str(e)}")
        return []


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_path: str,
    pretty_print: bool = True
) -> bool:
    """
    Menyimpan chunks dalam format JSON.
    
    Args:
        chunks (List[Dict[str, Any]]): List chunks untuk disimpan.
        output_path (str): Path file output JSON.
        pretty_print (bool): Jika True, format JSON dengan indentasi.
        
    Returns:
        bool: True jika berhasil, False jika gagal.
    """
    try:
        output_file = Path(output_path)
        
        # Buat direktori jika belum ada
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan ke JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            else:
                json.dump(chunks, f, ensure_ascii=False)
        
        logger.info(f"✓ Berhasil menyimpan {len(chunks)} chunks ke: {output_file}")
        logger.info(f"  - Ukuran file: {output_file.stat().st_size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saat menyimpan chunks ke {output_path}: {str(e)}")
        return False


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    strategy: str = "hi_res",
    include_metadata: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Memproses satu file PDF: partition, convert, dan save.
    
    Args:
        pdf_path (str): Path ke file PDF.
        output_dir (str): Direktori output untuk hasil chunking.
        strategy (str): Strategi partitioning (default: 'hi_res').
        include_metadata (bool): Sertakan metadata dalam chunks.
        
    Returns:
        Optional[List[Dict[str, Any]]]: List chunks jika berhasil, None jika gagal.
    """
    try:
        logger.info(f"Memproses PDF: {Path(pdf_path).name}")
        
        # 1. Load PDF
        valid_path = load_pdf(pdf_path)
        if not valid_path:
            return None
        
        # 2. Partition document
        elements = partition_document(valid_path, strategy=strategy)
        if not elements:
            logger.warning(f"Tidak ada elemen yang diekstrak dari {Path(pdf_path).name}")
            return None
        
        # 3. Convert elements to chunks
        chunks = convert_elements_to_chunks(elements, include_metadata=include_metadata)
        if not chunks:
            logger.warning(f"Tidak ada chunks yang dihasilkan dari {Path(pdf_path).name}")
            return None
        
        # 4. Save chunks
        output_filename = Path(pdf_path).stem + "_chunks.json"
        output_path = Path(output_dir) / output_filename
        
        success = save_chunks(chunks, str(output_path), pretty_print=True)
        
        if success:
            return chunks
        else:
            return None
        
    except Exception as e:
        logger.error(f"Error saat memproses PDF {pdf_path}: {str(e)}")
        return None


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


def run_element_based_chunking(
    input_dir: str = "data/raw",
    output_dir: str = "data/chunked/element_based",
    strategy: str = "hi_res",
    include_metadata: bool = True,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Menjalankan element-based chunking untuk semua PDF di direktori input.
    
    Args:
        input_dir (str): Direktori berisi file PDF input.
        output_dir (str): Direktori output untuk hasil chunking.
        strategy (str): Strategi partitioning (default: 'hi_res' untuk akurasi maksimal).
        include_metadata (bool): Sertakan metadata dalam chunks.
        skip_existing (bool): Skip file yang sudah diproses.
        
    Returns:
        Dict[str, Any]: Statistik hasil chunking.
    """
    if partition_pdf is None:
        logger.error("Library 'unstructured' atau dependencies-nya tidak tersedia.")
        logger.error(f"Error: {_import_error}")
        logger.error("Install dengan: pip install unstructured[pdf]")
        logger.error("Atau lengkap: pip install unstructured[all-docs]")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0,
            'duration': 0
        }
    
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Memulai Element-Based Chunking Pipeline")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Include metadata: {include_metadata}")
    logger.info(f"Skip existing: {skip_existing}")
    
    # Buat direktori output
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
            'total_chunks': 0,
            'duration': 0
        }
    
    # Proses setiap PDF
    stats: Dict[str, Any] = {
        'total_files': len(pdf_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_chunks': 0,
        'output_files': []
    }
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        # Cek apakah file output sudah ada
        output_filename = pdf_path.stem + "_chunks.json"
        output_file = Path(output_dir) / output_filename
        
        if skip_existing and output_file.exists():
            logger.info(f"⊙ File output sudah ada, skip: {output_filename}")
            stats['skipped'] += 1
            continue
        
        # Proses PDF
        chunks = process_single_pdf(
            str(pdf_path),
            output_dir,
            strategy=strategy,
            include_metadata=include_metadata
        )
        
        if chunks:
            stats['processed'] += 1
            stats['total_chunks'] += len(chunks)
            stats['output_files'].append(str(output_file))
        else:
            stats['failed'] += 1
    
    # Hitung durasi
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats['duration'] = duration
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info("Element-Based Chunking Selesai")
    logger.info("="*70)
    logger.info(f"Total file PDF: {stats['total_files']}")
    logger.info(f"Berhasil diproses: {stats['processed']}")
    logger.info(f"Di-skip (sudah ada): {stats['skipped']}")
    logger.info(f"Gagal: {stats['failed']}")
    logger.info(f"Total chunks dihasilkan: {stats['total_chunks']}")
    logger.info(f"Durasi: {duration:.2f} detik")
    
    if stats['processed'] > 0:
        avg_chunks = stats['total_chunks'] / stats['processed']
        logger.info(f"Rata-rata chunks per dokumen: {avg_chunks:.2f}")
    
    logger.info("="*70)
    
    return stats


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
