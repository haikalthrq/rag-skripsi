"""
Modul Element-Based Chunking menggunakan library Unstructured.

Modul ini melakukan ekstraksi dan chunking dokumen PDF berdasarkan struktur elemen
seperti Title, Paragraph, ListItem, Table, dan lainnya menggunakan partition_pdf.

DESIGN PRINCIPLE:
- Menggunakan partition_pdf(strategy="hi_res") untuk ekstraksi layout-aware
- Tidak 1 elemen = 1 chunk, tapi membangun COMPOSITE CHUNKS
- Gabung elemen berurutan yang masih dalam konteks struktural sama
- Judul sebagai boundary section
- Tabel dipertahankan utuh sebagai unit standalone
- Hindari chunk terlalu kecil (orphan chunks) melalui backward merge
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


def categorize_element(elem_type: str) -> Tuple[str, int]:
    """
    Kategorikan tipe elemen Unstructured ke kategori internal.
    
    Priority untuk penanganan:
    - title (priority 1): Boundary struktural
    - table (priority 2): Unit standalone
    - text (priority 3): Dapat digabung
    - other (priority 0): Diabaikan
    
    Args:
        elem_type: Nama class tipe elemen dari Unstructured
        
    Returns:
        Tuple[kategori, priority]
    """
    # Title elements - highest priority as boundary
    if elem_type in ('Title', 'Header', 'Subheadline', 'Headline'):
        return ('title', 1)
    
    # Table elements - standalone unit
    elif elem_type in ('Table', 'FormattedTable'):
        return ('table', 2)
    
    # Text elements - composite building blocks
    elif elem_type in ('Text', 'NarrativeText', 'ListItem', 'BulletedText', 'NumberedList'):
        return ('text', 3)
    
    # Page break - structural boundary (handled specially)
    elif elem_type == 'PageBreak':
        return ('page_break', 0)
    
    # Everything else - ignore for main content
    else:
        return ('other', 0)


def merge_small_chunks_backward(
    chunks: List[Dict[str, Any]], 
    min_chunk_chars: int
) -> List[Dict[str, Any]]:
    """
    Post-processing: Merge chunks yang terlalu kecil ke chunk sebelumnya.
    
    Strategi:
    - Iterasi dari belakang (kecuali chunk pertama)
    - Jika chunk[i] < min_chunk_chars, gabung ke chunk[i-1] jika:
      * Tipe sama (text+text) atau
      * Section title sama
    - Tabel tidak digabung ke chunk lain (tetap standalone)
    
    Args:
        chunks: List chunks hasil chunking
        min_chunk_chars: Minimal karakter untuk chunk valid
        
    Returns:
        List chunks setelah merge
    """
    if not chunks or min_chunk_chars <= 0:
        return chunks
    
    result = list(chunks)  # Copy
    
    # Iterasi dari belakang (kecuali index 0)
    i = len(result) - 1
    while i > 0:
        current = result[i]
        prev = result[i - 1]
        
        current_text_len = len(current.get('text', ''))
        
        # Jika chunk terlalu kecil dan BUKAN table
        if (current_text_len < min_chunk_chars and 
            current.get('metadata', {}).get('chunk_type') != 'table'):
            
            # Cek apakah bisa merge dengan previous
            prev_type = prev.get('metadata', {}).get('chunk_type')
            current_type = current.get('metadata', {}).get('chunk_type')
            prev_section = prev.get('metadata', {}).get('section_title')
            current_section = current.get('metadata', {}).get('section_title')
            
            # Merge jika:
            # 1. Previous juga bukan table (table tetap standalone)
            # 2. Section title sama (atau salah satu None)
            can_merge = (
                prev_type != 'table' and
                (prev_section == current_section or 
                 prev_section is None or 
                 current_section is None)
            )
            
            if can_merge:
                # Gabungkan teks
                prev_text = prev.get('text', '')
                current_text = current.get('text', '')
                
                if prev_text and current_text:
                    prev['text'] = prev_text + "\n\n" + current_text
                else:
                    prev['text'] = prev_text + current_text
                
                # Update metadata
                if 'metadata' in prev and 'metadata' in current:
                    # Merge element types
                    prev_elem_types = set(prev['metadata'].get('element_types', []))
                    curr_elem_types = set(current['metadata'].get('element_types', []))
                    prev['metadata']['element_types'] = list(prev_elem_types | curr_elem_types)
                    
                    # Update element count
                    prev_count = prev['metadata'].get('element_count', 1)
                    curr_count = current['metadata'].get('element_count', 1)
                    prev['metadata']['element_count'] = prev_count + curr_count
                    
                    # Update page range
                    prev_pages = set(prev['metadata'].get('page_numbers', []))
                    curr_pages = set(current['metadata'].get('page_numbers', []))
                    all_pages = sorted(list(prev_pages | curr_pages))
                    prev['metadata']['page_numbers'] = all_pages
                    if all_pages:
                        prev['metadata']['page_range'] = f"{all_pages[0]}-{all_pages[-1]}" if len(all_pages) > 1 else str(all_pages[0])
                    
                    # Update char count
                    prev['metadata']['num_characters'] = len(prev['text'])
                
                # Hapus chunk current
                result.pop(i)
                logger.debug(f"Merged small chunk {i} ({current_text_len} chars) into chunk {i-1}")
            
            i -= 1
        else:
            i -= 1
    
    # Re-assign chunk_id setelah merge
    for idx, chunk in enumerate(result):
        chunk['chunk_id'] = idx
        if 'metadata' in chunk:
            chunk['metadata']['order_index'] = idx
    
    return result


def convert_elements_to_chunks(
    elements: List[Any],
    include_metadata: bool = True,
    target_chunk_chars: int = 1500,
    max_chunk_chars: int = 3000,
    min_chunk_chars: int = 300
) -> List[Dict[str, Any]]:
    """
    Konversi elemen dokumen menjadi COMPOSITE CHUNKS.
    
    ALUR KERJA:
    1. Filter dan kategorikan elemen (title, table, text, other)
    2. Iterasi elemen dengan buffer current_chunk
    3. PageBreak = cek sebelum filter empty-text; flush jika chunk sudah besar
    4. Title = flush chunk saat ini (termasuk pending list), mulai section baru
    5. Table = flush chunk saat ini (termasuk pending list), simpan table utuh
    6. Text/ListItem = akumulasi ke chunk; list items dikelompokkan di buffer terpisah
    7. Flush selalu merge pending list items terlebih dahulu (B1 fix)
    8. Post-processing: merge backward untuk chunks < min_chunk_chars
    
    Args:
        elements (List[Any]): List elemen hasil partitioning (Unstructured).
        include_metadata (bool): Jika True, metadata detil disertakan.
        target_chunk_chars (int): Ukuran optimal chunk (default: 1500).
        max_chunk_chars (int): Batas keras chunk (default: 3000).
        min_chunk_chars (int): Minimal ukuran chunk, yang lebih kecil di-merge backward (default: 300).
        
    Returns:
        List[Dict[str, Any]]: List composite chunks dengan metadata lengkap.
    """
    chunks: List[Dict[str, Any]] = []
    
    try:
        # --- State tracking ---
        active_title: Optional[str] = None
        current_source_file: Optional[str] = None   # Persists across flush via closure
        current_chunk: Optional[Dict[str, Any]] = None
        prev_was_list: bool = False

        # List item buffer terpisah dari current_chunk['text'] untuk menghindari
        # hilangnya items saat flush dipanggil di boundary (B1 fix)
        list_group_buffer: List[str] = []
        list_group_pages: List[int] = []

        # ------------------------------------------------------------------
        def init_chunk() -> Dict[str, Any]:
            """Inisialisasi chunk baru dengan state aktif saat ini."""
            return {
                'text': '',
                'metadata': {
                    'chunk_type': 'text',
                    'element_types': [],
                    'section_title': active_title,
                    'page_numbers': [],
                    'source_file': current_source_file,       # B6 fix: field yang diminta
                    'source_filename': current_source_file,   # backward compat
                    'element_count': 0,
                    'order_index': -1,                        # Diisi saat flush
                }
            }

        def flush_pending_list() -> None:
            """
            Merge list_group_buffer ke current_chunk sebelum chunk disimpan.
            Harus dipanggil sebelum setiap operasi flush.
            Jika chunk masih kosong dan ada active_title, prepend title sebagai
            konteks (lazy title prepend — simetris dengan text handler).
            """
            nonlocal list_group_buffer, list_group_pages, prev_was_list
            if not list_group_buffer:
                return
            list_text = "\n".join(list_group_buffer)
            if current_chunk['text'].strip():
                current_chunk['text'] += "\n\n" + list_text
            elif active_title:
                # Lazy title prepend: chunk baru setelah title, list item pertama
                current_chunk['text'] = active_title + "\n\n" + list_text
            else:
                current_chunk['text'] = list_text
            if 'ListGroup' not in current_chunk['metadata']['element_types']:
                current_chunk['metadata']['element_types'].append('ListGroup')
            current_chunk['metadata']['element_count'] += len(list_group_buffer)
            current_chunk['metadata']['page_numbers'].extend(list_group_pages)
            list_group_buffer = []
            list_group_pages = []
            prev_was_list = False

        def flush_chunk(forced_type: Optional[str] = None) -> None:
            """
            Simpan current_chunk ke list chunks jika valid.
            Selalu merge pending list items terlebih dahulu.
            
            Args:
                forced_type: Jika diset, override chunk_type (untuk table).
            """
            nonlocal current_chunk
            # B1 fix: merge list buffer sebelum save
            flush_pending_list()

            if current_chunk and current_chunk['text'].strip():
                pages = sorted(list(set(current_chunk['metadata']['page_numbers'])))
                current_chunk['metadata']['page_numbers'] = pages
                current_chunk['metadata']['page_range'] = (
                    f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
                ) if pages else "Unknown"
                current_chunk['metadata']['num_characters'] = len(current_chunk['text'])

                if forced_type:
                    current_chunk['metadata']['chunk_type'] = forced_type

                chunk_id = len(chunks)
                current_chunk['chunk_id'] = chunk_id
                current_chunk['metadata']['order_index'] = chunk_id

                if not include_metadata:
                    current_chunk['metadata'] = {
                        'chunk_type': current_chunk['metadata'].get('chunk_type', 'text'),
                        'page_range': current_chunk['metadata'].get('page_range', 'Unknown'),
                    }

                chunks.append(current_chunk)

            current_chunk = init_chunk()

        # ------------------------------------------------------------------

        # Inisialisasi chunk pertama
        current_chunk = init_chunk()

        for idx, element in enumerate(elements):
            elem_type = type(element).__name__

            # B2 fix: cek PageBreak SEBELUM filter empty-text
            # (PageBreak tidak punya text, akan di-skip oleh guard di bawah)
            if elem_type == 'PageBreak':
                pending_len = len("\n".join(list_group_buffer))
                effective_len = len(current_chunk['text']) + pending_len
                if effective_len >= target_chunk_chars:
                    flush_chunk()
                continue

            text = str(element.text) if hasattr(element, 'text') else str(element)
            if not text or not text.strip():
                continue

            category, _ = categorize_element(elem_type)
            if category == 'other':
                continue

            # Ekstrak metadata native dari Unstructured
            elem_metadata = element.metadata if hasattr(element, 'metadata') else None
            page_num = getattr(elem_metadata, 'page_number', None) if elem_metadata else None
            filename = getattr(elem_metadata, 'filename', None) if elem_metadata else None

            # B6 fix: track source file di level fungsi agar persist lintas flush
            # (init_chunk() membaca current_source_file via closure)
            if filename and current_source_file is None:
                current_source_file = filename
                current_chunk['metadata']['source_file'] = filename
                current_chunk['metadata']['source_filename'] = filename

            # ----------------------------------------------------------
            # TITLE: Boundary struktural
            # ----------------------------------------------------------
            if category == 'title':
                # flush_chunk() sudah memanggil flush_pending_list() di dalamnya.
                # Jika current_chunk tidak punya konten nyata (text kosong),
                # flush_chunk() tidak akan menyimpannya → tidak ada orphan title chunk.
                flush_chunk()
                active_title = text.strip()
                current_chunk['metadata']['section_title'] = active_title
                # TIDAK menulis title ke current_chunk['text'] di sini (lazy prepend).
                # Title baru diprepend saat konten nyata pertama (teks/list) tiba.
                # Ini mencegah orphan chunk untuk pola Title→Title, Title→EOF,
                # Title→PageBreak→Title.
                current_chunk['metadata']['element_types'].append(elem_type)
                current_chunk['metadata']['element_count'] += 1
                if page_num:
                    current_chunk['metadata']['page_numbers'].append(page_num)
                prev_was_list = False
                continue

            # ----------------------------------------------------------
            # TABLE: Unit standalone utuh
            # ----------------------------------------------------------
            if category == 'table':
                # flush_chunk() menangani semua kasus:
                # - Ada konten text/list → disimpan dulu sebelum table
                # - Chunk kosong (baru setelah title) → tidak disimpan (no orphan)
                # - Ada pending list buffer → flush_pending_list() di dalam flush_chunk
                flush_chunk()

                # B3 & B4 fix: set source_file dan section_title eksplisit
                current_chunk['text'] = text.strip()
                current_chunk['metadata']['chunk_type'] = 'table'
                current_chunk['metadata']['section_title'] = active_title  # B4 fix
                current_chunk['metadata']['element_types'] = [elem_type]
                current_chunk['metadata']['element_count'] = 1
                current_chunk['metadata']['page_numbers'] = [page_num] if page_num else []
                # B3 fix: source_file eksplisit untuk table chunk
                src = current_source_file or filename
                current_chunk['metadata']['source_file'] = src
                current_chunk['metadata']['source_filename'] = src

                if elem_metadata and hasattr(elem_metadata, 'text_as_html') and elem_metadata.text_as_html:
                    current_chunk['metadata']['text_as_html'] = elem_metadata.text_as_html

                flush_chunk(forced_type='table')
                prev_was_list = False
                continue

            # ----------------------------------------------------------
            # TEXT (termasuk ListItem)
            # ----------------------------------------------------------
            if category == 'text':
                is_list_item = elem_type in ('ListItem', 'BulletedText', 'NumberedList')

                if is_list_item:
                    # Akumulasi ke list buffer terpisah (B1 fix)
                    list_group_buffer.append(text.strip())
                    if page_num:
                        list_group_pages.append(page_num)
                    prev_was_list = True
                    continue

                # Bukan list item: merge pending list ke current_chunk dulu
                flush_pending_list()
                prev_was_list = False

                text_stripped = text.strip()
                current_len = len(current_chunk['text'])
                text_len = len(text_stripped)

                # Flush jika menambahkan teks ini akan melewati max_chunk_chars
                # dan sudah ada konten meaningful (>= min_chunk_chars).
                # Jika current sangat kecil (< min), biarkan melebihi max daripada
                # membuat orphan kecil.
                if (current_len >= min_chunk_chars
                        and (current_len + text_len + 2) > max_chunk_chars):
                    flush_chunk()

                # Append teks ke chunk aktif.
                # Lazy title prepend: jika chunk masih kosong setelah title (atau
                # setelah flush mid-section), prepend active_title sebagai konteks.
                if current_chunk['text'].strip():
                    current_chunk['text'] += "\n\n" + text_stripped
                elif active_title:
                    current_chunk['text'] = active_title + "\n\n" + text_stripped
                else:
                    current_chunk['text'] = text_stripped

                if elem_type not in current_chunk['metadata']['element_types']:
                    current_chunk['metadata']['element_types'].append(elem_type)
                current_chunk['metadata']['element_count'] += 1
                if page_num:
                    current_chunk['metadata']['page_numbers'].append(page_num)

        # Flush sisa (termasuk list buffer yang mungkin tersisa di akhir dokumen)
        flush_chunk()

        # POST-PROCESSING: Merge backward untuk chunks terlalu kecil
        if min_chunk_chars > 0 and chunks:
            original_count = len(chunks)
            chunks = merge_small_chunks_backward(chunks, min_chunk_chars)
            if len(chunks) < original_count:
                logger.info(
                    f"Post-processing: merged {original_count - len(chunks)} "
                    f"small chunks (min: {min_chunk_chars} chars)"
                )

        # Logging hasil
        logger.info(
            f"Berhasil konversi menjadi {len(chunks)} composite chunks "
            f"dari {len(elements)} elemen ekstraksi"
        )

        if chunks:
            total_chars = sum(len(c['text']) for c in chunks)
            avg_chars = total_chars / len(chunks)
            logger.info(f"Total karakter: {total_chars}")
            logger.info(f"Rata-rata karakter per chunk: {avg_chars:.2f}")

            type_counts: Dict[str, int] = {}
            for c in chunks:
                ctype = (
                    c.get('metadata', {}).get('chunk_type', 'unknown')
                    if include_metadata else 'unknown'
                )
                type_counts[ctype] = type_counts.get(ctype, 0) + 1

            logger.info("Distribusi tipe chunk:")
            for k, v in type_counts.items():
                logger.info(f"  - {k}: {v}")

            sizes = [len(c['text']) for c in chunks]
            logger.info(
                f"Ukuran chunk: min={min(sizes)}, max={max(sizes)}, "
                f"median={sorted(sizes)[len(sizes)//2]}"
            )

        return chunks

    except Exception as e:
        logger.error(f"Error saat konversi elemen ke composite chunks: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
    include_metadata: bool = True,
    target_chunk_chars: int = 1500,
    max_chunk_chars: int = 3000,
    min_chunk_chars: int = 300
) -> Optional[List[Dict[str, Any]]]:
    """
    Memproses satu file PDF: partition, convert composite chunks, dan save.
    
    Args:
        pdf_path (str): Path ke file PDF.
        output_dir (str): Direktori output untuk hasil chunking.
        strategy (str): Strategi partitioning (default: 'hi_res').
        include_metadata (bool): Sertakan metadata dalam chunks.
        target_chunk_chars (int): Ukuran target chunk (default: 1500).
        max_chunk_chars (int): Batas maksimal chunk (default: 3000).
        min_chunk_chars (int): Minimal chunk, yang lebih kecil di-merge (default: 300).
        
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
        
        # 3. Convert elements to COMPOSITE chunks
        chunks = convert_elements_to_chunks(
            elements, 
            include_metadata=include_metadata,
            target_chunk_chars=target_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars
        )
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
    skip_existing: bool = True,
    target_chunk_chars: int = 1500,
    max_chunk_chars: int = 3000,
    min_chunk_chars: int = 300
) -> Dict[str, Any]:
    """
    Menjalankan element-based chunking untuk semua PDF di direktori input.
    
    Args:
        input_dir (str): Direktori berisi file PDF input.
        output_dir (str): Direktori output untuk hasil chunking.
        strategy (str): Strategi partitioning (default: 'hi_res' untuk akurasi maksimal).
        include_metadata (bool): Sertakan metadata dalam chunks.
        skip_existing (bool): Skip file yang sudah diproses.
        target_chunk_chars (int): Ukuran target chunk (default: 1500).
        max_chunk_chars (int): Batas maksimal chunk (default: 3000).
        min_chunk_chars (int): Minimal chunk, yang lebih kecil di-merge (default: 300).
        
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
    logger.info(f"Target chunk size: {target_chunk_chars} chars")
    logger.info(f"Max chunk size: {max_chunk_chars} chars")
    logger.info(f"Min chunk size: {min_chunk_chars} chars (merge backward if smaller)")
    
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
        
        # Proses PDF dengan parameter ukuran chunk
        chunks = process_single_pdf(
            str(pdf_path),
            output_dir,
            strategy=strategy,
            include_metadata=include_metadata,
            target_chunk_chars=target_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars
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
        '--target-chars',
        type=int,
        default=1500,
        help='Target ukuran chunk dalam karakter (default: 1500)'
    )
    
    parser.add_argument(
        '--max-chars',
        type=int,
        default=3000,
        help='Batas maksimal chunk dalam karakter (default: 3000)'
    )
    
    parser.add_argument(
        '--min-chars',
        type=int,
        default=300,
        help='Minimal ukuran chunk, yang lebih kecil di-merge backward (default: 300)'
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
            include_metadata=not args.no_metadata,
            target_chunk_chars=args.target_chars,
            max_chunk_chars=args.max_chars,
            min_chunk_chars=args.min_chars
        )
        exit(0 if chunks else 1)
    else:
        # Mode batch
        stats = run_element_based_chunking(
            input_dir=args.input,
            output_dir=args.output,
            strategy=args.strategy,
            include_metadata=not args.no_metadata,
            skip_existing=not args.no_skip,
            target_chunk_chars=args.target_chars,
            max_chunk_chars=args.max_chars,
            min_chunk_chars=args.min_chars
        )
        
        exit(0 if stats['processed'] > 0 else 1)
