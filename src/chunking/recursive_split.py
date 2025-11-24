"""
Modul Recursive Chunking menggunakan RecursiveCharacterTextSplitter dari LangChain.

Modul ini memanggil RecursiveCharacterTextSplitter dari library langchain-text-splitters
untuk melakukan recursive text splitting berdasarkan hierarki separator.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-not-found, import-untyped]
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment, misc]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text(text_path: str) -> Optional[str]:
    """
    Memuat file teks dari data/cleaned/.
    
    Args:
        text_path (str): Path ke file teks.
        
    Returns:
        Optional[str]: Isi file teks atau None jika gagal.
    """
    try:
        text_file = Path(text_path)
        
        if not text_file.exists():
            logger.error(f"File tidak ditemukan: {text_path}")
            return None
        
        if not text_file.suffix.lower() == '.txt':
            logger.error(f"File bukan .txt: {text_path}")
            return None
        
        logger.info(f"Memuat teks: {text_file.name}")
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"✓ Berhasil memuat {len(text)} karakter")
        return text
        
    except Exception as e:
        logger.error(f"Error saat memuat teks {text_path}: {str(e)}")
        return None


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    length_function: Any = len,
    is_separator_regex: bool = False
) -> Optional[Any]:
    """
    Membuat instance RecursiveCharacterTextSplitter dari LangChain.
    
    Args:
        chunk_size (int): Maximum size per chunk dalam characters (default: 1000).
        chunk_overlap (int): Overlap antara chunks (default: 200).
        separators (Optional[List[str]]): Hierarki separator. Default: ["\n\n", "\n", " ", ""]
        length_function (Any): Fungsi untuk mengukur panjang (default: len).
        is_separator_regex (bool): Apakah separator adalah regex (default: False).
        
    Returns:
        Optional[Any]: Instance RecursiveCharacterTextSplitter atau None jika gagal.
    """
    if RecursiveCharacterTextSplitter is None:
        logger.error("Library 'langchain-text-splitters' tidak terinstall.")
        logger.error("Install dengan: pip install langchain-text-splitters")
        return None
    
    try:
        # Default separators sesuai LangChain API
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        logger.info("Membuat RecursiveCharacterTextSplitter...")
        logger.info(f"  - chunk_size: {chunk_size}")
        logger.info(f"  - chunk_overlap: {chunk_overlap}")
        logger.info(f"  - separators: {separators}")
        
        # Buat splitter dari LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            separators=separators
        )
        
        logger.info("✓ Text splitter berhasil dibuat")
        return text_splitter
        
    except Exception as e:
        logger.error(f"Error saat membuat text splitter: {str(e)}")
        return None


def run_recursive_splitter(
    text: str,
    text_splitter: Any
) -> Optional[List[str]]:
    """
    Menjalankan recursive text splitting menggunakan splitter dari LangChain.
    
    Args:
        text (str): Teks yang akan di-split.
        text_splitter (Any): Instance RecursiveCharacterTextSplitter.
        
    Returns:
        Optional[List[str]]: List chunks atau None jika gagal.
    """
    try:
        logger.info("Menjalankan recursive text splitting...")
        
        # Panggil split_text dari LangChain (tidak implementasi ulang!)
        chunks = text_splitter.split_text(text)
        
        logger.info(f"✓ Recursive splitting selesai")
        logger.info(f"  - Total chunks: {len(chunks)}")
        
        # Log statistik chunks
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            logger.info(f"  - Rata-rata karakter per chunk: {sum(chunk_sizes) / len(chunks):.2f}")
            logger.info(f"  - Min karakter per chunk: {min(chunk_sizes)}")
            logger.info(f"  - Max karakter per chunk: {max(chunk_sizes)}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error saat recursive splitting: {str(e)}")
        return None


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


def convert_chunks_to_dict(
    chunks: List[str],
    source_filename: str,
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Konversi list chunks (strings) menjadi format dictionary dengan metadata.
    
    Args:
        chunks (List[str]): List chunks dari splitter.
        source_filename (str): Nama file sumber.
        include_metadata (bool): Jika True, sertakan metadata.
        
    Returns:
        List[Dict[str, Any]]: List chunks dengan metadata.
    """
    chunk_dicts = []
    
    try:
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_dict: Dict[str, Any] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'num_characters': len(chunk_text)
            }
            
            if include_metadata:
                chunk_dict['metadata'] = {
                    'source_file': source_filename,
                    'chunking_method': 'recursive_character_text_splitter',
                    'chunk_length': len(chunk_text)
                }
            
            chunk_dicts.append(chunk_dict)
        
        logger.info(f"✓ Konversi {len(chunks)} chunks ke format dictionary")
        
        return chunk_dicts
        
    except Exception as e:
        logger.error(f"Error saat konversi chunks: {str(e)}")
        return []


def process_single_text(
    text_path: str,
    output_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    include_metadata: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Memproses satu file teks: load, split, dan save.
    
    Args:
        text_path (str): Path ke file teks.
        output_dir (str): Direktori output untuk hasil chunking.
        chunk_size (int): Maximum size per chunk.
        chunk_overlap (int): Overlap antara chunks.
        separators (Optional[List[str]]): Hierarki separator.
        include_metadata (bool): Sertakan metadata dalam chunks.
        
    Returns:
        Optional[List[Dict[str, Any]]]: List chunks jika berhasil, None jika gagal.
    """
    try:
        logger.info(f"Memproses teks: {Path(text_path).name}")
        
        # 1. Load text
        text = load_text(text_path)
        if not text:
            return None
        
        # 2. Create text splitter
        text_splitter = create_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        if text_splitter is None:
            return None
        
        # 3. Run recursive splitting
        chunks = run_recursive_splitter(text, text_splitter)
        if not chunks or len(chunks) == 0:
            logger.warning(f"Tidak ada chunks yang dihasilkan dari {Path(text_path).name}")
            return None
        
        # 4. Convert to dictionary format
        chunk_dicts = convert_chunks_to_dict(
            chunks,
            Path(text_path).name,
            include_metadata=include_metadata
        )
        if not chunk_dicts:
            return None
        
        # 5. Save chunks
        output_filename = Path(text_path).stem + "_chunks.json"
        output_path = Path(output_dir) / output_filename
        
        success = save_chunks(chunk_dicts, str(output_path), pretty_print=True)
        
        if success:
            return chunk_dicts
        else:
            return None
        
    except Exception as e:
        logger.error(f"Error saat memproses teks {text_path}: {str(e)}")
        return None


def get_text_files(input_dir: str) -> List[Path]:
    """
    Mendapatkan daftar semua file .txt dalam direktori.
    
    Args:
        input_dir (str): Path ke direktori input.
        
    Returns:
        List[Path]: List path file .txt.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Direktori input tidak ditemukan: {input_dir}")
        return []
    
    text_files = list(input_path.glob("*.txt"))
    logger.info(f"Ditemukan {len(text_files)} file .txt di {input_dir}")
    
    return text_files


def run_recursive_chunking(
    input_dir: str = "data/cleaned",
    output_dir: str = "data/chunked/recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    include_metadata: bool = True,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Menjalankan recursive chunking untuk semua file teks di direktori input.
    
    Args:
        input_dir (str): Direktori berisi file teks input (default: data/cleaned).
        output_dir (str): Direktori output untuk hasil chunking (default: data/chunked/recursive).
        chunk_size (int): Maximum size per chunk (default: 1000).
        chunk_overlap (int): Overlap antara chunks (default: 200).
        separators (Optional[List[str]]): Hierarki separator (default: ["\n\n", "\n", " ", ""]).
        include_metadata (bool): Sertakan metadata (default: True).
        skip_existing (bool): Skip file yang sudah diproses (default: True).
        
    Returns:
        Dict[str, Any]: Statistik hasil chunking.
    """
    if RecursiveCharacterTextSplitter is None:
        logger.error("Library 'langchain-text-splitters' tidak terinstall.")
        logger.error("Install dengan: pip install langchain-text-splitters")
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
    logger.info("Memulai Recursive Chunking Pipeline")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    
    # Format separators untuk logging
    default_seps = ["\n\n", "\n", " ", ""]
    sep_display = separators if separators else default_seps
    logger.info(f"Separators: {sep_display}")
    
    # Buat direktori output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dapatkan daftar file teks
    text_files = get_text_files(input_dir)
    
    if not text_files:
        logger.warning("Tidak ada file teks yang ditemukan untuk diproses")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0,
            'duration': 0
        }
    
    # Proses setiap file teks
    stats: Dict[str, Any] = {
        'total_files': len(text_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_chunks': 0,
        'output_files': []
    }
    
    for i, text_path in enumerate(text_files, 1):
        logger.info(f"\n[{i}/{len(text_files)}] Processing: {text_path.name}")
        
        # Cek apakah file output sudah ada
        output_filename = text_path.stem + "_chunks.json"
        output_file = Path(output_dir) / output_filename
        
        if skip_existing and output_file.exists():
            logger.info(f"⊙ File output sudah ada, skip: {output_filename}")
            stats['skipped'] += 1
            continue
        
        # Proses file teks
        chunks = process_single_text(
            str(text_path),
            output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
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
    logger.info("Recursive Chunking Selesai")
    logger.info("="*70)
    logger.info(f"Total file teks: {stats['total_files']}")
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
        description="Recursive chunking untuk dokumen teks menggunakan LangChain"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/cleaned',
        help='Direktori input yang berisi file .txt (default: data/cleaned)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/chunked/recursive',
        help='Direktori output untuk hasil chunking (default: data/chunked/recursive)'
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
    
    parser.add_argument(
        '--single',
        type=str,
        help='Proses satu file .txt saja (berikan path ke file)'
    )
    
    args = parser.parse_args()
    
    # Jalankan chunking
    if args.single:
        # Mode single file
        chunks = process_single_text(
            args.single,
            args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            include_metadata=not args.no_metadata
        )
        exit(0 if chunks else 1)
    else:
        # Mode batch
        stats = run_recursive_chunking(
            input_dir=args.input,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            include_metadata=not args.no_metadata,
            skip_existing=not args.no_skip
        )
        
        exit(0 if stats['processed'] > 0 else 1)
