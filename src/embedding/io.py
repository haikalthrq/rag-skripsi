"""
I/O utilities untuk load/save chunks dan embeddings.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_chunks_from_json(json_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load chunks dari file JSON hasil chunking.
    
    Supports 3 formats:
    1. Element-based: [{"text": "...", "metadata": {...}}]
    2. MaxMin/Recursive: [{"text": "...", "id": "...", "metadata": {...}}]
    
    Args:
        json_path: Path ke file JSON
        
    Returns:
        List of chunk dictionaries atau None jika error
    """
    try:
        json_file = Path(json_path)
        
        if not json_file.exists():
            logger.error(f"File tidak ditemukan: {json_path}")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            logger.error(f"Expected list, got {type(chunks)}")
            return None
        
        logger.info(f"Loaded {len(chunks)} chunks from {json_file.name}")
        
        return chunks
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {json_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading {json_path}: {str(e)}")
        return None


def _html_table_to_text(html: str) -> str:
    """Parse HTML table menjadi teks baris pipe-separated.
    
    Digunakan untuk menggantikan teks OCR yang korup pada table chunks
    element_based dengan representasi teks yang bersih dari HTML.
    """
    from html.parser import HTMLParser as _HTMLParser

    class _TblParser(_HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: List[List[str]] = []
            self._row: List[str] = []
            self._cell: List[str] = []
            self._in_cell = False

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
                self._in_cell = True

        def handle_data(self, data: str) -> None:
            if self._in_cell:
                t = " ".join(data.split())
                if t:
                    self._cell.append(t)

        def handle_endtag(self, tag: str) -> None:
            if tag in ("td", "th"):
                self._row.append(" ".join(self._cell).strip())
                self._cell = []
                self._in_cell = False
            elif tag == "tr" and self._row:
                self.rows.append(self._row)
                self._row = []

    p = _TblParser()
    p.feed(html)
    return "\n".join(" | ".join(c for c in row if c) for row in p.rows)


def _is_noise_text(text: str) -> bool:
    """Deteksi apakah teks adalah noise OCR/header PDF, bukan judul bermakna.

    Noise patterns:
    - Baris pertama berisi "PROVINSI/PROVINCE" (header tabel berulang)
    - Teks yang mayoritas adalah angka, simbol, atau karakter non-alfabet
    - Teks yang dimulai dengan nomor halaman atau kode kolom ("(1)", "(2)", dll)
    """
    if not text:
        return True
    first_line = text.split("\n")[0].strip()
    # Terlalu pendek untuk jadi judul bermakna (< 3 kata)
    words = first_line.split()
    if len(words) < 2:
        return True
    # Header tabel berulang dari PDF dua kolom
    if "PROVINSI/PROVINCE" in first_line.upper():
        return True
    if "INDONESIA" == first_line.upper().strip():
        return True
    # Baris pertama hanya angka/simbol (nomor halaman, kode kolom)
    import re as _re
    stripped = _re.sub(r"[\d\s\.,\-\(\)\|/]", "", first_line)
    if len(stripped) < 3:  # hampir tidak ada huruf bermakna
        return True
    return False


def enrich_table_chunk_texts(chunks: List[Dict[str, Any]]) -> int:
    """Enrichment untuk element_based table chunks: ganti teks OCR yang korup
    dengan teks yang di-parse dari text_as_html metadata, lalu tambahkan
    judul/section dari chunk teks sebelumnya sebagai prefix konteks.

    Fungsi ini memodifikasi list ``chunks`` secara in-place sehingga:
    - ``chunk['text']`` berisi representasi bersih dari tabel HTML
    - Chunk tabel mendapat prefix judul agar embedding-nya lebih dekat ke query
    - Embedding yang dihasilkan dan teks yang disimpan ke ChromaDB konsisten

    Hanya berdampak pada chunk yang memiliki ``metadata.text_as_html`` dan
    ``metadata.chunk_type == 'table'``. Aman dijalankan pada metode chunking
    lain (maxmin_semantic, recursive) karena mereka tidak punya text_as_html.

    Returns:
        Jumlah chunk yang berhasil di-enrich.
    """
    enriched = 0
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata") or {}
        html = meta.get("text_as_html") or ""
        if not html.strip():
            continue
        if meta.get("chunk_type") != "table":
            continue
        table_text = _html_table_to_text(html)
        if not table_text.strip():
            continue

        # Cari judul/heading dari chunk teks sebelumnya sebagai prefix konteks.
        # Ini penting agar embedding tabel (yang hanya berisi angka/simbol)
        # mendapat konteks semantik yang membantu kemiripan dengan query natural.
        prefix = ""
        curr_pgs = set(str(meta.get("page_numbers") or "").replace("[", "")
                       .replace("]", "").replace(" ", "").split(","))
        for back in range(1, min(4, idx + 1)):
            prev = chunks[idx - back]
            prev_meta = prev.get("metadata") or {}
            if prev_meta.get("chunk_type") == "table":
                continue  # skip jika prev juga tabel
            prev_text = (prev.get("text") or "").strip()
            if not prev_text or len(prev_text) > 300:
                continue  # terlalu panjang – bukan heading
            # Tolak teks yang merupakan noise OCR/header PDF berulang
            if _is_noise_text(prev_text):
                continue
            # Cek halaman overlap
            prev_pgs = set(str(prev_meta.get("page_numbers") or "")
                           .replace("[", "").replace("]", "")
                           .replace(" ", "").split(","))
            if curr_pgs & prev_pgs:  # ada halaman yang sama
                prefix = prev_text + "\n\n"
                break

        # Fallback: jika tidak ada prev chunk valid, gunakan section_title metadata
        if not prefix:
            section_title = (meta.get("section_title") or "").strip()
            if section_title and not _is_noise_text(section_title):
                prefix = section_title + "\n\n"

        chunk["text"] = prefix + table_text
        enriched += 1

    if enriched:
        logger.info(f"enrich_table_chunk_texts: {enriched} table chunks enriched from HTML (with title prefix)")
    return enriched


def clean_and_filter_chunks(chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    """
    Clean whitespace dan filter chunk kosong.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Tuple of (cleaned_texts, valid_indices)
        - cleaned_texts: List of cleaned text strings
        - valid_indices: Original indices of valid chunks
    """
    cleaned_texts = []
    valid_indices = []
    
    for idx, chunk in enumerate(chunks):
        # Extract text
        text = chunk.get('text', '').strip()
        
        # Skip empty chunks
        if not text:
            logger.debug(f"Skipping empty chunk at index {idx}")
            continue
        
        # Clean excessive whitespace
        text = ' '.join(text.split())
        
        cleaned_texts.append(text)
        valid_indices.append(idx)
    
    skipped = len(chunks) - len(cleaned_texts)
    if skipped > 0:
        logger.info(f"Filtered out {skipped} empty chunks ({len(cleaned_texts)} valid chunks remaining)")
    
    return cleaned_texts, valid_indices


def save_embeddings(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    valid_indices: List[int],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save embeddings dan metadata ke file.
    
    Format output:
    {
        "metadata": {
            "source_file": "...",
            "chunking_method": "...",
            "embedding_model": "...",
            "embedding_dim": ...,
            "num_chunks": ...,
            "timestamp": "..."
        },
        "embeddings": [...],  # List of lists (untuk JSON serialization)
        "chunks": [...]       # Original chunks dengan embeddings
    }
    
    Args:
        embeddings: Numpy array dengan shape (n_chunks, embedding_dim)
        chunks: Original chunk dictionaries
        valid_indices: Indices of valid chunks yang di-embed
        output_path: Path untuk output file
        metadata: Optional metadata tambahan
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        from datetime import datetime
        
        output_data = {
            "metadata": {
                "embedding_dim": int(embeddings.shape[1]),
                "num_chunks": len(valid_indices),
                "num_original_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            },
            "embeddings": embeddings.tolist(),  # Convert numpy to list for JSON
            "chunks": []
        }
        
        # Add chunks dengan embeddings
        for i, chunk_idx in enumerate(valid_indices):
            chunk_data = chunks[chunk_idx].copy()
            chunk_data['embedding_index'] = i
            chunk_data['original_index'] = chunk_idx
            output_data['chunks'].append(chunk_data)
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Embeddings saved to: {output_file.name}")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  - Number of embeddings: {len(valid_indices)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving embeddings to {output_path}: {str(e)}")
        return False


def load_embeddings(embedding_path: str) -> Optional[Dict[str, Any]]:
    """
    Load embeddings dari file yang sudah di-save.
    
    Args:
        embedding_path: Path ke embedding file
        
    Returns:
        Dictionary dengan keys: 'metadata', 'embeddings', 'chunks'
        atau None jika error
    """
    try:
        embedding_file = Path(embedding_path)
        
        if not embedding_file.exists():
            logger.error(f"Embedding file tidak ditemukan: {embedding_path}")
            return None
        
        with open(embedding_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert embeddings back to numpy
        data['embeddings'] = np.array(data['embeddings'], dtype=np.float32)
        
        logger.info(f"Loaded embeddings from {embedding_file.name}")
        logger.info(f"  - Embedding dimension: {data['metadata']['embedding_dim']}")
        logger.info(f"  - Number of embeddings: {data['metadata']['num_chunks']}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading embeddings from {embedding_path}: {str(e)}")
        return None


def save_embeddings_numpy(
    embeddings: np.ndarray,
    output_path: str
) -> bool:
    """
    Save embeddings dalam format numpy (.npy) untuk efisiensi.
    
    Args:
        embeddings: Numpy array
        output_path: Path untuk output file (.npy)
        
    Returns:
        True jika berhasil
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_file, embeddings)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Embeddings saved (numpy) to: {output_file.name}")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving numpy embeddings: {str(e)}")
        return False


def load_embeddings_numpy(embedding_path: str) -> Optional[np.ndarray]:
    """
    Load embeddings dari file numpy (.npy).
    
    Args:
        embedding_path: Path ke .npy file
        
    Returns:
        Numpy array atau None jika error
    """
    try:
        embedding_file = Path(embedding_path)
        
        if not embedding_file.exists():
            logger.error(f"Numpy embedding file tidak ditemukan: {embedding_path}")
            return None
        
        embeddings = np.load(embedding_file)
        
        logger.info(f"Loaded numpy embeddings from {embedding_file.name}")
        logger.info(f"  - Shape: {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error loading numpy embeddings: {str(e)}")
        return None
