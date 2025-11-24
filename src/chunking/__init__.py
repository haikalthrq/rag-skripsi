"""
Modul Chunking untuk RAG System
================================

Modul ini menyediakan berbagai metode chunking untuk dokumen:
1. Element-based chunking menggunakan Unstructured
2. MaxMin semantic chunking menggunakan maxmin_chunker library
3. Recursive chunking menggunakan LangChain RecursiveCharacterTextSplitter

Struktur Modul:
--------------
- element_based.py: Element-based chunking dengan Unstructured
- maxmin_chunker.py: MaxMin semantic chunking
- recursive_split.py: Recursive character text splitting dengan LangChain

Penggunaan:
----------
Sebagai script langsung:
    python chunk_element.py --input data/raw --output data/chunked/element_based
    python chunk_maxmin.py --input data/cleaned_text --output data/chunked/maxmin_semantic
    python chunk_recursive.py --input data/cleaned_text --output data/chunked/recursive
    
Atau import dalam kode Python:
    from src.chunking.element_based import run_element_based_chunking
    from src.chunking.maxmin_chunker import run_maxmin_chunking
    from src.chunking.recursive_split import run_recursive_chunking
"""

# Element-based chunking
from .element_based import (
    load_pdf,
    partition_document,
    convert_elements_to_chunks,
    convert_elements_to_text_list,
    save_chunks as save_element_chunks,
    process_single_pdf,
    get_pdf_files,
    run_element_based_chunking
)

# MaxMin semantic chunking
from .maxmin_chunker import (
    initialize_embedding_model,
    load_text,
    split_sentences,
    embed_sentences,
    apply_maxmin_chunking,
    save_chunks as save_maxmin_chunks,
    convert_paragraphs_to_chunks,
    process_single_text,
    get_text_files,
    run_maxmin_chunking
)

# Recursive chunking dengan LangChain
from .recursive_split import (
    load_text as load_text_recursive,
    create_text_splitter,
    run_recursive_splitter,
    save_chunks as save_recursive_chunks,
    convert_chunks_to_dict,
    process_single_text as process_single_text_recursive,
    get_text_files as get_text_files_recursive,
    run_recursive_chunking
)

__version__ = '1.0.0'

__all__ = [
    # Element-based chunking
    'load_pdf',
    'partition_document',
    'convert_elements_to_chunks',
    'convert_elements_to_text_list',
    'save_element_chunks',
    'process_single_pdf',
    'get_pdf_files',
    'run_element_based_chunking',
    
    # MaxMin semantic chunking
    'initialize_embedding_model',
    'load_text',
    'split_sentences',
    'embed_sentences',
    'apply_maxmin_chunking',
    'save_maxmin_chunks',
    'convert_paragraphs_to_chunks',
    'process_single_text',
    'get_text_files',
    'run_maxmin_chunking',
    
    # Recursive chunking
    'load_text_recursive',
    'create_text_splitter',
    'run_recursive_splitter',
    'save_recursive_chunks',
    'convert_chunks_to_dict',
    'process_single_text_recursive',
    'get_text_files_recursive',
    'run_recursive_chunking',
]
