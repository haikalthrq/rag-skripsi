"""Implementasi chunking untuk pipeline RAG.

Package ini mengekspor tiga pendekatan:

1. Composite element-based chunking dengan Unstructured.
2. MaxMin semantic chunking dengan algoritma lokal dan sentence embeddings.
3. Recursive character splitting dengan LangChain.

Entry point CLI berada di modul implementasi masing-masing::

    python src/chunking/element_based.py
    python src/chunking/maxmin_chunker.py
    python src/chunking/recursive_split.py

MaxMin dan recursive menggunakan ``data/cleaned`` sebagai input default,
sedangkan element-based menggunakan ``data/raw``.
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
