"""
Modul ChromaDB untuk load embeddings dan manage vector store.

Pipeline:
1. Load embeddings dari JSON hasil embedding
2. Create ChromaDB collections (per chunking method)
3. Add documents dengan embeddings ke ChromaDB
4. Provide query interface untuk retrieval

Supports:
- Local persistent storage
- Batch processing untuk large datasets
- Metadata filtering
- Multiple chunking methods dalam separate collections
"""

from .client import (
    initialize_chroma_client,
    get_or_create_collection,
    delete_collection,
    list_collections
)

from .loader import (
    load_to_chroma,
    load_all_embeddings_to_chroma,
    batch_add_documents
)

from .query import (
    ChromaRetriever,
    similarity_search,
    similarity_search_with_score,
    search_by_vector
)

__all__ = [
    # Client management
    'initialize_chroma_client',
    'get_or_create_collection',
    'delete_collection',
    'list_collections',
    
    # Loading
    'load_to_chroma',
    'load_all_embeddings_to_chroma',
    'batch_add_documents',
    
    # Querying
    'ChromaRetriever',
    'similarity_search',
    'similarity_search_with_score',
    'search_by_vector',
]

__version__ = '1.0.0'
