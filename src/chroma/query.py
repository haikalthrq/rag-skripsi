"""Helper query ChromaDB untuk vector NumPy dan filter metadata opsional."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """Retriever untuk pencarian teks atau vector pada satu collection.

    Pencarian teks memerlukan ``embedding_function``. Semua metode mendukung
    filter metadata, dan hasil dengan score mengembalikan distance mentah dari
    Chroma, bukan similarity score atau MMR.
    """
    
    def __init__(
        self,
        collection: Any,
        embedding_function: Optional[Any] = None,
        k: int = 5
    ):
        """
        Initialize retriever.
        
        Args:
            collection: ChromaDB collection instance
            embedding_function: Function untuk generate query embeddings
            k: Default number of results
        """
        self.collection = collection
        self.embedding_function = embedding_function
        self.k = k
        
        logger.info(f"ChromaRetriever initialized")
        logger.info(f"  - Collection: {collection.name}")
        logger.info(f"  - Document count: {collection.count()}")
        logger.info(f"  - Default k: {k}")
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Similarity search dengan query text.
        
        Args:
            query: Query text
            k: Number of results (None = use default)
            filter: Metadata filter dictionary
            
        Returns:
            List of document dictionaries
        """
        if k is None:
            k = self.k
        
        # Generate query embedding
        if self.embedding_function is None:
            logger.error("No embedding function provided")
            return []
        
        query_embedding = self.embedding_function(query)
        
        return self.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter
        )
    
    def similarity_search_by_vector(
        self,
        embedding: np.ndarray,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Similarity search dengan vector embedding.
        
        Args:
            embedding: Query embedding vector
            k: Number of results
            filter: Metadata filter dictionary
            
        Returns:
            List of document dictionaries
        """
        if k is None:
            k = self.k
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k,
                where=filter
            )
            
            # Parse results
            documents = []
            
            if results and 'documents' in results:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Similarity search dengan scores.
        
        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter dictionary
            
        Returns:
            List of (document, score) tuples
        """
        if k is None:
            k = self.k
        
        # Generate query embedding
        if self.embedding_function is None:
            logger.error("No embedding function provided")
            return []
        
        query_embedding = self.embedding_function(query)
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filter
            )
            
            # Parse results with scores
            documents_with_scores = []
            
            if results and 'documents' in results:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    }
                    score = results['distances'][0][i] if 'distances' in results else 0.0
                    documents_with_scores.append((doc, score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []


def similarity_search(
    collection: Any,
    query_embedding: np.ndarray,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Standalone similarity search function.
    
    Args:
        collection: ChromaDB collection instance
        query_embedding: Query embedding vector
        k: Number of results
        filter: Metadata filter dictionary
        
    Returns:
        List of document dictionaries
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter
        )
        
        documents = []
        
        if results and 'documents' in results:
            for i in range(len(results['documents'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return []


def similarity_search_with_score(
    collection: Any,
    query_embedding: np.ndarray,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Standalone similarity search dengan scores.
    
    Args:
        collection: ChromaDB collection instance
        query_embedding: Query embedding vector
        k: Number of results
        filter: Metadata filter dictionary
        
    Returns:
        List of (document, score) tuples
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter
        )
        
        documents_with_scores = []
        
        if results and 'documents' in results:
            for i in range(len(results['documents'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                }
                # Nilai ini berasal langsung dari field `distances` Chroma,
                # bukan similarity terbalik. Umumnya nilai lebih kecil berarti
                # hasil lebih dekat, bergantung pada metrik collection.
                score = results['distances'][0][i] if 'distances' in results else 0.0
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
        
    except Exception as e:
        logger.error(f"Error in similarity search with score: {str(e)}")
        return []


def search_by_vector(
    collection: Any,
    embedding: np.ndarray,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
    include_distances: bool = True
) -> Dict[str, Any]:
    """
    Search menggunakan vector embedding dengan full results.
    
    Args:
        collection: ChromaDB collection instance
        embedding: Query embedding vector
        k: Number of results
        filter: Metadata filter dictionary
        include_distances: Include distance scores in results
        
    Returns:
        Dictionary dengan full results dari ChromaDB
    """
    try:
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where=filter,
            include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search by vector: {str(e)}")
        return {}


def get_documents_by_ids(
    collection: Any,
    ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Get documents by IDs.
    
    Args:
        collection: ChromaDB collection instance
        ids: List of document IDs
        
    Returns:
        List of document dictionaries
    """
    try:
        results = collection.get(ids=ids)
        
        documents = []
        
        if results and 'documents' in results:
            for i in range(len(results['documents'])):
                doc = {
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                }
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error getting documents by IDs: {str(e)}")
        return []


def filter_by_metadata(
    collection: Any,
    filter: Dict[str, Any],
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter documents by metadata.
    
    Args:
        collection: ChromaDB collection instance
        filter: Metadata filter dictionary
        limit: Maximum number of results
        
    Returns:
        List of filtered documents
    """
    try:
        results = collection.get(
            where=filter,
            limit=limit
        )
        
        documents = []
        
        if results and 'documents' in results:
            for i in range(len(results['documents'])):
                doc = {
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                }
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error filtering by metadata: {str(e)}")
        return []
