"""
Load embeddings dari JSON ke ChromaDB collections.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .client import get_or_create_collection

logger = logging.getLogger(__name__)

# Batch size untuk add documents
DEFAULT_BATCH_SIZE = 1000


def batch_add_documents(
    collection: Any,
    ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> bool:
    """
    Add documents ke collection dalam batches.
    
    Args:
        collection: ChromaDB collection instance
        ids: List document IDs
        embeddings: Numpy array embeddings
        documents: List document texts
        metadatas: List metadata dictionaries
        batch_size: Size per batch
        
    Returns:
        True jika berhasil, False jika gagal
    """
    def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata untuk ChromaDB compatibility."""
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert list to string
            if isinstance(value, list):
                if value and isinstance(value[0], str):
                    cleaned[key] = " | ".join(value)  # Join dengan separator
                else:
                    cleaned[key] = str(value)
            
            # Keep primitives
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            
            # Skip complex objects (coordinates, etc)
            elif "object at" not in str(value):
                cleaned[key] = str(value)
        
        return cleaned
    
    try:
        total = len(ids)
        
        if total == 0:
            logger.warning("No documents to add")
            return True
        
        logger.info(f"Adding {total} documents in batches of {batch_size}...")
        
        # Clean metadata untuk semua documents
        if metadatas:
            metadatas = [clean_metadata(m) for m in metadatas]
        
        # Process in batches
        for i in tqdm(range(0, total, batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, total)
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx] if metadatas else None
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        
        logger.info(f"✓ Added {total} documents successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error adding documents to collection: {str(e)}")
        return False


def load_to_chroma(
    client: Any,
    embedding_file: str,
    collection_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset_collection: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Load embeddings dari single JSON file ke ChromaDB.
    
    Args:
        client: ChromaDB client instance
        embedding_file: Path ke embedding JSON file
        collection_name: Nama collection untuk store embeddings
        batch_size: Batch size untuk add documents
        reset_collection: Reset collection sebelum add (delete existing data)
        
    Returns:
        Dictionary dengan stats atau None jika gagal
    """
    try:
        file_path = Path(embedding_file)
        
        if not file_path.exists():
            logger.error(f"Embedding file tidak ditemukan: {embedding_file}")
            return None
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Loading: {file_path.name}")
        logger.info(f"{'='*70}")
        
        # Load embedding data
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        embeddings_list = data.get('embeddings', [])
        chunks = data.get('chunks', [])
        
        if len(embeddings_list) == 0:
            logger.error("No embeddings found in file")
            return None
        
        logger.info(f"Loaded {len(embeddings_list)} embeddings")
        logger.info(f"  - Embedding dim: {metadata.get('embedding_dim', 'unknown')}")
        logger.info(f"  - Chunking method: {metadata.get('chunking_method', 'unknown')}")
        
        # Convert embeddings to numpy
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate ID (gunakan original_index jika ada, atau generate)
            chunk_id = f"{file_path.stem}_{chunk.get('original_index', chunk.get('embedding_index', 0))}"
            ids.append(chunk_id)
            
            # Document text
            documents.append(chunk.get('text', ''))
            
            # Metadata
            chunk_metadata = chunk.get('metadata', {}).copy()
            chunk_metadata['source_file'] = metadata.get('source_file', file_path.stem)
            chunk_metadata['chunking_method'] = metadata.get('chunking_method', 'unknown')
            metadatas.append(chunk_metadata)
        
        # Get or create collection
        if reset_collection:
            from .client import reset_collection as reset_col
            collection = reset_col(client, collection_name)
        else:
            collection = get_or_create_collection(
                client=client,
                collection_name=collection_name,
                embedding_dim=metadata.get('embedding_dim'),
                metadata={
                    'chunking_method': metadata.get('chunking_method'),
                    'embedding_model': metadata.get('embedding_model'),
                }
            )
        
        if collection is None:
            return None
        
        # Add documents to collection
        success = batch_add_documents(
            collection=collection,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            batch_size=batch_size
        )
        
        if not success:
            return None
        
        # Return stats
        stats = {
            'file': file_path.name,
            'collection': collection_name,
            'documents_added': len(ids),
            'embedding_dim': metadata.get('embedding_dim'),
            'chunking_method': metadata.get('chunking_method'),
        }
        
        logger.info("✓ Loading completed successfully")
        logger.info(f"  - Collection: {collection_name}")
        logger.info(f"  - Total documents: {collection.count()}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error loading to ChromaDB: {str(e)}")
        return None


def load_all_embeddings_to_chroma(
    client: Any,
    embeddings_dir: str = "data/embeddings",
    batch_size: int = DEFAULT_BATCH_SIZE,
    methods: Optional[List[str]] = None,
    reset_collections: bool = False
) -> Dict[str, Any]:
    """
    Load semua embeddings ke ChromaDB (3 metode chunking).
    
    Membuat separate collections untuk setiap chunking method:
    - collection_element_based
    - collection_maxmin_semantic
    - collection_recursive
    
    Args:
        client: ChromaDB client instance
        embeddings_dir: Directory berisi embeddings
        batch_size: Batch size untuk add documents
        methods: List metode chunking untuk diload (None = semua)
        reset_collections: Reset collections sebelum load
        
    Returns:
        Dictionary dengan statistics
    """
    logger.info("="*70)
    logger.info("LOADING EMBEDDINGS TO CHROMADB")
    logger.info("="*70)
    logger.info(f"Source directory: {embeddings_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")
    
    if methods is None:
        methods = ['element_based', 'maxmin_semantic', 'recursive']
    
    stats = {
        'total_files': 0,
        'loaded': 0,
        'failed': 0,
        'by_method': {}
    }
    
    for method in methods:
        method_dir = Path(embeddings_dir) / method
        
        if not method_dir.exists():
            logger.warning(f"Directory not found: {method_dir}")
            continue
        
        # Find all embedding files
        embed_files = list(method_dir.glob("*_embeddings.json"))
        
        if len(embed_files) == 0:
            logger.warning(f"No embedding files found in {method_dir}")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing method: {method.upper()}")
        logger.info(f"Found {len(embed_files)} files")
        logger.info(f"{'='*70}")
        
        collection_name = f"collection_{method}"
        
        method_stats = {
            'total': len(embed_files),
            'loaded': 0,
            'failed': 0,
            'documents': 0
        }
        
        # Load each file
        for i, embed_file in enumerate(embed_files, 1):
            logger.info(f"\n[{i}/{len(embed_files)}] Processing: {embed_file.name}")
            
            result = load_to_chroma(
                client=client,
                embedding_file=str(embed_file),
                collection_name=collection_name,
                batch_size=batch_size,
                reset_collection=(reset_collections and i == 1)  # Reset only on first file
            )
            
            if result:
                method_stats['loaded'] += 1
                method_stats['documents'] += result['documents_added']
                stats['loaded'] += 1
            else:
                method_stats['failed'] += 1
                stats['failed'] += 1
        
        stats['by_method'][method] = method_stats
        stats['total_files'] += method_stats['total']
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("LOADING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Loaded: {stats['loaded']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    
    for method, method_stats in stats['by_method'].items():
        logger.info(f"{method}:")
        logger.info(f"  Files loaded: {method_stats['loaded']}/{method_stats['total']}")
        logger.info(f"  Total documents: {method_stats['documents']}")
        if method_stats['failed'] > 0:
            logger.info(f"  Failed: {method_stats['failed']}")
    
    logger.info(f"{'='*70}")
    
    return stats
