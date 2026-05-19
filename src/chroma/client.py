"""
ChromaDB client initialization dan collection management.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment, misc]
    _CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_PERSIST_DIRECTORY = "data/chroma"


def _recover_stale_sqlite_journal(persist_path: Path) -> None:
    """
    Move a stale Chroma SQLite rollback journal out of the way.

    On Windows, an interrupted Streamlit/Chroma process can leave
    chroma.sqlite3-journal behind. The main DB can still be valid, but normal
    SQLite open tries to rollback the journal and may fail with disk I/O error.
    We only quarantine the journal when the DB passes immutable quick_check.
    """
    db_path = persist_path / "chroma.sqlite3"
    journal_path = persist_path / "chroma.sqlite3-journal"

    if not db_path.exists() or not journal_path.exists():
        return

    try:
        db_uri = f"file:{db_path.resolve().as_posix()}?immutable=1"
        with sqlite3.connect(db_uri, uri=True, timeout=5) as conn:
            result = conn.execute("PRAGMA quick_check;").fetchone()
        if not result or result[0] != "ok":
            logger.warning(
                "ChromaDB journal exists but DB quick_check is not OK; "
                "leaving journal in place."
            )
            return

        backup_dir = persist_path.parent.parent / "backup" / "chroma_journals"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"chroma.sqlite3-journal_{stamp}"
        journal_path.replace(backup_path)
        logger.warning(f"Moved stale ChromaDB journal to: {backup_path}")
    except Exception as e:
        logger.warning(f"Failed to recover stale ChromaDB journal: {e}")


def initialize_chroma_client(
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    in_memory: bool = False
) -> Optional[Any]:
    """
    Initialize ChromaDB client.
    
    Menggunakan persistent storage by default (recommended untuk production).
    
    Args:
        persist_directory: Directory untuk persistent storage
        in_memory: Jika True, gunakan in-memory storage (tidak persistent)
        
    Returns:
        ChromaDB client instance atau None jika gagal
    """
    if not _CHROMADB_AVAILABLE:
        logger.error("chromadb tidak terinstall. Install dengan: pip install chromadb")
        return None
    
    try:
        if in_memory:
            logger.info("Initializing ChromaDB client (IN-MEMORY mode)")
            logger.warning("⚠️ Data akan hilang setelah program berhenti!")
            client = chromadb.Client()
        else:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            _recover_stale_sqlite_journal(persist_path)
            
            logger.info(f"Initializing ChromaDB client (PERSISTENT mode)")
            logger.info(f"  - Storage path: {persist_path.absolute()}")
            
            try:
                client = chromadb.PersistentClient(
                    path=str(persist_path.absolute())
                )
            except Exception as e:
                if "disk I/O error" not in str(e):
                    raise
                _recover_stale_sqlite_journal(persist_path)
                client = chromadb.PersistentClient(
                    path=str(persist_path.absolute())
                )
        
        logger.info("✓ ChromaDB client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {str(e)}")
        return None


def get_or_create_collection(
    client: Any,
    collection_name: str,
    embedding_dim: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Get existing collection atau create new jika belum ada.
    
    Args:
        client: ChromaDB client instance
        collection_name: Nama collection
        embedding_dim: Dimensi embedding (optional, untuk validasi)
        metadata: Metadata untuk collection
        
    Returns:
        Collection instance atau None jika gagal
    """
    try:
        # Try to get existing collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✓ Found existing collection: {collection_name}")
            logger.info(f"  - Document count: {collection.count()}")
            return collection
        except Exception:
            # Collection doesn't exist, create new
            logger.info(f"Creating new collection: {collection_name}")
            
            collection_metadata = metadata or {}
            if embedding_dim:
                collection_metadata['embedding_dim'] = embedding_dim
            
            collection = client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            
            logger.info(f"✓ Collection created: {collection_name}")
            return collection
            
    except Exception as e:
        logger.error(f"Error getting/creating collection {collection_name}: {str(e)}")
        return None


def delete_collection(client: Any, collection_name: str) -> bool:
    """
    Delete collection dari ChromaDB.
    
    Args:
        client: ChromaDB client instance
        collection_name: Nama collection yang akan dihapus
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"✓ Collection deleted: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False


def list_collections(client: Any) -> List[str]:
    """
    List semua collections di ChromaDB.
    
    Args:
        client: ChromaDB client instance
        
    Returns:
        List nama collections
    """
    try:
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        logger.info(f"Found {len(collection_names)} collections:")
        for name in collection_names:
            col = client.get_collection(name)
            logger.info(f"  - {name}: {col.count()} documents")
        
        return collection_names
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return []


def get_collection_info(client: Any, collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information tentang collection.
    
    Args:
        client: ChromaDB client instance
        collection_name: Nama collection
        
    Returns:
        Dictionary dengan info collection atau None jika tidak ditemukan
    """
    try:
        collection = client.get_collection(name=collection_name)
        
        info = {
            'name': collection.name,
            'count': collection.count(),
            'metadata': collection.metadata
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting collection info {collection_name}: {str(e)}")
        return None


def reset_collection(client: Any, collection_name: str) -> Optional[Any]:
    """
    Reset collection (delete dan recreate).
    
    Args:
        client: ChromaDB client instance
        collection_name: Nama collection
        
    Returns:
        New collection instance atau None jika gagal
    """
    try:
        # Get metadata before deleting
        try:
            old_collection = client.get_collection(name=collection_name)
            metadata = old_collection.metadata
        except Exception:
            metadata = {}
        
        # Delete collection
        delete_collection(client, collection_name)
        
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        logger.info(f"✓ Collection reset: {collection_name}")
        return collection
        
    except Exception as e:
        logger.error(f"Error resetting collection {collection_name}: {str(e)}")
        return None
