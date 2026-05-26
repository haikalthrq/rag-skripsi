"""
Load embeddings dari data/embeddings/ ke ChromaDB.

Usage:
    python scripts/load_embeddings_to_chroma.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chroma.client import initialize_chroma_client
from src.chroma.loader import load_all_embeddings_to_chroma

def main():
    print("=" * 70)
    print("LOADING EMBEDDINGS TO CHROMADB")
    print("=" * 70)
    
    # Initialize ChromaDB client
    chroma_path = ROOT / "data/chroma"
    embeddings_dir = ROOT / "data/embeddings"
    
    print(f"ChromaDB path: {chroma_path}")
    print(f"Embeddings dir: {embeddings_dir}")
    print()
    
    client = initialize_chroma_client(persist_directory=str(chroma_path))
    if client is None:
        print("[ERROR] Failed to initialize ChromaDB client")
        sys.exit(1)
    
    # Load all embeddings
    stats = load_all_embeddings_to_chroma(
        client=client,
        embeddings_dir=str(embeddings_dir),
        batch_size=1000,
        methods=['element_based', 'maxmin_semantic', 'recursive'],
        reset_collections=True  # Reset collections sebelum load
    )
    
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    
    if stats.get('failed', 0) > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
