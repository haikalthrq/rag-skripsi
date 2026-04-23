"""
Script untuk load embeddings ke ChromaDB.

Usage:
    # Load all embeddings
    python load_to_chroma.py

    # Load specific method
    python load_to_chroma.py --methods maxmin_semantic

    # Reset collections
    python load_to_chroma.py --reset

    # Custom paths
    python load_to_chroma.py --input data/embeddings --output data/chroma
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chroma.client import initialize_chroma_client, list_collections  # type: ignore[import-not-found]
from chroma.loader import load_all_embeddings_to_chroma  # type: ignore[import-not-found]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load embeddings ke ChromaDB vector database"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/embeddings',
        help='Directory berisi embeddings (default: data/embeddings)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/chroma',
        help='Directory untuk ChromaDB storage (default: data/chroma)'
    )
    
    parser.add_argument(
        '--methods', '-m',
        nargs='+',
        choices=['element_based', 'maxmin_semantic', 'recursive'],
        help='Specific chunking methods untuk diload (default: semua)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size untuk add documents (default: 1000)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset collections sebelum load (delete existing data)'
    )
    
    parser.add_argument(
        '--in-memory',
        action='store_true',
        help='Use in-memory storage (tidak persistent)'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Hanya list existing collections tanpa load data'
    )
    
    args = parser.parse_args()
    
    # Initialize ChromaDB client
    logger.info("="*70)
    logger.info("CHROMADB INTEGRATION")
    logger.info("="*70)
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Storage directory: {args.output}")
    logger.info(f"In-memory mode: {args.in_memory}")
    logger.info("")
    
    client = initialize_chroma_client(
        persist_directory=args.output,
        in_memory=args.in_memory
    )
    
    if client is None:
        logger.error("Failed to initialize ChromaDB client")
        sys.exit(1)
    
    # List only mode
    if args.list_only:
        logger.info("")
        list_collections(client)
        sys.exit(0)
    
    # Load embeddings
    stats = load_all_embeddings_to_chroma(
        client=client,
        embeddings_dir=args.input,
        batch_size=args.batch_size,
        methods=args.methods,
        reset_collections=args.reset
    )
    
    # List collections after loading
    logger.info("")
    logger.info("="*70)
    logger.info("CURRENT COLLECTIONS")
    logger.info("="*70)
    list_collections(client)
    
    # Exit code based on results
    if stats.get('failed', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)
