"""
Embed single chunk JSON file menggunakan HF model.
Digunakan untuk maxmin_semantic yang OOM jika diproses bersamaan.

Usage:
    /venv/main/bin/python3 scripts/embed_single_file.py data/chunked/maxmin_semantic/FILE.json
"""
import sys
import json
import logging
from pathlib import Path

# Add repo root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        print("Usage: python embed_single_file.py <path_to_chunk_json>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)

    # Determine output path
    method_dir = input_file.parent.name
    out_name = input_file.stem + "_embeddings.json"
    output_dir = Path("data/embeddings") / method_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / out_name

    if output_file.exists():
        logger.info(f"[SKIP] Already exists: {output_file}")
        sys.exit(0)

    # Load chunks (bisa dict atau list)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        chunks = data.get('chunks', [])
    else:
        chunks = data
    texts = [c.get('text', '') for c in chunks if isinstance(c, dict) and c.get('text', '').strip()]

    logger.info(f"Loaded {len(texts)} chunks from {input_file.name}")

    if not texts:
        logger.warning("No valid texts found")
        sys.exit(0)

    # Initialize embedder
    from src.embedding.embedder import initialize_hf_embedder
    embedder = initialize_hf_embedder(
        model_name="models/Qwen3-Embedding-4B",
        device="cuda",
        normalize=True
    )
    if embedder is None:
        logger.error("Failed to initialize embedder")
        sys.exit(1)

    # Embed in smaller sub-batches to avoid OOM
    all_embeddings = []
    sub_batch_size = 1  # Batch size 1 untuk chunk panjang maxmin_semantic
    for i in range(0, len(texts), sub_batch_size):
        batch = texts[i:i + sub_batch_size]
        emb = embedder.embed(batch)
        all_embeddings.extend(emb.tolist())
        logger.info(f"  Batch {i // sub_batch_size + 1}/{(len(texts) - 1) // sub_batch_size + 1}: {len(batch)} chunks")

    # Save
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_data = {
        'metadata': {
            'source_file': input_file.name,
            'chunking_method': method_dir,
            'num_chunks': len(texts),
            'embedding_dim': len(all_embeddings[0]) if all_embeddings else 0,
        },
        'chunks': chunks,
        'embeddings': all_embeddings
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[OK] Saved {len(all_embeddings)} embeddings to {output_file}")


if __name__ == "__main__":
    main()
