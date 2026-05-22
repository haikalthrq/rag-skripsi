"""
Download embedding model ke models/ lokal.

Model : Qwen/Qwen3-Embedding-4B (HuggingFace, non-GGUF)
Repo  : Qwen/Qwen3-Embedding-4B
Ukuran: ~8 GB (BF16)

Usage:
  python scripts/download_embedding_model.py
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

ROOT           = Path(__file__).resolve().parent.parent
EMBEDDING_REPO = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIR  = ROOT / "models" / "Qwen3-Embedding-4B"


def main() -> None:
    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("  DOWNLOAD EMBEDDING MODEL")
    print("=" * 60)
    print(f"  Repo  : {EMBEDDING_REPO}")
    print(f"  Target: {EMBEDDING_DIR}")
    print(f"  Ukuran: ~8 GB (BF16)\n")

    if EMBEDDING_DIR.exists() and any(EMBEDDING_DIR.glob("*.safetensors")):
        size_gb = sum(
            f.stat().st_size for f in EMBEDDING_DIR.glob("*.safetensors")
        ) / (1024 ** 3)
        print(f"[SKIP] Sudah ada: {EMBEDDING_DIR} ({size_gb:.1f} GB safetensors)")
        return

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id   = EMBEDDING_REPO,
        local_dir = str(EMBEDDING_DIR),
    )
    print(f"\n[OK] Embedding model tersimpan: {path}")


if __name__ == "__main__":
    main()
