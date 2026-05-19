"""
Download embedding model ke models/ lokal.

Model : Qwen3-Embedding-4B-Q8_0.gguf
Repo  : Qwen/Qwen3-Embedding-4B-GGUF (HuggingFace)
Ukuran: ~4.3 GB

Usage:
  python scripts/download_embedding_model.py
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

ROOT           = Path(__file__).resolve().parent.parent
EMBEDDING_REPO = "Qwen/Qwen3-Embedding-4B-GGUF"
EMBEDDING_FILE = "Qwen3-Embedding-4B-Q8_0.gguf"
EMBEDDING_PATH = ROOT / "models" / EMBEDDING_FILE


def main() -> None:
    from huggingface_hub import hf_hub_download

    print("=" * 60)
    print("  DOWNLOAD EMBEDDING MODEL")
    print("=" * 60)
    print(f"  Repo  : {EMBEDDING_REPO}")
    print(f"  File  : {EMBEDDING_FILE}")
    print(f"  Target: {EMBEDDING_PATH}\n")

    if EMBEDDING_PATH.exists():
        size_gb = EMBEDDING_PATH.stat().st_size / (1024 ** 3)
        print(f"[SKIP] Sudah ada: {EMBEDDING_PATH} ({size_gb:.1f} GB)")
        return

    path = hf_hub_download(
        repo_id   = EMBEDDING_REPO,
        filename  = EMBEDDING_FILE,
        local_dir = str(ROOT / "models"),
    )
    size_gb = Path(path).stat().st_size / (1024 ** 3)
    print(f"\n[OK] Embedding model tersimpan: {path} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
