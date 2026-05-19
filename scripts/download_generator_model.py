"""
Download generator model ke models/ lokal.

Model : Qwen3-4B-Instruct-2507-FP8
Repo  : Qwen/Qwen3-4B-Instruct-2507-FP8 (HuggingFace)
Ukuran: ~8 GB (FP8 quantized)

Setelah download selesai, run_generation_eval.py akan otomatis
menggunakan model lokal dari models/Qwen3-4B-Instruct-2507-FP8/.

Usage:
  python scripts/download_generator_model.py
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

ROOT          = Path(__file__).resolve().parent.parent
GENERATOR_REPO = "Qwen/Qwen3-4B-Instruct-2507-FP8"
GENERATOR_DIR  = ROOT / "models/Qwen3-4B-Instruct-2507-FP8"


def main() -> None:
    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("  DOWNLOAD GENERATOR MODEL")
    print("=" * 60)
    print(f"  Repo  : {GENERATOR_REPO}")
    print(f"  Target: {GENERATOR_DIR}")
    print(f"  Ukuran: ~8 GB (FP8 quantized)\n")

    if GENERATOR_DIR.exists() and any(GENERATOR_DIR.glob("*.safetensors")):
        size_gb = sum(
            f.stat().st_size for f in GENERATOR_DIR.glob("*.safetensors")
        ) / (1024 ** 3)
        print(f"[SKIP] Sudah ada: {GENERATOR_DIR} ({size_gb:.1f} GB safetensors)")
        return

    GENERATOR_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id   = GENERATOR_REPO,
        local_dir = str(GENERATOR_DIR),
    )
    print(f"\n[OK] Generator model tersimpan: {path}")
    print("     Jalankan evaluasi: python scripts/run_generation_eval.py")


if __name__ == "__main__":
    main()
