"""
Download generator model ke models/ lokal.

Model : Qwen3-4B-Instruct-2507
Repo  : Qwen/Qwen3-4B-Instruct-2507 (HuggingFace)
Format: Safetensors, BF16 tensor type
Ukuran: ~8-9 GB

Catatan model:
- Model ini adalah non-thinking mode only.
- Tidak menghasilkan blok <think> pada output.
- Parameter enable_thinking=False tidak diperlukan.
- Gunakan transformers >= 4.51.0 untuk menghindari KeyError: 'qwen3'.

Token HuggingFace (opsional):
- Jika repo memerlukan autentikasi, set environment variable:
    export HF_TOKEN=hf_xxxxxxxxxxxx
  atau:
    export HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxx
- Jika token tidak ada, script tetap mencoba download tanpa token.

Setelah download selesai, pipeline akan otomatis menggunakan model
lokal dari models/Qwen3-4B-Instruct-2507/.

Usage:
  python scripts/download_generator_model.py
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

ROOT           = Path(__file__).resolve().parent.parent
GENERATOR_REPO = "Qwen/Qwen3-4B-Instruct-2507"
GENERATOR_DIR  = ROOT / "models/Qwen3-4B-Instruct-2507"


def _check_transformers_version() -> None:
    """Peringatkan jika transformers < 4.51.0 (menyebabkan KeyError: 'qwen3')."""
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("4.51.0"):
            print(f"[WARN] transformers {transformers.__version__} < 4.51.0")
            print("       Upgrade: pip install -U transformers")
    except Exception:
        pass


def main() -> None:
    from huggingface_hub import snapshot_download

    _check_transformers_version()

    print("=" * 60)
    print("  DOWNLOAD GENERATOR MODEL")
    print("=" * 60)
    print(f"  Model : Qwen3-4B-Instruct-2507")
    print(f"  Repo  : {GENERATOR_REPO}")
    print(f"  Format: Safetensors, BF16")
    print(f"  Target: {GENERATOR_DIR}\n")

    if GENERATOR_DIR.exists() and any(GENERATOR_DIR.glob("*.safetensors")):
        size_gb = sum(
            f.stat().st_size for f in GENERATOR_DIR.glob("*.safetensors")
        ) / (1024 ** 3)
        print(f"[SKIP] Sudah ada: {GENERATOR_DIR} ({size_gb:.1f} GB safetensors)")
        return

    # Token opsional dari environment variable — tidak di-hardcode
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if token:
        print(f"[INFO] HF_TOKEN ditemukan, menggunakan autentikasi.")
    else:
        print(f"[INFO] Tidak ada HF_TOKEN, mencoba download tanpa autentikasi.")

    GENERATOR_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id   = GENERATOR_REPO,
        local_dir = str(GENERATOR_DIR),
        token     = token,
    )
    print(f"\n[OK] Generator model tersimpan: {path}")
    print("     Jalankan evaluasi: python scripts/run_generation_eval.py")


if __name__ == "__main__":
    main()
