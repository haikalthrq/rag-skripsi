# Backup Downloaders

These scripts are kept for the laptop setup only. They write the GGUF embedding
model and FP8 generator to the repository-root `models/` directory.

Do not use them for Vast.ai RTX 3090 runs. Use the active entry point instead:

```bash
python scripts/download_vast_assets.py
```

The faculty source-code ZIP excludes these legacy scripts and all model files.
