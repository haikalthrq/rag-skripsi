# Developer Handoff

This document defines the supported source-code handoff for the thesis project.
Run commands from the repository root with Python 3.11.9.

## Installation Profiles

Create and activate a virtual environment, then select the required profile:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Development and tests:

```bash
pip install -r requirements-dev.txt
```

Notebook visualization:

```bash
pip install -r requirements-visualization.txt
```

GGUF backend, CPU build:

```bash
pip install -r requirements-gguf.txt
```

GGUF backend with CUDA requires a platform-specific `llama-cpp-python` build.
For a typical source build:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

The dependency files intentionally list direct, high-level dependencies rather
than claiming an exact lock. CUDA, PyTorch, `llama-cpp-python`, and system OCR
packages depend on the target hardware and operating system. Before reproducing
reported measurements, create a lock from a validated Python 3.11.9 environment
and record the CUDA/driver versions.

## System Dependencies

- Poppler and Tesseract are required by Unstructured PDF strategies.
- Tesseract language data `ind` is required by the default element-based OCR
  path.
- NVIDIA GPU and a compatible CUDA/PyTorch stack are recommended for model
  inference.

## Runtime Assets

The source ZIP does not include large runtime assets. Download them separately:

```bash
python scripts/download_vast_assets.py --asset all
```

Expected destinations:

- Models: `models/`
- Embeddings: `data/embeddings/`
- ChromaDB: `data/chroma/`

The downloader validates file size but not a cryptographic checksum. For strict
reproduction, record SHA-256 checksums after obtaining the approved asset set.

## Supported Entry Points

```bash
python -m src.preprocessing.pipeline --input data/raw --output data/cleaned
python src/chunking/element_based.py --help
python src/chunking/maxmin_chunker.py --help
python src/chunking/recursive_split.py --help
python -m src.embedding.embed_chunks --help
python scripts/load_embeddings_to_chroma.py
python scripts/run_retrieval_eval.py --help
python scripts/run_generation_eval.py --help
streamlit run src/streamlit/rag_chat.py
```

`src/streamlit/rag_chat.py` uses the curated 30-question workbook as a question
selector; it is not a free-text chat interface. The separate annotation app
requires an evidence-aware candidate workbook that is not included in the
faculty ZIP.

## Verification

```bash
python -m compileall -q src scripts tests
python -m pytest tests/test_generation_eval_timing.py tests/test_evaluation.py tests/test_create_faculty_submission.py
```

The current automated tests cover evaluation metrics, latency persistence, and
the submission archive policy. They do not execute model inference or the full
PDF-to-RAG pipeline.

## Faculty ZIP

Build the curated package with:

```bash
python scripts/create_faculty_submission.py
```

The command writes a ZIP and a matching `.sha256` file under `dist/`. Notebook
outputs and execution counters are stripped in the archive. `PACKAGE_INFO.md`
and `SHA256SUMS.txt` inside the ZIP record provenance and file checksums.

The package includes source code, active scripts, tests, current documentation,
sanitized notebooks, minimal ground truth, the README-linked Top-1 to Top-10
analysis figures/tables, and the canonical latency summary. It excludes all
large or local runtime artifacts listed in `DISTRIBUTION_NOTES.md`.
