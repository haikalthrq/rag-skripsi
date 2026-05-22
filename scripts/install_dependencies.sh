#!/bin/bash
# Install semua dependencies untuk RAG pipeline di RTX 3090 + CUDA
# Jalankan: bash scripts/install_dependencies.sh
#
# Urutan penting:
# 1. PyTorch dengan CUDA 12.1 (kompatibel dengan driver CUDA 13.2)
# 2. transformers + accelerate + sentence-transformers
# 3. llama-cpp-python dengan CUDA (jika belum)
# 4. Dependencies lainnya dari requirements.txt

set -e

PYTHON=/venv/main/bin/python3
PIP=/venv/main/bin/pip

echo "============================================================"
echo "  INSTALL RAG SKRIPSI DEPENDENCIES (RTX 3090 + CUDA)"
echo "============================================================"
echo "  Python : $($PYTHON --version)"
echo "  Pip    : $($PIP --version)"
echo ""

# Step 1: PyTorch dengan CUDA 12.1
echo "[1/5] Installing PyTorch (CUDA 12.1)..."
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

echo ""
echo "  Verifikasi CUDA:"
$PYTHON -c "import torch; print(f'  torch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 2: transformers, accelerate, sentence-transformers
echo ""
echo "[2/5] Installing transformers + accelerate + sentence-transformers..."
$PIP install \
    "transformers>=4.51.0" \
    "accelerate>=0.26.0" \
    "sentence-transformers>=2.7.0" \
    "huggingface_hub>=0.23.0" \
    --quiet

# Step 3: ChromaDB + vector store
echo ""
echo "[3/5] Installing ChromaDB..."
$PIP install chromadb --quiet

# Step 4: Core dependencies
echo ""
echo "[4/5] Installing core dependencies..."
$PIP install \
    numpy scipy pandas openpyxl tqdm \
    langchain-text-splitters nltk \
    PyMuPDF pdfplumber \
    streamlit \
    python-dotenv \
    ragas rank-eval rapidfuzz sacrebleu rouge-score \
    loguru \
    --quiet

# Step 5: Cek llama-cpp-python (CUDA)
echo ""
echo "[5/5] Checking llama-cpp-python..."
if $PYTHON -c "from llama_cpp import Llama; print('  llama-cpp-python: OK')" 2>/dev/null; then
    echo "  Already installed, checking CUDA support..."
    $PYTHON -c "
from llama_cpp import Llama
import llama_cpp
print(f'  Version: {llama_cpp.__version__}')
" 2>/dev/null || true
else
    echo "  Installing llama-cpp-python with CUDA (RTX 3090, sm86)..."
    CMAKE_ARGS="-DGGML_CUDA=on" $PIP install llama-cpp-python --force-reinstall --no-cache-dir --quiet
fi

echo ""
echo "============================================================"
echo "  INSTALLATION COMPLETE"
echo "============================================================"
$PYTHON -c "
import torch
print(f'  PyTorch     : {torch.__version__}')
print(f'  CUDA        : {torch.version.cuda}')
print(f'  GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')

import transformers
print(f'  transformers: {transformers.__version__}')

import sentence_transformers
print(f'  sent-transf : {sentence_transformers.__version__}')

import chromadb
print(f'  chromadb    : {chromadb.__version__}')
"

echo ""
echo "Langkah selanjutnya:"
echo "  1. Download models   : python scripts/download_embedding_model.py"
echo "  2.                     python scripts/download_generator_model.py"
echo "  3. Embed chunks      : python -m src.embedding.embed_chunks (atau python embed_chunks.py)"
echo "  4. Load ChromaDB     : python load_to_chroma.py"
echo "  5. Jalankan Streamlit: streamlit run src/streamlit/rag_chat.py"
