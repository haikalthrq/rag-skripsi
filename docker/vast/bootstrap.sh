#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RAG_PROJECT_ROOT:-/workspace/rag-skripsi}"

ensure_directory() {
  local path="$1"
  if [ -L "${path}" ]; then
    rm "${path}"
  elif [ -e "${path}" ] && [ ! -d "${path}" ]; then
    echo "Refusing to replace non-directory path: ${path}" >&2
    exit 1
  fi
  mkdir -p "${path}"
}

ensure_directory "${PROJECT_ROOT}/models"
ensure_directory "${PROJECT_ROOT}/data/chroma"
ensure_directory "${PROJECT_ROOT}/data/embeddings"

echo "RAG Vast runtime ready"
echo "  project : ${PROJECT_ROOT}"
echo "  models  : ${PROJECT_ROOT}/models"
echo "  chroma  : ${PROJECT_ROOT}/data/chroma"
echo "  embeds  : ${PROJECT_ROOT}/data/embeddings"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
