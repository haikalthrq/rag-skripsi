#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RAG_PROJECT_ROOT:-/opt/workspace-internal/rag-skripsi}"
ASSET_ROOT="${RAG_ASSET_ROOT:-/workspace}"

mkdir -p \
  "${ASSET_ROOT}/models" \
  "${ASSET_ROOT}/chroma" \
  "${ASSET_ROOT}/embeddings" \
  "${PROJECT_ROOT}/data"

link_directory() {
  local link_path="$1"
  local target_path="$2"

  mkdir -p "$(dirname "${link_path}")"
  if [ -L "${link_path}" ]; then
    rm "${link_path}"
  elif [ -e "${link_path}" ]; then
    if [ -n "$(find "${link_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "Refusing to replace non-empty directory: ${link_path}" >&2
      exit 1
    fi
    rmdir "${link_path}"
  fi
  ln -s "${target_path}" "${link_path}"
}

link_directory "${PROJECT_ROOT}/models" "${ASSET_ROOT}/models"
link_directory "${PROJECT_ROOT}/data/chroma" "${ASSET_ROOT}/chroma"
link_directory "${PROJECT_ROOT}/data/embeddings" "${ASSET_ROOT}/embeddings"

echo "RAG Vast runtime ready"
echo "  project : ${PROJECT_ROOT}"
echo "  assets  : ${ASSET_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
