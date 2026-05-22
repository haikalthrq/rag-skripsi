#!/bin/bash
# Embed maxmin_semantic files satu per satu dengan proses Python baru
# Agar GPU memory dibersihkan antar file

set -e

cd /workspace/rag-skripsi

FILES=$(find data/chunked/maxmin_semantic -name "*_chunks.json" | sort)

echo "Embedding maxmin_semantic files one-by-one..."
for f in $FILES; do
    fname=$(basename "$f")
    outname="${fname%.json}_embeddings.json"
    if [ -f "data/embeddings/maxmin_semantic/$outname" ]; then
        echo "[SKIP] $outname"
        continue
    fi
    echo "[EMBED] $fname"
    /venv/main/bin/python3 -c "
from src.embedding.embed_chunks import embed_all_chunks
stats = embed_all_chunks(
    chunked_dir='data/chunked',
    output_dir='data/embeddings',
    mode='huggingface',
    hf_model_name='models/Qwen3-Embedding-4B',
    device='cuda',
    methods=['maxmin_semantic']
)
" 2>&1 | grep -E "$fname|ERROR|Processed"
done

echo "Done!"
