# Vast.ai Docker Runtime

`Dockerfile.vast` packages the Python/CUDA runtime for the HuggingFace
RTX 3090 workflow. It deliberately does not contain model weights, ChromaDB,
or precomputed embeddings.

## Build

Build and publish the image from the repository root. Replace the image name
with a registry that the Vast template can access:

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.vast \
  -t ghcr.io/OWNER/rag-skripsi:vast-rtx3090 \
  --push .
```

The base image is pinned to:

```text
vastai/pytorch:2.6.0-cuda-12.6.3-py312
```

## Vast Template

Configure the template with:

- GPU: RTX 3090 with at least 24 GB VRAM.
- Image: the published `rag-skripsi:vast-rtx3090` image.
- Runtime: SSH direct.
- Persistent storage: preserve the direct project asset directories.
- On-start command: `/usr/local/bin/rag-vast-bootstrap`.

Vast SSH mode replaces the image entrypoint, so the bootstrap should be set as
the template on-start command. It creates the direct asset directories:

```text
/workspace/rag-skripsi/models
/workspace/rag-skripsi/data/chroma
/workspace/rag-skripsi/data/embeddings
```

## First Run

Run from the project directory after SSH connects:

```bash
cd /workspace/rag-skripsi
python scripts/download_vast_assets.py --asset all --workers 4
python scripts/run_generation_eval.py --top_k 1
```

The first command downloads the public Google Drive assets to the persistent
volume. It can be interrupted and rerun; `.part` files are resumed. Do not run
`scripts/load_embeddings_to_chroma.py` when the Chroma folder was downloaded
successfully because the persistent Chroma collections are already present.

## Later Runs

After the volume and image are reused, only run:

```bash
cd /workspace/rag-skripsi
python scripts/run_generation_eval.py
```

The Docker image contains dependencies; the direct project directories contain
models, ChromaDB, and embeddings.
