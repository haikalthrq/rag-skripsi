# Embedding Module

Modul untuk generate vector embeddings dari chunks hasil chunking menggunakan Qwen3-Embedding-4B model.

## Catatan Implementasi Saat Ini

Entry point berada di module, bukan `embed_chunks.py` pada root:

```bash
python -m src.embedding.embed_chunks --help
```

Sebelum embedding, pipeline melakukan transformasi tambahan:

- Tabel element-based dapat diperkaya dari `text_as_html` dan mengubah teks
  chunk yang kemudian disimpan.
- MaxMin dan recursive mendapat prefix maksimal 200 karakter dari chunk
  sebelumnya. Prefix hanya dipakai untuk embedding dan tidak disimpan sebagai
  teks chunk.

Akibatnya, teks chunk tersimpan tidak selalu identik dengan teks persis yang
masuk ke model embedding.

Parameter `batch_size` pada `QwenEmbedder.embed()` belum diteruskan ke backend.
GGUF diproses satu per satu, sedangkan HuggingFace memakai batch size 1 dan
memotong input di atas 4096 karakter untuk mengurangi risiko OOM. Opsi
`--device` terutama relevan untuk mode HuggingFace.

Mode GGUF tetap mengimpor `torch` melalui entry point. Pastikan dependency root
terpasang meskipun model embedding yang dipilih adalah GGUF.

## Features

- ✅ **Dual Mode**: Support GGUF (via llama-cpp) dan HuggingFace (via sentence-transformers)
- ✅ **Auto Cleaning**: Clean whitespace dan filter chunk kosong otomatis
- ✅ **Batch Processing**: Process semua chunks dari 3 metode chunking sekaligus
- ✅ **Flexible Output**: JSON format (dengan metadata) dan optional numpy format
- ✅ **Skip Existing**: Skip file yang sudah di-embed untuk efficiency
- ✅ **Error Handling**: Robust error handling dan logging

## Struktur

```
src/embedding/
├── __init__.py          # Package initialization
├── embedder.py          # QwenEmbedder class (GGUF + HuggingFace)
├── io.py                # Load/save utilities
└── embed_chunks.py      # Main entrypoint
```

## Pipeline

```
Input: data/chunked/{method}/*_chunks.json
  ↓
1. Load chunks dari JSON
  ↓
2. Clean whitespace & filter empty chunks
  ↓
3. Generate embeddings (Qwen3-Embedding-4B)
  ↓
4. Save embeddings + metadata
  ↓
Output: data/embeddings/{method}/*_embeddings.json
```

## Usage

### 1. Command Line (Recommended)

```bash
# Default: GGUF mode (recommended, efficient)
python -m src.embedding.embed_chunks

# HuggingFace mode
python -m src.embedding.embed_chunks --mode huggingface

# Custom paths
python -m src.embedding.embed_chunks --input data/chunked --output data/embeddings

# Process ulang file existing
python -m src.embedding.embed_chunks --no-skip

# Save juga dalam numpy format
python -m src.embedding.embed_chunks --save-numpy

# Custom GGUF model
python -m src.embedding.embed_chunks --gguf-model models/custom-model.gguf

# CPU only
python -m src.embedding.embed_chunks --mode huggingface --device cpu
```

### 2. Python API

```python
from src.embedding import embed_all_chunks, embed_single_file
from src.embedding import initialize_gguf_embedder, initialize_hf_embedder

# Initialize embedder (GGUF - recommended)
embedder = initialize_gguf_embedder(
    model_path="models/Qwen3-Embedding-4B-Q8_0.gguf",
    normalize=True
)

# Or HuggingFace
embedder = initialize_hf_embedder(
    model_name="Qwen/Qwen3-Embedding-4B",
    device='cuda',
    normalize=True
)

# Embed all chunks (3 metode)
stats = embed_all_chunks(
    chunked_dir="data/chunked",
    output_dir="data/embeddings",
    mode="gguf",
    normalize=True,
    skip_existing=True
)

print(f"Processed: {stats['processed']}")
print(f"Failed: {stats['failed']}")

# Embed single file
result = embed_single_file(
    json_path="data/chunked/maxmin_semantic/file_chunks.json",
    output_dir="data/embeddings",
    embedder=embedder,
    chunking_method="maxmin_semantic"
)
```

### 3. Load Embeddings

```python
from src.embedding import load_embeddings

# Load embeddings dari file
data = load_embeddings("data/embeddings/maxmin_semantic/file_embeddings.json")

print(data['metadata'])        # Metadata (model, dim, timestamp, dll)
print(data['embeddings'])      # Numpy array (n_chunks, embedding_dim)
print(data['chunks'])          # Original chunks dengan embedding_index

# Access embeddings
embeddings = data['embeddings']  # Shape: (n_chunks, 2560)
chunks = data['chunks']

# Get embedding for specific chunk
chunk_idx = 0
chunk_text = chunks[chunk_idx]['text']
chunk_embedding = embeddings[chunk_idx]
```

## Output Format

### JSON Output (`*_embeddings.json`)

```json
{
  "metadata": {
    "source_file": "dokumen_chunks.json",
    "source_path": "data/chunked/maxmin_semantic/dokumen_chunks.json",
    "chunking_method": "maxmin_semantic",
    "embedding_model": "gguf",
    "normalized": true,
    "embedding_dim": 2560,
    "num_chunks": 143,
    "num_original_chunks": 150,
    "timestamp": "2025-12-15T10:30:00"
  },
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // 2560 dimensions
    [0.234, -0.567, 0.890, ...],
    ...
  ],
  "chunks": [
    {
      "text": "Original chunk text...",
      "id": "chunk_1",
      "metadata": {...},
      "embedding_index": 0,
      "original_index": 0
    },
    ...
  ]
}
```

### Numpy Output (Optional, `*_embeddings.npy`)

Binary numpy array format untuk loading cepat:

```python
import numpy as np

embeddings = np.load("data/embeddings/maxmin_semantic/file_embeddings.npy")
print(embeddings.shape)  # (143, 2560)
```

## Supported Chunking Methods

1. **element_based**: Chunks dari `data/chunked/element_based/`
2. **maxmin_semantic**: Chunks dari `data/chunked/maxmin_semantic/`
3. **recursive**: Chunks dari `data/chunked/recursive/`

## Requirements

```bash
# GGUF mode (recommended)
pip install llama-cpp-python numpy

# HuggingFace mode
pip install sentence-transformers transformers torch numpy
```

## Model Information

### GGUF Model (Recommended)
- **Model**: Qwen3-Embedding-4B-Q8_0.gguf
- **Location**: `models/Qwen3-Embedding-4B-Q8_0.gguf`
- **Size**: ~4 GB
- **Embedding Dim**: 2560
- **Advantages**: Efficient memory usage, fast inference

### HuggingFace Model (Fallback)
- **Model**: Qwen/Qwen3-Embedding-4B
- **Size**: ~8 GB (full precision)
- **Embedding Dim**: 2560
- **Advantages**: Standard transformers API

## Performance Tips

1. **Use GGUF mode** untuk efficiency (lebih cepat, less VRAM)
2. **Enable skip_existing** untuk avoid re-processing
3. **Use GPU** untuk faster inference
4. **Normalize embeddings** untuk better similarity comparison
5. **Save numpy format** jika butuh loading cepat untuk downstream tasks

## Error Handling

Modul ini robust terhadap:
- Missing files
- Empty chunks
- JSON decode errors
- Model loading errors
- Memory errors

Semua error di-log dengan detail untuk debugging.

## Examples

### Example 1: Basic Usage

```bash
python -m src.embedding.embed_chunks
```

Output:
```
======================================================================
EMBEDDING CHUNKS PIPELINE
======================================================================
Input directory: data/chunked
Output directory: data/embeddings
Mode: gguf
Normalize: True

Initializing embedding model...
Loading GGUF model: models/Qwen3-Embedding-4B-Q8_0.gguf
  - GPU Layers: -1 (-1 = all)
✓ GGUF model loaded successfully
  - File size: 4081.40 MB
  - Embedding dimension: 2560

======================================================================
Processing method: MAXMIN_SEMANTIC
Found 10 files
======================================================================

[1/10] benchmark-indeks-konstruksi--2016-100---2018---2023_chunks.json
Loaded 143 chunks from benchmark-indeks-konstruksi--2016-100---2018---2023_chunks.json
Cleaning and filtering chunks...
Valid chunks: 143/143
Generating embeddings for 143 chunks...
✓ Generated 143 embeddings (dim: 2560)
Saving embeddings to: data/embeddings/maxmin_semantic/benchmark-indeks-konstruksi--2016-100---2018---2023_embeddings.json
✓ Embeddings saved to: benchmark-indeks-konstruksi--2016-100---2018---2023_embeddings.json
  - File size: 23.45 MB
  - Embedding dimension: 2560
  - Number of embeddings: 143
✓ Processing completed successfully
...
```

### Example 2: Custom Configuration

```python
from src.embedding import embed_all_chunks

stats = embed_all_chunks(
    chunked_dir="data/chunked",
    output_dir="data/embeddings",
    mode="gguf",
    gguf_model_path="models/Qwen3-Embedding-4B-Q8_0.gguf",
    normalize=True,
    save_numpy=True,
    skip_existing=True
)

print(f"Total: {stats['total_files']}")
print(f"Processed: {stats['processed']}")
print(f"Skipped: {stats['skipped']}")
print(f"Failed: {stats['failed']}")
```

## Troubleshooting

### Issue: `llama-cpp-python not found`
```bash
pip install llama-cpp-python
```

### Issue: `CUDA out of memory`
```bash
# Use CPU
python -m src.embedding.embed_chunks --mode huggingface --device cpu

# Or reduce batch processing (edit embedder.py)
```

### Issue: `Model file not found`
```bash
# Specify custom path
python -m src.embedding.embed_chunks --gguf-model path/to/your/model.gguf
```

### Issue: `Empty chunks`
Chunks dengan text kosong otomatis difilter. Check source JSON jika terlalu banyak empty chunks.

## Integration

Output embeddings bisa digunakan untuk:
1. **Retrieval Evaluation**: Calculate similarity untuk retrieval metrics
2. **ChromaDB**: Load ke vector database
3. **RAG Pipeline**: Query matching dan context retrieval
4. **Analysis**: Clustering, visualization, similarity analysis

Next steps: [RAG Module](../rag/README.md)
