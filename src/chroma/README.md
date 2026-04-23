# ChromaDB Integration Module

Module untuk integrasi dengan ChromaDB vector database untuk sistem RAG.

## Struktur

```
src/chroma/
├── __init__.py          # Package initialization
├── client.py            # Client management
├── loader.py            # Load embeddings ke ChromaDB
├── query.py             # Query dan retrieval interface
└── README.md            # Dokumentasi ini
```

## Fitur

### 1. Client Management (`client.py`)
- **`initialize_chroma_client()`**: Initialize ChromaDB PersistentClient
  - Default persist directory: `data/chroma`
  - Mendukung mode in-memory untuk testing
- **`get_or_create_collection()`**: Get atau create collection dengan metadata
- **`delete_collection()`**: Delete collection by name
- **`list_collections()`**: List semua collections

### 2. Data Loading (`loader.py`)
- **`batch_add_documents()`**: Add documents ke collection dengan batching
  - Default batch size: 1000 documents
  - Otomatis handle large datasets
- **`load_to_chroma()`**: Load embeddings dari single JSON file
- **`load_all_embeddings_to_chroma()`**: Batch load semua embeddings
  - Otomatis scan directory untuk .json files
  - Progress tracking dengan tqdm
  - Skip files yang sudah diload

### 3. Query Interface (`query.py`)
- **`ChromaRetriever`**: High-level retriever class
  - `similarity_search()`: Search by text query
  - `similarity_search_with_score()`: Search dengan similarity scores
- **`similarity_search()`**: Query dengan text input
- **`search_by_vector()`**: Query dengan vector langsung
- **`get_documents_by_ids()`**: Retrieve specific documents
- **`filter_by_metadata()`**: Filter berdasarkan metadata

## Usage

### Initialize Client
```python
from src.chroma import initialize_chroma_client

# Local persistent storage
client = initialize_chroma_client(persist_directory="data/chroma")

# In-memory (untuk testing)
client = initialize_chroma_client(in_memory=True)
```

### Create Collection
```python
from src.chroma import get_or_create_collection

collection = get_or_create_collection(
    client=client,
    collection_name="my_collection",
    embedding_dimension=2560  # Qwen3-Embedding-4B
)
```

### Load Embeddings
```python
from src.chroma import load_to_chroma, load_all_embeddings_to_chroma

# Load single file
load_to_chroma(
    client=client,
    collection_name="collection_element_based",
    embeddings_file="data/embeddings/element_based/doc_embeddings.json",
    batch_size=1000
)

# Load all files dari directory
load_all_embeddings_to_chroma(
    client=client,
    chunking_method="element_based",
    embeddings_dir="data/embeddings/element_based",
    batch_size=1000
)
```

### Query Collection
```python
from src.chroma import ChromaRetriever

# Initialize retriever
retriever = ChromaRetriever(
    client=client,
    collection_name="collection_element_based"
)

# Search by text
results = retriever.similarity_search(
    query_text="pertumbuhan ekonomi Indonesia",
    n_results=5
)

# Search dengan scores
results = retriever.similarity_search_with_score(
    query_text="pertumbuhan ekonomi Indonesia",
    n_results=5
)
```

### Metadata Filtering
```python
from src.chroma import filter_by_metadata

# Filter by source document
results = filter_by_metadata(
    collection=collection,
    filters={"source": "statistik-perdagangan-luar-negeri.txt"},
    n_results=10
)

# Advanced filters
results = filter_by_metadata(
    collection=collection,
    filters={
        "chunking_method": "element_based",
        "chunk_index": {"$lt": 50}
    },
    n_results=20
)
```

## Main Script

Script wrapper tersedia di root directory:

### `load_to_chroma.py`
Load embeddings dari JSON ke ChromaDB.

```bash
# Load semua embeddings dengan default settings
python load_to_chroma.py

# Custom batch size
python load_to_chroma.py --batch-size 500

# Reset collections (delete dulu sebelum load)
python load_to_chroma.py --reset

# List collections only (tidak load)
python load_to_chroma.py --list-only
```

**Arguments:**
- `--persist-dir`: ChromaDB persist directory (default: `data/chroma`)
- `--embeddings-dir`: Directory dengan embeddings (default: `data/embeddings`)
- `--batch-size`: Batch size untuk loading (default: 1000)
- `--reset`: Delete existing collections sebelum load
- `--list-only`: Hanya list collections, tidak load

## Data Structure

### Embeddings JSON Format
```json
{
  "document_name": "doc.txt",
  "chunking_method": "element_based",
  "chunks": [
    {
      "chunk_index": 0,
      "text": "chunk text...",
      "embedding": [0.1, 0.2, ...],
      "metadata": {
        "source": "doc.txt",
        "chunk_index": 0
      }
    }
  ]
}
```

### ChromaDB Collections
- **Collection per chunking method**: `collection_{method_name}`
  - `collection_element_based`
  - `collection_maxmin_semantic`
  - `collection_recursive`

### Metadata Fields
- `source`: Original document filename
- `chunk_index`: Index chunk dalam document
- `chunking_method`: Method yang dipakai (element_based/maxmin_semantic/recursive)
- `document_name`: Document name (untuk grouping)

## Statistics

Berdasarkan embeddings yang sudah di-generate:

| Chunking Method | Total Chunks | Files | Avg per File |
|----------------|--------------|-------|--------------|
| element_based | 35,548 | 10 | 3,555 |
| maxmin_semantic | 1,827 | 10 | 183 |
| recursive | 3,782 | 10 | 378 |
| **TOTAL** | **41,157** | **30** | **1,372** |

## Dependencies

```python
chromadb>=0.4.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Configuration

Default paths (defined in module):
```python
DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
DEFAULT_BATCH_SIZE = 1000
```

## Best Practices

1. **Batching**: Use batch_size=1000 untuk optimal performance
2. **Collections**: Separate collection per chunking method untuk comparison
3. **Metadata**: Selalu include metadata untuk filtering dan tracing
4. **Reset**: Use `--reset` flag saat re-load untuk avoid duplicates
5. **Testing**: Use in-memory mode untuk testing (`in_memory=True`)

## Troubleshooting

### Collection Already Exists
```python
# Delete collection first
from src.chroma import delete_collection
delete_collection(client, "collection_name")
```

### Memory Issues
```python
# Reduce batch size
load_to_chroma(..., batch_size=500)
```

### Dimension Mismatch
- Pastikan embedding dimension = 2560 (Qwen3-Embedding-4B)
- Check embeddings JSON format

## Next Steps

1. **Load embeddings**: Run `python load_to_chroma.py`
2. **Test retrieval**: Query collections untuk validasi
3. **Evaluation**: Build retrieval metrics on top of ChromaDB
4. **RAG Pipeline**: Integrate dengan generation module

## Author
RAG Skripsi Project - 2025
