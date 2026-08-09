# Modul ChromaDB

Modul ini mengelola client ChromaDB, memuat embedding JSON, dan menyediakan
helper retrieval untuk collection per metode chunking.

## API Publik

Nama berikut diekspor dari `src.chroma`.

### Client (`client.py`)

- `initialize_chroma_client(persist_directory="data/chroma", in_memory=False)`
  membuat persistent client atau in-memory client dan mengembalikan `None` jika
  inisialisasi gagal.
- `get_or_create_collection(client, collection_name, embedding_dim=None,
  metadata=None)` mengambil collection yang ada atau membuat yang baru.
  `embedding_dim` disimpan sebagai metadata saat collection dibuat.
- `delete_collection(client, collection_name)` menghapus collection dan
  mengembalikan status boolean.
- `list_collections(client)` mengembalikan daftar nama collection.

`client.py` juga memiliki `get_collection_info()` dan `reset_collection()`, tetapi
keduanya harus diimpor langsung dari `src.chroma.client`.

### Loader (`loader.py`)

- `batch_add_documents(collection, ids, embeddings, documents, metadatas=None,
  batch_size=1000)` menambahkan data per batch.
- `load_to_chroma(client, embedding_file, collection_name, batch_size=1000,
  reset_collection=False)` memuat satu file JSON dan mengembalikan statistik,
  atau `None` jika gagal.
- `load_all_embeddings_to_chroma(client, embeddings_dir="data/embeddings",
  batch_size=1000, methods=None, reset_collections=False)` memuat file
  `*_embeddings.json` dari subdirektori metode dan mengembalikan ringkasan.

Jika `methods=None`, loader memakai `element_based`, `maxmin_semantic`, dan
`recursive`. Setiap metode dimuat ke `collection_{method}`. Saat
`reset_collections=True`, collection di-reset hanya sebelum file pertama untuk
metode tersebut.

### Query (`query.py`)

- `ChromaRetriever(collection, embedding_function=None, k=5)` menyediakan
  `similarity_search(query, k=None, filter=None)`,
  `similarity_search_by_vector(embedding, k=None, filter=None)`, dan
  `similarity_search_with_score(query, k=None, filter=None)`.
- `similarity_search(collection, query_embedding, k=5, filter=None)`
  mengembalikan list dokumen beserta `distance`.
- `similarity_search_with_score(collection, query_embedding, k=5,
  filter=None)` mengembalikan tuple `(document, distance)`.
- `search_by_vector(collection, embedding, k=5, filter=None,
  include_distances=True)` mengembalikan hasil mentah `collection.query()`.

Parameter `filter` diteruskan sebagai `where` ke Chroma. Nilai yang dinamai
`score` oleh helper berasal langsung dari `distances`; umumnya nilai lebih kecil
berarti hasil lebih dekat, bergantung pada konfigurasi collection.

`get_documents_by_ids()` dan `filter_by_metadata()` tersedia melalui import
langsung dari `src.chroma.query`, bukan dari `src.chroma`.

## Contoh

### Memuat Satu File

```python
from src.chroma import initialize_chroma_client, load_to_chroma

client = initialize_chroma_client(persist_directory="data/chroma")
if client is None:
    raise RuntimeError("ChromaDB gagal diinisialisasi")

stats = load_to_chroma(
    client=client,
    embedding_file="data/embeddings/element_based/doc_embeddings.json",
    collection_name="collection_element_based",
    batch_size=1000,
    reset_collection=False,
)
```

### Query dengan Vector

`query_embedding` harus berupa `numpy.ndarray` yang dimensinya sama dengan
embedding dalam collection.

```python
from src.chroma import get_or_create_collection, similarity_search

collection = get_or_create_collection(
    client=client,
    collection_name="collection_element_based",
    embedding_dim=2560,
)
if collection is None:
    raise RuntimeError("Collection gagal dibuka")

results = similarity_search(
    collection=collection,
    query_embedding=query_embedding,
    k=5,
    filter={"source_file": "doc.txt"},
)
```

### Query Teks dengan Retriever

`embedding_function` menerima satu string dan harus mengembalikan vector yang
mendukung `.tolist()`, misalnya `numpy.ndarray`.

```python
from src.chroma import ChromaRetriever

retriever = ChromaRetriever(
    collection=collection,
    embedding_function=embed_query,
    k=5,
)
results = retriever.similarity_search(
    query="pertumbuhan ekonomi Indonesia",
    filter={"chunking_method": "element_based"},
)
```

## Format JSON Loader

`embeddings` dan `chunks` harus berupa array yang sejajar. `metadata` adalah
object top-level opsional yang dipakai untuk metadata collection dan dokumen.

```json
{
  "metadata": {
    "source_file": "doc.txt",
    "chunking_method": "element_based",
    "embedding_dim": 3,
    "embedding_model": "model-name"
  },
  "embeddings": [[0.1, 0.2, 0.3]],
  "chunks": [
    {
      "text": "Isi chunk",
      "metadata": {"page": 1},
      "embedding_index": 0,
      "original_index": 0
    }
  ]
}
```

ID dokumen dibentuk sebagai `{stem_file}_{original_index}`. Jika
`original_index` tidak ada, loader memakai `embedding_index`, lalu `0` sebagai
fallback. Metadata chunk dibersihkan agar hanya berisi nilai yang kompatibel
dengan Chroma.

## Script Pemuatan

```bash
python scripts/load_embeddings_to_chroma.py
```

Script tidak menerima argument CLI. Konfigurasinya menetapkan
`data/embeddings`, `data/chroma`, batch size 1000, ketiga metode, dan
`reset_collections=True`. Collection suatu metode akan dibangun ulang jika
subdirektorinya berisi file embedding yang cocok. Gunakan API Python langsung
untuk konfigurasi lain.
