# Embedding Module

Modul ini membuat embedding chunk dengan Qwen3-Embedding-4B melalui salah satu
backend berikut:

- GGUF via `llama-cpp-python` (default)
- HuggingFace via `sentence-transformers`

Entry point CLI:

```bash
python -m src.embedding.embed_chunks --help
python -m src.embedding.embed_chunks
```

Secara default, pipeline membaca `*_chunks.json` dari subdirektori
`element_based`, `maxmin_semantic`, dan `recursive` di `data/chunked`, lalu
menulis hasil ke subdirektori yang sama di `data/embeddings`. Nama output
mempertahankan stem input: `document_chunks.json` menjadi
`document_chunks_embeddings.json`.

## Transformasi Teks

Urutan pemrosesan per file:

1. Muat JSON berupa list chunk.
2. Untuk chunk tabel dengan `metadata.chunk_type == "table"` dan
   `metadata.text_as_html`, ganti `chunk["text"]` dengan tabel pipe-separated.
   Judul dari chunk sebelumnya pada halaman yang sama, atau `section_title`,
   ditambahkan jika tersedia dan lolos filter noise.
3. Buang chunk dengan teks kosong dan normalkan whitespace.
4. Untuk metode `maxmin_semantic` dan `recursive`, tambahkan maksimal 200
   karakter terakhir dari chunk valid sebelumnya ke teks yang di-embed.
5. Buat embedding, lakukan normalisasi L2 jika aktif, lalu simpan JSON dan
   opsional `.npy`.

Enrichment tabel memodifikasi chunk yang disimpan dalam output. Context prefix
hanya digunakan untuk embedding dan tidak disimpan. Backend HuggingFace juga
memotong setiap teks menjadi maksimal 4096 karakter sebelum encoding. Karena
itu, `chunks[*].text` pada output tidak selalu merekam teks persis yang diterima
model.

`QwenEmbedder.embed(..., batch_size=...)` mempertahankan parameter
`batch_size`, tetapi implementasi saat ini tidak menggunakannya. GGUF memproses
teks satu per satu; HuggingFace menggunakan `batch_size=1`.

## CLI

```bash
# HuggingFace; --device hanya diteruskan ke backend ini
python -m src.embedding.embed_chunks --mode huggingface --device cpu

# Path dan model custom
python -m src.embedding.embed_chunks \
  --input data/chunked \
  --output data/embeddings \
  --gguf-model models/custom-model.gguf

# Proses ulang output yang sudah ada dan simpan array NumPy
python -m src.embedding.embed_chunks --no-skip --save-numpy
```

Gunakan `--no-normalize` untuk menonaktifkan normalisasi. CLI selalu mengimpor
`torch`, termasuk saat memakai GGUF.

## Python API

```python
from src.embedding import embed_all_chunks, initialize_gguf_embedder

stats = embed_all_chunks(
    chunked_dir="data/chunked",
    output_dir="data/embeddings",
    mode="gguf",
    methods=["element_based"],  # Filter metode hanya tersedia melalui API.
)

embedder = initialize_gguf_embedder(
    model_path="models/Qwen3-Embedding-4B-Q8_0.gguf",
)
embeddings = embedder.embed(["teks pertama", "teks kedua"])
```

`initialize_gguf_embedder()` dan `initialize_hf_embedder()` mengembalikan
`None` jika dependency/model tidak tersedia atau inisialisasi gagal.
`embed_all_chunks()` mengembalikan statistik `total_files`, `processed`,
`skipped`, `failed`, dan `by_method`, atau dictionary dengan key `error` jika
inisialisasi gagal.

## Output JSON

```json
{
  "metadata": {
    "embedding_dim": 2,
    "num_chunks": 143,
    "num_original_chunks": 150,
    "timestamp": "...",
    "source_file": "document_chunks.json",
    "source_path": "data/chunked/recursive/document_chunks.json",
    "chunking_method": "recursive",
    "embedding_model": "gguf",
    "normalized": true
  },
  "embeddings": [[0.1, 0.2]],
  "chunks": [
    {
      "text": "teks chunk",
      "embedding_index": 0,
      "original_index": 0
    }
  ]
}
```

`load_embeddings()` mengubah `embeddings` kembali menjadi `np.float32`.
`chunks` hanya memuat chunk valid dan masing-masing menunjuk ke baris embedding
melalui `embedding_index`.

Dependency proyek tercantum di `requirements.txt`. Model default GGUF dicari di
`models/Qwen3-Embedding-4B-Q8_0.gguf`; model HuggingFace default adalah
`Qwen/Qwen3-Embedding-4B`.
