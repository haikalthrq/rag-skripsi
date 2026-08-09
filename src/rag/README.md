# Modul RAG

Modul ini menjalankan embedding query, retrieval dari ChromaDB, lalu generation
dengan backend GGUF atau Hugging Face.

## Konfigurasi Eksplisit

Selalu berikan backend dan path yang sesuai. Pemanggilan `build_pipeline()` tanpa
argumen tidak siap pakai: mode embedder default adalah GGUF, tetapi default
`embedder_path` menunjuk direktori model HF, sedangkan `generator_path` kosong.

Path lokal relatif di bawah ini di-resolve dari working directory proses:

| Komponen | Mode | Nilai path/model yang diharapkan |
|---|---|---|
| Embedder | `gguf` | File, mis. `models/Qwen3-Embedding-4B-Q8_0.gguf` |
| Embedder | `huggingface` | Direktori, mis. `models/Qwen3-Embedding-4B`, atau ID `Qwen/Qwen3-Embedding-4B` |
| Generator | `gguf` | File, mis. `models/Qwen3-4B-Instruct-Q8_0.gguf` |
| Generator | `hf` | Direktori lokal atau ID model HF, mis. `Qwen/Qwen3-4B-Thinking-2507-FP8` |
| ChromaDB | persistent | Direktori, biasanya `data/chroma` |

Contoh GGUF lengkap:

```python
from src.rag.pipeline import build_pipeline

pipeline = build_pipeline(
    chunking_method="element_based",
    embedder_mode="gguf",
    embedder_path="models/Qwen3-Embedding-4B-Q8_0.gguf",
    generator_type="gguf",
    generator_path="models/Qwen3-4B-Instruct-Q8_0.gguf",
    chroma_path="data/chroma",
)
result = pipeline.run("Pertanyaan pengguna")
```

Collection yang dipilih adalah `collection_element_based`,
`collection_maxmin_semantic`, atau `collection_recursive`, sesuai
`chunking_method`.

## Batas Konteks HF

`HFRAGGenerator` membatasi gabungan teks konteks retrieval hingga 1500 token,
dihitung dengan tokenizer HF. Chunk dipakai sesuai urutan retrieval; chunk yang
melewati sisa batas dipotong jika ruangnya lebih dari 50 token, lalu chunk
berikutnya tidak disertakan. Batas ini hanya untuk teks konteks, bukan seluruh
prompt atau token output. Backend GGUF tidak memakai batas 1500 ini dan mengikuti
`n_ctx` yang diberikan ke llama.cpp.

## ChromaDB Dan Retrieval

`RAGPipeline` memakai `get_or_create_collection()`. Jika nama collection tidak
ada, collection kosong akan dibuat. Constructor hanya menolak hasil `None`, bukan
collection dengan `count() == 0`; karena itu path Chroma yang salah atau database
yang belum diisi dapat terlihat berhasil diinisialisasi tetapi selalu memberi
hasil kosong. Periksa path log dan jumlah dokumen collection sebelum menjalankan
query.

`similarity_search()` menangkap error query ChromaDB, menulisnya ke log, lalu
mengembalikan `[]`. Nilai yang sama juga dipakai untuk retrieval yang benar-benar
tidak menghasilkan dokumen. `RAGPipeline.run()` tidak dapat membedakan kedua
kondisi itu dan mengembalikan jawaban "Tidak dapat menemukan informasi yang
relevan dalam dokumen." Periksa log untuk membedakan error retrieval dari no-hit.
Error embedding tidak ditangkap oleh fungsi retrieval dan dapat tetap diteruskan
sebagai exception.

## Kontrak Hasil

Jalur generation berhasil mengembalikan:

- `query`
- `answer`
- `thinking`
- `retrieved_chunks`
- `chunking_method`
- `num_chunks`
- `elapsed_seconds`

Pada hasil retrieval kosong, key `thinking` tidak ada; gunakan
`result.get("thinking")`. Initializer dan generator dapat melempar `RuntimeError`
ketika model/backend gagal atau jawaban kosong, sehingga pemanggil library harus
menanganinya.
