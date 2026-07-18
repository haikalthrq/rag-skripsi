# Modul RAG

Modul ini menggabungkan embedding query, retrieval ChromaDB, dan generation.

## Konfigurasi Backend

`build_pipeline()` menerima kombinasi berikut:

- `embedder_mode="gguf"`: `embedder_path` harus menunjuk file `.gguf`.
- `embedder_mode="huggingface"`: `embedder_path` harus menunjuk direktori atau
  nama model HuggingFace lengkap.
- `generator_type="gguf"`: `generator_path` harus menunjuk file model GGUF.
- `generator_type="hf"`: `generator_path` harus menunjuk direktori atau nama
  model HuggingFace.

Pemanggilan `build_pipeline()` tanpa argumen belum merupakan konfigurasi yang
aman karena default path dan backend tidak seluruhnya saling cocok. Berikan
path model secara eksplisit.

```python
from src.rag.pipeline import build_pipeline

pipeline = build_pipeline(
    chunking_method="element_based",
    embedder_mode="gguf",
    embedder_path="models/Qwen3-Embedding-4B-Q8_0.gguf",
    generator_type="gguf",
    generator_path="path/ke/model-generator.gguf",
    chroma_path="data/chroma",
)

result = pipeline.run("Pertanyaan pengguna")
```

## Kontrak Hasil

Pada jalur retrieval berhasil, `RAGPipeline.run()` mengembalikan dictionary
dengan key:

- `query`
- `answer`
- `thinking`
- `retrieved_chunks`
- `chunking_method`
- `num_chunks`
- `elapsed_seconds`

Jalur tanpa hasil retrieval tidak menyertakan key `thinking`, sehingga konsumen
sebaiknya membaca key opsional tersebut dengan `result.get("thinking")`.

## Penanganan Error

Initializer dan generator dapat menghasilkan `RuntimeError` ketika model gagal
dimuat, backend gagal, atau jawaban kosong. Pemanggil library harus menangani
exception tersebut. Aplikasi Streamlit dapat memiliki strategi penanganan yang
berbeda dari pemakaian library langsung.
