# RAG Skripsi

Project ini membandingkan tiga metode chunking untuk sistem RAG: element-based,
MaxMin semantic, dan recursive. Dokumen ini menjelaskan kondisi implementasi
saat ini agar checkout baru tidak hanya bergantung pada contoh lama di README
komponen.

## Status Penting

- `src/chunking/maxmin_chunker.py` saat ini memiliki signature tidak valid pada
  `embed_sentences()`. Karena `src.chunking` melakukan import eager, seluruh
  import melalui package tersebut akan gagal sampai signature diperbaiki.
- Dokumentasi ini hanya mencatat kondisi tersebut. Kode asli tidak diperbaiki
  dalam perubahan dokumentasi ini.
- Folder `graphify-out/`, cache, model, log, dan sebagian artefak data tidak
  tersedia dari Git karena diabaikan atau harus dibangun lokal.

## Persiapan

Jalankan semua command dari root repository.

```bash
pip install -r requirements.txt
```

Notebook visualisasi juga memerlukan dependency opsional:

```bash
pip install matplotlib seaborn
```

Untuk Vast.ai RTX 3090, gunakan downloader aktif berikut. Script ini menyiapkan
model, ChromaDB, dan embedding dari folder Google Drive publik:

```bash
python scripts/download_vast_assets.py
python scripts/download_vast_assets.py --asset models
python scripts/download_vast_assets.py --asset all --dry-run
```

Downloader laptop GGUF/FP8 dipindahkan ke `scripts/backup/` dan bukan bagian
dari workflow Vast. Downloader Drive hanya memakai standard library Python dan
mendukung resume melalui file `.part`.

## Docker Vast.ai

Build image runtime RTX 3090 dengan `Dockerfile.vast`. Image hanya berisi
dependency dan source code; model, ChromaDB, dan embedding diunduh langsung ke
`models/`, `data/chroma/`, dan `data/embeddings/`. Instruksi template tersedia di
`docker/vast/README.md`.

## Alur Data

1. PDF mentah: `data/raw/`
2. Teks bersih: `data/cleaned/`
3. Chunk: `data/chunked/{method}/`
4. Embedding: `data/embeddings/{method}/`
5. Vector store: `data/chroma/`
6. Evaluasi: `results/`

Preprocessing masih memiliki default `data/cleaned_text`. Gunakan output
`data/cleaned` secara eksplisit agar cocok dengan default MaxMin dan recursive.

```bash
python -m src.preprocessing.pipeline --input data/raw --output data/cleaned
```

Selama blocker parse MaxMin masih ada, jalankan file chunker secara langsung
untuk menghindari import eager package:

```bash
python src/chunking/element_based.py --help
python src/chunking/recursive_split.py --help
```

Embedding dijalankan melalui module entry point:

```bash
python -m src.embedding.embed_chunks --help
```

Loader berikut meminta reset untuk setiap metode yang memiliki file embedding.
Metode tanpa file input dilewati dan collection lamanya tidak disentuh:

```bash
python scripts/load_embeddings_to_chroma.py
```

Untuk model GGUF yang disiapkan downloader, retrieval evaluation perlu memilih
mode GGUF secara eksplisit:

```bash
python scripts/run_retrieval_eval.py --mode gguf
```

Standalone generation evaluation hanya mendukung Top-1 sampai Top-10 dan belum
menulis kolom `f1_at_k`. Workflow Streamlit memiliki kontrak output berbeda.

## Entry Point Streamlit

```bash
streamlit run src/streamlit/rag_chat.py
streamlit run src/streamlit/app.py
```

`src/streamlit/app.py` membutuhkan file kandidat evidence-aware yang mungkin
tidak tersedia pada checkout baru. Bangun atau pulihkan kandidat sebelum
menjalankan aplikasi anotasi.

## Notebook

Jalankan notebook dengan working directory `notebooks/` karena sebagian besar
path relatif menggunakan awalan `../`. Catat file CSV sumber yang dipilih saat
menjalankan visualisasi; beberapa notebook mencari beberapa run dengan pola
nama yang sama.

## Dokumentasi Komponen

- `src/preprocessing/README.md`
- `src/chunking/README.md`
- `src/embedding/README.md`
- `src/chroma/README.md`
- `src/rag/README.md`
- `src/evaluation/README.md`
- `docs/CODEMAP.md`
- `docs/PROJECT_CONTEXT.txt`
