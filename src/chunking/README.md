# Chunking

Modul ini menyediakan tiga implementasi chunking untuk menyiapkan dokumen RAG.
Jalankan perintah dari root repository.

| Implementasi | Input default | Output default | Cara kerja |
|---|---|---|---|
| `element_based.py` | PDF di `data/raw` | `data/chunked/element_based` | Ekstraksi layout dengan Unstructured lalu membentuk composite chunks |
| `maxmin_chunker.py` | TXT di `data/cleaned` | `data/chunked/maxmin_semantic` | Sentence embedding dan algoritma MaxMin lokal |
| `recursive_split.py` | TXT di `data/cleaned` | `data/chunked/recursive` | `RecursiveCharacterTextSplitter` dari LangChain |

Setiap file input menghasilkan `<nama_file>_chunks.json`. Mode batch melewati
file jika output tersebut sudah ada. Gunakan `--no-skip` untuk memproses ulang.
Mode `--single` tidak melakukan pemeriksaan skip dan dapat menimpa output lama.
Jika semua file batch dilewati, entry point CLI berakhir dengan status nonzero
karena tidak ada file yang diproses pada invocation tersebut.

## Element-Based

Element-based chunking mengekstrak struktur PDF dengan
`unstructured.partition_pdf`. Implementasi tidak membuat satu chunk per elemen:
judul menjadi batas section, elemen teks berurutan digabung, tabel disimpan
sebagai chunk mandiri, dan chunk kecil dapat digabung ke chunk sebelumnya.

Default ukuran composite chunk:

- target: 1500 karakter
- maksimum: 3000 karakter
- minimum: 300 karakter

Ukuran maksimum bukan jaminan mutlak. Implementasi dapat mempertahankan chunk
yang lebih besar agar tidak menghasilkan orphan chunk di bawah batas minimum.

### CLI

| Opsi | Default | Keterangan |
|---|---|---|
| `--input`, `-i` | `data/raw` | Direktori PDF untuk mode batch |
| `--output`, `-o` | `data/chunked/element_based` | Direktori output |
| `--strategy`, `-s` | `hi_res` | Salah satu `auto`, `hi_res`, `fast`, atau `ocr_only` |
| `--target-chars` | `1500` | Target ukuran chunk |
| `--max-chars` | `3000` | Batas ukuran sebelum flush |
| `--min-chars` | `300` | Chunk yang lebih kecil dicoba merge backward |
| `--no-metadata` | tidak aktif | Simpan metadata minimal saja |
| `--no-skip` | tidak aktif | Proses ulang output yang sudah ada |
| `--single PATH` | tidak diset | Proses satu PDF |

Contoh:

```bash
python src/chunking/element_based.py
python src/chunking/element_based.py --strategy fast --target-chars 1200 --max-chars 2400
python src/chunking/element_based.py --single data/raw/dokumen.pdf
```

### Catatan OCR

`partition_document()` menggunakan bahasa OCR `['ind']` jika `languages` tidak
diberikan. Opsi bahasa tidak diekspos oleh CLI, sehingga instalasi Tesseract
untuk alur OCR perlu menyediakan data bahasa Indonesia (`ind`). Untuk bahasa
lain, panggil API partition secara langsung:

```python
from src.chunking.element_based import partition_document

elements = partition_document("data/raw/document.pdf", languages=["eng"])
```

### Output

Dengan metadata lengkap, setiap item berisi `chunk_id`, `text`, dan `metadata`.
Metadata dapat memuat `chunk_type`, `element_types`, `section_title`,
`page_numbers`, `page_range`, `source_file`, `source_filename`, `element_count`,
`order_index`, dan `num_characters`. Tabel juga dapat memuat `text_as_html`. Dengan
`--no-metadata`, object `metadata` tetap ada tetapi hanya menyimpan
`chunk_type` dan `page_range`.

## MaxMin Semantic

MaxMin memecah teks menjadi kalimat dengan NLTK, membuat embedding, lalu
menjalankan `process_sentences()` yang diimplementasikan di modul ini, bukan
algoritma dari package tambahan. Kalimat di atas 32.000 karakter dilewati sebagai
perlindungan terhadap artefak parsing yang melebihi kapasitas model.

Default algoritma adalah `fixed_threshold=0.95`, `c=0.9`, dan
`init_constant=1.5`.

### Mode Model

Mode default adalah GGUF melalui `llama-cpp-python`:

- model: `models/Qwen3-Embedding-4B-Q8_0.gguf`
- GPU layers: `-1` (semua layer)
- context: 8192
- internal llama.cpp batch: 64

Gunakan `--no-gguf` untuk mode HuggingFace melalui `SentenceTransformer`.
Default mode ini adalah model `Qwen/Qwen3-Embedding-4B` pada device `cuda`.
`--device`, `--low-memory`, dan encoding `--batch-size` berlaku untuk mode
SentenceTransformer; `--low-memory` memuat model dengan float16.

### CLI

| Opsi | Default | Keterangan |
|---|---|---|
| `--input`, `-i` | `data/cleaned` | Direktori TXT untuk mode batch |
| `--output`, `-o` | `data/chunked/maxmin_semantic` | Direktori output |
| `--gguf PATH` | `models/Qwen3-Embedding-4B-Q8_0.gguf` | File model GGUF |
| `--no-gguf` | tidak aktif | Gunakan SentenceTransformer |
| `--n-gpu-layers` | `-1` | Layer GGUF yang dipindahkan ke GPU |
| `--model`, `-m` | `Qwen/Qwen3-Embedding-4B` | Model HuggingFace untuk `--no-gguf` |
| `--device` | `cuda` | `cpu` atau `cuda`, untuk `--no-gguf` |
| `--threshold`, `-t` | `0.95` | Fixed threshold MaxMin |
| `--c` | `0.9` | Koefisien adaptive threshold |
| `--init` | `1.5` | Initial similarity multiplier |
| `--batch-size` | `8` | Batch encoding SentenceTransformer |
| `--low-memory` | tidak aktif | Gunakan float16 pada SentenceTransformer |
| `--no-metadata` | tidak aktif | Hilangkan object metadata |
| `--no-skip` | tidak aktif | Proses ulang output yang sudah ada |
| `--single PATH` | tidak diset | Proses satu TXT |

Contoh GGUF:

```bash
python src/chunking/maxmin_chunker.py
python src/chunking/maxmin_chunker.py --gguf models/Qwen3-Embedding-4B-Q8_0.gguf --n-gpu-layers 0
python src/chunking/maxmin_chunker.py --single data/cleaned/dokumen.txt --threshold 0.95
```

Contoh HuggingFace/SentenceTransformer:

```bash
python src/chunking/maxmin_chunker.py --no-gguf --model Qwen/Qwen3-Embedding-4B --device cuda --batch-size 8
```

### Output

Setiap item selalu memiliki `chunk_id`, `text`, dan `num_sentences`. Jika
metadata aktif, object `metadata` berisi `source_file`, `chunking_method`,
`sentences`, `num_characters`, dan `page_numbers`. Marker halaman
`<<<PAGE_N>>>` dihapus dari `text` dan dipetakan ke `page_numbers`.

## Recursive

Recursive chunking menggunakan separator default `"\n\n"`, `"\n"`, spasi, lalu
string kosong. Ukuran dihitung dalam karakter dengan default chunk size 1000
dan overlap 200.

### CLI

| Opsi | Default | Keterangan |
|---|---|---|
| `--input`, `-i` | `data/cleaned` | Direktori TXT untuk mode batch |
| `--output`, `-o` | `data/chunked/recursive` | Direktori output |
| `--chunk-size`, `-c` | `1000` | Ukuran maksimum chunk |
| `--overlap` | `200` | Overlap antarchunk |
| `--no-metadata` | tidak aktif | Hilangkan object metadata |
| `--no-skip` | tidak aktif | Proses ulang output yang sudah ada |
| `--single PATH` | tidak diset | Proses satu TXT |

Contoh:

```bash
python src/chunking/recursive_split.py
python src/chunking/recursive_split.py --chunk-size 1200 --overlap 150
python src/chunking/recursive_split.py --single data/cleaned/dokumen.txt
```

### Output

Setiap item selalu memiliki `chunk_id`, `text`, dan `num_characters`. Jika
metadata aktif, object `metadata` berisi `source_file`, `chunking_method`,
`chunk_length`, dan `page_numbers`. Seperti MaxMin, marker halaman dihapus dari
`text` dan dipetakan ke `page_numbers`.

## Python API

Fungsi batch utama diekspor dari package:

```python
from src.chunking import (
    run_element_based_chunking,
    run_maxmin_chunking,
    run_recursive_chunking,
)

element_stats = run_element_based_chunking()
maxmin_stats = run_maxmin_chunking()
recursive_stats = run_recursive_chunking()
```

Ketiga fungsi mengembalikan statistik batch dengan field `total_files`,
`processed`, `skipped`, `failed`, `total_chunks`, dan `duration`. Jika ada file
yang diproses, hasil juga memuat `output_files`.

## Dependencies

- Element-based: `unstructured[pdf]` serta Poppler/Tesseract sesuai strategi.
- MaxMin umum: `numpy`, `nltk`, dan `scikit-learn`.
- MaxMin GGUF: `llama-cpp-python` dan file model GGUF lokal.
- MaxMin HuggingFace: `sentence-transformers` beserta backend PyTorch-nya.
- Recursive: `langchain-text-splitters`.
