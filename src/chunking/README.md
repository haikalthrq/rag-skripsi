# Modul Chunking

Modul ini menyediakan berbagai metode chunking untuk dokumen RAG system.

## 📁 Metode Chunking

### 1. **Element-Based Chunking** (`element_based.py`)
Chunking berdasarkan struktur elemen dokumen menggunakan library **Unstructured**.

**Input**: PDF files (`data/raw/`)  
**Output**: JSON chunks (`data/chunked/element_based/`)  
**Library**: `unstructured`

### 2. **MaxMin Semantic Chunking** (`maxmin_chunker.py`)
Chunking berdasarkan semantic similarity menggunakan library **maxmin_chunker**.

**Input**: Cleaned text files (`data/cleaned_text/`)  
**Output**: JSON chunks (`data/chunked/maxmin_semantic/`)  
**Library**: `maxmin_chunker`, `nltk`, `langchain_huggingface`

---

# Element-Based Chunking

Melakukan chunking berdasarkan struktur elemen dokumen PDF (Title, Paragraph, Table, dll).

## 📖 Dokumentasi Lengkap

Lihat [Element-Based Chunking Documentation](#element-based-chunking-documentation) di bawah.

---

# MaxMin Semantic Chunking

Melakukan chunking berdasarkan semantic similarity antar kalimat menggunakan threshold dinamis.

## 🔧 Fungsi Utama

### 1. `initialize_embedding_model(model_name, device)`
Inisialisasi model embedding untuk generate sentence embeddings.

**Parameter:**
- `model_name` (str): Nama model HuggingFace (default: `"Alibaba-NLP/gte-Qwen2-1.5B-instruct"` - Qwen3-Embedding)
- `device` (str): Device untuk inference (`"cpu"` atau `"cuda"`)

**Return:**
- Model embedding HuggingFace atau None jika gagal

**Contoh:**
```python
from src.chunking import initialize_embedding_model

model = initialize_embedding_model(
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    device="cpu"
)
```

### 2. `load_text(text_path)`
Memuat file teks dari `data/cleaned_text/`.

**Parameter:**
- `text_path` (str): Path ke file .txt

**Return:**
- String berisi isi file atau None jika gagal

### 3. `split_sentences(text)`
Split teks menjadi kalimat menggunakan NLTK `sent_tokenize`.

**Parameter:**
- `text` (str): Teks yang akan di-split

**Return:**
- List kalimat atau None jika gagal

**Contoh:**
```python
from src.chunking import split_sentences

text = "Ini kalimat pertama. Ini kalimat kedua. Ini kalimat ketiga."
sentences = split_sentences(text)
# Output: ['Ini kalimat pertama.', 'Ini kalimat kedua.', 'Ini kalimat ketiga.']
```

### 4. `embed_sentences(sentences, embedding_model)`
Generate embeddings untuk setiap kalimat menggunakan model embedding.

**Parameter:**
- `sentences` (List[str]): List kalimat
- `embedding_model` (Any): Model embedding dari HuggingFace

**Return:**
- NumPy array dengan shape `(n_sentences, embedding_dim)` atau None jika gagal

**Contoh:**
```python
from src.chunking import initialize_embedding_model, embed_sentences

model = initialize_embedding_model()
sentences = ["Kalimat pertama.", "Kalimat kedua."]
embeddings = embed_sentences(sentences, model)
print(embeddings.shape)  # Output: (2, 1536) untuk Qwen3-Embedding
```

### 5. `apply_maxmin_chunking(sentences, embeddings, fixed_threshold, c, init_constant)`
**Memanggil `process_sentences()` dari library `maxmin_chunker`** (tidak mengimplementasi ulang algoritma).

**Parameter:**
- `sentences` (List[str]): List kalimat
- `embeddings` (np.ndarray): Array embeddings dengan shape `(n_sentences, embedding_dim)`
- `fixed_threshold` (float): Fixed threshold untuk similarity (default: `0.6`)
- `c` (float): Parameter untuk adaptive threshold (default: `0.9`)
- `init_constant` (float): Initial constant untuk threshold (default: `1.5`)

**Return:**
- List of chunks, dimana setiap chunk adalah list of sentences (List[List[str]])

**Contoh:**
```python
from src.chunking import apply_maxmin_chunking

# sentences dan embeddings sudah di-generate sebelumnya
chunks = apply_maxmin_chunking(
    sentences,
    embeddings,
    fixed_threshold=0.6,
    c=0.9,
    init_constant=1.5
)

# Output: [['sent1', 'sent2'], ['sent3', 'sent4', 'sent5'], ...]
```

### 6. `run_maxmin_chunking(input_dir, output_dir, model_name, device, ...)`
Menjalankan MaxMin chunking untuk semua file .txt di direktori input.

**Parameter:**
- `input_dir` (str): Direktori input (default: `'data/cleaned_text'`)
- `output_dir` (str): Direktori output (default: `'data/chunked/maxmin_semantic'`)
- `model_name` (str): Nama model embedding (default: `'Alibaba-NLP/gte-Qwen2-1.5B-instruct'`)
- `device` (str): Device (`'cpu'` atau `'cuda'`, default: `'cpu'`)
- `fixed_threshold` (float): Threshold (default: `0.6`)
- `c` (float): Parameter c (default: `0.9`)
- `init_constant` (float): Parameter init (default: `1.5`)
- `include_metadata` (bool): Sertakan metadata (default: `True`)
- `skip_existing` (bool): Skip file yang sudah ada (default: `True`)

**Return:**
- Dictionary statistik hasil chunking

**Contoh:**
```python
from src.chunking import run_maxmin_chunking

stats = run_maxmin_chunking(
    input_dir='data/cleaned_text',
    output_dir='data/chunked/maxmin_semantic',
    model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    device='cpu',
    fixed_threshold=0.6,
    c=0.9,
    init_constant=1.5
)

print(f"Berhasil: {stats['processed']}/{stats['total_files']}")
print(f"Total chunks: {stats['total_chunks']}")
```

## 🚀 Cara Penggunaan

### Sebagai Script (Command Line)

#### 1. Proses semua file .txt dalam folder

```bash
# Menggunakan default settings
python chunk_maxmin.py

# Dengan custom directories
python chunk_maxmin.py --input data/cleaned_text --output data/chunked/maxmin_semantic

# Dengan custom model (gunakan GPU)
python chunk_maxmin.py --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --device cuda

# Tune parameters
python chunk_maxmin.py --threshold 0.7 --c 0.85 --init 1.3

# Tanpa metadata
python chunk_maxmin.py --no-metadata

# Force reprocess
python chunk_maxmin.py --no-skip
```

#### 2. Proses satu file .txt saja

```bash
python chunk_maxmin.py --single "data/cleaned_text/dokumen.txt"
```

### Sebagai Import dalam Python

#### 1. Import dan jalankan batch processing

```python
from src.chunking import run_maxmin_chunking

# Proses semua .txt files
stats = run_maxmin_chunking(
    input_dir='data/cleaned_text',
    output_dir='data/chunked/maxmin_semantic',
    model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    device='cpu'
)

print(f"Total: {stats['total_files']}")
print(f"Berhasil: {stats['processed']}")
print(f"Chunks: {stats['total_chunks']}")
```

#### 2. Proses individual dengan kontrol penuh

```python
from src.chunking import (
    initialize_embedding_model,
    load_text,
    split_sentences,
    embed_sentences,
    apply_maxmin_chunking,
    save_maxmin_chunks,
    convert_paragraphs_to_chunks
)

# 1. Initialize model
model = initialize_embedding_model(
    model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    device='cpu'
)

# 2. Load text
text = load_text('data/cleaned_text/dokumen.txt')

# 3. Split into sentences
sentences = split_sentences(text)
print(f"Total sentences: {len(sentences)}")

# 4. Generate embeddings
embeddings = embed_sentences(sentences, model)
print(f"Embeddings shape: {embeddings.shape}")

# 5. Apply MaxMin chunking (memanggil library maxmin_chunker)
paragraphs = apply_maxmin_chunking(
    sentences,
    embeddings,
    fixed_threshold=0.6,
    c=0.9,
    init_constant=1.5
)
print(f"Total chunks: {len(paragraphs)}")

# 6. Convert to JSON format
chunks = convert_paragraphs_to_chunks(paragraphs, 'dokumen.txt')

# 7. Save
save_maxmin_chunks(chunks, 'data/chunked/maxmin_semantic/dokumen_chunks.json')
```

## 📊 Output

### Format JSON Output
```json
[
  {
    "chunk_id": 0,
    "text": "Kalimat pertama. Kalimat kedua yang related.",
    "num_sentences": 2,
    "metadata": {
      "source_file": "dokumen.txt",
      "chunking_method": "maxmin_semantic",
      "sentences": [
        "Kalimat pertama.",
        "Kalimat kedua yang related."
      ],
      "num_characters": 52
    }
  },
  {
    "chunk_id": 1,
    "text": "Kalimat ketiga dengan topik berbeda.",
    "num_sentences": 1,
    "metadata": {
      "source_file": "dokumen.txt",
      "chunking_method": "maxmin_semantic",
      "sentences": ["Kalimat ketiga dengan topik berbeda."],
      "num_characters": 37
    }
  }
]
```

### Contoh Output Log
```
INFO - Memulai MaxMin Semantic Chunking Pipeline
INFO - Input directory: data/cleaned_text
INFO - Output directory: data/chunked/maxmin_semantic
INFO - Model: Alibaba-NLP/gte-Qwen2-1.5B-instruct
INFO - Device: cpu
INFO - Parameters: threshold=0.6, c=0.9, init=1.5

INFO - Inisialisasi embedding model...
INFO - ✓ Model embedding berhasil diinisialisasi

[1/10] Processing: dokumen1.txt
INFO - Memuat teks: dokumen1.txt
INFO - ✓ Berhasil memuat 45890 karakter
INFO - Splitting teks menjadi kalimat...
INFO - ✓ Berhasil split menjadi 234 kalimat
INFO - Generating embeddings untuk 234 kalimat...
INFO - ✓ Embeddings berhasil digenerate
INFO -   - Shape: (234, 1536)
INFO - Menjalankan MaxMin chunking...
INFO - ✓ MaxMin chunking selesai
INFO -   - Total chunks: 45
INFO -   - Rata-rata kalimat per chunk: 5.20
INFO - ✓ Berhasil menyimpan 45 chunks

...

INFO - MaxMin Semantic Chunking Selesai
INFO - Total file teks: 10
INFO - Berhasil diproses: 10
INFO - Total chunks dihasilkan: 453
INFO - Durasi: 125.43 detik
INFO - Rata-rata chunks per dokumen: 45.30
```

## 📦 Dependencies

```bash
# Install dependencies
pip install maxmin-chunker
pip install nltk
pip install langchain-huggingface
pip install sentence-transformers
pip install numpy
```

## ⚙️ Parameter MaxMin Chunking

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `fixed_threshold` | 0.6 | Fixed similarity threshold (0-1) |
| `c` | 0.9 | Coefficient untuk adaptive threshold |
| `init_constant` | 1.5 | Initial constant untuk threshold calculation |

**Cara Tuning:**
1. **fixed_threshold** (0.5 - 0.7):
   - Lebih rendah → chunks lebih besar, lebih permisif
   - Lebih tinggi → chunks lebih kecil, lebih strict
   
2. **c** (0.8 - 0.95):
   - Mempengaruhi adaptive threshold
   - Default 0.9 sudah optimal untuk kebanyakan kasus

3. **init_constant** (1.0 - 2.0):
   - Mempengaruhi threshold awal
   - Default 1.5 sudah optimal

## 💡 Tips

1. **Model Embedding**: Gunakan Qwen3-Embedding (default) untuk bahasa Indonesia/multilingual
2. **GPU**: Gunakan `--device cuda` jika ada GPU untuk speed up
3. **Threshold Tuning**: Start dengan default (0.6), tune berdasarkan hasil
4. **Input**: Pastikan text sudah di-cleaning (gunakan modul preprocessing dulu)
5. **Batch Processing**: Lebih efisien untuk banyak file

## 🐛 Troubleshooting

### Error: "Library 'maxmin_chunker' tidak terinstall"
```bash
pip install maxmin-chunker
```

### Error: "Library 'nltk' tidak terinstall"
```bash
pip install nltk
```

### Error: "punkt tokenizer not found"
Script akan otomatis download, tapi bisa manual:
```python
import nltk
nltk.download('punkt')
```

### Error: Model download gagal
```bash
# Pre-download model
huggingface-cli login
huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

### Chunks terlalu besar/kecil
- Tune `fixed_threshold`: lebih rendah = chunks lebih besar
- Cek distribusi chunk size di log output
- Evaluasi dengan sample documents

---

# Element-Based Chunking Documentation

### 1. `partition_document(pdf_path, strategy="hi_res")`
Melakukan partitioning dokumen PDF menggunakan `unstructured.partition_pdf`.

**Parameter:**
- `pdf_path` (str): Path ke file PDF
- `strategy` (str): Strategi partitioning. Default: `"hi_res"` untuk akurasi maksimal
  - `"hi_res"`: Akurasi tinggi, lebih lambat (recommended)
  - `"fast"`: Cepat tapi kurang akurat
  - `"auto"`: Otomatis pilih strategi
  - `"ocr_only"`: Hanya OCR

**Parameter Unstructured yang digunakan:**
- `strategy="hi_res"`: Untuk pembacaan struktur PDF yang lebih akurat
- `infer_table_structure=True`: Ekstrak struktur tabel
- `extract_image_block_types=["table"]`: Ekstrak tabel dari gambar
- `include_page_breaks=True`: Sertakan informasi page breaks

**Return:**
- List elemen dokumen (Title, Paragraph, Table, ListItem, dll) atau None jika gagal

**Contoh:**
```python
from src.chunking import partition_document

elements = partition_document('data/raw/dokumen.pdf', strategy='hi_res')
print(f"Ditemukan {len(elements)} elemen")
```

### 2. `convert_elements_to_chunks(elements, include_metadata=True)`
Konversi elemen dokumen menjadi list chunks dengan metadata.

**Parameter:**
- `elements` (List[Any]): List elemen dari partition_pdf
- `include_metadata` (bool): Jika True, sertakan metadata dalam chunk

**Return:**
- List dictionary dengan struktur:
  ```python
  {
      'chunk_id': 0,
      'text': 'Teks dari elemen...',
      'metadata': {
          'element_type': 'Paragraph',
          'page_number': 1,
          'element_id': '...',
          'coordinates': '...',
          'filename': 'dokumen.pdf'
      }
  }
  ```

**Contoh:**
```python
from src.chunking import partition_document, convert_elements_to_chunks

elements = partition_document('data/raw/dokumen.pdf')
chunks = convert_elements_to_chunks(elements, include_metadata=True)

for chunk in chunks[:3]:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
    print(f"Type: {chunk['metadata']['element_type']}")
```

### 3. `convert_elements_to_text_list(elements)`
Konversi elemen dokumen menjadi list string sederhana (hanya text, tanpa metadata).

**Parameter:**
- `elements` (List[Any]): List elemen dari partition_pdf

**Return:**
- List[str]: List string dari setiap elemen

**Contoh:**
```python
from src.chunking import partition_document, convert_elements_to_text_list

elements = partition_document('data/raw/dokumen.pdf')
text_chunks = convert_elements_to_text_list(elements)

for i, text in enumerate(text_chunks[:5]):
    print(f"Chunk {i}: {text[:100]}...")
```

### 4. `save_chunks(chunks, output_path, pretty_print=True)`
Menyimpan chunks dalam format JSON.

**Parameter:**
- `chunks` (List[Dict[str, Any]]): List chunks untuk disimpan
- `output_path` (str): Path file output JSON
- `pretty_print` (bool): Jika True, format JSON dengan indentasi

**Return:**
- bool: True jika berhasil, False jika gagal

### 5. `process_single_pdf(pdf_path, output_dir, strategy="hi_res", include_metadata=True)`
Memproses satu file PDF: partition, convert, dan save.

**Parameter:**
- `pdf_path` (str): Path ke file PDF
- `output_dir` (str): Direktori output untuk hasil chunking
- `strategy` (str): Strategi partitioning (default: `"hi_res"`)
- `include_metadata` (bool): Sertakan metadata dalam chunks

**Return:**
- List chunks jika berhasil, None jika gagal

**Contoh:**
```python
from src.chunking import process_single_pdf

chunks = process_single_pdf(
    'data/raw/dokumen.pdf',
    'data/chunked/element_based',
    strategy='hi_res',
    include_metadata=True
)

if chunks:
    print(f"Berhasil: {len(chunks)} chunks")
```

### 6. `run_element_based_chunking(input_dir, output_dir, strategy="hi_res", include_metadata=True, skip_existing=True)`
Menjalankan element-based chunking untuk semua PDF di direktori input.

**Parameter:**
- `input_dir` (str): Direktori berisi file PDF input (default: `'data/raw'`)
- `output_dir` (str): Direktori output (default: `'data/chunked/element_based'`)
- `strategy` (str): Strategi partitioning (default: `'hi_res'`)
- `include_metadata` (bool): Sertakan metadata (default: `True`)
- `skip_existing` (bool): Skip file yang sudah diproses (default: `True`)

**Return:**
- Dictionary statistik:
  ```python
  {
      'total_files': 10,
      'processed': 10,
      'skipped': 0,
      'failed': 0,
      'total_chunks': 1523,
      'duration': 45.32,
      'output_files': [...]
  }
  ```

**Contoh:**
```python
from src.chunking import run_element_based_chunking

stats = run_element_based_chunking(
    input_dir='data/raw',
    output_dir='data/chunked/element_based',
    strategy='hi_res',
    include_metadata=True
)

print(f"Berhasil: {stats['processed']}/{stats['total_files']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Durasi: {stats['duration']:.2f} detik")
```

## 🚀 Cara Penggunaan

### Sebagai Script (Command Line)

#### 1. Proses semua PDF dalam folder (batch mode)

```bash
# Menggunakan default (hi_res strategy)
python chunk_element.py

# Dengan custom directories
python chunk_element.py --input data/raw --output data/chunked/element_based

# Dengan strategy berbeda
python chunk_element.py --strategy fast

# Tanpa metadata
python chunk_element.py --no-metadata

# Force reprocess file yang sudah ada
python chunk_element.py --no-skip
```

#### 2. Proses satu file PDF saja

```bash
python chunk_element.py --single "data/raw/dokumen.pdf"
```

#### 3. Jalankan langsung dari modul

```bash
python -m src.chunking.element_based --input data/raw --output data/chunked/element_based
```

### Sebagai Import dalam Python

#### 1. Import dan jalankan batch processing

```python
from src.chunking import run_element_based_chunking

# Proses semua PDF dengan hi_res strategy
stats = run_element_based_chunking(
    input_dir='data/raw',
    output_dir='data/chunked/element_based',
    strategy='hi_res',
    include_metadata=True
)

print(f"Total file: {stats['total_files']}")
print(f"Berhasil: {stats['processed']}")
print(f"Total chunks: {stats['total_chunks']}")
```

#### 2. Proses individual dengan kontrol penuh

```python
from src.chunking import partition_document, convert_elements_to_chunks, save_chunks
from pathlib import Path

# 1. Partition PDF
pdf_path = 'data/raw/dokumen.pdf'
elements = partition_document(pdf_path, strategy='hi_res')

if elements:
    print(f"Ditemukan {len(elements)} elemen")
    
    # 2. Convert ke chunks
    chunks = convert_elements_to_chunks(elements, include_metadata=True)
    print(f"Dihasilkan {len(chunks)} chunks")
    
    # 3. Save ke JSON
    output_path = 'data/chunked/element_based/dokumen_chunks.json'
    success = save_chunks(chunks, output_path)
    
    if success:
        print(f"Berhasil disimpan ke {output_path}")
```

#### 3. Hanya ekstrak text (tanpa metadata)

```python
from src.chunking import partition_document, convert_elements_to_text_list

# Partition dan convert ke text list
elements = partition_document('data/raw/dokumen.pdf', strategy='hi_res')
text_chunks = convert_elements_to_text_list(elements)

# Cetak beberapa chunk pertama
for i, text in enumerate(text_chunks[:5]):
    print(f"\n--- Chunk {i} ---")
    print(text)
```

## 📊 Output

### Format JSON Output
Setiap PDF yang diproses menghasilkan file JSON dengan format:

```json
[
  {
    "chunk_id": 0,
    "text": "Teks dari elemen pertama...",
    "metadata": {
      "element_type": "Title",
      "element_id": "abc123",
      "page_number": 1,
      "coordinates": "...",
      "filename": "dokumen.pdf"
    }
  },
  {
    "chunk_id": 1,
    "text": "Teks dari paragraf...",
    "metadata": {
      "element_type": "Paragraph",
      "page_number": 1,
      "filename": "dokumen.pdf"
    }
  }
]
```

### Tipe Elemen yang Dideteksi
- `Title`: Judul/heading
- `Paragraph`: Paragraf teks
- `ListItem`: Item dalam list
- `Table`: Tabel (dengan struktur)
- `Header`: Header halaman
- `Footer`: Footer halaman
- `PageBreak`: Pemisah halaman
- Dan lainnya...

### Contoh Output Log
```
INFO - Memulai Element-Based Chunking Pipeline
INFO - Input directory: data/raw
INFO - Output directory: data/chunked/element_based
INFO - Strategy: hi_res
INFO - Ditemukan 10 file PDF di data/raw

[1/10] Processing: dokumen1.pdf
INFO - Memulai partitioning dokumen: dokumen1.pdf
INFO - Strategi: hi_res
INFO - Berhasil mempartisi dokumen: 156 elemen ditemukan
INFO - Distribusi tipe elemen:
INFO -   - Paragraph: 98
INFO -   - Title: 24
INFO -   - Table: 15
INFO -   - ListItem: 19
INFO - Berhasil konversi 156 chunks dari 156 elemen
INFO - Total karakter: 45890
INFO - Rata-rata karakter per chunk: 294.17
INFO - ✓ Berhasil menyimpan 156 chunks ke: data/chunked/element_based/dokumen1_chunks.json

...

INFO - Element-Based Chunking Selesai
INFO - Total file PDF: 10
INFO - Berhasil diproses: 10
INFO - Total chunks dihasilkan: 1523
INFO - Durasi: 125.43 detik
INFO - Rata-rata chunks per dokumen: 152.30
```

## 📦 Dependencies

```bash
# Install Unstructured dengan dukungan PDF
pip install "unstructured[pdf]"

# Atau install dependencies lengkap untuk hi_res strategy
pip install unstructured
pip install "unstructured[pdf]"
pip install unstructured-inference
pip install pdf2image
pip install pytesseract
pip install pillow
```

### Dependencies Tambahan untuk Hi-Res Strategy

**Windows:**
```bash
# Install poppler (untuk pdf2image)
# Download dari: https://github.com/oschwartz10612/poppler-windows/releases
# Extract dan tambahkan ke PATH

# Install Tesseract OCR
# Download dari: https://github.com/UB-Mannheim/tesseract/wiki
# Install dan tambahkan ke PATH
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr

# MacOS
brew install poppler tesseract
```

## ⚙️ Perbandingan Strategy

| Strategy | Kecepatan | Akurasi | Use Case |
|----------|-----------|---------|----------|
| `hi_res` | Lambat | Sangat Tinggi | **Recommended** - Dokumen kompleks dengan tabel/struktur |
| `fast` | Cepat | Sedang | Dokumen sederhana, perlu speed |
| `auto` | Variabel | Tinggi | Otomatis pilih berdasarkan konten |
| `ocr_only` | Lambat | Tinggi | Dokumen scan/gambar |

## 💡 Tips

1. **Hi-Res Strategy**: Default dan recommended untuk akurasi maksimal dalam ekstraksi tabel dan struktur
2. **Batch Processing**: Gunakan `run_element_based_chunking()` untuk efisiensi saat proses banyak file
3. **Skip Existing**: Default skip file yang sudah diproses untuk efisiensi
4. **Metadata**: Sangat berguna untuk tracking sumber chunk saat retrieval
5. **Memory**: Untuk PDF sangat besar, proses satu per satu dengan `process_single_pdf()`

## 🐛 Troubleshooting

### Error: "Library 'unstructured' tidak terinstall"
```bash
pip install "unstructured[pdf]"
```

### Error: "poppler not found" atau "tesseract not found"
- Install poppler dan tesseract sesuai OS Anda (lihat Dependencies)
- Pastikan sudah ditambahkan ke PATH

### Proses sangat lambat
- Strategy `hi_res` memang lebih lambat tapi lebih akurat
- Gunakan `strategy='fast'` jika perlu kecepatan
- Atau proses satu per satu di background

### Tabel tidak terdeteksi
- Pastikan menggunakan `strategy='hi_res'`
- Parameter `infer_table_structure=True` sudah diset (default)
- Parameter `extract_image_block_types=["table"]` sudah diset (default)

### Hasil chunks terlalu banyak
- Ini normal untuk element-based, setiap elemen jadi 1 chunk
- Untuk chunk lebih besar, pertimbangkan recursive atau semantic chunking
