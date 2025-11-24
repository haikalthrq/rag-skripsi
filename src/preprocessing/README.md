# Modul Preprocessing

Modul ini menyediakan pipeline lengkap untuk preprocessing dokumen PDF, meliputi ekstraksi teks dan pembersihan teks menggunakan regex.

## 📁 Struktur Modul

```
src/preprocessing/
├── __init__.py          # Export fungsi utama
├── pdf_extractor.py     # Ekstraksi teks dari PDF
├── text_cleaner.py      # Pembersihan teks dengan regex
└── pipeline.py          # Pipeline preprocessing lengkap
```

## 🔧 Fungsi Utama

### 1. Ekstraksi PDF (`pdf_extractor.py`)

#### `extract_text(pdf_path: str) -> Optional[str]`
Mengekstrak teks dari file PDF menggunakan PyMuPDF.

**Parameter:**
- `pdf_path`: Path ke file PDF yang akan diekstrak

**Return:**
- String berisi teks yang diekstrak, atau `None` jika gagal

**Contoh:**
```python
from src.preprocessing import extract_text

text = extract_text('data/raw/dokumen.pdf')
print(f"Berhasil ekstrak {len(text)} karakter")
```

#### `extract_text_with_metadata(pdf_path: str) -> Optional[dict]`
Mengekstrak teks beserta metadata dokumen PDF.

**Return:**
- Dictionary berisi `text`, `metadata`, dan `filename`

### 2. Pembersihan Teks (`text_cleaner.py`)

#### `clean_text(text: str) -> str`
Membersihkan teks hasil ekstraksi PDF menggunakan berbagai pola regex.

**Proses pembersihan meliputi:**
- ✅ Menghapus nomor halaman (Page 1, Halaman 1, 1 of 10, dll)
- ✅ Menghapus header dan footer umum
- ✅ Menghapus URL dan email
- ✅ Menghapus karakter kontrol dan tidak relevan
- ✅ Menghapus whitespace berlebih
- ✅ Normalisasi line breaks
- ✅ Menghapus bullet points berlebih
- ✅ Menghapus referensi footnote/endnote
- ✅ Menghapus pola repetitif (... --- ___ ===)

**Contoh:**
```python
from src.preprocessing import clean_text

raw_text = """
Page 1
Ini adalah contoh     teks.


Dengan whitespace    berlebih.
"""

cleaned = clean_text(raw_text)
print(cleaned)
# Output: "Ini adalah contoh teks.\nDengan whitespace berlebih."
```

#### `clean_text_advanced(text: str, remove_numbers: bool, remove_punctuation: bool) -> str`
Pembersihan teks dengan opsi tambahan untuk menghapus angka dan tanda baca.

#### `remove_headers_footers(text: str, pattern_list: list) -> str`
Menghapus header/footer spesifik berdasarkan pola regex yang diberikan.

### 3. Pipeline Lengkap (`pipeline.py`)

#### `run_preprocessing(input_dir, output_dir, save_metadata, skip_existing) -> dict`
Menjalankan pipeline preprocessing lengkap untuk semua PDF dalam direktori.

**Parameter:**
- `input_dir` (str): Direktori berisi file PDF (default: `'data/raw'`)
- `output_dir` (str): Direktori output teks bersih (default: `'data/cleaned_text'`)
- `save_metadata` (bool): Simpan metadata PDF (default: `False`)
- `skip_existing` (bool): Skip file yang sudah diproses (default: `True`)

**Return:**
- Dictionary berisi statistik: `total_files`, `processed`, `skipped`, `failed`, `duration`

**Contoh:**
```python
from src.preprocessing import run_preprocessing

stats = run_preprocessing(
    input_dir='data/raw',
    output_dir='data/cleaned_text',
    save_metadata=False,
    skip_existing=True
)

print(f"Berhasil: {stats['processed']}, Gagal: {stats['failed']}")
```

#### `run_preprocessing_single(pdf_path, output_dir, save_metadata) -> bool`
Memproses satu file PDF saja.

#### `process_single_pdf(pdf_path, output_dir, save_metadata) -> Tuple[bool, Optional[str]]`
Memproses satu file PDF dan return status serta path output.

## 🚀 Cara Penggunaan

### Sebagai Script (Command Line)

#### 1. Proses semua PDF dalam folder (batch mode)

```bash
# Menggunakan default directories (data/raw -> data/cleaned_text)
python preprocess.py

# Dengan custom directories
python preprocess.py --input data/raw --output data/cleaned_text

# Dengan menyimpan metadata
python preprocess.py --metadata

# Force reprocess file yang sudah ada
python preprocess.py --no-skip
```

#### 2. Proses satu file PDF saja

```bash
python preprocess.py --single "data/raw/dokumen.pdf"
```

#### 3. Jalankan langsung dari modul

```bash
python -m src.preprocessing.pipeline --input data/raw --output data/cleaned_text
```

### Sebagai Import dalam Python

#### 1. Import dan jalankan batch processing

```python
from src.preprocessing import run_preprocessing

# Proses semua PDF
stats = run_preprocessing(
    input_dir='data/raw',
    output_dir='data/cleaned_text'
)

print(f"Total: {stats['total_files']}")
print(f"Berhasil: {stats['processed']}")
print(f"Gagal: {stats['failed']}")
print(f"Durasi: {stats['duration']:.2f} detik")
```

#### 2. Proses individual dengan kontrol penuh

```python
from src.preprocessing import extract_text, clean_text
from pathlib import Path

# Ekstrak teks
pdf_path = 'data/raw/dokumen.pdf'
raw_text = extract_text(pdf_path)

# Bersihkan teks
cleaned_text = clean_text(raw_text)

# Simpan hasil
output_path = Path('data/cleaned_text') / (Path(pdf_path).stem + '.txt')
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(cleaned_text, encoding='utf-8')

print(f"Hasil disimpan ke: {output_path}")
```

#### 3. Dengan custom text cleaning

```python
from src.preprocessing import extract_text, clean_text_advanced, remove_headers_footers

# Ekstrak
text = extract_text('data/raw/dokumen.pdf')

# Custom header/footer patterns
custom_patterns = [
    r'^\s*Badan Pusat Statistik.*$',
    r'^\s*www\.example\.com.*$',
]
text = remove_headers_footers(text, custom_patterns)

# Advanced cleaning (hapus angka dan tanda baca)
text = clean_text_advanced(text, remove_numbers=True, remove_punctuation=False)

print(f"Teks bersih: {len(text)} karakter")
```

## 📊 Output

### File Output
Setiap PDF yang diproses akan menghasilkan:
- **File teks bersih**: `data/cleaned_text/<nama_pdf>.txt`
- **File metadata** (opsional): `data/cleaned_text/<nama_pdf>_metadata.txt`

### Log File
Log preprocessing disimpan di: `preprocessing_<timestamp>.log`

### Contoh Output Log
```
INFO - Memulai Pipeline Preprocessing
INFO - Input directory: data/raw
INFO - Output directory: data/cleaned_text
INFO - Ditemukan 10 file PDF di data/raw

[1/10] Processing: dokumen1.pdf
INFO - Berhasil mengekstrak 45 halaman dari dokumen1.pdf
INFO - Total karakter yang diekstrak: 125340
INFO - Pembersihan teks selesai:
INFO -   - Karakter awal: 125340
INFO -   - Karakter akhir: 118920
INFO -   - Pengurangan: 5.12%
INFO - ✓ Berhasil disimpan ke: data/cleaned_text/dokumen1.txt

...

INFO - Preprocessing Selesai
INFO - Total file PDF: 10
INFO - Berhasil diproses: 10
INFO - Di-skip (sudah ada): 0
INFO - Gagal: 0
INFO - Durasi: 45.32 detik
```

## 🔍 Detail Regex Patterns

Berikut adalah pattern regex yang digunakan untuk pembersihan:

| Pattern | Deskripsi |
|---------|-----------|
| `\b[Pp]age\s+\d+\b` | Nomor halaman "Page 1" |
| `\b[Hh]alaman\s+\d+\b` | Nomor halaman "Halaman 1" |
| `\b\d+\s+of\s+\d+\b` | Format "1 of 10" |
| `^[\s\-_=]{3,}$` | Garis horizontal separator |
| `©\s*\d{4}.*$` | Copyright notices |
| `http[s]?://...` | URL |
| `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}` | Email |
| `\[\d+\]` | Referensi footnote [1] |
| `[ \t]{2,}` | Multiple spaces |
| `\n{3,}` | Multiple newlines |

## 📦 Dependencies

```
PyMuPDF (fitz)  # Untuk ekstraksi PDF
```

## 🧪 Testing

### Test individual functions

```python
# Test ekstraksi
from src.preprocessing.pdf_extractor import extract_text
text = extract_text('data/raw/test.pdf')
print(f"Extracted {len(text)} characters")

# Test pembersihan
from src.preprocessing.text_cleaner import clean_text
test_text = "Page 1\n\nIni teks     dengan    spasi.\n\n\n"
cleaned = clean_text(test_text)
print(cleaned)
```

### Test pipeline lengkap

```python
from src.preprocessing import run_preprocessing_single

success = run_preprocessing_single(
    'data/raw/test.pdf',
    'data/cleaned_text'
)
print(f"Success: {success}")
```

## 💡 Tips

1. **Batch Processing**: Untuk efisiensi, gunakan `run_preprocessing()` yang otomatis skip file yang sudah diproses
2. **Custom Cleaning**: Jika perlu pattern khusus, gunakan `remove_headers_footers()` dengan custom patterns
3. **Logging**: Check file log untuk detail setiap proses dan debugging
4. **Memory**: Untuk PDF besar, proses dilakukan per-file untuk efisiensi memory

## 🐛 Troubleshooting

### Error: "File tidak ditemukan"
- Pastikan path file benar
- Gunakan absolute path atau relative path dari root project

### Error: "File bukan PDF"
- Pastikan ekstensi file adalah `.pdf`
- Check file tidak corrupt

### Hasil teks kosong
- Check log untuk detail error
- Pastikan PDF bukan hasil scan (gunakan OCR terlebih dahulu)
- Beberapa PDF memiliki proteksi yang mencegah ekstraksi teks

### Teks hasil tidak sempurna
- Sesuaikan regex patterns di `text_cleaner.py`
- Tambahkan custom patterns dengan `remove_headers_footers()`
