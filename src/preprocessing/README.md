# Modul Preprocessing

Pipeline ini mengekstrak PDF dan membersihkan teks sebelum chunking.

## Penggunaan

Jalankan dari root repository:

```bash
python -m src.preprocessing.pipeline --input data/raw --output data/cleaned
```

Default `output_dir` di API dan CLI adalah `data/cleaned`, sama dengan input
default chunking MaxMin dan recursive.

Opsi CLI:

```bash
python -m src.preprocessing.pipeline --metadata
python -m src.preprocessing.pipeline --no-skip
python -m src.preprocessing.pipeline --single data/raw/dokumen.pdf --output data/cleaned
```

API utama:

```python
from src.preprocessing import run_preprocessing

stats = run_preprocessing(
    input_dir="data/raw",
    output_dir="data/cleaned",
    save_metadata=False,
    skip_existing=True,
)
```

`stats` memuat `total_files`, `processed`, `skipped`, `failed`, `output_files`,
dan `duration` ketika PDF ditemukan. Untuk input tanpa PDF, hasil tidak memuat
`output_files`.

## Mode Ekstraksi

`save_metadata` juga memilih extractor, bukan sekadar mengaktifkan file tambahan:

- `False` memakai `extract_text()`: tabel diekstrak terstruktur, narasi diambil
  dari blok di luar tabel, dan `<<<PAGE_N>>>` ditambahkan. Heuristik dua kolom
  menganggap dokumen bilingual dan membuang kolom kanan.
- `True` memakai `extract_text_with_metadata()`: setiap halaman dibaca dengan
  `page.get_text()`, tanpa ekstraksi tabel hybrid dan tanpa penanda halaman,
  lalu metadata dokumen disimpan terpisah.

Kedua mode dapat menghasilkan corpus teks yang berbeda secara material.

## Pembersihan Teks

`clean_text()` menghapus atau menormalkan BOM/karakter zero-width, karakter
kontrol, pola nomor halaman umum (`Page 1`, `Halaman 1`, `1 of 10`), URL, email,
copyright, referensi `[1]`, bullet di awal baris, separator, spasi, tab, newline,
dan pola repetitif tertentu.

Fungsi ini sengaja tidak:

- menghapus penanda `<<<PAGE_N>>>`, karena dipakai chunker untuk metadata halaman;
- menghapus semua angka atau tanda baca;
- menghapus header/footer spesifik BPS seperti nama instansi, katalog, ISSN, atau
  ISBN.

Untuk header/footer spesifik, panggil `remove_headers_footers()` secara eksplisit.
`clean_text_advanced()` menyediakan opsi terpisah untuk menghapus angka atau
tanda baca.

## Output Dan Log

Dengan `--output data/cleaned`, setiap PDF menghasilkan:

- `data/cleaned/<nama_pdf>.txt`
- `data/cleaned/<nama_pdf>_metadata.txt` jika `--metadata` digunakan dan metadata
  tersedia

Log dibuat di `logs/preprocessing_<timestamp>.log`, relatif terhadap working
directory saat proses dimulai.

Saat ini `pipeline.py` memanggil `setup_logging()` pada level modul. Akibatnya,
`import src.preprocessing` juga membuat direktori/file log dan memasang handler,
meskipun pipeline belum dijalankan.

## Exit Code Dan Skip

Batch CLI keluar dengan status 0 hanya jika sedikitnya satu PDF diproses berhasil.
Jika semua output sudah ada dan seluruh PDF di-skip, statusnya 1 walaupun tidak
ada kegagalan pemrosesan. Input tanpa PDF dan batch dengan semua proses gagal juga
keluar dengan status 1. Gunakan isi `stats` saat memanggil API untuk membedakan
kondisi tersebut.

Dependency ekstraksi PDF: PyMuPDF (`fitz`). PDF hasil scan memerlukan OCR sebelum
pipeline ini digunakan.
