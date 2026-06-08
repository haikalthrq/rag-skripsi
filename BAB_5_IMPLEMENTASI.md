# BAB 5 IMPLEMENTASI

## 5.1 Implementasi Lingkungan dan Struktur Sistem

Implementasi sistem dilakukan dalam workspace `rag-skripsi` dengan struktur kode yang memisahkan proses prapemrosesan dokumen, pembentukan chunk, embedding, penyimpanan vektor, pipeline RAG, ground truth, serta evaluasi. Pemisahan ini digunakan agar setiap tahap dapat dijalankan dan diaudit secara terpisah. Alur implementasi dimulai dari dokumen PDF pada `data/raw/`, dilanjutkan dengan ekstraksi dan pembersihan teks, pembentukan chunk dengan tiga metode, pembuatan embedding, pemuatan embedding ke ChromaDB, eksekusi pipeline RAG, serta evaluasi melalui antarmuka Streamlit.

Struktur implementasi sistem dapat dilihat pada Tabel 5.1 berikut.

Tabel 5.1 Struktur implementasi sistem

<table>
<thead>
<tr><th>No</th><th>Bagian Sistem</th><th>Path/Artefak</th><th>Fungsi Implementasi</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Dokumen sumber</td><td><code>data/raw/*.pdf</code></td><td>Menyimpan dokumen publikasi BPS yang menjadi input awal sistem.</td></tr>
<tr><td>2</td><td>Prapemrosesan</td><td><code>src/preprocessing/</code></td><td>Mengekstrak, membersihkan, dan menyimpan teks hasil ekstraksi PDF.</td></tr>
<tr><td>3</td><td>Chunking</td><td><code>src/chunking/</code></td><td>Membentuk chunk dokumen menggunakan metode element-based, max-min semantic, dan recursive.</td></tr>
<tr><td>4</td><td>Embedding</td><td><code>src/embedding/</code></td><td>Mengubah chunk dan query menjadi representasi vektor menggunakan model embedding.</td></tr>
<tr><td>5</td><td>Vector store</td><td><code>src/chroma/</code> dan <code>data/chroma/</code></td><td>Mengelola penyimpanan dan pencarian vektor pada ChromaDB.</td></tr>
<tr><td>6</td><td>Pipeline RAG</td><td><code>src/rag/</code></td><td>Menggabungkan proses retrieval, penyusunan konteks, dan generation jawaban.</td></tr>
<tr><td>7</td><td>Evaluasi dan antarmuka</td><td><code>src/evaluation/</code> dan <code>src/streamlit/rag_chat.py</code></td><td>Menghitung metrik retrieval dan generation melalui alur evaluasi final berbasis Streamlit.</td></tr>
<tr><td>8</td><td>Hasil evaluasi</td><td><code>results/final/generation/*.csv</code></td><td>Menyimpan hasil evaluasi final yang digunakan untuk visualisasi dan pembahasan.</td></tr>
</tbody>
</table>

Penjelasan struktur implementasi sistem dapat dilihat pada Tabel 5.2 berikut.

Tabel 5.2 Penjelasan struktur implementasi sistem

<table>
<thead>
<tr><th>Bagian Struktur</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Bagian 1-3</td><td>Menunjukkan alur awal sistem dari dokumen PDF, tahap prapemrosesan teks, sampai pembentukan chunk dengan tiga metode yang dibandingkan.</td></tr>
<tr><td>Bagian 4-6</td><td>Menunjukkan komponen inti RAG, yaitu embedding, penyimpanan vektor, retrieval, penyusunan konteks, dan generation jawaban.</td></tr>
<tr><td>Bagian 7-8</td><td>Menunjukkan komponen evaluasi dan penyimpanan hasil final yang menjadi dasar visualisasi serta analisis pada bab berikutnya.</td></tr>
</tbody>
</table>

Struktur tersebut digunakan sebagai dasar penulisan Bab 5. Fokus pembahasan pada bab ini adalah implementasi aktual dalam kode, bukan pembahasan teori atau analisis hasil. Pembahasan nilai metrik dan perbandingan performa metode chunking tidak dimasukkan pada bab ini karena termasuk pembahasan Bab 6.

## 5.2 Implementasi Prapemrosesan Dokumen

Prapemrosesan dokumen diimplementasikan pada modul `src/preprocessing/`. Input tahap ini berupa file PDF publikasi BPS pada `data/raw/`. Implementasi membaca PDF menggunakan PyMuPDF, mengekstrak teks per halaman, mempertahankan penanda halaman dalam format `<<<PAGE_N>>>`, kemudian membersihkan teks hasil ekstraksi. Output prapemrosesan digunakan sebagai input tahap chunking dan disimpan pada direktori hasil prapemrosesan sesuai konfigurasi eksekusi pipeline.

Implementasi prapemrosesan dokumen dapat dilihat pada Tabel 5.3 berikut.

Tabel 5.3 Kode sumber implementasi prapemrosesan dokumen

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 1: Ekstraksi Teks PDF</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/preprocessing/pdf_extractor.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">doc = fitz.open(str(pdf_path_obj))</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">extracted_text = []</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">page_count = len(doc)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">for page_num in range(page_count):</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    page = doc[page_num]</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    page_text = _extract_page_hybrid(page)</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    if page_text.strip():</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">        extracted_text.append(f"&lt;&lt;&lt;PAGE_{page_num + 1}&gt;&gt;&gt;\n{page_text}")</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">doc.close()</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">full_text = "\n".join(extracted_text)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 2: Pembersihan Teks</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/preprocessing/text_cleaner.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">cleaned = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', cleaned)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">cleaned = re.sub(r'\b[Pp]age\s+\d+\b', '', cleaned)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">cleaned = re.sub(r'\b[Hh]alaman\s+\d+\b', '', cleaned)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">cleaned = re.sub(r'(?&lt;=\n)\n(\s*\d{1,3}\s*)\n(?=\n)', '\n', cleaned)</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">cleaned = re.sub(r'^[ \t]+', '', cleaned, flags=re.MULTILINE)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 3: Pipeline Prapemrosesan</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/preprocessing/pipeline.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">raw_text = extract_text(str(pdf_path))</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">if not raw_text:</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    return False, None</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">cleaned_text = clean_text(raw_text)</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">if not cleaned_text or len(cleaned_text.strip()) == 0:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    return False, None</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">output_filename = pdf_path.stem + ".txt"</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">output_file = Path(output_dir) / output_filename</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">with open(output_file, 'w', encoding='utf-8') as f:</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    f.write(cleaned_text)</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber prapemrosesan dokumen dapat dilihat pada Tabel 5.4 berikut.

Tabel 5.4 Penjelasan kode sumber prapemrosesan dokumen

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-14</td><td>Dokumen PDF dibuka menggunakan `fitz.open()`. Setiap halaman diekstrak dengan `_extract_page_hybrid()`, lalu hasilnya diberi penanda halaman. Penanda ini penting karena tahap chunking dapat menggunakannya untuk metadata halaman.</td></tr>
<tr><td>Baris 1-8</td><td>Fungsi pembersihan menghapus karakter tidak terlihat, pola nomor halaman umum, serta spasi berlebih. Penanda `<<<PAGE_N>>>` tidak dihapus agar informasi halaman tetap dapat diteruskan ke tahap berikutnya.</td></tr>
<tr><td>Baris 1-13</td><td>Pipeline memanggil ekstraksi PDF, membersihkan teks, memvalidasi agar output tidak kosong, lalu menulis file `.txt` ke direktori output.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, tahap prapemrosesan berfungsi sebagai penghubung antara dokumen PDF dan tiga metode chunking. Tahap ini tidak melakukan penilaian relevansi dan tidak menghitung metrik evaluasi.

## 5.3 Implementasi Metode Chunking

Tiga metode chunking diimplementasikan secara terpisah agar dapat dibandingkan pada tahap evaluasi. Modul `src/chunking/` menyediakan fungsi untuk Element-Based Chunking, Max-Min Semantic Chunking, dan Recursive Chunking. Pemisahan fungsi ini membuat setiap metode menghasilkan file chunk JSON pada subfolder berbeda, yaitu `data/chunked/element_based/`, `data/chunked/maxmin_semantic/`, dan `data/chunked/recursive/`.

Kode sumber pemanggilan metode chunking dapat dilihat pada Tabel 5.5 berikut.

Tabel 5.5 Kode sumber pemanggilan metode chunking

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 4: Pemanggilan Metode Chunking</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/__init__.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">from .element_based import (</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    partition_document,</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    convert_elements_to_chunks,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    run_element_based_chunking</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">from .maxmin_chunker import (</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    split_sentences,</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    embed_sentences,</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    apply_maxmin_chunking,</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    run_maxmin_chunking</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">from .recursive_split import (</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    create_text_splitter,</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">    run_recursive_splitter,</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">    run_recursive_chunking</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber pemanggilan metode chunking dapat dilihat pada Tabel 5.6 berikut.

Tabel 5.6 Penjelasan kode sumber pemanggilan metode chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-6</td><td>Mengekspos fungsi utama Element-Based Chunking dari `element_based.py`.</td></tr>
<tr><td>Baris 8-13</td><td>Mengekspos fungsi utama Max-Min Semantic Chunking dari `maxmin_chunker.py`.</td></tr>
<tr><td>Baris 15-19</td><td>Mengekspos fungsi utama Recursive Chunking dari `recursive_split.py`.</td></tr>
</tbody>
</table>

Subbab berikutnya menjelaskan implementasi masing-masing metode secara lebih spesifik berdasarkan file sumber yang aktif.

### 5.3.1 Implementasi Element-Based Chunking

Element-Based Chunking diimplementasikan pada `src/chunking/element_based.py`. Metode ini memproses dokumen PDF dengan `partition_pdf` dari `unstructured`, kemudian mengelompokkan elemen dokumen menjadi chunk berbasis struktur. Elemen seperti judul, teks, daftar, dan tabel diperlakukan berbeda. Tabel disimpan sebagai unit mandiri, sedangkan elemen teks dapat digabung menjadi composite chunk. Metadata yang disimpan meliputi tipe chunk, tipe elemen, judul bagian, nomor halaman, dan sumber file.

Implementasi Element-Based Chunking dapat dilihat pada Tabel 5.7 berikut.

Tabel 5.7 Kode sumber implementasi Element-Based Chunking

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 5: Partisi PDF Element-Based</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/element_based.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">elements = partition_pdf(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    filename=pdf_path,</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    strategy=strategy,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    infer_table_structure=True,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    extract_image_block_types=["table"],</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    extract_images_in_pdf=False,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    include_page_breaks=True,</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    languages=languages,</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 6: Inisialisasi Chunk Element-Based</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/element_based.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def init_chunk():</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    return {</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">        'text': '',</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">        'metadata': {</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">            'chunk_type': 'text',</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">            'element_types': [],</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">            'section_title': active_title,</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">            'page_numbers': [],</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">            'source_file': current_source_file,</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">            'source_filename': current_source_file,</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">            'element_count': 0,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">            'order_index': -1,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        }</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    }</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 7: Pemrosesan Elemen Tabel</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/element_based.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">if category == 'table':</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    flush_chunk()</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    current_chunk['text'] = text.strip()</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    current_chunk['metadata']['chunk_type'] = 'table'</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    current_chunk['metadata']['section_title'] = active_title</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    current_chunk['metadata']['element_types'] = [elem_type]</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    current_chunk['metadata']['element_count'] = 1</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    current_chunk['metadata']['page_numbers'] = [page_num] if page_num else []</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    src = current_source_file or filename</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    current_chunk['metadata']['source_file'] = src</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    current_chunk['metadata']['source_filename'] = src</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    if elem_metadata and hasattr(elem_metadata, 'text_as_html') and elem_metadata.text_as_html:</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">        current_chunk['metadata']['text_as_html'] = elem_metadata.text_as_html</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">    flush_chunk(forced_type='table')</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Element-Based Chunking dapat dilihat pada Tabel 5.8 berikut.

Tabel 5.8 Penjelasan kode sumber Element-Based Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-10</td><td>PDF dipartisi menggunakan `partition_pdf` dengan strategi `hi_res`, struktur tabel diaktifkan, dan informasi page break disertakan.</td></tr>
<tr><td>Baris 1-15</td><td>Fungsi internal `init_chunk()` membentuk struktur awal chunk beserta metadata seperti `chunk_type`, `section_title`, `page_numbers`, dan `source_file`.</td></tr>
<tr><td>Baris 1-17</td><td>Elemen tabel diperlakukan sebagai unit tersendiri. Jika `text_as_html` tersedia pada metadata elemen, struktur HTML tabel ikut disimpan untuk dipakai pada tahap embedding atau formatting konteks.</td></tr>
</tbody>
</table>

Implementasi ini menghasilkan chunk JSON yang mempertahankan metadata struktural dokumen. Bab ini tidak membahas penilaian performa karena analisis hasil evaluasi ditempatkan pada Bab 6.

### 5.3.2 Implementasi Max-Min Semantic Chunking

Max-Min Semantic Chunking diimplementasikan pada `src/chunking/maxmin_chunker.py`. Alur implementasinya memuat teks bersih, memecah teks menjadi kalimat, membuat embedding kalimat, mengelompokkan kalimat berdasarkan kemiripan semantik, lalu menyimpan hasilnya sebagai chunk JSON. Parameter yang digunakan pada fungsi utama antara lain `fixed_threshold`, `c`, `init_constant`, `batch_size`, dan pilihan backend embedding. Uraian pada subbab ini difokuskan pada alur embedding kalimat dan pengelompokan Max-Min sebagaimana ditampilkan pada potongan kode. Detail parameter mengikuti konfigurasi fungsi yang digunakan pada proses eksekusi chunking.

Implementasi Max-Min Semantic Chunking dapat dilihat pada Tabel 5.9 berikut.

Tabel 5.9 Kode sumber implementasi Max-Min Semantic Chunking

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 8: Pemecahan Kalimat Max-Min</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/maxmin_chunker.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">sentences = sent_tokenize(text)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">sentences = [s.strip() for s in sentences if s.strip()]</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">SKIP_CHARS_LIMIT = 32000</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">filtered, skipped = [], 0</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">for s in sentences:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    if len(s) &lt;= SKIP_CHARS_LIMIT:</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">        filtered.append(s)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    else:</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">        skipped += 1</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">sentences = filtered</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 9: Pengelompokan Kalimat Max-Min</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/maxmin_chunker.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">for i in range(1, len(sentences)):</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    cluster_embeddings = embeddings[cluster_start:cluster_end]</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    cluster_size = cluster_end - cluster_start</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    new_sentence_embedding = embeddings[i].reshape(1, -1)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    new_sentence_similarities = cosine_similarity(</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">        new_sentence_embedding,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        cluster_embeddings</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    )[0]</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    final_threshold = max(adjusted_threshold, fixed_threshold)</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    if new_sentence_similarity &gt; final_threshold:</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">        current_paragraph.append(sentences[i])</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        cluster_end += 1</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    else:</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">        paragraphs.append(current_paragraph)</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">        current_paragraph = [sentences[i]]</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">        cluster_start = i</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">        cluster_end = i + 1</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 10: Pipeline Max-Min Semantic Chunking</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/maxmin_chunker.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">text = load_text(text_path)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">sentences = split_sentences(text)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">embeddings = embed_sentences(</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    sentences,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    embedding_model,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    batch_size=batch_size,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    use_gguf=use_gguf</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">paragraphs = apply_maxmin_chunking(</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    sentences,</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    embeddings,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    fixed_threshold=fixed_threshold,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    c=c,</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    init_constant=init_constant</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">chunks = convert_paragraphs_to_chunks(</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">    paragraphs,</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">    Path(text_path).name,</code></td></tr>
<tr><td>20</td><td><code style="white-space: pre;">    include_metadata=include_metadata</code></td></tr>
<tr><td>21</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Max-Min Semantic Chunking dapat dilihat pada Tabel 5.10 berikut.

Tabel 5.10 Penjelasan kode sumber Max-Min Semantic Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-12</td><td>Teks dibagi menjadi kalimat dengan `sent_tokenize`, kemudian kalimat kosong dan kalimat yang terlalu panjang akibat artefak parsing disaring.</td></tr>
<tr><td>Baris 1-19</td><td>Fungsi `process_sentences()` membentuk cluster kalimat. Kalimat baru dibandingkan dengan cluster aktif menggunakan cosine similarity, kemudian diputuskan apakah digabung atau memulai cluster baru.</td></tr>
<tr><td>Baris 1-21</td><td>Fungsi `process_single_text()` mengorkestrasi pemuatan teks, sentence splitting, embedding kalimat, penerapan Max-Min, konversi ke format chunk, dan penyimpanan output JSON.</td></tr>
</tbody>
</table>

Implementasi Max-Min menggunakan representasi embedding kalimat untuk menentukan pengelompokan semantik. Bab ini hanya menjelaskan proses implementasi, sedangkan analisis perbandingan metode ditempatkan pada Bab 6.

### 5.3.3 Implementasi Recursive Chunking

Recursive Chunking diimplementasikan pada `src/chunking/recursive_split.py` dengan `RecursiveCharacterTextSplitter`. Implementasi ini menggunakan parameter `chunk_size`, `chunk_overlap`, dan daftar separator. Teks dipotong secara rekursif berdasarkan hierarki separator, kemudian hasil potongan dikonversi menjadi dictionary dengan metadata sumber file, metode chunking, panjang chunk, dan nomor halaman jika marker halaman tersedia.

Implementasi Recursive Chunking dapat dilihat pada Tabel 5.11 berikut.

Tabel 5.11 Kode sumber implementasi Recursive Chunking

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 11: Inisialisasi Recursive Text Splitter</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/recursive_split.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">if separators is None:</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    separators = ["\n\n", "\n", " ", ""]</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">text_splitter = RecursiveCharacterTextSplitter(</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    chunk_size=chunk_size,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    chunk_overlap=chunk_overlap,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    length_function=length_function,</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    is_separator_regex=is_separator_regex,</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    separators=separators</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 12: Pemecahan Teks Recursive</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/recursive_split.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">chunks = text_splitter.split_text(text)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">if chunks:</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    chunk_sizes = [len(chunk) for chunk in chunks]</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    logger.info(f"Rata-rata karakter per chunk: {sum(chunk_sizes) / len(chunks):.2f}")</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    logger.info(f"Min karakter per chunk: {min(chunk_sizes)}")</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    logger.info(f"Max karakter per chunk: {max(chunk_sizes)}")</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 13: Konversi Chunk Recursive</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chunking/recursive_split.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">page_numbers = sorted({int(m) for m in _PAGE_MARKER.findall(chunk_text)})</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">chunk_text = _PAGE_MARKER.sub('', chunk_text).strip()</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">chunk_dict = {</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    'chunk_id': chunk_id,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    'text': chunk_text,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    'num_characters': len(chunk_text)</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">}</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">chunk_dict['metadata'] = {</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    'source_file': source_filename,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    'chunking_method': 'recursive_character_text_splitter',</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    'chunk_length': len(chunk_text),</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    'page_numbers': page_numbers if page_numbers else None,</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">}</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Recursive Chunking dapat dilihat pada Tabel 5.12 berikut.

Tabel 5.12 Penjelasan kode sumber Recursive Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-11</td><td>Splitter dibuat menggunakan `RecursiveCharacterTextSplitter` dengan parameter ukuran chunk, overlap, dan separator.</td></tr>
<tr><td>Baris 1-8</td><td>Fungsi `run_recursive_splitter()` menjalankan `split_text()` dan mencatat statistik panjang chunk.</td></tr>
<tr><td>Baris 1-16</td><td>Hasil chunk string dikonversi menjadi dictionary. Marker halaman diekstrak untuk metadata dan dihapus dari teks chunk.</td></tr>
</tbody>
</table>

Hasil dari metode ini berupa file JSON pada `data/chunked/recursive/`. Metadata yang disimpan membuat chunk tetap dapat ditelusuri ke file sumber dan halaman asalnya.

## 5.4 Implementasi Embedding

Tahap embedding diimplementasikan pada `src/embedding/`. Model embedding yang digunakan dalam implementasi adalah Qwen3-Embedding-4B, baik melalui backend HuggingFace maupun GGUF sesuai environment. Fungsi `QwenEmbedder` menerima teks tunggal atau daftar teks, menghasilkan embedding, dan melakukan normalisasi L2 jika parameter `normalize` aktif. Tahap batch embedding membaca chunk JSON, melakukan enrichment pada chunk tabel jika metadata HTML tersedia, menambahkan context prefix untuk metode Max-Min dan Recursive, lalu menyimpan embedding ke `data/embeddings/`.

Implementasi embedding dapat dilihat pada Tabel 5.13 berikut.

Tabel 5.13 Kode sumber implementasi embedding

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 14: Inisialisasi Qwen Embedder</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/embedding/embedder.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def embed(self, texts, batch_size=32):</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    if isinstance(texts, str):</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">        texts = [texts]</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    if len(texts) == 0:</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">        return np.array([])</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    if self.mode == 'gguf':</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">        embeddings = self._embed_gguf(texts)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    elif self.mode == 'huggingface':</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">        embeddings = self._embed_hf(texts)</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    else:</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        raise ValueError(f"Unknown mode: {self.mode}")</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    if self.normalize:</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">        embeddings = self._normalize_embeddings(embeddings)</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">    return embeddings</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 15: Enrichment Chunk Tabel</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/embedding/io.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">html = meta.get("text_as_html") or ""</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">if not html.strip():</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    continue</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">if meta.get("chunk_type") != "table":</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    continue</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">table_text = _html_table_to_text(html)</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">if not table_text.strip():</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    continue</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">section_title = (meta.get("section_title") or "").strip()</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">if section_title and not _is_noise_text(section_title):</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    prefix = section_title + "\n\n"</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">chunk["text"] = prefix + table_text</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 16: Pembuatan Embedding Chunk</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/embedding/embed_chunks.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">chunks = load_chunks_from_json(json_path)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">n_enriched = enrich_table_chunk_texts(chunks)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">cleaned_texts, valid_indices = clean_and_filter_chunks(chunks)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">valid_chunks = [chunks[i] for i in valid_indices]</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">if chunking_method in _METHODS_WITH_CONTEXT_PREFIX:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    embed_texts = inject_context_prefix(valid_chunks, CONTEXT_PREFIX_CHARS)</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">else:</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    embed_texts = [c.get("text", "") for c in valid_chunks]</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">embed_texts = [' '.join(t.split()) for t in embed_texts]</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">embeddings = embedder.embed(embed_texts)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>4</td>
<td><strong>Algoritma 17: Penyimpanan Embedding</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/embedding/embed_chunks.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">output_path = Path(output_dir) / chunking_method / f"{json_file.stem}_embeddings.json"</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">metadata = {</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    "source_file": json_file.name,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    "source_path": str(json_file),</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    "chunking_method": chunking_method,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    "embedding_model": embedder.mode,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    "normalized": embedder.normalize</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">}</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">success = save_embeddings(</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    embeddings=embeddings,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    chunks=chunks,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    valid_indices=valid_indices,</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    output_path=str(output_path),</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    metadata=metadata</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber embedding dapat dilihat pada Tabel 5.14 berikut.

Tabel 5.14 Penjelasan kode sumber embedding

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-19</td><td>Fungsi `embed()` menerima teks, memilih backend GGUF atau HuggingFace, lalu menormalisasi embedding bila konfigurasi `normalize` bernilai benar.</td></tr>
<tr><td>Baris 1-16</td><td>Fungsi `enrich_table_chunk_texts()` memanfaatkan `text_as_html` untuk membentuk teks tabel yang lebih terstruktur dan menambahkan konteks judul jika tersedia.</td></tr>
<tr><td>Baris 1-13</td><td>Fungsi `embed_single_file()` membaca chunk JSON, melakukan enrichment, membersihkan teks, menerapkan context prefix untuk Max-Min dan Recursive, lalu membuat embedding.</td></tr>
<tr><td>Baris 1-17</td><td>Embedding dan metadata disimpan ke file JSON dalam subfolder metode chunking pada `data/embeddings/`.</td></tr>
</tbody>
</table>

Tahap embedding tidak mengubah file chunk asli pada `data/chunked/`. Enrichment dan context prefix digunakan pada teks yang di-embed dan disimpan untuk kebutuhan downstream, terutama ChromaDB.

## 5.5 Implementasi Vector Database dan Retrieval

Vector database diimplementasikan menggunakan ChromaDB pada modul `src/chroma/`. Embedding JSON dari `data/embeddings/` dimuat ke persistent storage `data/chroma/`. Setiap metode chunking memiliki collection terpisah, yaitu `collection_element_based`, `collection_maxmin_semantic`, dan `collection_recursive`. Retrieval dilakukan dengan query embedding yang dikirim ke `collection.query()`, kemudian hasilnya dikembalikan sebagai daftar dictionary berisi `id`, `document`, `metadata`, dan `distance`.

Implementasi Vector Database dan Retrieval dapat dilihat pada Tabel 5.15 berikut.

Tabel 5.15 Kode sumber implementasi Vector Database dan Retrieval

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 18: Inisialisasi ChromaDB Client</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chroma/client.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">persist_path = Path(persist_directory)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">persist_path.mkdir(parents=True, exist_ok=True)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">_recover_stale_sqlite_journal(persist_path)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">client = chromadb.PersistentClient(</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    path=str(persist_path.absolute())</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 19: Persiapan Data Embedding</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chroma/loader.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">embeddings = np.array(embeddings_list, dtype=np.float32)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">ids = []</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">documents = []</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">metadatas = []</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">for chunk in chunks:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    chunk_id = f"{file_path.stem}_{chunk.get('original_index', chunk.get('embedding_index', 0))}"</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    ids.append(chunk_id)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    documents.append(chunk.get('text', ''))</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    chunk_metadata = chunk.get('metadata', {}).copy()</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    chunk_metadata['source_file'] = metadata.get('source_file', file_path.stem)</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    chunk_metadata['chunking_method'] = metadata.get('chunking_method', 'unknown')</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    metadatas.append(chunk_metadata)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 20: Pemuatan Data ke ChromaDB</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chroma/loader.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">success = batch_add_documents(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    collection=collection,</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    ids=ids,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    embeddings=embeddings,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    documents=documents,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    metadatas=metadatas,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    batch_size=batch_size</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>4</td>
<td><strong>Algoritma 21: Similarity Search ChromaDB</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/chroma/query.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">results = collection.query(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    query_embeddings=[query_embedding.tolist()],</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    n_results=k,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    where=filter</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">documents = []</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">for i in range(len(results['documents'][0])):</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    doc = {</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">        'id': results['ids'][0][i],</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">        'document': results['documents'][0][i],</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        'distance': results['distances'][0][i] if 'distances' in results else None</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    }</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    documents.append(doc)</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Vector Database dan Retrieval dapat dilihat pada Tabel 5.16 berikut.

Tabel 5.16 Penjelasan kode sumber Vector Database dan Retrieval

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-8</td><td>ChromaDB diinisialisasi sebagai persistent client pada direktori `data/chroma/` sehingga data vektor dapat digunakan kembali setelah aplikasi berhenti.</td></tr>
<tr><td>Baris 1-14</td><td>File embedding JSON dibaca, embedding dikonversi ke `numpy`, lalu ID, dokumen, dan metadata chunk disiapkan untuk ChromaDB.</td></tr>
<tr><td>Baris 1-9</td><td>Data dimasukkan ke collection menggunakan `batch_add_documents()` dalam ukuran batch tertentu.</td></tr>
<tr><td>Baris 1-16</td><td>Fungsi `similarity_search()` melakukan pencarian berdasarkan query embedding dan mengubah output ChromaDB menjadi list dictionary yang digunakan oleh pipeline RAG.</td></tr>
</tbody>
</table>

Implementasi pemuatan embedding ke ChromaDB berada pada `src/chroma/loader.py`. File `scripts/load_embeddings_to_chroma.py` hanya berperan sebagai entry point yang memanggil fungsi loader tersebut, sedangkan logika penyimpanan dan retrieval tetap berada pada modul `src/chroma/`.

## 5.6 Implementasi RAG Pipeline dan Generator

Pipeline RAG diimplementasikan pada `src/rag/pipeline.py`, sedangkan generator diimplementasikan pada `src/rag/generator.py`. Alur runtime dimulai dari query pengguna atau pertanyaan QA gold, dilanjutkan dengan embedding query, retrieval chunk dari collection ChromaDB sesuai metode chunking, formatting konteks, dan generation answer. Pipeline ini juga dipanggil oleh `src/streamlit/rag_chat.py` untuk menjalankan chat dan evaluasi batch. Model generator yang digunakan oleh antarmuka tersebut diarahkan ke Qwen3-4B-Instruct-2507 sesuai konfigurasi auto-detect pada `rag_chat.py`.

Implementasi RAG Pipeline dan Generator dapat dilihat pada Tabel 5.17 berikut.

Tabel 5.17 Kode sumber implementasi RAG Pipeline dan Generator

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 22: Pemilihan Collection RAG</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/rag/pipeline.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">COLLECTION_NAMES = {</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    "element_based":   "collection_element_based",</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    "maxmin_semantic": "collection_maxmin_semantic",</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    "recursive":       "collection_recursive",</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">}</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">collection_name = COLLECTION_NAMES[chunking_method]</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">self.collection = get_or_create_collection(chroma_client, collection_name)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 23: Retrieval Query</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/rag/pipeline.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">query_embedding = self.embedder.embed(query)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">query_vec = query_embedding[0]</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">results = similarity_search(self.collection, query_vec, k=k)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">return results</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 24: Eksekusi Pipeline RAG</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/rag/pipeline.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">retrieved = self.retrieve(query, k=k)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">contexts = [self._format_context(doc) for doc in retrieved]</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">raw = self.generator.generate(query, contexts)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">if isinstance(raw, tuple):</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    answer, thinking = raw</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">else:</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    answer, thinking = raw, ""</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">return {</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    "query": query,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    "answer": answer,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    "thinking": thinking,</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    "retrieved_chunks": retrieved,</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    "chunking_method": self.chunking_method,</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">    "num_chunks": len(retrieved),</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">    "elapsed_seconds": elapsed,</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">}</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>4</td>
<td><strong>Algoritma 25: Penyusunan Prompt Generator</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/rag/generator.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">context_block = "\n\n".join(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    f"[Konteks {i + 1}]\n{ctx.strip()}"</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    for i, ctx in enumerate(contexts)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">user_content = (</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    f"Konteks:\n{context_block}\n\n"</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    f"Pertanyaan: {query}\n\n"</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    f"Jawaban:"</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">)</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">return [</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    {"role": "system", "content": self.system_prompt},</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    {"role": "user", "content": user_content},</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">]</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber RAG Pipeline dan Generator dapat dilihat pada Tabel 5.18 berikut.

Tabel 5.18 Penjelasan kode sumber RAG Pipeline dan Generator

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-9</td><td>Pipeline memetakan nama metode chunking ke collection ChromaDB yang sesuai.</td></tr>
<tr><td>Baris 1-6</td><td>Fungsi `retrieve()` membuat embedding query lalu melakukan similarity search pada collection aktif.</td></tr>
<tr><td>Baris 1-19</td><td>Fungsi `run()` menjalankan alur lengkap retrieval, formatting konteks, pemanggilan generator, dan pengembalian hasil.</td></tr>
<tr><td>Baris 1-16</td><td>Generator membangun pesan chat berisi system prompt, konteks hasil retrieval, pertanyaan, dan instruksi jawaban.</td></tr>
</tbody>
</table>

Prompt generator mengarahkan model untuk menjawab berdasarkan konteks dan menyatakan bahwa informasi tidak memadai jika konteks tidak cukup. Bagian ini hanya menjelaskan mekanisme pembentukan jawaban, bukan kualitas jawaban yang dihasilkan.

## 5.7 Implementasi Ground Truth Retrieval dan Dataset Evaluasi

Dataset evaluasi dibangun dari QA gold, kandidat chunk, anotasi manual, dan konversi label ke JSON ground truth retrieval. Kandidat chunk dibuat oleh `scripts/build_candidates_v3.py` dengan membaca QA gold dan seluruh chunk JSON dari tiga metode. Kandidat tersebut kemudian diberi label manual melalui `src/streamlit/app.py`. Hasil anotasi disimpan sebagai `retrieval_labels_final.csv` dan `retrieval_labels_final.xlsx`. Setelah itu, `scripts/convert_ground_truth_to_json.py` mengubah label final menjadi `qa_pairs_binary.json` dengan skema binary relevance, yaitu label `0` sebagai tidak relevan dan label `>= 1` sebagai relevan.

Implementasi Ground Truth Retrieval dan Dataset Evaluasi dapat dilihat pada Tabel 5.19 berikut.

Tabel 5.19 Kode sumber implementasi Ground Truth Retrieval dan Dataset Evaluasi

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 26: Pemuatan QA dan Chunk</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: scripts/build_candidates_v3.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">METHODS = ["element_based", "maxmin_semantic", "recursive"]</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">PRE_K_DEFAULT = 10</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">TOP_K_DEFAULT = 5</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">def load_qa_gold():</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    df = pd.read_excel(str(QA_XLSX), sheet_name="qa_gold", dtype=str).fillna("")</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    rows = df.to_dict("records")</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    return rows</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">def load_chunks(doc_id, method):</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    stem = DOC_MAP.get(doc_id)</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    fp = CHUNK_DIR / method / f"{stem}_chunks.json"</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    with open(fp, encoding="utf-8") as f:</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">        return json.load(f)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 27: Pembangunan Kandidat Retrieval</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: scripts/build_candidates_v3.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">for method in METHODS:</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    candidates = build_candidates_for_group(qa, method, pre_k, top_k)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    if not candidates:</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">        rows.append({</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">            "query_id": qid,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">            "doc_id": doc_id,</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">            "method": method,</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">            "chunk_id": "",</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">            "match_type": "not_found",</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">            "suggested_label": "0",</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">            "status": "needs_manual_validation",</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        })</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">        continue</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 28: Penyimpanan Label Anotasi</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/app.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">OUTPUT_XLSX = ROOT / "data/ground_truth/retrieval_labels_final.xlsx"</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">OUTPUT_CSV = ROOT / "data/ground_truth/retrieval_labels_final.csv"</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">def apply_label(qid, method, chunk_id, label):</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    df = st.session_state.df</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    mask = (</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        (df["query_id"] == qid)</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">        &amp; (df["method"] == method)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">        &amp; (df["chunk_id"] == chunk_id)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    )</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    df.loc[mask, "label"] = label</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    df.loc[mask, "annotator"] = st.session_state.get("annotator_name", "")</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    st.session_state.df = df</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">    save_data(df)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>4</td>
<td><strong>Algoritma 29: Konversi Ground Truth Retrieval</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: scripts/convert_ground_truth_to_json.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">if label &lt; threshold:</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    continue</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">pipeline_method = CSV_METHOD_TO_CODE[method_csv]</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">file_stem = DOC_TO_FILE_STEM[doc_id]</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">chroma_id = f"{file_stem}_{chunk_id_int}"</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">result[q_id][pipeline_method].append(chroma_id)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">relevant_chunk_ids = {</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    m: query_labels.get(m, []) for m in ALL_METHODS</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">}</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Ground Truth Retrieval dan Dataset Evaluasi dapat dilihat pada Tabel 5.20 berikut.

Tabel 5.20 Penjelasan kode sumber Ground Truth Retrieval dan Dataset Evaluasi

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-15</td><td>Script kandidat menetapkan tiga metode chunking, membaca QA gold, dan memuat chunk JSON sesuai `doc_id` dan metode.</td></tr>
<tr><td>Baris 1-15</td><td>Untuk setiap QA dan metode, kandidat dibangun dan disiapkan dalam format yang dapat divalidasi manual. Jika kandidat tidak ditemukan, baris tetap dibuat dengan status `needs_manual_validation`.</td></tr>
<tr><td>Baris 1-15</td><td>Aplikasi anotasi menyimpan label manual ke CSV dan XLSX final. Fungsi `apply_label()` memperbarui label berdasarkan kombinasi `query_id`, `method`, dan `chunk_id`.</td></tr>
<tr><td>Baris 1-13</td><td>Konverter membaca label final, memasukkan chunk dengan label minimal sesuai `relevance_threshold`, membentuk ID ChromaDB, dan menyusun `relevant_chunk_ids` per metode.</td></tr>
</tbody>
</table>

Ground truth retrieval yang digunakan oleh evaluasi akhir adalah `data/ground_truth/qa_pairs_binary.json`. File tersebut dibaca oleh evaluasi batch, bukan dibentuk ulang saat evaluasi dijalankan.

## 5.8 Implementasi Evaluasi Retrieval dan Generation

Evaluasi akhir diimplementasikan melalui tab Evaluasi Batch pada `src/streamlit/rag_chat.py`. Fitur ini membaca QA gold dari `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx` dan ground truth retrieval binary dari `data/ground_truth/qa_pairs_binary.json`. Evaluasi dijalankan untuk tiga metode chunking dan dapat menggunakan rentang top-k yang dipilih pada antarmuka. Proses evaluasi melakukan pre-compute query embedding, retrieval per metode, generation answer, perhitungan metrik retrieval dan generation, lalu menyimpan hasil ke CSV pada `results/final/generation/`. Implementasi evaluasi akhir tidak dijalankan dari `scripts/run_generation_eval.py` maupun `scripts/run_retrieval_eval.py`.

Implementasi evaluasi retrieval dan generation dapat dilihat pada Tabel 5.21 berikut.

Tabel 5.21 Kode sumber implementasi evaluasi retrieval dan generation

<table>
<thead>
<tr><th>No</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Algoritma 30: Pemuatan Data Evaluasi</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/rag_chat.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">EVAL_RESULTS_DIR = ROOT / "results" / "final" / "generation"</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">def _load_qa_gold():</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    qa_path = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    if qa_path.exists():</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        _QA_GOLD_DF = pd.read_excel(qa_path, sheet_name="qa_gold")</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    return _QA_GOLD_DF</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">def _load_ground_truth():</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    gt_path = ROOT / "data/ground_truth/qa_pairs_binary.json"</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    if gt_path.exists():</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        with open(gt_path, encoding="utf-8") as f:</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">            _GT_BINARY = json.load(f)</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    return _GT_BINARY</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>2</td>
<td><strong>Algoritma 31: Konfigurasi Evaluasi Batch</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/rag_chat.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">top_k_min = st.number_input("Min Top-K", min_value=1, max_value=10, value=1)</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">top_k_max = st.number_input("Max Top-K", min_value=1, max_value=10, value=10)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">def _run_eval_and_save(qa_subset, mode_tag, top_k_range):</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    gt_data = _load_ground_truth()</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    gt_lookup = {item["id"]: item for item in gt_data}</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    min_k, max_k = top_k_range</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    query_embeddings = {}</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    for _, qa_row in qa_subset.iterrows():</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">        q_id = str(qa_row["query_id"])</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">        question = str(qa_row["question"])</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">        q_vec = pipeline.embedder.embed(question)[0]</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">        query_embeddings[q_id] = (q_vec, True)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>3</td>
<td><strong>Algoritma 32: Loop Evaluasi per Metode</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/rag_chat.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">for current_k in range(min_k, max_k + 1):</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    for _, qa_row in qa_subset.iterrows():</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">        question = str(qa_row["question"])</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">        gold_ans = str(qa_row["gold_answer"])</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">        q_id = str(qa_row["query_id"])</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">        gt_item = gt_lookup.get(q_id)</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        q_vec, embed_ok = query_embeddings.get(q_id, (None, False))</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">        for method in METHODS:</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">            p = RAGPipeline(</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">                embedder=pipeline.embedder,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">                generator=pipeline.generator,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">                chroma_client=pipeline.chroma_client,</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">                chunking_method=method,</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">                top_k=current_k,</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">            )</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">            retrieved = p.retrieve_by_vector(q_vec, k=current_k) if embed_ok else p.retrieve(question, k=current_k)</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">            retrieved_ids = [doc.get("id", "") for doc in retrieved]</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>4</td>
<td><strong>Algoritma 33: Perhitungan Metrik dan Generation</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/rag_chat.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">if rel_ids:</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    precision_val = compute_precision_at_k(retrieved_ids, rel_ids, current_k)</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    recall_val = compute_recall_at_k(retrieved_ids, rel_ids, current_k)</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    mrr_val = compute_mrr(retrieved_ids, rel_ids)</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">else:</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    precision_val = recall_val = mrr_val = "N/A"</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">contexts = [p._format_context(doc) for doc in retrieved]</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">raw = pipeline.generator.generate(question, contexts)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">gen_answer = raw[0] if isinstance(raw, tuple) else raw</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">bleu_val = compute_bleu(gen_answer, gold_ans)</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">rouge_val = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>5</td>
<td><strong>Algoritma 34: Penyimpanan Hasil Evaluasi</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/streamlit/rag_chat.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">rows.append({</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    "query_id": q_id,</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    "method": METHOD_LABELS[method],</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    "question": question,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    "gold_answer": gold_ans,</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    "generated_answer": gen_answer,</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    "precision_at_k": round(precision_val, 4) if isinstance(precision_val, (int, float)) else precision_val,</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    "recall_at_k": round(recall_val, 4) if isinstance(recall_val, (int, float)) else recall_val,</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    "mrr": round(mrr_val, 4) if isinstance(mrr_val, (int, float)) else mrr_val,</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    "bleu": round(bleu_val, 4) if isinstance(bleu_val, (int, float)) else bleu_val,</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    "rouge_l_recall": round(rouge_val, 4) if isinstance(rouge_val, (int, float)) else rouge_val,</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">    "error": error_msg,</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    "hardware_info": hw_info_str,</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">})</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;"></code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">df_result = pd.DataFrame(rows)</code></td></tr>
<tr><td>18</td><td><code style="white-space: pre;">save_path = EVAL_RESULTS_DIR / f"eval_{ts_wib}_{mode_tag}_top{current_k}.csv"</code></td></tr>
<tr><td>19</td><td><code style="white-space: pre;">df_result.to_csv(save_path, index=False)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>6</td>
<td><strong>Algoritma 35: Precision@k</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/evaluation/metrics.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def compute_precision_at_k(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    retrieved_ids: List[str],</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    relevant_ids: List[str],</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    k: int,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">) -&gt; float:</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    if k &lt;= 0:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        return 0.0</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    top_k = retrieved_ids[:k]</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    relevant_set = set(relevant_ids)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    hits = sum(1 for r_id in top_k if r_id in relevant_set)</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    return hits / k</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>7</td>
<td><strong>Algoritma 36: Recall@k</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/evaluation/metrics.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def compute_recall_at_k(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    retrieved_ids: List[str],</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    relevant_ids: List[str],</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    k: int,</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">) -&gt; float:</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    if not relevant_ids:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        return 0.0</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    top_k = retrieved_ids[:k]</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    relevant_set = set(relevant_ids)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    hits = sum(1 for r_id in top_k if r_id in relevant_set)</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    return hits / len(relevant_ids)</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>8</td>
<td><strong>Algoritma 37: MRR</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/evaluation/metrics.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def compute_mrr(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    retrieved_ids: List[str],</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    relevant_ids: List[str],</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">) -&gt; float:</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    relevant_set = set(relevant_ids)</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    for rank, r_id in enumerate(retrieved_ids, start=1):</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        if r_id in relevant_set:</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">            return 1.0 / rank</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    return 0.0</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>9</td>
<td><strong>Algoritma 38: BLEU</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/evaluation/metrics.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def compute_bleu(response: str, reference: str) -&gt; float:</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    try:</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">        from sacrebleu import corpus_bleu</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">        result = corpus_bleu([response], [[reference]])</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">        return result.score / 100.0</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">    except Exception as e:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">        logger.error(f"compute_bleu error: {e}")</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">        return 0.0</code></td></tr>
</tbody>
</table></td>
</tr>
<tr>
<td>10</td>
<td><strong>Algoritma 39: ROUGE-L</strong>
<table>
<thead>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><code style="white-space: pre;"># Path: src/evaluation/metrics.py</code></td></tr>
<tr><td>2</td><td><code style="white-space: pre;">def compute_rouge(</code></td></tr>
<tr><td>3</td><td><code style="white-space: pre;">    response: str,</code></td></tr>
<tr><td>4</td><td><code style="white-space: pre;">    reference: str,</code></td></tr>
<tr><td>5</td><td><code style="white-space: pre;">    rouge_type: str = "rougeL",</code></td></tr>
<tr><td>6</td><td><code style="white-space: pre;">    mode: str = "recall",</code></td></tr>
<tr><td>7</td><td><code style="white-space: pre;">) -&gt; float:</code></td></tr>
<tr><td>8</td><td><code style="white-space: pre;">    from rouge_score import rouge_scorer</code></td></tr>
<tr><td>9</td><td><code style="white-space: pre;">    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)</code></td></tr>
<tr><td>10</td><td><code style="white-space: pre;">    scores = scorer.score(reference, response)</code></td></tr>
<tr><td>11</td><td><code style="white-space: pre;">    rouge_score = scores[rouge_type]</code></td></tr>
<tr><td>12</td><td><code style="white-space: pre;">    if mode == "precision":</code></td></tr>
<tr><td>13</td><td><code style="white-space: pre;">        return rouge_score.precision</code></td></tr>
<tr><td>14</td><td><code style="white-space: pre;">    elif mode == "recall":</code></td></tr>
<tr><td>15</td><td><code style="white-space: pre;">        return rouge_score.recall</code></td></tr>
<tr><td>16</td><td><code style="white-space: pre;">    elif mode == "fmeasure":</code></td></tr>
<tr><td>17</td><td><code style="white-space: pre;">        return rouge_score.fmeasure</code></td></tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber evaluasi retrieval dan generation dapat dilihat pada Tabel 5.22 berikut.

Tabel 5.22 Penjelasan kode sumber evaluasi retrieval dan generation

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1-16</td><td>Evaluasi batch menyiapkan direktori output final, membaca QA gold dari file Excel, dan membaca ground truth binary dari `qa_pairs_binary.json`.</td></tr>
<tr><td>Baris 1-15</td><td>Antarmuka menyediakan konfigurasi mode QA dan rentang top-k. Fungsi evaluasi melakukan pre-compute embedding query agar embedding tidak dihitung berulang untuk setiap metode dan top-k.</td></tr>
<tr><td>Baris 1-19</td><td>Evaluasi melakukan loop untuk setiap top-k, setiap QA, dan setiap metode chunking. Untuk setiap metode, pipeline baru dibuat dengan collection yang sesuai, lalu retrieval dilakukan menggunakan query vector yang sudah dihitung.</td></tr>
<tr><td>Baris 1-13</td><td>Metrik retrieval dihitung jika query memiliki chunk relevan pada ground truth. Setelah itu, konteks hasil retrieval dikirim ke generator untuk menghasilkan jawaban dan menghitung BLEU serta ROUGE-L Recall.</td></tr>
<tr><td>Baris 1-19</td><td>Hasil evaluasi disusun dalam row CSV dengan kolom query, metode, jawaban gold, jawaban generated, metrik retrieval, metrik generation, error, dan hardware info. File disimpan per top-k di `results/final/generation/`.</td></tr>
<tr><td>Baris 1-12</td><td>Fungsi `compute_precision_at_k()` mengambil hasil retrieval sampai cutoff `k`, menghitung jumlah chunk yang cocok dengan ground truth, lalu membagi jumlah hit dengan nilai `k`.</td></tr>
<tr><td>Baris 1-12</td><td>Fungsi `compute_recall_at_k()` menghitung jumlah chunk relevan yang ditemukan pada top-k dan membaginya dengan jumlah chunk relevan pada ground truth.</td></tr>
<tr><td>Baris 1-10</td><td>Fungsi `compute_mrr()` mencari posisi chunk relevan pertama pada hasil retrieval dan mengembalikan nilai reciprocal rank.</td></tr>
<tr><td>Baris 1-9</td><td>Fungsi `compute_bleu()` menggunakan `sacrebleu.corpus_bleu()` untuk menghitung skor BLEU dan mengonversi skala skor ke rentang 0 sampai 1.</td></tr>
<tr><td>Baris 1-17</td><td>Fungsi `compute_rouge()` menggunakan `rouge_scorer.RougeScorer` dan mengembalikan nilai ROUGE sesuai mode yang digunakan, yaitu precision, recall, atau f-measure.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, evaluasi akhir dilakukan dari tab Evaluasi Batch pada `rag_chat.py`. Output yang dihasilkan berupa file CSV per top-k. Bab ini tidak menyajikan nilai metrik atau interpretasi performa karena pembahasan hasil evaluasi ditempatkan pada Bab 6.
