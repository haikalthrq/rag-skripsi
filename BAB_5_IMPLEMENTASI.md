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

### 5.2.1 Implementasi Ekstraksi Teks PDF

Implementasi Ekstraksi Teks PDF dapat dilihat pada Tabel 5.3 berikut.

Tabel 5.3 Kode sumber Ekstraksi Teks PDF

<table>
<thead>
<tr><th colspan="2">Algoritma 1: Ekstraksi Teks PDF</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/preprocessing/pdf_extractor.py
import fitz

doc = fitz.open(str(pdf_path_obj))
extracted_text = []
page_count = len(doc)

for page_num in range(page_count):
    page = doc[page_num]
    page_text = _extract_page_hybrid(page)

    if page_text.strip():
        extracted_text.append(f"&lt;&lt;&lt;PAGE_{page_num + 1}&gt;&gt;&gt;\n{page_text}")

doc.close()
full_text = "\n".join(extracted_text)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Ekstraksi Teks PDF dapat dilihat pada Tabel 5.4 berikut.

Tabel 5.4 Penjelasan kode sumber Ekstraksi Teks PDF

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber potongan kode ekstraksi PDF.</td></tr>
<tr><td>Baris 2</td><td>Mengimpor PyMuPDF yang digunakan untuk membuka dan membaca dokumen PDF.</td></tr>
<tr><td>Baris 4-6</td><td>Membuka file PDF, menyiapkan list penampung teks, dan menghitung jumlah halaman.</td></tr>
<tr><td>Baris 8-13</td><td>Melakukan iterasi setiap halaman, mengekstrak teks dengan metode hybrid, lalu menambahkan penanda halaman jika teks tidak kosong.</td></tr>
<tr><td>Baris 15-16</td><td>Menutup dokumen PDF dan menggabungkan seluruh hasil ekstraksi menjadi satu teks utuh.</td></tr>
</tbody>
</table>

### 5.2.2 Implementasi Pembersihan Teks

Implementasi Pembersihan Teks dapat dilihat pada Tabel 5.5 berikut.

Tabel 5.5 Kode sumber Pembersihan Teks

<table>
<thead>
<tr><th colspan="2">Algoritma 2: Pembersihan Teks</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/preprocessing/text_cleaner.py
import re

cleaned = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', cleaned)
cleaned = re.sub(r'\b[Pp]age\s+\d+\b', '', cleaned)
cleaned = re.sub(r'\b[Hh]alaman\s+\d+\b', '', cleaned)
cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)
cleaned = re.sub(r'(?&lt;=\n)\n(\s*\d{1,3}\s*)\n(?=\n)', '\n', cleaned)
cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
cleaned = re.sub(r'^[ \t]+', '', cleaned, flags=re.MULTILINE)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Pembersihan Teks dapat dilihat pada Tabel 5.6 berikut.

Tabel 5.6 Penjelasan kode sumber Pembersihan Teks

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber potongan kode pembersihan teks.</td></tr>
<tr><td>Baris 2</td><td>Mengimpor modul `re` yang digunakan untuk operasi regular expression.</td></tr>
<tr><td>Baris 4-8</td><td>Menghapus karakter tidak terlihat dan pola nomor halaman umum tanpa menghapus penanda halaman.</td></tr>
<tr><td>Baris 9-10</td><td>Membersihkan spasi berlebih di akhir dan awal baris agar teks siap diproses pada tahap chunking.</td></tr>
</tbody>
</table>

### 5.2.3 Implementasi Pipeline Prapemrosesan Dokumen

Implementasi Pipeline Prapemrosesan Dokumen dapat dilihat pada Tabel 5.7 berikut.

Tabel 5.7 Kode sumber Pipeline Prapemrosesan Dokumen

<table>
<thead>
<tr><th colspan="2">Algoritma 3: Pipeline Prapemrosesan Dokumen</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/preprocessing/pipeline.py
from pathlib import Path
from .pdf_extractor import extract_text
from .text_cleaner import clean_text

raw_text = extract_text(str(pdf_path))
if not raw_text:
    return False, None

cleaned_text = clean_text(raw_text)
if not cleaned_text or len(cleaned_text.strip()) == 0:
    return False, None

output_filename = pdf_path.stem + ".txt"
output_file = Path(output_dir) / output_filename
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Pipeline Prapemrosesan Dokumen dapat dilihat pada Tabel 5.8 berikut.

Tabel 5.8 Penjelasan kode sumber Pipeline Prapemrosesan Dokumen

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber pipeline prapemrosesan dokumen.</td></tr>
<tr><td>Baris 2-4</td><td>Mengimpor `Path`, fungsi ekstraksi teks, dan fungsi pembersihan teks yang digunakan pipeline.</td></tr>
<tr><td>Baris 6-8</td><td>Mengekstrak teks dari PDF dan menghentikan proses jika hasil ekstraksi kosong.</td></tr>
<tr><td>Baris 10-12</td><td>Membersihkan teks hasil ekstraksi dan memvalidasi agar output tidak kosong.</td></tr>
<tr><td>Baris 14-17</td><td>Membentuk path output dan menyimpan teks bersih ke file `.txt`.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, tahap prapemrosesan berfungsi sebagai penghubung antara dokumen PDF dan tiga metode chunking. Tahap ini tidak melakukan penilaian relevansi dan tidak menghitung metrik evaluasi.

## 5.3 Implementasi Metode Chunking

Tiga metode chunking diimplementasikan secara terpisah agar dapat dibandingkan pada tahap evaluasi. Modul `src/chunking/` menyediakan fungsi untuk Element-Based Chunking, Max-Min Semantic Chunking, dan Recursive Chunking. Pemisahan fungsi ini membuat setiap metode menghasilkan file chunk JSON pada subfolder berbeda, yaitu `data/chunked/element_based/`, `data/chunked/maxmin_semantic/`, dan `data/chunked/recursive/`.

Implementasi Pemanggilan Metode Chunking dapat dilihat pada Tabel 5.9 berikut.

Tabel 5.9 Kode sumber Pemanggilan Metode Chunking

<table>
<thead>
<tr><th colspan="2">Algoritma 4: Pemanggilan Metode Chunking</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chunking/__init__.py
from .element_based import (
    partition_document,
    convert_elements_to_chunks,
    run_element_based_chunking
)

from .maxmin_chunker import (
    split_sentences,
    embed_sentences,
    apply_maxmin_chunking,
    run_maxmin_chunking
)

from .recursive_split import (
    create_text_splitter,
    run_recursive_splitter,
    run_recursive_chunking
)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Pemanggilan Metode Chunking dapat dilihat pada Tabel 5.10 berikut.

Tabel 5.10 Penjelasan kode sumber Pemanggilan Metode Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber inisialisasi modul chunking.</td></tr>
<tr><td>Baris 2-6</td><td>Mengimpor dan mengekspos fungsi utama Element-Based Chunking dari `element_based.py`.</td></tr>
<tr><td>Baris 8-13</td><td>Mengimpor dan mengekspos fungsi utama Max-Min Semantic Chunking dari `maxmin_chunker.py`.</td></tr>
<tr><td>Baris 15-19</td><td>Mengimpor dan mengekspos fungsi utama Recursive Chunking dari `recursive_split.py`.</td></tr>
</tbody>
</table>

Subbab berikutnya menjelaskan implementasi masing-masing metode secara lebih spesifik berdasarkan file sumber yang aktif.

### 5.3.1 Implementasi Element-Based Chunking

Element-Based Chunking diimplementasikan pada `src/chunking/element_based.py`. Metode ini memproses dokumen PDF dengan `partition_pdf` dari `unstructured`, kemudian mengelompokkan elemen dokumen menjadi chunk berbasis struktur. Elemen seperti judul, teks, daftar, dan tabel diperlakukan berbeda. Tabel disimpan sebagai unit mandiri, sedangkan elemen teks dapat digabung menjadi composite chunk. Metadata yang disimpan meliputi tipe chunk, tipe elemen, judul bagian, nomor halaman, dan sumber file.

Implementasi Element-Based Chunking dapat dilihat pada Tabel 5.11 berikut.

Tabel 5.11 Kode sumber Element-Based Chunking

<table>
<thead>
<tr><th colspan="2">Algoritma 5: Element-Based Chunking</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chunking/element_based.py
from typing import Any, Dict, List, Optional
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename=pdf_path,
    strategy=strategy,
    infer_table_structure=True,
    extract_image_block_types=["table"],
    extract_images_in_pdf=False,
    include_page_breaks=True,
    languages=languages,
)

chunks: List[Dict[str, Any]] = []
active_title: Optional[str] = None
current_source_file: Optional[str] = None
current_chunk: Optional[Dict[str, Any]] = None
prev_was_list: bool = False
list_group_buffer: List[str] = []
list_group_pages: List[int] = []

def init_chunk() -&gt; Dict[str, Any]:
    return {
        'text': '',
        'metadata': {
            'chunk_type': 'text',
            'element_types': [],
            'section_title': active_title,
            'page_numbers': [],
            'source_file': current_source_file,
            'source_filename': current_source_file,
            'element_count': 0,
            'order_index': -1,
        }
    }

def flush_pending_list() -&gt; None:
    nonlocal list_group_buffer, list_group_pages, prev_was_list
    if not list_group_buffer:
        return
    list_text = "\n".join(list_group_buffer)
    if current_chunk['text'].strip():
        current_chunk['text'] += "\n\n" + list_text
    elif active_title:
        current_chunk['text'] = active_title + "\n\n" + list_text
    else:
        current_chunk['text'] = list_text
    if 'ListGroup' not in current_chunk['metadata']['element_types']:
        current_chunk['metadata']['element_types'].append('ListGroup')
    current_chunk['metadata']['element_count'] += len(list_group_buffer)
    current_chunk['metadata']['page_numbers'].extend(list_group_pages)
    list_group_buffer = []
    list_group_pages = []
    prev_was_list = False

def flush_chunk(forced_type: Optional[str] = None) -&gt; None:
    nonlocal current_chunk
    flush_pending_list()
    if current_chunk and current_chunk['text'].strip():
        pages = sorted(list(set(current_chunk['metadata']['page_numbers'])))
        current_chunk['metadata']['page_numbers'] = pages
        current_chunk['metadata']['page_range'] = (
            f"{pages[0]}-{pages[-1]}" if len(pages) &gt; 1 else str(pages[0])
        ) if pages else "Unknown"
        current_chunk['metadata']['num_characters'] = len(current_chunk['text'])
        if forced_type:
            current_chunk['metadata']['chunk_type'] = forced_type
        chunk_id = len(chunks)
        current_chunk['chunk_id'] = chunk_id
        current_chunk['metadata']['order_index'] = chunk_id
        chunks.append(current_chunk)
    current_chunk = init_chunk()

current_chunk = init_chunk()
for idx, element in enumerate(elements):
    elem_type = type(element).__name__
    if elem_type == 'PageBreak':
        pending_len = len("\n".join(list_group_buffer))
        effective_len = len(current_chunk['text']) + pending_len
        if effective_len &gt;= target_chunk_chars:
            flush_chunk()
        continue

    text = str(element.text) if hasattr(element, 'text') else str(element)
    if not text or not text.strip():
        continue

    category, _ = categorize_element(elem_type)
    if category == 'other':
        continue

    elem_metadata = element.metadata if hasattr(element, 'metadata') else None
    page_num = getattr(elem_metadata, 'page_number', None) if elem_metadata else None
    filename = getattr(elem_metadata, 'filename', None) if elem_metadata else None

    if filename and current_source_file is None:
        current_source_file = filename
        current_chunk['metadata']['source_file'] = filename
        current_chunk['metadata']['source_filename'] = filename

    if category == 'title':
        flush_chunk()
        active_title = text.strip()
        current_chunk['metadata']['section_title'] = active_title
        current_chunk['metadata']['element_types'].append(elem_type)
        current_chunk['metadata']['element_count'] += 1
        if page_num:
            current_chunk['metadata']['page_numbers'].append(page_num)
        prev_was_list = False
        continue

    if category == 'table':
        flush_chunk()
        current_chunk['text'] = text.strip()
        current_chunk['metadata']['chunk_type'] = 'table'
        current_chunk['metadata']['section_title'] = active_title
        current_chunk['metadata']['element_types'] = [elem_type]
        current_chunk['metadata']['element_count'] = 1
        current_chunk['metadata']['page_numbers'] = [page_num] if page_num else []
        src = current_source_file or filename
        current_chunk['metadata']['source_file'] = src
        current_chunk['metadata']['source_filename'] = src
        if elem_metadata and hasattr(elem_metadata, 'text_as_html') and elem_metadata.text_as_html:
            current_chunk['metadata']['text_as_html'] = elem_metadata.text_as_html
        flush_chunk(forced_type='table')
        prev_was_list = False
        continue

    if category == 'text':
        is_list_item = elem_type in ('ListItem', 'BulletedText', 'NumberedList')
        if is_list_item:
            list_group_buffer.append(text.strip())
            if page_num:
                list_group_pages.append(page_num)
            prev_was_list = True
            continue
        flush_pending_list()
        prev_was_list = False
        text_stripped = text.strip()
        current_len = len(current_chunk['text'])
        text_len = len(text_stripped)
        if (current_len &gt;= min_chunk_chars and (current_len + text_len + 2) &gt; max_chunk_chars):
            flush_chunk()
        if current_chunk['text'].strip():
            current_chunk['text'] += "\n\n" + text_stripped
        elif active_title:
            current_chunk['text'] = active_title + "\n\n" + text_stripped
        else:
            current_chunk['text'] = text_stripped
        if elem_type not in current_chunk['metadata']['element_types']:
            current_chunk['metadata']['element_types'].append(elem_type)
        current_chunk['metadata']['element_count'] += 1
        if page_num:
            current_chunk['metadata']['page_numbers'].append(page_num)

flush_chunk()
if min_chunk_chars &gt; 0 and chunks:
    chunks = merge_small_chunks_backward(chunks, min_chunk_chars)
return chunks</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Element-Based Chunking dapat dilihat pada Tabel 5.12 berikut.

Tabel 5.12 Penjelasan kode sumber Element-Based Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>

<tr><td>Baris 1</td><td>Menunjukkan path file sumber implementasi Element-Based Chunking.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor tipe data anotasi dan `partition_pdf` yang digunakan untuk mempartisi PDF berdasarkan elemen layout.</td></tr>
<tr><td>Baris 5-13</td><td>Memanggil `partition_pdf` dengan strategi, struktur tabel, page break, dan konfigurasi bahasa yang digunakan pada proses element-based.</td></tr>
<tr><td>Baris 15-22</td><td>Menyiapkan state chunking, termasuk judul section aktif, sumber file, buffer list item, dan daftar chunk hasil.</td></tr>
<tr><td>Baris 24-37</td><td>Membentuk struktur awal chunk beserta metadata dasar seperti tipe chunk, judul section, halaman, sumber file, dan urutan chunk.</td></tr>
<tr><td>Baris 39-57</td><td>Menggabungkan list item yang tertunda ke chunk aktif agar bullet atau numbered list tidak hilang saat boundary chunk terjadi.</td></tr>
<tr><td>Baris 59-80</td><td>Menyimpan chunk aktif ke daftar hasil dengan metadata halaman, rentang halaman, jumlah karakter, tipe chunk, dan order index.</td></tr>
<tr><td>Baris 82-118</td><td>Melakukan iterasi elemen dokumen, menangani page break, mengambil teks dan metadata, serta memperbarui sumber file dan title section.</td></tr>
<tr><td>Baris 120-139</td><td>Memperlakukan elemen tabel sebagai chunk mandiri dan menyimpan `text_as_html` jika tersedia.</td></tr>
<tr><td>Baris 129-155</td><td>Memproses elemen teks dan list item, melakukan flush saat ukuran melewati batas, menambahkan konteks title, dan menyimpan metadata elemen.</td></tr>
<tr><td>Baris 157-160</td><td>Melakukan flush akhir, merge chunk kecil secara backward, dan mengembalikan daftar chunk.</td></tr>
</tbody>
</table>

Implementasi ini menghasilkan chunk JSON yang mempertahankan metadata struktural dokumen. Bab ini tidak membahas penilaian performa karena analisis hasil evaluasi ditempatkan pada Bab 6.

### 5.3.2 Implementasi Max-Min Semantic Chunking

Max-Min Semantic Chunking diimplementasikan pada `src/chunking/maxmin_chunker.py`. Alur implementasinya memuat teks bersih, memecah teks menjadi kalimat, membuat embedding kalimat, mengelompokkan kalimat berdasarkan kemiripan semantik, lalu menyimpan hasilnya sebagai chunk JSON. Parameter yang digunakan pada fungsi utama antara lain `fixed_threshold`, `c`, `init_constant`, `batch_size`, dan pilihan backend embedding. Uraian pada subbab ini difokuskan pada alur embedding kalimat dan pengelompokan Max-Min sebagaimana ditampilkan pada potongan kode. Detail parameter mengikuti konfigurasi fungsi yang digunakan pada proses eksekusi chunking.

Implementasi Max-Min Semantic Chunking dapat dilihat pada Tabel 5.13 berikut.

Tabel 5.13 Kode sumber Max-Min Semantic Chunking

<table>
<thead>
<tr><th colspan="2">Algoritma 6: Max-Min Semantic Chunking</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chunking/maxmin_chunker.py
import numpy as np
from pathlib import Path
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

sentences = sent_tokenize(text)
sentences = [s.strip() for s in sentences if s.strip()]

SKIP_CHARS_LIMIT = 32000
filtered, skipped = [], 0
for s in sentences:
    if len(s) &lt;= SKIP_CHARS_LIMIT:
        filtered.append(s)
    else:
        skipped += 1
sentences = filtered

paragraphs: List[List[str]] = []
current_paragraph: List[str] = [sentences[0]]
cluster_start: int = 0
cluster_end: int = 1
pairwise_min: float = float('-inf')

for i in range(1, len(sentences)):
    cluster_embeddings = embeddings[cluster_start:cluster_end]
    cluster_size = cluster_end - cluster_start
    new_sentence_embedding = embeddings[i].reshape(1, -1)
    new_sentence_similarities = cosine_similarity(
        new_sentence_embedding,
        cluster_embeddings
    )[0]

    if cluster_size &gt; 1:
        adjusted_threshold = pairwise_min * c * sigmoid(cluster_size - 1)
        new_sentence_similarity = float(np.max(new_sentence_similarities))
        pairwise_min = min(float(np.min(new_sentence_similarities)), pairwise_min)
    else:
        adjusted_threshold = 0.0
        pairwise_min = float(new_sentence_similarities[0])
        new_sentence_similarity = init_constant * pairwise_min

    final_threshold = max(adjusted_threshold, fixed_threshold)
    if new_sentence_similarity &gt; final_threshold:
        current_paragraph.append(sentences[i])
        cluster_end += 1
    else:
        paragraphs.append(current_paragraph)
        current_paragraph = [sentences[i]]
        cluster_start = i
        cluster_end = i + 1
        pairwise_min = float('-inf')

paragraphs.append(current_paragraph)

text = load_text(text_path)
sentences = split_sentences(text)
embeddings = embed_sentences(
    sentences,
    embedding_model,
    batch_size=batch_size,
    use_gguf=use_gguf
)
paragraphs = apply_maxmin_chunking(
    sentences,
    embeddings,
    fixed_threshold=fixed_threshold,
    c=c,
    init_constant=init_constant
)
chunks = convert_paragraphs_to_chunks(
    paragraphs,
    Path(text_path).name,
    include_metadata=include_metadata
)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Max-Min Semantic Chunking dapat dilihat pada Tabel 5.14 berikut.

Tabel 5.14 Penjelasan kode sumber Max-Min Semantic Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>

<tr><td>Baris 1</td><td>Menunjukkan path file sumber Max-Min Semantic Chunking.</td></tr>
<tr><td>Baris 2-5</td><td>Mengimpor NumPy, `Path`, tokenizer kalimat, dan cosine similarity yang digunakan pada proses semantic chunking.</td></tr>
<tr><td>Baris 7-17</td><td>Membagi teks menjadi kalimat, membersihkan kalimat kosong, dan menyaring kalimat terlalu panjang yang biasanya berasal dari artefak parsing.</td></tr>
<tr><td>Baris 19-24</td><td>Menginisialisasi paragraph awal, indeks cluster, dan nilai minimum pairwise similarity.</td></tr>
<tr><td>Baris 26-33</td><td>Mengambil embedding cluster aktif dan menghitung similarity kalimat baru terhadap cluster.</td></tr>
<tr><td>Baris 35-42</td><td>Menghitung adaptive threshold dan nilai similarity akhir berdasarkan ukuran cluster.</td></tr>
<tr><td>Baris 44-54</td><td>Menentukan apakah kalimat digabung ke cluster aktif atau memulai paragraph baru berdasarkan threshold akhir.</td></tr>
<tr><td>Baris 56</td><td>Menambahkan paragraph terakhir setelah seluruh kalimat diproses.</td></tr>
<tr><td>Baris 58-75</td><td>Mengorkestrasi pemuatan teks, sentence splitting, embedding, penerapan Max-Min, dan konversi hasil menjadi chunk.</td></tr>
</tbody>
</table>

Implementasi Max-Min menggunakan representasi embedding kalimat untuk menentukan pengelompokan semantik. Bab ini hanya menjelaskan proses implementasi, sedangkan analisis perbandingan metode ditempatkan pada Bab 6.

### 5.3.3 Implementasi Recursive Chunking

Recursive Chunking diimplementasikan pada `src/chunking/recursive_split.py` dengan `RecursiveCharacterTextSplitter`. Implementasi ini menggunakan parameter `chunk_size`, `chunk_overlap`, dan daftar separator. Teks dipotong secara rekursif berdasarkan hierarki separator, kemudian hasil potongan dikonversi menjadi dictionary dengan metadata sumber file, metode chunking, panjang chunk, dan nomor halaman jika marker halaman tersedia.

Implementasi Recursive Chunking dapat dilihat pada Tabel 5.15 berikut.

Tabel 5.15 Kode sumber Recursive Chunking

<table>
<thead>
<tr><th colspan="2">Algoritma 7: Recursive Chunking</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chunking/recursive_split.py
import re as _re
from langchain_text_splitters import RecursiveCharacterTextSplitter

if separators is None:
    separators = ["\n\n", "\n", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=length_function,
    is_separator_regex=is_separator_regex,
    separators=separators
)

chunks = text_splitter.split_text(text)

if chunks:
    chunk_sizes = [len(chunk) for chunk in chunks]
    logger.info(f"Rata-rata karakter per chunk: {sum(chunk_sizes) / len(chunks):.2f}")
    logger.info(f"Min karakter per chunk: {min(chunk_sizes)}")
    logger.info(f"Max karakter per chunk: {max(chunk_sizes)}")

_PAGE_MARKER = _re.compile(r'&lt;&lt;&lt;PAGE_(\d+)&gt;&gt;&gt;')
chunk_dicts = []
last_seen_page: Optional[int] = None

for chunk_id, chunk_text in enumerate(chunks):
    page_numbers = sorted({int(m) for m in _PAGE_MARKER.findall(chunk_text)})
    chunk_text = _PAGE_MARKER.sub('', chunk_text).strip()

    if page_numbers:
        last_seen_page = page_numbers[-1]
    elif last_seen_page is not None:
        page_numbers = [last_seen_page]

    chunk_dict: Dict[str, Any] = {
        'chunk_id': chunk_id,
        'text': chunk_text,
        'num_characters': len(chunk_text)
    }

    if include_metadata:
        chunk_dict['metadata'] = {
            'source_file': source_filename,
            'chunking_method': 'recursive_character_text_splitter',
            'chunk_length': len(chunk_text),
            'page_numbers': page_numbers if page_numbers else None,
        }

    chunk_dicts.append(chunk_dict)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Recursive Chunking dapat dilihat pada Tabel 5.16 berikut.

Tabel 5.16 Penjelasan kode sumber Recursive Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>

<tr><td>Baris 1</td><td>Menunjukkan path file sumber Recursive Chunking.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor modul regex dan `RecursiveCharacterTextSplitter` yang digunakan pada proses recursive splitting.</td></tr>
<tr><td>Baris 5-13</td><td>Menentukan separator default dan membuat text splitter dengan ukuran chunk serta overlap yang ditentukan.</td></tr>
<tr><td>Baris 15-21</td><td>Menjalankan proses `split_text()` dan mencatat statistik panjang chunk jika hasil tersedia.</td></tr>
<tr><td>Baris 23-26</td><td>Membuat pola marker halaman dan menyiapkan container hasil chunk dictionary.</td></tr>
<tr><td>Baris 28-36</td><td>Mengekstrak marker halaman dari setiap chunk dan melakukan forward propagation halaman jika chunk tidak memiliki marker baru.</td></tr>
<tr><td>Baris 38-51</td><td>Membentuk dictionary chunk beserta metadata sumber file, metode chunking, panjang chunk, dan nomor halaman.</td></tr>
<tr><td>Baris 51</td><td>Menambahkan chunk dictionary ke daftar hasil.</td></tr>
</tbody>
</table>

Hasil dari metode ini berupa file JSON pada `data/chunked/recursive/`. Metadata yang disimpan membuat chunk tetap dapat ditelusuri ke file sumber dan halaman asalnya.

## 5.4 Implementasi Embedding

Tahap embedding diimplementasikan pada `src/embedding/`. Model embedding yang digunakan dalam implementasi adalah Qwen3-Embedding-4B, baik melalui backend HuggingFace maupun GGUF sesuai environment. Fungsi `QwenEmbedder` menerima teks tunggal atau daftar teks, menghasilkan embedding, dan melakukan normalisasi L2 jika parameter `normalize` aktif. Tahap batch embedding membaca chunk JSON, melakukan enrichment pada chunk tabel jika metadata HTML tersedia, menambahkan context prefix untuk metode Max-Min dan Recursive, lalu menyimpan embedding ke `data/embeddings/`. Pada proses embedding, chunk tabel yang memiliki metadata `text_as_html` diproses terlebih dahulu menjadi representasi teks agar dapat digunakan sebagai input embedding. Proses ini dilakukan sebagai bagian dari tahap embedding chunk dan tidak diperlakukan sebagai metode chunking tersendiri.

### 5.4.1 Implementasi Embedder

Implementasi Embedder dapat dilihat pada Tabel 5.17 berikut.

Tabel 5.17 Kode sumber Embedder

<table>
<thead>
<tr><th colspan="2">Algoritma 8: Embedder</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/embedding/embedder.py
import numpy as np

def embed(self, texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]

    if len(texts) == 0:
        return np.array([])

    if self.mode == 'gguf':
        embeddings = self._embed_gguf(texts)
    elif self.mode == 'huggingface':
        embeddings = self._embed_hf(texts)
    else:
        raise ValueError(f"Unknown mode: {self.mode}")

    if self.normalize:
        embeddings = self._normalize_embeddings(embeddings)

    return embeddings</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Embedder dapat dilihat pada Tabel 5.18 berikut.

Tabel 5.18 Penjelasan kode sumber Embedder

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber embedder.</td></tr>
<tr><td>Baris 2</td><td>Mengimpor NumPy untuk representasi array embedding.</td></tr>
<tr><td>Baris 4-8</td><td>Menormalkan input menjadi list teks dan mengembalikan array kosong jika tidak ada teks.</td></tr>
<tr><td>Baris 10-15</td><td>Memilih backend embedding berdasarkan mode yang aktif.</td></tr>
<tr><td>Baris 17-21</td><td>Menormalisasi embedding jika konfigurasi aktif dan mengembalikan hasil embedding.</td></tr>
</tbody>
</table>

### 5.4.2 Implementasi Embedding Chunk

Implementasi Embedding Chunk dapat dilihat pada Tabel 5.19 berikut.

Tabel 5.19 Kode sumber Embedding Chunk

<table>
<thead>
<tr><th colspan="2">Algoritma 9: Embedding Chunk</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/embedding/embed_chunks.py
from pathlib import Path
from .io import load_chunks_from_json, enrich_table_chunk_texts, save_embeddings

chunks = load_chunks_from_json(json_path)
n_enriched = enrich_table_chunk_texts(chunks)
cleaned_texts, valid_indices = clean_and_filter_chunks(chunks)

valid_chunks = [chunks[i] for i in valid_indices]
if chunking_method in _METHODS_WITH_CONTEXT_PREFIX:
    embed_texts = inject_context_prefix(valid_chunks, CONTEXT_PREFIX_CHARS)
else:
    embed_texts = [c.get("text", "") for c in valid_chunks]

embed_texts = [' '.join(t.split()) for t in embed_texts]
embeddings = embedder.embed(embed_texts)

output_path = Path(output_dir) / chunking_method / f"{json_file.stem}_embeddings.json"
metadata = {
    "source_file": json_file.name,
    "source_path": str(json_file),
    "chunking_method": chunking_method,
    "embedding_model": embedder.mode,
    "normalized": embedder.normalize
}

success = save_embeddings(
    embeddings=embeddings,
    chunks=chunks,
    valid_indices=valid_indices,
    output_path=str(output_path),
    metadata=metadata
)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Embedding Chunk dapat dilihat pada Tabel 5.20 berikut.

Tabel 5.20 Penjelasan kode sumber Embedding Chunk

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber proses embedding chunk.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor `Path` dan fungsi I/O embedding yang digunakan pada potongan proses ini.</td></tr>
<tr><td>Baris 5-16</td><td>Memuat chunk JSON, menjalankan enrichment pada chunk tabel jika metadata HTML tersedia, memfilter teks valid, dan menyiapkan teks embedding sesuai metode chunking.</td></tr>
<tr><td>Baris 18-25</td><td>Membentuk path output dan metadata embedding yang akan disimpan.</td></tr>
<tr><td>Baris 27-33</td><td>Menyimpan embedding, chunk, indeks valid, dan metadata ke file output.</td></tr>
</tbody>
</table>

Tahap embedding tidak mengubah file chunk asli pada `data/chunked/`. Enrichment dan context prefix digunakan pada teks yang di-embed dan disimpan untuk kebutuhan downstream, terutama ChromaDB.

## 5.5 Implementasi Vector Database dan Retrieval

Vector database diimplementasikan menggunakan ChromaDB pada modul `src/chroma/`. Embedding JSON dari `data/embeddings/` dimuat ke persistent storage `data/chroma/`. Setiap metode chunking memiliki collection terpisah, yaitu `collection_element_based`, `collection_maxmin_semantic`, dan `collection_recursive`. Retrieval dilakukan dengan query embedding yang dikirim ke `collection.query()`, kemudian hasilnya dikembalikan sebagai daftar dictionary berisi `id`, `document`, `metadata`, dan `distance`.

### 5.5.1 Implementasi ChromaDB Client

Implementasi ChromaDB Client dapat dilihat pada Tabel 5.21 berikut.

Tabel 5.21 Kode sumber ChromaDB Client

<table>
<thead>
<tr><th colspan="2">Algoritma 10: ChromaDB Client</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chroma/client.py
from pathlib import Path
import chromadb

persist_path = Path(persist_directory)
persist_path.mkdir(parents=True, exist_ok=True)
_recover_stale_sqlite_journal(persist_path)

client = chromadb.PersistentClient(
    path=str(persist_path.absolute())
)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber ChromaDB Client dapat dilihat pada Tabel 5.22 berikut.

Tabel 5.22 Penjelasan kode sumber ChromaDB Client

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber client ChromaDB.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor `Path` dan ChromaDB yang digunakan untuk inisialisasi persistent client.</td></tr>
<tr><td>Baris 5-7</td><td>Menyiapkan direktori persistensi dan memulihkan jurnal SQLite yang tertinggal jika ada.</td></tr>
<tr><td>Baris 9-11</td><td>Membuat `PersistentClient` ChromaDB pada path persistensi yang sudah disiapkan.</td></tr>
</tbody>
</table>

### 5.5.2 Implementasi Loader ChromaDB

Implementasi Loader ChromaDB dapat dilihat pada Tabel 5.23 berikut.

Tabel 5.23 Kode sumber Loader ChromaDB

<table>
<thead>
<tr><th colspan="2">Algoritma 11: Loader ChromaDB</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chroma/loader.py
import numpy as np

embeddings = np.array(embeddings_list, dtype=np.float32)
ids = []
documents = []
metadatas = []

for chunk in chunks:
    chunk_id = f"{file_path.stem}_{chunk.get('original_index', chunk.get('embedding_index', 0))}"
    ids.append(chunk_id)
    documents.append(chunk.get('text', ''))
    chunk_metadata = chunk.get('metadata', {}).copy()
    chunk_metadata['source_file'] = metadata.get('source_file', file_path.stem)
    chunk_metadata['chunking_method'] = metadata.get('chunking_method', 'unknown')
    metadatas.append(chunk_metadata)

success = batch_add_documents(
    collection=collection,
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    batch_size=batch_size
)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Loader ChromaDB dapat dilihat pada Tabel 5.24 berikut.

Tabel 5.24 Penjelasan kode sumber Loader ChromaDB

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber loader ChromaDB.</td></tr>
<tr><td>Baris 2</td><td>Mengimpor NumPy untuk mengonversi list embedding menjadi array numerik.</td></tr>
<tr><td>Baris 4-8</td><td>Mengonversi embedding dan menyiapkan list ID, dokumen, serta metadata.</td></tr>
<tr><td>Baris 10-17</td><td>Membentuk ID chunk, mengisi dokumen, dan melengkapi metadata sumber serta metode chunking.</td></tr>
<tr><td>Baris 19-25</td><td>Memasukkan dokumen, embedding, dan metadata ke collection ChromaDB secara batch.</td></tr>
</tbody>
</table>

### 5.5.3 Implementasi Query ChromaDB

Implementasi Query ChromaDB dapat dilihat pada Tabel 5.25 berikut.

Tabel 5.25 Kode sumber Query ChromaDB

<table>
<thead>
<tr><th colspan="2">Algoritma 12: Query ChromaDB</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/chroma/query.py
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=k,
    where=filter
)

documents = []
for i in range(len(results['documents'][0])):
    doc = {
        'id': results['ids'][0][i],
        'document': results['documents'][0][i],
        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
        'distance': results['distances'][0][i] if 'distances' in results else None
    }
    documents.append(doc)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Query ChromaDB dapat dilihat pada Tabel 5.26 berikut.

Tabel 5.26 Penjelasan kode sumber Query ChromaDB

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber query ChromaDB.</td></tr>
<tr><td>Baris 2-5</td><td>Menjalankan query ChromaDB menggunakan query embedding, jumlah hasil, dan filter metadata jika tersedia.</td></tr>
<tr><td>Baris 7-16</td><td>Mengubah struktur hasil ChromaDB menjadi list dictionary berisi ID, dokumen, metadata, dan distance.</td></tr>
</tbody>
</table>

Implementasi pemuatan embedding ke ChromaDB berada pada `src/chroma/loader.py`. File `scripts/load_embeddings_to_chroma.py` hanya berperan sebagai entry point yang memanggil fungsi loader tersebut, sedangkan logika penyimpanan dan retrieval tetap berada pada modul `src/chroma/`.

## 5.6 Implementasi RAG Pipeline dan Generator

Pipeline RAG diimplementasikan pada `src/rag/pipeline.py`, sedangkan generator diimplementasikan pada `src/rag/generator.py`. Alur runtime dimulai dari query pengguna atau pertanyaan QA gold, dilanjutkan dengan embedding query, retrieval chunk dari collection ChromaDB sesuai metode chunking, formatting konteks, dan generation answer. Pipeline ini juga dipanggil oleh `src/streamlit/rag_chat.py` untuk menjalankan chat dan evaluasi batch. Model generator yang digunakan oleh antarmuka tersebut diarahkan ke Qwen3-4B-Instruct-2507 sesuai konfigurasi auto-detect pada `rag_chat.py`.

### 5.6.1 Implementasi RAG Pipeline

Implementasi RAG Pipeline dapat dilihat pada Tabel 5.27 berikut.

Tabel 5.27 Kode sumber RAG Pipeline

<table>
<thead>
<tr><th colspan="2">Algoritma 13: RAG Pipeline</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/rag/pipeline.py
from ..chroma.client import get_or_create_collection
from ..chroma.query import similarity_search

COLLECTION_NAMES = {
    "element_based":   "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive":       "collection_recursive",
}

collection_name = COLLECTION_NAMES[chunking_method]
self.collection = get_or_create_collection(chroma_client, collection_name)

query_embedding = self.embedder.embed(query)
query_vec = query_embedding[0]

results = similarity_search(self.collection, query_vec, k=k)
return results

retrieved = self.retrieve(query, k=k)
contexts = [self._format_context(doc) for doc in retrieved]

raw = self.generator.generate(query, contexts)
if isinstance(raw, tuple):
    answer, thinking = raw
else:
    answer, thinking = raw, ""

return {
    "query": query,
    "answer": answer,
    "thinking": thinking,
    "retrieved_chunks": retrieved,
    "chunking_method": self.chunking_method,
    "num_chunks": len(retrieved),
    "elapsed_seconds": elapsed,
}</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber RAG Pipeline dapat dilihat pada Tabel 5.28 berikut.

Tabel 5.28 Penjelasan kode sumber RAG Pipeline

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber pipeline RAG.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor fungsi akses collection ChromaDB dan similarity search yang digunakan pipeline.</td></tr>
<tr><td>Baris 5-13</td><td>Memetakan metode chunking ke nama collection dan mengambil collection yang sesuai.</td></tr>
<tr><td>Baris 15-18</td><td>Membuat embedding query dan menjalankan similarity search untuk retrieval.</td></tr>
<tr><td>Baris 20-27</td><td>Menjalankan retrieval, membentuk konteks, memanggil generator, dan memisahkan jawaban dari thinking jika tersedia.</td></tr>
<tr><td>Baris 29-37</td><td>Mengembalikan struktur hasil RAG berisi query, jawaban, chunk yang diambil, metode chunking, jumlah chunk, dan waktu eksekusi.</td></tr>
</tbody>
</table>

### 5.6.2 Implementasi Generator

Implementasi Generator dapat dilihat pada Tabel 5.29 berikut.

Tabel 5.29 Kode sumber Generator

<table>
<thead>
<tr><th colspan="2">Algoritma 14: Generator</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/rag/generator.py
context_block = "\n\n".join(
    f"[Konteks {i + 1}]\n{ctx.strip()}"
    for i, ctx in enumerate(contexts)
)

user_content = (
    f"Konteks:\n{context_block}\n\n"
    f"Pertanyaan: {query}\n\n"
    f"Jawaban:"
)

return [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": user_content},
]</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Generator dapat dilihat pada Tabel 5.30 berikut.

Tabel 5.30 Penjelasan kode sumber Generator

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber generator.</td></tr>
<tr><td>Baris 2-5</td><td>Menyusun blok konteks dari hasil retrieval dengan penomoran konteks.</td></tr>
<tr><td>Baris 7-11</td><td>Membentuk pesan pengguna yang berisi konteks, pertanyaan, dan instruksi jawaban.</td></tr>
<tr><td>Baris 13-16</td><td>Mengembalikan daftar pesan chat dengan system prompt dan user content.</td></tr>
</tbody>
</table>

Prompt generator mengarahkan model untuk menjawab berdasarkan konteks dan menyatakan bahwa informasi tidak memadai jika konteks tidak cukup. Bagian ini hanya menjelaskan mekanisme pembentukan jawaban, bukan kualitas jawaban yang dihasilkan.

## 5.7 Implementasi Ground Truth Retrieval dan Dataset Evaluasi

Dataset evaluasi dibangun dari QA gold, kandidat chunk, anotasi manual, dan konversi label ke JSON ground truth retrieval. Kandidat chunk dibuat oleh `scripts/build_candidates_v3.py` dengan membaca QA gold dan seluruh chunk JSON dari tiga metode. Kandidat tersebut kemudian diberi label manual melalui `src/streamlit/app.py`. Hasil anotasi disimpan sebagai `retrieval_labels_final.csv` dan `retrieval_labels_final.xlsx`. Setelah itu, `scripts/convert_ground_truth_to_json.py` mengubah label final menjadi `qa_pairs_binary.json` dengan skema binary relevance, yaitu label `0` sebagai tidak relevan dan label `>= 1` sebagai relevan.

### 5.7.1 Implementasi Kandidat Ground Truth

Implementasi Kandidat Ground Truth dapat dilihat pada Tabel 5.31 berikut.

Tabel 5.31 Kode sumber Kandidat Ground Truth

<table>
<thead>
<tr><th colspan="2">Algoritma 15: Kandidat Ground Truth</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: scripts/build_candidates_v3.py
import json
import pandas as pd

METHODS = ["element_based", "maxmin_semantic", "recursive"]
PRE_K_DEFAULT = 10
TOP_K_DEFAULT = 5

def load_qa_gold():
    df = pd.read_excel(str(QA_XLSX), sheet_name="qa_gold", dtype=str).fillna("")
    rows = df.to_dict("records")
    return rows

def load_chunks(doc_id, method):
    stem = DOC_MAP.get(doc_id)
    fp = CHUNK_DIR / method / f"{stem}_chunks.json"
    with open(fp, encoding="utf-8") as f:
        return json.load(f)

for method in METHODS:
    candidates = build_candidates_for_group(qa, method, pre_k, top_k)

    if not candidates:
        rows.append({
            "query_id": qid,
            "doc_id": doc_id,
            "method": method,
            "chunk_id": "",
            "match_type": "not_found",
            "suggested_label": "0",
            "status": "needs_manual_validation",
        })
        continue</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Kandidat Ground Truth dapat dilihat pada Tabel 5.32 berikut.

Tabel 5.32 Penjelasan kode sumber Kandidat Ground Truth

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber pembentukan kandidat ground truth.</td></tr>
<tr><td>Baris 2-3</td><td>Mengimpor JSON dan pandas untuk membaca QA gold serta chunk JSON.</td></tr>
<tr><td>Baris 5-7</td><td>Menetapkan daftar metode chunking dan nilai default kandidat.</td></tr>
<tr><td>Baris 9-18</td><td>Membaca QA gold dari Excel dan memuat chunk berdasarkan dokumen serta metode chunking.</td></tr>
<tr><td>Baris 20-33</td><td>Membangun kandidat untuk setiap metode dan tetap membuat baris validasi manual jika kandidat tidak ditemukan.</td></tr>
</tbody>
</table>

### 5.7.2 Implementasi Anotasi Ground Truth

Implementasi Anotasi Ground Truth dapat dilihat pada Tabel 5.33 berikut.

Tabel 5.33 Kode sumber Anotasi Ground Truth

<table>
<thead>
<tr><th colspan="2">Algoritma 16: Anotasi Ground Truth</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/streamlit/app.py
from datetime import datetime
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUTPUT_XLSX = ROOT / "data/ground_truth/retrieval_labels_final.xlsx"
OUTPUT_CSV = ROOT / "data/ground_truth/retrieval_labels_final.csv"

def load_data() -&gt; pd.DataFrame:
    df_fresh = _load_fresh(_mtime_c, _mtime_q)
    if OUTPUT_XLSX.exists():
        df_saved = pd.read_excel(str(OUTPUT_XLSX), sheet_name="labels", dtype=str).fillna("")
        keys = ["query_id", "method", "chunk_id"]
        if all(k in df_saved.columns for k in keys):
            restore_cols = [c for c in ["label", "annotator", "rationale"] if c in df_saved.columns]
            saved_map = df_saved.set_index(keys)[restore_cols]
            idx = df_fresh.set_index(keys)
            for col in restore_cols:
                if col not in idx.columns:
                    idx[col] = ""
                idx[col] = saved_map[col].reindex(idx.index).fillna(idx[col])
            return idx.reset_index()
    return df_fresh

def save_data(df: pd.DataFrame) -&gt; None:
    out_cols = [
        "query_id", "doc_id", "question_preview", "evidence_type", "method",
        "chunk_id", "label", "strength_score", "rationale",
        "chunk_page_start", "chunk_page_end", "chunk_text_excerpt",
        "annotator", "status",
    ]
    df_out = df[[c for c in out_cols if c in df.columns]].copy()
    df_out["status"] = df_out["label"].apply(
        lambda x: "labeled" if x.strip() not in ("", "None") else "needs_manual_validation"
    )
    df_out.to_csv(str(OUTPUT_CSV), index=False, encoding="utf-8")

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    h_font = Font(bold=True, color="FFFFFF")
    h_fill = PatternFill("solid", fgColor="6366F1")
    ws.append(list(df_out.columns))
    for cell in ws[1]:
        cell.font = h_font
        cell.fill = h_fill
        cell.alignment = Alignment(horizontal="center")
    for _, row in df_out.iterrows():
        ws.append(list(row))
    wb.save(str(OUTPUT_XLSX))
    st.session_state.last_saved = datetime.now()

def apply_label(qid: str, method: str, chunk_id: str, label: str) -&gt; None:
    df = st.session_state.df
    mask = (
        (df["query_id"] == qid)
        &amp; (df["method"] == method)
        &amp; (df["chunk_id"] == chunk_id)
    )
    df.loc[mask, "label"] = label
    df.loc[mask, "annotator"] = st.session_state.get("annotator_name", "")
    st.session_state.df = df
    save_data(df)

def render_chunk_card(row: pd.Series, active: bool) -&gt; None:
    label = str(row.get("label", "") or "").strip()
    qid = str(row.get("query_id", ""))
    method = str(row.get("method", ""))
    chunk_id = str(row.get("chunk_id", ""))
    if st.button("1 ? Relevan", type="primary" if label == "1" else "secondary"):
        apply_label(qid, method, chunk_id, "1")
        st.rerun()
    if st.button("0 ? Tidak Relevan", type="primary" if label == "0" else "secondary"):
        apply_label(qid, method, chunk_id, "0")
        st.rerun()
    if st.button("Review", type="primary" if label == "needs_review" else "secondary"):
        apply_label(qid, method, chunk_id, "needs_review")
        st.rerun()

def main() -&gt; None:
    if "df" not in st.session_state:
        df = load_data()
        init_state(df)
    filters = render_sidebar(st.session_state.df)
    groups = get_groups(st.session_state.df, filters)
    if not groups:
        st.info("Tidak ada data sesuai filter. Ubah filter di sidebar.")
        return
    qid, method = groups[st.session_state.group_idx]
    mask = (
        (st.session_state.df["query_id"] == qid)
        &amp; (st.session_state.df["method"] == method)
    )
    group_df = sort_group_df(st.session_state.df[mask].copy())
    for _, row in group_df.iterrows():
        render_chunk_card(row, active=False)</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Anotasi Ground Truth dapat dilihat pada Tabel 5.34 berikut.

Tabel 5.34 Penjelasan kode sumber Anotasi Ground Truth

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>

<tr><td>Baris 1</td><td>Menunjukkan path file sumber aplikasi anotasi ground truth retrieval.</td></tr>
<tr><td>Baris 2-6</td><td>Mengimpor dependensi untuk waktu simpan, dataframe, antarmuka Streamlit, dan ekspor Excel.</td></tr>
<tr><td>Baris 8-9</td><td>Menetapkan file output final label retrieval dalam format XLSX dan CSV.</td></tr>
<tr><td>Baris 11-26</td><td>Memuat data kandidat fresh dan mengembalikan label, annotator, serta rationale dari file output sebelumnya jika proses anotasi sudah pernah disimpan.</td></tr>
<tr><td>Baris 28-56</td><td>Menyusun kolom output, membentuk status validasi, menyimpan CSV, lalu membuat workbook Excel dengan header dan isi label.</td></tr>
<tr><td>Baris 58-69</td><td>Memperbarui label berdasarkan kombinasi `query_id`, `method`, dan `chunk_id`, menyimpan nama annotator, lalu menjalankan auto-save.</td></tr>
<tr><td>Baris 71-86</td><td>Menampilkan tombol label untuk kelas relevan, tidak relevan, dan review, kemudian memanggil `apply_label()` sesuai pilihan annotator.</td></tr>
<tr><td>Baris 88-98</td><td>Menjalankan alur utama aplikasi: memuat state data, menerapkan filter, memilih grup query-metode, mengambil kandidat chunk, dan merender kartu anotasi.</td></tr>
</tbody>
</table>

### 5.7.3 Implementasi Konversi Ground Truth Retrieval

Implementasi Konversi Ground Truth Retrieval dapat dilihat pada Tabel 5.35 berikut.

Tabel 5.35 Kode sumber Konversi Ground Truth Retrieval

<table>
<thead>
<tr><th colspan="2">Algoritma 17: Konversi Ground Truth Retrieval</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: scripts/convert_ground_truth_to_json.py
if label &lt; threshold:
    continue

pipeline_method = CSV_METHOD_TO_CODE[method_csv]
file_stem = DOC_TO_FILE_STEM[doc_id]
chroma_id = f"{file_stem}_{chunk_id_int}"

result[q_id][pipeline_method].append(chroma_id)

relevant_chunk_ids = {
    m: query_labels.get(m, []) for m in ALL_METHODS
}</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Konversi Ground Truth Retrieval dapat dilihat pada Tabel 5.36 berikut.

Tabel 5.36 Penjelasan kode sumber Konversi Ground Truth Retrieval

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber konversi ground truth retrieval.</td></tr>
<tr><td>Baris 2-3</td><td>Melewati label yang berada di bawah threshold relevansi.</td></tr>
<tr><td>Baris 5-9</td><td>Memetakan nama metode dan dokumen menjadi ID ChromaDB yang digunakan pipeline.</td></tr>
<tr><td>Baris 11-13</td><td>Menyusun daftar chunk relevan untuk setiap metode chunking.</td></tr>
</tbody>
</table>

Ground truth retrieval yang digunakan oleh evaluasi akhir adalah `data/ground_truth/qa_pairs_binary.json`. File tersebut dibaca oleh evaluasi batch, bukan dibentuk ulang saat evaluasi dijalankan.

## 5.8 Implementasi Evaluasi Retrieval dan Generation

Evaluasi akhir diimplementasikan melalui tab Evaluasi Batch pada `src/streamlit/rag_chat.py`. Fitur ini membaca QA gold dari `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx` dan ground truth retrieval binary dari `data/ground_truth/qa_pairs_binary.json`. Evaluasi dijalankan untuk tiga metode chunking dan dapat menggunakan rentang top-k yang dipilih pada antarmuka. Proses evaluasi melakukan pre-compute query embedding, retrieval per metode, generation answer, perhitungan metrik retrieval dan generation, lalu menyimpan hasil ke CSV pada `results/final/generation/`. Implementasi evaluasi akhir tidak dijalankan dari `scripts/run_generation_eval.py` maupun `scripts/run_retrieval_eval.py`.

### 5.8.1 Implementasi Evaluasi Batch

Implementasi Evaluasi Batch dapat dilihat pada Tabel 5.37 berikut.

Tabel 5.37 Kode sumber Evaluasi Batch

<table>
<thead>
<tr><th colspan="2">Algoritma 18: Evaluasi Batch</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/streamlit/rag_chat.py
import json
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import torch
from src.rag.pipeline import RAGPipeline
from src.evaluation.metrics import compute_bleu, compute_mrr, compute_precision_at_k, compute_recall_at_k, compute_rouge

EVAL_RESULTS_DIR = ROOT / "results" / "final" / "generation"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

top_k_min = st.number_input("Min Top-K", min_value=1, max_value=10, value=1)
top_k_max = st.number_input("Max Top-K", min_value=1, max_value=10, value=10)
run_btn = st.button("? Jalankan Evaluasi", use_container_width=True, type="primary")

def _run_eval_and_save(qa_subset: pd.DataFrame, mode_tag: str, top_k_range: tuple) -&gt; list:
    gt_data = _load_ground_truth()
    if not gt_data:
        st.error("Ground truth binary tidak ditemukan.")
        return []

    gt_lookup = {item["id"]: item for item in gt_data}
    hw_info = get_hardware_info()
    hw_info_str = json.dumps(hw_info, ensure_ascii=False)
    min_k, max_k = top_k_range
    total_steps = len(qa_subset) * len(METHODS) * (max_k - min_k + 1)
    all_results = []
    step = 0

    query_embeddings = {}
    for _, qa_row in qa_subset.iterrows():
        q_id = str(qa_row["query_id"])
        question = str(qa_row["question"])
        try:
            q_vec = pipeline.embedder.embed(question)[0]
            query_embeddings[q_id] = (q_vec, True)
        except Exception:
            query_embeddings[q_id] = (None, False)

    for current_k in range(min_k, max_k + 1):
        rows = []
        for _, qa_row in qa_subset.iterrows():
            question = str(qa_row["question"])
            gold_ans = str(qa_row["gold_answer"])
            q_id = str(qa_row["query_id"])
            gt_item = gt_lookup.get(q_id)
            q_vec, embed_ok = query_embeddings.get(q_id, (None, False))

            for method in METHODS:
                step += 1
                if gt_item:
                    rel_all = gt_item.get("relevant_chunk_ids", {})
                    rel_ids = rel_all.get(method, []) if isinstance(rel_all, dict) else rel_all
                else:
                    rel_ids = []

                precision_val = recall_val = mrr_val = None
                gen_answer = bleu_val = rouge_val = None
                error_msg = ""

                try:
                    p = RAGPipeline(
                        embedder=pipeline.embedder,
                        generator=pipeline.generator,
                        chroma_client=pipeline.chroma_client,
                        chunking_method=method,
                        top_k=current_k,
                    )
                    retrieved = p.retrieve_by_vector(q_vec, k=current_k) if embed_ok else p.retrieve(question, k=current_k)
                    retrieved_ids = [doc.get("id", "") for doc in retrieved]

                    if rel_ids:
                        precision_val = compute_precision_at_k(retrieved_ids, rel_ids, current_k)
                        recall_val = compute_recall_at_k(retrieved_ids, rel_ids, current_k)
                        mrr_val = compute_mrr(retrieved_ids, rel_ids)
                    else:
                        precision_val = recall_val = mrr_val = "N/A"

                    contexts = [p._format_context(doc) for doc in retrieved]
                    raw = pipeline.generator.generate(question, contexts)
                    gen_answer = raw[0] if isinstance(raw, tuple) else raw
                    bleu_val = compute_bleu(gen_answer, gold_ans)
                    rouge_val = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")

                except torch.cuda.OutOfMemoryError as oom_exc:
                    gen_answer = "[OOM - Out of Memory]"
                    precision_val = recall_val = mrr_val = "OOM"
                    bleu_val = rouge_val = "OOM"
                    error_msg = f"OOM at top-{current_k}: {str(oom_exc)}"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    gen_answer = f"[ERROR] {exc}"
                    error_msg = str(exc)

                rows.append({
                    "query_id": q_id,
                    "method": METHOD_LABELS[method],
                    "question": question,
                    "gold_answer": gold_ans,
                    "generated_answer": gen_answer,
                    "precision_at_k": round(precision_val, 4) if isinstance(precision_val, (int, float)) else precision_val,
                    "recall_at_k": round(recall_val, 4) if isinstance(recall_val, (int, float)) else recall_val,
                    "mrr": round(mrr_val, 4) if isinstance(mrr_val, (int, float)) else mrr_val,
                    "bleu": round(bleu_val, 4) if isinstance(bleu_val, (int, float)) else bleu_val,
                    "rouge_l_recall": round(rouge_val, 4) if isinstance(rouge_val, (int, float)) else rouge_val,
                    "error": error_msg,
                    "hardware_info": hw_info_str,
                })

        df_result = pd.DataFrame(rows)
        ts_wib = (datetime.now() + timedelta(hours=7)).strftime("%Y%m%d_%H%M%S")
        save_path = EVAL_RESULTS_DIR / f"eval_{ts_wib}_{mode_tag}_top{current_k}.csv"
        df_result.to_csv(save_path, index=False)
        all_results.append((df_result, save_path))

    return all_results

if run_btn:
    qa_df = _load_qa_gold()
    mode_tag = "quick" if eval_mode.startswith("Quick") else "full"
    qa_subset = qa_df[qa_df["query_id"].isin(QUICK_EVAL_IDS)] if mode_tag == "quick" else qa_df
    all_results = _run_eval_and_save(qa_subset, mode_tag, (top_k_min, top_k_max))</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Evaluasi Batch dapat dilihat pada Tabel 5.38 berikut.

Tabel 5.38 Penjelasan kode sumber Evaluasi Batch

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>

<tr><td>Baris 1</td><td>Menunjukkan path file sumber evaluasi batch pada Streamlit.</td></tr>
<tr><td>Baris 2-8</td><td>Mengimpor JSON, waktu, pandas, Streamlit, PyTorch, pipeline RAG, dan fungsi metrik retrieval serta generation.</td></tr>
<tr><td>Baris 10-15</td><td>Menyiapkan direktori output final dan input antarmuka untuk rentang top-k serta tombol eksekusi evaluasi.</td></tr>
<tr><td>Baris 17-29</td><td>Mendefinisikan fungsi evaluasi batch, memuat ground truth binary, menyiapkan lookup ground truth, metadata hardware, jumlah langkah, dan container hasil.</td></tr>
<tr><td>Baris 31-39</td><td>Melakukan pre-compute embedding query agar embedding dapat digunakan ulang untuk seluruh metode dan top-k.</td></tr>
<tr><td>Baris 41-58</td><td>Melakukan loop untuk setiap top-k, setiap QA, dan setiap metode chunking, lalu mengambil daftar chunk relevan dari ground truth.</td></tr>
<tr><td>Baris 60-84</td><td>Membuat pipeline sesuai metode chunking, menjalankan retrieval, mengambil ID chunk, dan menghitung Precision@k, Recall@k, serta MRR jika ground truth relevan tersedia.</td></tr>
<tr><td>Baris 86-91</td><td>Membentuk konteks hasil retrieval, menjalankan generator, lalu menghitung BLEU dan ROUGE-L.</td></tr>
<tr><td>Baris 93-103</td><td>Menangani error OOM pada GPU dan membersihkan cache CUDA jika tersedia.</td></tr>
<tr><td>Baris 104-106</td><td>Menangani error umum dengan menyimpan pesan error ke hasil evaluasi.</td></tr>
<tr><td>Baris 108-122</td><td>Menyusun row hasil evaluasi yang memuat query, metode, jawaban, metrik retrieval, metrik generation, error, dan hardware info.</td></tr>
<tr><td>Baris 112-116</td><td>Menyimpan hasil evaluasi per top-k ke CSV final dengan timestamp WIB dan memasukkan file ke daftar hasil.</td></tr>
<tr><td>Baris 118-124</td><td>Mengembalikan semua hasil dan menjalankan evaluasi saat tombol antarmuka ditekan.</td></tr>
</tbody>
</table>

### 5.8.2 Implementasi Fungsi Metrik Evaluasi

Implementasi Fungsi Metrik Evaluasi dapat dilihat pada Tabel 5.39 berikut.

Tabel 5.39 Kode sumber Fungsi Metrik Evaluasi

<table>
<thead>
<tr><th colspan="2">Algoritma 19: Fungsi Metrik Evaluasi</th></tr>
<tr><th>Baris</th><th>Kode</th></tr>
</thead>
<tbody>
<tr>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace;">1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62</pre></td>
<td style="vertical-align: top;"><pre style="margin:0; padding:0; line-height:1.2; font-family:'Courier New', monospace; white-space:pre; overflow-x:auto;"><code class="language-python"># Path: src/evaluation/metrics.py
from typing import List

def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -&gt; float:
    if k &lt;= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r_id in top_k if r_id in relevant_set)
    return hits / k

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -&gt; float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r_id in top_k if r_id in relevant_set)
    return hits / len(relevant_ids)

def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -&gt; float:
    relevant_set = set(relevant_ids)
    for rank, r_id in enumerate(retrieved_ids, start=1):
        if r_id in relevant_set:
            return 1.0 / rank
    return 0.0

def compute_bleu(response: str, reference: str) -&gt; float:
    try:
        from sacrebleu import corpus_bleu
        result = corpus_bleu([response], [[reference]])
        return result.score / 100.0
    except Exception as e:
        logger.error(f"compute_bleu error: {e}")
        return 0.0

def compute_rouge(
    response: str,
    reference: str,
    rouge_type: str = "rougeL",
    mode: str = "recall",
) -&gt; float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
    scores = scorer.score(reference, response)
    rouge_score = scores[rouge_type]
    if mode == "precision":
        return rouge_score.precision
    elif mode == "recall":
        return rouge_score.recall
    elif mode == "fmeasure":
        return rouge_score.fmeasure</code></pre></td>
</tr>
</tbody>
</table>

Penjelasan kode sumber Fungsi Metrik Evaluasi dapat dilihat pada Tabel 5.40 berikut.

Tabel 5.40 Penjelasan kode sumber Fungsi Metrik Evaluasi

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 1</td><td>Menunjukkan path file sumber fungsi metrik evaluasi.</td></tr>
<tr><td>Baris 2</td><td>Mengimpor `List` untuk anotasi tipe parameter daftar ID retrieval dan ground truth.</td></tr>
<tr><td>Baris 4-13</td><td>Menghitung Precision@k dari jumlah chunk relevan yang muncul pada hasil retrieval sampai cutoff k.</td></tr>
<tr><td>Baris 15-25</td><td>Menghitung Recall@k berdasarkan jumlah chunk relevan yang berhasil ditemukan dibanding total chunk relevan.</td></tr>
<tr><td>Baris 27-35</td><td>Menghitung MRR dari posisi chunk relevan pertama pada daftar hasil retrieval.</td></tr>
<tr><td>Baris 37-44</td><td>Menghitung BLEU menggunakan `sacrebleu` dan mengonversi skor ke rentang 0 sampai 1.</td></tr>
<tr><td>Baris 46-61</td><td>Menghitung ROUGE-L menggunakan `rouge_scorer` dan mengembalikan nilai sesuai mode evaluasi.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, evaluasi akhir dilakukan dari tab Evaluasi Batch pada `rag_chat.py`. Output yang dihasilkan berupa file CSV per top-k. Bab ini tidak menyajikan nilai metrik atau interpretasi performa karena pembahasan hasil evaluasi ditempatkan pada Bab 6.
