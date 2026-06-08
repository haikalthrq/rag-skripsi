# BAB 5 IMPLEMENTASI

## 5.1 Implementasi Lingkungan dan Struktur Sistem

Implementasi sistem dilakukan dalam workspace `rag-skripsi` dengan struktur kode yang memisahkan proses prapemrosesan dokumen, pembentukan chunk, embedding, penyimpanan vektor, pipeline RAG, ground truth, serta evaluasi. Pemisahan ini digunakan agar setiap tahap dapat dijalankan dan diaudit secara terpisah. Alur implementasi dimulai dari dokumen PDF pada `data/raw/`, dilanjutkan dengan ekstraksi dan pembersihan teks, pembentukan chunk dengan tiga metode, pembuatan embedding, pemuatan embedding ke ChromaDB, eksekusi pipeline RAG, serta evaluasi melalui antarmuka Streamlit.

Tabel 5.1 Kode sumber struktur implementasi sistem

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: docs/CODEMAP.md
project_structure = {
    "src/preprocessing/": "ekstraksi dan pembersihan teks PDF",
    "src/chunking/": "element_based, maxmin_semantic, recursive",
    "src/embedding/": "pembuatan embedding chunk",
    "src/chroma/": "client, loader, dan retrieval ChromaDB",
    "src/rag/": "pipeline RAG dan generator",
    "src/evaluation/": "fungsi metrik evaluasi",
    "src/streamlit/": "aplikasi anotasi dan evaluasi batch",
}

implementation_flow = [
    "data/raw/*.pdf",
    "data/cleaned/*.txt",
    "data/chunked/{element_based,maxmin_semantic,recursive}/*.json",
    "data/embeddings/*.json",
    "data/chroma/",
    "src/rag/pipeline.py",
    "src/streamlit/rag_chat.py",
    "results/final/generation/*.csv",
]</code></pre></td>
</tr>
</tbody>
</table>

Tabel 5.2 Penjelasan kode sumber struktur implementasi sistem

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Blok `project_structure`</td><td>Menunjukkan pemisahan modul utama sistem. Setiap folder pada `src/` memiliki tanggung jawab implementasi yang berbeda sehingga proses dapat ditelusuri dari input dokumen sampai evaluasi.</td></tr>
<tr><td>Blok `implementation_flow`</td><td>Menunjukkan alur data utama dari PDF, teks bersih, chunk JSON, embedding, ChromaDB, pipeline RAG, evaluasi batch, hingga output CSV evaluasi. Alur ini mengikuti pemetaan pada `docs/CODEMAP.md`.</td></tr>
</tbody>
</table>

Struktur tersebut digunakan sebagai dasar penulisan Bab 5. Fokus pembahasan pada bab ini adalah implementasi aktual dalam kode, bukan pembahasan teori atau analisis hasil. Pembahasan nilai metrik dan perbandingan performa metode chunking tidak dimasukkan pada bab ini karena termasuk pembahasan Bab 6.

## 5.2 Implementasi Prapemrosesan Dokumen

Prapemrosesan dokumen diimplementasikan pada modul `src/preprocessing/`. Input tahap ini berupa file PDF publikasi BPS pada `data/raw/`. Implementasi membaca PDF menggunakan PyMuPDF, mengekstrak teks per halaman, mempertahankan penanda halaman dalam format `<<<PAGE_N>>>`, kemudian membersihkan teks hasil ekstraksi. Output tahap ini berupa file teks bersih yang digunakan oleh tahap chunking. Pada alur project, folder output yang digunakan adalah `data/cleaned/`. Namun, default parameter `run_preprocessing()` pada `src/preprocessing/pipeline.py` masih tertulis `data/cleaned_text`, sehingga kesesuaian default tersebut perlu diperiksa ketika menjalankan pipeline secara langsung. [PERLU VERIFIKASI]

Tabel 5.3 Kode sumber implementasi prapemrosesan dokumen

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/preprocessing/pdf_extractor.py
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
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/preprocessing/text_cleaner.py
cleaned = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', cleaned)
cleaned = re.sub(r'\b[Pp]age\s+\d+\b', '', cleaned)
cleaned = re.sub(r'\b[Hh]alaman\s+\d+\b', '', cleaned)
cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)
cleaned = re.sub(r'(?&lt;=\n)\n(\s*\d{1,3}\s*)\n(?=\n)', '\n', cleaned)
cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
cleaned = re.sub(r'^[ \t]+', '', cleaned, flags=re.MULTILINE)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/preprocessing/pipeline.py
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

Tabel 5.4 Penjelasan kode sumber prapemrosesan dokumen

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 115-128 pada `src/preprocessing/pdf_extractor.py`</td><td>Dokumen PDF dibuka menggunakan `fitz.open()`. Setiap halaman diekstrak dengan `_extract_page_hybrid()`, lalu hasilnya diberi penanda halaman. Penanda ini penting karena tahap chunking dapat menggunakannya untuk metadata halaman.</td></tr>
<tr><td>Baris 43-56 dan 84-87 pada `src/preprocessing/text_cleaner.py`</td><td>Fungsi pembersihan menghapus karakter tidak terlihat, pola nomor halaman umum, serta spasi berlebih. Penanda `<<<PAGE_N>>>` tidak dihapus agar informasi halaman tetap dapat diteruskan ke tahap berikutnya.</td></tr>
<tr><td>Baris 103-133 pada `src/preprocessing/pipeline.py`</td><td>Pipeline memanggil ekstraksi PDF, membersihkan teks, memvalidasi agar output tidak kosong, lalu menulis file `.txt` ke direktori output.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, tahap prapemrosesan berfungsi sebagai penghubung antara dokumen PDF dan tiga metode chunking. Tahap ini tidak melakukan penilaian relevansi dan tidak menghitung metrik evaluasi.

## 5.3 Implementasi Metode Chunking

Tiga metode chunking diimplementasikan secara terpisah agar dapat dibandingkan pada tahap evaluasi. Modul `src/chunking/` menyediakan fungsi untuk Element-Based Chunking, Max-Min Semantic Chunking, dan Recursive Chunking. Pemisahan fungsi ini membuat setiap metode menghasilkan file chunk JSON pada subfolder berbeda, yaitu `data/chunked/element_based/`, `data/chunked/maxmin_semantic/`, dan `data/chunked/recursive/`.

Tabel 5.5 Kode sumber pemanggilan metode chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/chunking/__init__.py
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

Tabel 5.6 Penjelasan kode sumber pemanggilan metode chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 29-39 pada `src/chunking/__init__.py`</td><td>Mengekspos fungsi utama Element-Based Chunking dari `element_based.py`.</td></tr>
<tr><td>Baris 41-53 pada `src/chunking/__init__.py`</td><td>Mengekspos fungsi utama Max-Min Semantic Chunking dari `maxmin_chunker.py`.</td></tr>
<tr><td>Baris 55-65 pada `src/chunking/__init__.py`</td><td>Mengekspos fungsi utama Recursive Chunking dari `recursive_split.py`.</td></tr>
</tbody>
</table>

Subbab berikutnya menjelaskan implementasi masing-masing metode secara lebih spesifik berdasarkan file sumber yang aktif.

### 5.3.1 Implementasi Element-Based Chunking

Element-Based Chunking diimplementasikan pada `src/chunking/element_based.py`. Metode ini memproses dokumen PDF dengan `partition_pdf` dari `unstructured`, kemudian mengelompokkan elemen dokumen menjadi chunk berbasis struktur. Elemen seperti judul, teks, daftar, dan tabel diperlakukan berbeda. Tabel disimpan sebagai unit mandiri, sedangkan elemen teks dapat digabung menjadi composite chunk. Metadata yang disimpan meliputi tipe chunk, tipe elemen, judul bagian, nomor halaman, dan sumber file.

Tabel 5.7 Kode sumber implementasi Element-Based Chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/chunking/element_based.py
elements = partition_pdf(
    filename=pdf_path,
    strategy=strategy,
    infer_table_structure=True,
    extract_image_block_types=["table"],
    extract_images_in_pdf=False,
    include_page_breaks=True,
    languages=languages,
)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/chunking/element_based.py
def init_chunk():
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
    }</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/chunking/element_based.py
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

    flush_chunk(forced_type='table')</code></pre></td>
</tr>
</tbody>
</table>

Tabel 5.8 Penjelasan kode sumber Element-Based Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 100-109 pada `src/chunking/element_based.py`</td><td>PDF dipartisi menggunakan `partition_pdf` dengan strategi `hi_res`, struktur tabel diaktifkan, dan informasi page break disertakan.</td></tr>
<tr><td>Baris 316-330 pada `src/chunking/element_based.py`</td><td>Fungsi internal `init_chunk()` membentuk struktur awal chunk beserta metadata seperti `chunk_type`, `section_title`, `page_numbers`, dan `source_file`.</td></tr>
<tr><td>Baris 456-478 pada `src/chunking/element_based.py`</td><td>Elemen tabel diperlakukan sebagai unit tersendiri. Jika `text_as_html` tersedia pada metadata elemen, struktur HTML tabel ikut disimpan untuk dipakai pada tahap embedding atau formatting konteks.</td></tr>
</tbody>
</table>

Implementasi ini menghasilkan chunk JSON yang mempertahankan metadata struktural dokumen. Bab ini tidak menyimpulkan apakah struktur tersebut menghasilkan performa lebih baik karena penilaian performa dibahas pada Bab 6.

### 5.3.2 Implementasi Max-Min Semantic Chunking

Max-Min Semantic Chunking diimplementasikan pada `src/chunking/maxmin_chunker.py`. Alur implementasinya memuat teks bersih, memecah teks menjadi kalimat, membuat embedding kalimat, mengelompokkan kalimat berdasarkan kemiripan semantik, lalu menyimpan hasilnya sebagai chunk JSON. Parameter yang digunakan pada fungsi utama antara lain `fixed_threshold`, `c`, `init_constant`, `batch_size`, dan pilihan backend embedding. Pada kondisi file saat ini, signature fungsi `embed_sentences()` menampilkan parameter yang tidak lengkap sebelum `batch_size`, sehingga bagian detail signature fungsi tersebut ditandai [PERLU VERIFIKASI]. Penjelasan implementasi tetap mengacu pada pemanggilan fungsi dan algoritme yang dapat diverifikasi dari kode.

Tabel 5.9 Kode sumber implementasi Max-Min Semantic Chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/chunking/maxmin_chunker.py
sentences = sent_tokenize(text)
sentences = [s.strip() for s in sentences if s.strip()]

SKIP_CHARS_LIMIT = 32000
filtered, skipped = [], 0
for s in sentences:
    if len(s) &lt;= SKIP_CHARS_LIMIT:
        filtered.append(s)
    else:
        skipped += 1
sentences = filtered</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/chunking/maxmin_chunker.py
for i in range(1, len(sentences)):
    cluster_embeddings = embeddings[cluster_start:cluster_end]
    cluster_size = cluster_end - cluster_start
    new_sentence_embedding = embeddings[i].reshape(1, -1)
    new_sentence_similarities = cosine_similarity(
        new_sentence_embedding,
        cluster_embeddings
    )[0]

    final_threshold = max(adjusted_threshold, fixed_threshold)
    if new_sentence_similarity &gt; final_threshold:
        current_paragraph.append(sentences[i])
        cluster_end += 1
    else:
        paragraphs.append(current_paragraph)
        current_paragraph = [sentences[i]]
        cluster_start = i
        cluster_end = i + 1</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/chunking/maxmin_chunker.py
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

Tabel 5.10 Penjelasan kode sumber Max-Min Semantic Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 384-435 pada `src/chunking/maxmin_chunker.py`</td><td>Teks dibagi menjadi kalimat dengan `sent_tokenize`, kemudian kalimat kosong dan kalimat yang terlalu panjang akibat artefak parsing disaring.</td></tr>
<tr><td>Baris 124-183 pada `src/chunking/maxmin_chunker.py`</td><td>Fungsi `process_sentences()` membentuk cluster kalimat. Kalimat baru dibandingkan dengan cluster aktif menggunakan cosine similarity, kemudian diputuskan apakah digabung atau memulai cluster baru.</td></tr>
<tr><td>Baris 699-779 pada `src/chunking/maxmin_chunker.py`</td><td>Fungsi `process_single_text()` mengorkestrasi pemuatan teks, sentence splitting, embedding kalimat, penerapan Max-Min, konversi ke format chunk, dan penyimpanan output JSON.</td></tr>
<tr><td>Catatan fungsi `embed_sentences()`</td><td>Fungsi dipanggil pada baris 742-747, tetapi signature fungsi pada file saat ini memerlukan verifikasi karena parameter sebelum `use_gguf` tidak lengkap. [PERLU VERIFIKASI]</td></tr>
</tbody>
</table>

Implementasi Max-Min menggunakan representasi embedding kalimat untuk menentukan pengelompokan semantik. Bab ini hanya menjelaskan proses implementasi dan tidak menyatakan bahwa metode ini lebih unggul dibanding metode lain.

### 5.3.3 Implementasi Recursive Chunking

Recursive Chunking diimplementasikan pada `src/chunking/recursive_split.py` dengan `RecursiveCharacterTextSplitter`. Implementasi ini menggunakan parameter `chunk_size`, `chunk_overlap`, dan daftar separator. Teks dipotong secara rekursif berdasarkan hierarki separator, kemudian hasil potongan dikonversi menjadi dictionary dengan metadata sumber file, metode chunking, panjang chunk, dan nomor halaman jika marker halaman tersedia.

Tabel 5.11 Kode sumber implementasi Recursive Chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/chunking/recursive_split.py
if separators is None:
    separators = ["\n\n", "\n", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=length_function,
    is_separator_regex=is_separator_regex,
    separators=separators
)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/chunking/recursive_split.py
chunks = text_splitter.split_text(text)

if chunks:
    chunk_sizes = [len(chunk) for chunk in chunks]
    logger.info(f"Rata-rata karakter per chunk: {sum(chunk_sizes) / len(chunks):.2f}")
    logger.info(f"Min karakter per chunk: {min(chunk_sizes)}")
    logger.info(f"Max karakter per chunk: {max(chunk_sizes)}")</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/chunking/recursive_split.py
page_numbers = sorted({int(m) for m in _PAGE_MARKER.findall(chunk_text)})
chunk_text = _PAGE_MARKER.sub('', chunk_text).strip()

chunk_dict = {
    'chunk_id': chunk_id,
    'text': chunk_text,
    'num_characters': len(chunk_text)
}

chunk_dict['metadata'] = {
    'source_file': source_filename,
    'chunking_method': 'recursive_character_text_splitter',
    'chunk_length': len(chunk_text),
    'page_numbers': page_numbers if page_numbers else None,
}</code></pre></td>
</tr>
</tbody>
</table>

Tabel 5.12 Penjelasan kode sumber Recursive Chunking

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 61-106 pada `src/chunking/recursive_split.py`</td><td>Splitter dibuat menggunakan `RecursiveCharacterTextSplitter` dengan parameter ukuran chunk, overlap, dan separator.</td></tr>
<tr><td>Baris 113-143 pada `src/chunking/recursive_split.py`</td><td>Fungsi `run_recursive_splitter()` menjalankan `split_text()` dan mencatat statistik panjang chunk.</td></tr>
<tr><td>Baris 189-242 pada `src/chunking/recursive_split.py`</td><td>Hasil chunk string dikonversi menjadi dictionary. Marker halaman diekstrak untuk metadata dan dihapus dari teks chunk.</td></tr>
</tbody>
</table>

Hasil dari metode ini berupa file JSON pada `data/chunked/recursive/`. Metadata yang disimpan membuat chunk tetap dapat ditelusuri ke file sumber dan halaman asalnya.

## 5.4 Implementasi Embedding

Tahap embedding diimplementasikan pada `src/embedding/`. Model embedding yang digunakan dalam implementasi adalah Qwen3-Embedding-4B, baik melalui backend HuggingFace maupun GGUF sesuai environment. Fungsi `QwenEmbedder` menerima teks tunggal atau daftar teks, menghasilkan embedding, dan melakukan normalisasi L2 jika parameter `normalize` aktif. Tahap batch embedding membaca chunk JSON, melakukan enrichment pada chunk tabel jika metadata HTML tersedia, menambahkan context prefix untuk metode Max-Min dan Recursive, lalu menyimpan embedding ke `data/embeddings/`.

Tabel 5.13 Kode sumber implementasi embedding

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/embedding/embedder.py
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
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/embedding/io.py
html = meta.get("text_as_html") or ""
if not html.strip():
    continue
if meta.get("chunk_type") != "table":
    continue

table_text = _html_table_to_text(html)
if not table_text.strip():
    continue

section_title = (meta.get("section_title") or "").strip()
if section_title and not _is_noise_text(section_title):
    prefix = section_title + "\n\n"

chunk["text"] = prefix + table_text</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/embedding/embed_chunks.py
chunks = load_chunks_from_json(json_path)
n_enriched = enrich_table_chunk_texts(chunks)
cleaned_texts, valid_indices = clean_and_filter_chunks(chunks)

valid_chunks = [chunks[i] for i in valid_indices]
if chunking_method in _METHODS_WITH_CONTEXT_PREFIX:
    embed_texts = inject_context_prefix(valid_chunks, CONTEXT_PREFIX_CHARS)
else:
    embed_texts = [c.get("text", "") for c in valid_chunks]

embed_texts = [' '.join(t.split()) for t in embed_texts]
embeddings = embedder.embed(embed_texts)</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python"># Path: src/embedding/embed_chunks.py
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

Tabel 5.14 Penjelasan kode sumber embedding

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 64-94 pada `src/embedding/embedder.py`</td><td>Fungsi `embed()` menerima teks, memilih backend GGUF atau HuggingFace, lalu menormalisasi embedding bila konfigurasi `normalize` bernilai benar.</td></tr>
<tr><td>Baris 125-190 pada `src/embedding/io.py`</td><td>Fungsi `enrich_table_chunk_texts()` memanfaatkan `text_as_html` untuk membentuk teks tabel yang lebih terstruktur dan menambahkan konteks judul jika tersedia.</td></tr>
<tr><td>Baris 105-143 pada `src/embedding/embed_chunks.py`</td><td>Fungsi `embed_single_file()` membaca chunk JSON, melakukan enrichment, membersihkan teks, menerapkan context prefix untuk Max-Min dan Recursive, lalu membuat embedding.</td></tr>
<tr><td>Baris 151-169 pada `src/embedding/embed_chunks.py`</td><td>Embedding dan metadata disimpan ke file JSON dalam subfolder metode chunking pada `data/embeddings/`.</td></tr>
</tbody>
</table>

Tahap embedding tidak mengubah file chunk asli pada `data/chunked/`. Enrichment dan context prefix digunakan pada teks yang di-embed dan disimpan untuk kebutuhan downstream, terutama ChromaDB.

## 5.5 Implementasi Vector Database dan Retrieval

Vector database diimplementasikan menggunakan ChromaDB pada modul `src/chroma/`. Embedding JSON dari `data/embeddings/` dimuat ke persistent storage `data/chroma/`. Setiap metode chunking memiliki collection terpisah, yaitu `collection_element_based`, `collection_maxmin_semantic`, dan `collection_recursive`. Retrieval dilakukan dengan query embedding yang dikirim ke `collection.query()`, kemudian hasilnya dikembalikan sebagai daftar dictionary berisi `id`, `document`, `metadata`, dan `distance`.

Tabel 5.15 Kode sumber implementasi Vector Database dan Retrieval

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/chroma/client.py
persist_path = Path(persist_directory)
persist_path.mkdir(parents=True, exist_ok=True)
_recover_stale_sqlite_journal(persist_path)

client = chromadb.PersistentClient(
    path=str(persist_path.absolute())
)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/chroma/loader.py
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
    metadatas.append(chunk_metadata)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/chroma/loader.py
success = batch_add_documents(
    collection=collection,
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    batch_size=batch_size
)</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python"># Path: src/chroma/query.py
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

Tabel 5.16 Penjelasan kode sumber Vector Database dan Retrieval

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 62-108 pada `src/chroma/client.py`</td><td>ChromaDB diinisialisasi sebagai persistent client pada direktori `data/chroma/` sehingga data vektor dapat digunakan kembali setelah aplikasi berhenti.</td></tr>
<tr><td>Baris 135-172 pada `src/chroma/loader.py`</td><td>File embedding JSON dibaca, embedding dikonversi ke `numpy`, lalu ID, dokumen, dan metadata chunk disiapkan untuk ChromaDB.</td></tr>
<tr><td>Baris 192-200 pada `src/chroma/loader.py`</td><td>Data dimasukkan ke collection menggunakan `batch_add_documents()` dalam ukuran batch tertentu.</td></tr>
<tr><td>Baris 181-222 pada `src/chroma/query.py`</td><td>Fungsi `similarity_search()` melakukan pencarian berdasarkan query embedding dan mengubah output ChromaDB menjadi list dictionary yang digunakan oleh pipeline RAG.</td></tr>
</tbody>
</table>

Implementasi pemuatan embedding ke ChromaDB berada pada `src/chroma/loader.py`. File `scripts/load_embeddings_to_chroma.py` hanya berperan sebagai entry point yang memanggil fungsi loader tersebut, sedangkan logika penyimpanan dan retrieval tetap berada pada modul `src/chroma/`.

## 5.6 Implementasi RAG Pipeline dan Generator

Pipeline RAG diimplementasikan pada `src/rag/pipeline.py`, sedangkan generator diimplementasikan pada `src/rag/generator.py`. Alur runtime dimulai dari query pengguna atau pertanyaan QA gold, dilanjutkan dengan embedding query, retrieval chunk dari collection ChromaDB sesuai metode chunking, formatting konteks, dan generation answer. Pipeline ini juga dipanggil oleh `src/streamlit/rag_chat.py` untuk menjalankan chat dan evaluasi batch. Model generator yang digunakan oleh antarmuka tersebut diarahkan ke Qwen3-4B-Instruct-2507 sesuai konfigurasi auto-detect pada `rag_chat.py`.

Tabel 5.17 Kode sumber implementasi RAG Pipeline dan Generator

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/rag/pipeline.py
COLLECTION_NAMES = {
    "element_based":   "collection_element_based",
    "maxmin_semantic": "collection_maxmin_semantic",
    "recursive":       "collection_recursive",
}

collection_name = COLLECTION_NAMES[chunking_method]
self.collection = get_or_create_collection(chroma_client, collection_name)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/rag/pipeline.py
query_embedding = self.embedder.embed(query)
query_vec = query_embedding[0]

results = similarity_search(self.collection, query_vec, k=k)
return results</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/rag/pipeline.py
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
<tr>
<td>4</td>
<td><pre><code class="language-python"># Path: src/rag/generator.py
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

Tabel 5.18 Penjelasan kode sumber RAG Pipeline dan Generator

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 34-38 dan 85-87 pada `src/rag/pipeline.py`</td><td>Pipeline memetakan nama metode chunking ke collection ChromaDB yang sesuai.</td></tr>
<tr><td>Baris 172-195 pada `src/rag/pipeline.py`</td><td>Fungsi `retrieve()` membuat embedding query lalu melakukan similarity search pada collection aktif.</td></tr>
<tr><td>Baris 218-279 pada `src/rag/pipeline.py`</td><td>Fungsi `run()` menjalankan alur lengkap retrieval, formatting konteks, pemanggilan generator, dan pengembalian hasil.</td></tr>
<tr><td>Baris 97-126 pada `src/rag/generator.py`</td><td>Generator membangun pesan chat berisi system prompt, konteks hasil retrieval, pertanyaan, dan instruksi jawaban.</td></tr>
</tbody>
</table>

Prompt generator mengarahkan model untuk menjawab berdasarkan konteks dan menyatakan bahwa informasi tidak memadai jika konteks tidak cukup. Bagian ini hanya menjelaskan mekanisme pembentukan jawaban, bukan kualitas jawaban yang dihasilkan.

## 5.7 Implementasi Ground Truth Retrieval dan Dataset Evaluasi

Dataset evaluasi dibangun dari QA gold, kandidat chunk, anotasi manual, dan konversi label ke JSON ground truth retrieval. Kandidat chunk dibuat oleh `scripts/build_candidates_v3.py` dengan membaca QA gold dan seluruh chunk JSON dari tiga metode. Kandidat tersebut kemudian diberi label manual melalui `src/streamlit/app.py`. Hasil anotasi disimpan sebagai `retrieval_labels_final.csv` dan `retrieval_labels_final.xlsx`. Setelah itu, `scripts/convert_ground_truth_to_json.py` mengubah label final menjadi `qa_pairs_binary.json` dengan skema binary relevance, yaitu label `0` sebagai tidak relevan dan label `>= 1` sebagai relevan.

Tabel 5.19 Kode sumber implementasi Ground Truth Retrieval dan Dataset Evaluasi

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: scripts/build_candidates_v3.py
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
        return json.load(f)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: scripts/build_candidates_v3.py
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
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/streamlit/app.py
OUTPUT_XLSX = ROOT / "data/ground_truth/retrieval_labels_final.xlsx"
OUTPUT_CSV = ROOT / "data/ground_truth/retrieval_labels_final.csv"

def apply_label(qid, method, chunk_id, label):
    df = st.session_state.df
    mask = (
        (df["query_id"] == qid)
        &amp; (df["method"] == method)
        &amp; (df["chunk_id"] == chunk_id)
    )
    df.loc[mask, "label"] = label
    df.loc[mask, "annotator"] = st.session_state.get("annotator_name", "")
    st.session_state.df = df
    save_data(df)</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python"># Path: scripts/convert_ground_truth_to_json.py
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

Tabel 5.20 Penjelasan kode sumber Ground Truth Retrieval dan Dataset Evaluasi

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 93-96, 553-570 pada `scripts/build_candidates_v3.py`</td><td>Script kandidat menetapkan tiga metode chunking, membaca QA gold, dan memuat chunk JSON sesuai `doc_id` dan metode.</td></tr>
<tr><td>Baris 659-708 pada `scripts/build_candidates_v3.py`</td><td>Untuk setiap QA dan metode, kandidat dibangun dan disiapkan dalam format yang dapat divalidasi manual. Jika kandidat tidak ditemukan, baris tetap dibuat dengan status `needs_manual_validation`.</td></tr>
<tr><td>Baris 50-51 dan 472-482 pada `src/streamlit/app.py`</td><td>Aplikasi anotasi menyimpan label manual ke CSV dan XLSX final. Fungsi `apply_label()` memperbarui label berdasarkan kombinasi `query_id`, `method`, dan `chunk_id`.</td></tr>
<tr><td>Baris 134-196 dan 199-239 pada `scripts/convert_ground_truth_to_json.py`</td><td>Konverter membaca label final, memasukkan chunk dengan label minimal sesuai `relevance_threshold`, membentuk ID ChromaDB, dan menyusun `relevant_chunk_ids` per metode.</td></tr>
</tbody>
</table>

Ground truth retrieval yang digunakan oleh evaluasi akhir adalah `data/ground_truth/qa_pairs_binary.json`. File tersebut dibaca oleh evaluasi batch, bukan dibentuk ulang saat evaluasi dijalankan.

## 5.8 Implementasi Evaluasi Retrieval dan Generation

Evaluasi akhir diimplementasikan melalui tab Evaluasi Batch pada `src/streamlit/rag_chat.py`. Fitur ini membaca QA gold dari `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx` dan ground truth retrieval binary dari `data/ground_truth/qa_pairs_binary.json`. Evaluasi dijalankan untuk tiga metode chunking dan dapat menggunakan rentang top-k yang dipilih pada antarmuka. Proses evaluasi melakukan pre-compute query embedding, retrieval per metode, generation answer, perhitungan metrik retrieval dan generation, lalu menyimpan hasil ke CSV pada `results/final/generation/`. Implementasi evaluasi akhir tidak dijalankan dari `scripts/run_generation_eval.py` maupun `scripts/run_retrieval_eval.py`.

Tabel 5.21 Kode sumber implementasi evaluasi retrieval dan generation

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python"># Path: src/streamlit/rag_chat.py
EVAL_RESULTS_DIR = ROOT / "results" / "final" / "generation"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_qa_gold():
    qa_path = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"
    if qa_path.exists():
        _QA_GOLD_DF = pd.read_excel(qa_path, sheet_name="qa_gold")
    return _QA_GOLD_DF

def _load_ground_truth():
    gt_path = ROOT / "data/ground_truth/qa_pairs_binary.json"
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            _GT_BINARY = json.load(f)
    return _GT_BINARY</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python"># Path: src/streamlit/rag_chat.py
top_k_min = st.number_input("Min Top-K", min_value=1, max_value=10, value=1)
top_k_max = st.number_input("Max Top-K", min_value=1, max_value=10, value=10)

def _run_eval_and_save(qa_subset, mode_tag, top_k_range):
    gt_data = _load_ground_truth()
    gt_lookup = {item["id"]: item for item in gt_data}
    min_k, max_k = top_k_range

    query_embeddings = {}
    for _, qa_row in qa_subset.iterrows():
        q_id = str(qa_row["query_id"])
        question = str(qa_row["question"])
        q_vec = pipeline.embedder.embed(question)[0]
        query_embeddings[q_id] = (q_vec, True)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python"># Path: src/streamlit/rag_chat.py
for current_k in range(min_k, max_k + 1):
    for _, qa_row in qa_subset.iterrows():
        question = str(qa_row["question"])
        gold_ans = str(qa_row["gold_answer"])
        q_id = str(qa_row["query_id"])
        gt_item = gt_lookup.get(q_id)
        q_vec, embed_ok = query_embeddings.get(q_id, (None, False))

        for method in METHODS:
            p = RAGPipeline(
                embedder=pipeline.embedder,
                generator=pipeline.generator,
                chroma_client=pipeline.chroma_client,
                chunking_method=method,
                top_k=current_k,
            )
            retrieved = p.retrieve_by_vector(q_vec, k=current_k) if embed_ok else p.retrieve(question, k=current_k)
            retrieved_ids = [doc.get("id", "") for doc in retrieved]</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python"># Path: src/streamlit/rag_chat.py
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
rouge_val = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")</code></pre></td>
</tr>
<tr>
<td>5</td>
<td><pre><code class="language-python"># Path: src/streamlit/rag_chat.py
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
save_path = EVAL_RESULTS_DIR / f"eval_{ts_wib}_{mode_tag}_top{current_k}.csv"
df_result.to_csv(save_path, index=False)</code></pre></td>
</tr>
<tr>
<td>6</td>
<td><pre><code class="language-python"># Path: src/evaluation/metrics.py
top_k = retrieved_ids[:k]
relevant_set = set(relevant_ids)
hits = sum(1 for r_id in top_k if r_id in relevant_set)
precision = hits / k

recall = hits / len(relevant_ids)

for rank, r_id in enumerate(retrieved_ids, start=1):
    if r_id in relevant_set:
        mrr = 1.0 / rank

result = corpus_bleu([response], [[reference]])
bleu = result.score / 100.0

scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
scores = scorer.score(reference, response)
rouge_l_recall = scores[rouge_type].recall</code></pre></td>
</tr>
</tbody>
</table>

Tabel 5.22 Penjelasan kode sumber evaluasi retrieval dan generation

<table>
<thead>
<tr><th>Baris Kode</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td>Baris 59-60, 154-169, dan 172-197 pada `src/streamlit/rag_chat.py`</td><td>Evaluasi batch menyiapkan direktori output final, membaca QA gold dari file Excel, dan membaca ground truth binary dari `qa_pairs_binary.json`.</td></tr>
<tr><td>Baris 752-781 dan 784-832 pada `src/streamlit/rag_chat.py`</td><td>Antarmuka menyediakan konfigurasi mode QA dan rentang top-k. Fungsi evaluasi melakukan pre-compute embedding query agar embedding tidak dihitung berulang untuk setiap metode dan top-k.</td></tr>
<tr><td>Baris 834-884 pada `src/streamlit/rag_chat.py`</td><td>Evaluasi melakukan loop untuk setiap top-k, setiap QA, dan setiap metode chunking. Untuk setiap metode, pipeline baru dibuat dengan collection yang sesuai, lalu retrieval dilakukan menggunakan query vector yang sudah dihitung.</td></tr>
<tr><td>Baris 886-900 pada `src/streamlit/rag_chat.py`</td><td>Metrik retrieval dihitung jika query memiliki chunk relevan pada ground truth. Setelah itu, konteks hasil retrieval dikirim ke generator untuk menghasilkan jawaban dan menghitung BLEU serta ROUGE-L Recall.</td></tr>
<tr><td>Baris 921-942 pada `src/streamlit/rag_chat.py`</td><td>Hasil evaluasi disusun dalam row CSV dengan kolom query, metode, jawaban gold, jawaban generated, metrik retrieval, metrik generation, error, dan hardware info. File disimpan per top-k di `results/final/generation/`.</td></tr>
<tr><td>Baris 22-116 dan 142-176 pada `src/evaluation/metrics.py`</td><td>Fungsi metrik menghitung Precision@k, Recall@k, MRR, BLEU, dan ROUGE-L Recall. Potongan kode metrik pada tabel diringkas dari fungsi aktual agar hanya menampilkan inti perhitungan.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, evaluasi akhir dilakukan dari tab Evaluasi Batch pada `rag_chat.py`. Output yang dihasilkan berupa file CSV per top-k. Bab ini tidak menyajikan nilai metrik atau interpretasi performa karena pembahasan hasil evaluasi ditempatkan pada Bab 6.
