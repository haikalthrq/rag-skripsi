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
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/preprocessing/pdf_extractor.py
2   doc = fitz.open(str(pdf_path_obj))
3   extracted_text = []
4   page_count = len(doc)
5   
6   for page_num in range(page_count):
7       page = doc[page_num]
8       page_text = _extract_page_hybrid(page)
9   
10      if page_text.strip():
11          extracted_text.append(f"&lt;&lt;&lt;PAGE_{page_num + 1}&gt;&gt;&gt;\n{page_text}")
12  
13  doc.close()
14  full_text = "\n".join(extracted_text)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/preprocessing/text_cleaner.py
2   cleaned = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', cleaned)
3   cleaned = re.sub(r'\b[Pp]age\s+\d+\b', '', cleaned)
4   cleaned = re.sub(r'\b[Hh]alaman\s+\d+\b', '', cleaned)
5   cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)
6   cleaned = re.sub(r'(?&lt;=\n)\n(\s*\d{1,3}\s*)\n(?=\n)', '\n', cleaned)
7   cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
8   cleaned = re.sub(r'^[ \t]+', '', cleaned, flags=re.MULTILINE)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/preprocessing/pipeline.py
2   raw_text = extract_text(str(pdf_path))
3   if not raw_text:
4       return False, None
5   
6   cleaned_text = clean_text(raw_text)
7   if not cleaned_text or len(cleaned_text.strip()) == 0:
8       return False, None
9   
10  output_filename = pdf_path.stem + ".txt"
11  output_file = Path(output_dir) / output_filename
12  with open(output_file, 'w', encoding='utf-8') as f:
13      f.write(cleaned_text)</code></pre></td>
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
<tr><td>Kode 1 Baris 1-14</td><td>Dokumen PDF dibuka menggunakan `fitz.open()`. Setiap halaman diekstrak dengan `_extract_page_hybrid()`, lalu hasilnya diberi penanda halaman. Penanda ini penting karena tahap chunking dapat menggunakannya untuk metadata halaman.</td></tr>
<tr><td>Kode 2 Baris 1-8</td><td>Fungsi pembersihan menghapus karakter tidak terlihat, pola nomor halaman umum, serta spasi berlebih. Penanda `<<<PAGE_N>>>` tidak dihapus agar informasi halaman tetap dapat diteruskan ke tahap berikutnya.</td></tr>
<tr><td>Kode 3 Baris 1-13</td><td>Pipeline memanggil ekstraksi PDF, membersihkan teks, memvalidasi agar output tidak kosong, lalu menulis file `.txt` ke direktori output.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, tahap prapemrosesan berfungsi sebagai penghubung antara dokumen PDF dan tiga metode chunking. Tahap ini tidak melakukan penilaian relevansi dan tidak menghitung metrik evaluasi.

## 5.3 Implementasi Metode Chunking

Tiga metode chunking diimplementasikan secara terpisah agar dapat dibandingkan pada tahap evaluasi. Modul `src/chunking/` menyediakan fungsi untuk Element-Based Chunking, Max-Min Semantic Chunking, dan Recursive Chunking. Pemisahan fungsi ini membuat setiap metode menghasilkan file chunk JSON pada subfolder berbeda, yaitu `data/chunked/element_based/`, `data/chunked/maxmin_semantic/`, dan `data/chunked/recursive/`.

Kode sumber pemanggilan metode chunking dapat dilihat pada Tabel 5.5 berikut.

Tabel 5.5 Kode sumber pemanggilan metode chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/chunking/__init__.py
2   from .element_based import (
3       partition_document,
4       convert_elements_to_chunks,
5       run_element_based_chunking
6   )
7   
8   from .maxmin_chunker import (
9       split_sentences,
10      embed_sentences,
11      apply_maxmin_chunking,
12      run_maxmin_chunking
13  )
14  
15  from .recursive_split import (
16      create_text_splitter,
17      run_recursive_splitter,
18      run_recursive_chunking
19  )</code></pre></td>
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
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/chunking/element_based.py
2   elements = partition_pdf(
3       filename=pdf_path,
4       strategy=strategy,
5       infer_table_structure=True,
6       extract_image_block_types=["table"],
7       extract_images_in_pdf=False,
8       include_page_breaks=True,
9       languages=languages,
10  )</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/chunking/element_based.py
2   def init_chunk():
3       return {
4           'text': '',
5           'metadata': {
6               'chunk_type': 'text',
7               'element_types': [],
8               'section_title': active_title,
9               'page_numbers': [],
10              'source_file': current_source_file,
11              'source_filename': current_source_file,
12              'element_count': 0,
13              'order_index': -1,
14          }
15      }</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/chunking/element_based.py
2   if category == 'table':
3       flush_chunk()
4       current_chunk['text'] = text.strip()
5       current_chunk['metadata']['chunk_type'] = 'table'
6       current_chunk['metadata']['section_title'] = active_title
7       current_chunk['metadata']['element_types'] = [elem_type]
8       current_chunk['metadata']['element_count'] = 1
9       current_chunk['metadata']['page_numbers'] = [page_num] if page_num else []
10      src = current_source_file or filename
11      current_chunk['metadata']['source_file'] = src
12      current_chunk['metadata']['source_filename'] = src
13  
14      if elem_metadata and hasattr(elem_metadata, 'text_as_html') and elem_metadata.text_as_html:
15          current_chunk['metadata']['text_as_html'] = elem_metadata.text_as_html
16  
17      flush_chunk(forced_type='table')</code></pre></td>
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
<tr><td>Kode 1 Baris 1-10</td><td>PDF dipartisi menggunakan `partition_pdf` dengan strategi `hi_res`, struktur tabel diaktifkan, dan informasi page break disertakan.</td></tr>
<tr><td>Kode 2 Baris 1-15</td><td>Fungsi internal `init_chunk()` membentuk struktur awal chunk beserta metadata seperti `chunk_type`, `section_title`, `page_numbers`, dan `source_file`.</td></tr>
<tr><td>Kode 3 Baris 1-17</td><td>Elemen tabel diperlakukan sebagai unit tersendiri. Jika `text_as_html` tersedia pada metadata elemen, struktur HTML tabel ikut disimpan untuk dipakai pada tahap embedding atau formatting konteks.</td></tr>
</tbody>
</table>

Implementasi ini menghasilkan chunk JSON yang mempertahankan metadata struktural dokumen. Bab ini tidak membahas penilaian performa karena analisis hasil evaluasi ditempatkan pada Bab 6.

### 5.3.2 Implementasi Max-Min Semantic Chunking

Max-Min Semantic Chunking diimplementasikan pada `src/chunking/maxmin_chunker.py`. Alur implementasinya memuat teks bersih, memecah teks menjadi kalimat, membuat embedding kalimat, mengelompokkan kalimat berdasarkan kemiripan semantik, lalu menyimpan hasilnya sebagai chunk JSON. Parameter yang digunakan pada fungsi utama antara lain `fixed_threshold`, `c`, `init_constant`, `batch_size`, dan pilihan backend embedding. Uraian pada subbab ini difokuskan pada alur embedding kalimat dan pengelompokan Max-Min sebagaimana ditampilkan pada potongan kode. Detail parameter mengikuti konfigurasi fungsi yang digunakan pada proses eksekusi chunking.

Implementasi Max-Min Semantic Chunking dapat dilihat pada Tabel 5.9 berikut.

Tabel 5.9 Kode sumber implementasi Max-Min Semantic Chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/chunking/maxmin_chunker.py
2   sentences = sent_tokenize(text)
3   sentences = [s.strip() for s in sentences if s.strip()]
4   
5   SKIP_CHARS_LIMIT = 32000
6   filtered, skipped = [], 0
7   for s in sentences:
8       if len(s) &lt;= SKIP_CHARS_LIMIT:
9           filtered.append(s)
10      else:
11          skipped += 1
12  sentences = filtered</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/chunking/maxmin_chunker.py
2   for i in range(1, len(sentences)):
3       cluster_embeddings = embeddings[cluster_start:cluster_end]
4       cluster_size = cluster_end - cluster_start
5       new_sentence_embedding = embeddings[i].reshape(1, -1)
6       new_sentence_similarities = cosine_similarity(
7           new_sentence_embedding,
8           cluster_embeddings
9       )[0]
10  
11      final_threshold = max(adjusted_threshold, fixed_threshold)
12      if new_sentence_similarity &gt; final_threshold:
13          current_paragraph.append(sentences[i])
14          cluster_end += 1
15      else:
16          paragraphs.append(current_paragraph)
17          current_paragraph = [sentences[i]]
18          cluster_start = i
19          cluster_end = i + 1</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/chunking/maxmin_chunker.py
2   text = load_text(text_path)
3   sentences = split_sentences(text)
4   embeddings = embed_sentences(
5       sentences,
6       embedding_model,
7       batch_size=batch_size,
8       use_gguf=use_gguf
9   )
10  paragraphs = apply_maxmin_chunking(
11      sentences,
12      embeddings,
13      fixed_threshold=fixed_threshold,
14      c=c,
15      init_constant=init_constant
16  )
17  chunks = convert_paragraphs_to_chunks(
18      paragraphs,
19      Path(text_path).name,
20      include_metadata=include_metadata
21  )</code></pre></td>
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
<tr><td>Kode 1 Baris 1-12</td><td>Teks dibagi menjadi kalimat dengan `sent_tokenize`, kemudian kalimat kosong dan kalimat yang terlalu panjang akibat artefak parsing disaring.</td></tr>
<tr><td>Kode 2 Baris 1-19</td><td>Fungsi `process_sentences()` membentuk cluster kalimat. Kalimat baru dibandingkan dengan cluster aktif menggunakan cosine similarity, kemudian diputuskan apakah digabung atau memulai cluster baru.</td></tr>
<tr><td>Kode 3 Baris 1-21</td><td>Fungsi `process_single_text()` mengorkestrasi pemuatan teks, sentence splitting, embedding kalimat, penerapan Max-Min, konversi ke format chunk, dan penyimpanan output JSON.</td></tr>
</tbody>
</table>

Implementasi Max-Min menggunakan representasi embedding kalimat untuk menentukan pengelompokan semantik. Bab ini hanya menjelaskan proses implementasi, sedangkan analisis perbandingan metode ditempatkan pada Bab 6.

### 5.3.3 Implementasi Recursive Chunking

Recursive Chunking diimplementasikan pada `src/chunking/recursive_split.py` dengan `RecursiveCharacterTextSplitter`. Implementasi ini menggunakan parameter `chunk_size`, `chunk_overlap`, dan daftar separator. Teks dipotong secara rekursif berdasarkan hierarki separator, kemudian hasil potongan dikonversi menjadi dictionary dengan metadata sumber file, metode chunking, panjang chunk, dan nomor halaman jika marker halaman tersedia.

Implementasi Recursive Chunking dapat dilihat pada Tabel 5.11 berikut.

Tabel 5.11 Kode sumber implementasi Recursive Chunking

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/chunking/recursive_split.py
2   if separators is None:
3       separators = ["\n\n", "\n", " ", ""]
4   
5   text_splitter = RecursiveCharacterTextSplitter(
6       chunk_size=chunk_size,
7       chunk_overlap=chunk_overlap,
8       length_function=length_function,
9       is_separator_regex=is_separator_regex,
10      separators=separators
11  )</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/chunking/recursive_split.py
2   chunks = text_splitter.split_text(text)
3   
4   if chunks:
5       chunk_sizes = [len(chunk) for chunk in chunks]
6       logger.info(f"Rata-rata karakter per chunk: {sum(chunk_sizes) / len(chunks):.2f}")
7       logger.info(f"Min karakter per chunk: {min(chunk_sizes)}")
8       logger.info(f"Max karakter per chunk: {max(chunk_sizes)}")</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/chunking/recursive_split.py
2   page_numbers = sorted({int(m) for m in _PAGE_MARKER.findall(chunk_text)})
3   chunk_text = _PAGE_MARKER.sub('', chunk_text).strip()
4   
5   chunk_dict = {
6       'chunk_id': chunk_id,
7       'text': chunk_text,
8       'num_characters': len(chunk_text)
9   }
10  
11  chunk_dict['metadata'] = {
12      'source_file': source_filename,
13      'chunking_method': 'recursive_character_text_splitter',
14      'chunk_length': len(chunk_text),
15      'page_numbers': page_numbers if page_numbers else None,
16  }</code></pre></td>
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
<tr><td>Kode 1 Baris 1-11</td><td>Splitter dibuat menggunakan `RecursiveCharacterTextSplitter` dengan parameter ukuran chunk, overlap, dan separator.</td></tr>
<tr><td>Kode 2 Baris 1-8</td><td>Fungsi `run_recursive_splitter()` menjalankan `split_text()` dan mencatat statistik panjang chunk.</td></tr>
<tr><td>Kode 3 Baris 1-16</td><td>Hasil chunk string dikonversi menjadi dictionary. Marker halaman diekstrak untuk metadata dan dihapus dari teks chunk.</td></tr>
</tbody>
</table>

Hasil dari metode ini berupa file JSON pada `data/chunked/recursive/`. Metadata yang disimpan membuat chunk tetap dapat ditelusuri ke file sumber dan halaman asalnya.

## 5.4 Implementasi Embedding

Tahap embedding diimplementasikan pada `src/embedding/`. Model embedding yang digunakan dalam implementasi adalah Qwen3-Embedding-4B, baik melalui backend HuggingFace maupun GGUF sesuai environment. Fungsi `QwenEmbedder` menerima teks tunggal atau daftar teks, menghasilkan embedding, dan melakukan normalisasi L2 jika parameter `normalize` aktif. Tahap batch embedding membaca chunk JSON, melakukan enrichment pada chunk tabel jika metadata HTML tersedia, menambahkan context prefix untuk metode Max-Min dan Recursive, lalu menyimpan embedding ke `data/embeddings/`.

Implementasi embedding dapat dilihat pada Tabel 5.13 berikut.

Tabel 5.13 Kode sumber implementasi embedding

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/embedding/embedder.py
2   def embed(self, texts, batch_size=32):
3       if isinstance(texts, str):
4           texts = [texts]
5   
6       if len(texts) == 0:
7           return np.array([])
8   
9       if self.mode == 'gguf':
10          embeddings = self._embed_gguf(texts)
11      elif self.mode == 'huggingface':
12          embeddings = self._embed_hf(texts)
13      else:
14          raise ValueError(f"Unknown mode: {self.mode}")
15  
16      if self.normalize:
17          embeddings = self._normalize_embeddings(embeddings)
18  
19      return embeddings</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/embedding/io.py
2   html = meta.get("text_as_html") or ""
3   if not html.strip():
4       continue
5   if meta.get("chunk_type") != "table":
6       continue
7   
8   table_text = _html_table_to_text(html)
9   if not table_text.strip():
10      continue
11  
12  section_title = (meta.get("section_title") or "").strip()
13  if section_title and not _is_noise_text(section_title):
14      prefix = section_title + "\n\n"
15  
16  chunk["text"] = prefix + table_text</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/embedding/embed_chunks.py
2   chunks = load_chunks_from_json(json_path)
3   n_enriched = enrich_table_chunk_texts(chunks)
4   cleaned_texts, valid_indices = clean_and_filter_chunks(chunks)
5   
6   valid_chunks = [chunks[i] for i in valid_indices]
7   if chunking_method in _METHODS_WITH_CONTEXT_PREFIX:
8       embed_texts = inject_context_prefix(valid_chunks, CONTEXT_PREFIX_CHARS)
9   else:
10      embed_texts = [c.get("text", "") for c in valid_chunks]
11  
12  embed_texts = [' '.join(t.split()) for t in embed_texts]
13  embeddings = embedder.embed(embed_texts)</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python">1   # Path: src/embedding/embed_chunks.py
2   output_path = Path(output_dir) / chunking_method / f"{json_file.stem}_embeddings.json"
3   metadata = {
4       "source_file": json_file.name,
5       "source_path": str(json_file),
6       "chunking_method": chunking_method,
7       "embedding_model": embedder.mode,
8       "normalized": embedder.normalize
9   }
10  
11  success = save_embeddings(
12      embeddings=embeddings,
13      chunks=chunks,
14      valid_indices=valid_indices,
15      output_path=str(output_path),
16      metadata=metadata
17  )</code></pre></td>
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
<tr><td>Kode 1 Baris 1-19</td><td>Fungsi `embed()` menerima teks, memilih backend GGUF atau HuggingFace, lalu menormalisasi embedding bila konfigurasi `normalize` bernilai benar.</td></tr>
<tr><td>Kode 2 Baris 1-16</td><td>Fungsi `enrich_table_chunk_texts()` memanfaatkan `text_as_html` untuk membentuk teks tabel yang lebih terstruktur dan menambahkan konteks judul jika tersedia.</td></tr>
<tr><td>Kode 3 Baris 1-13</td><td>Fungsi `embed_single_file()` membaca chunk JSON, melakukan enrichment, membersihkan teks, menerapkan context prefix untuk Max-Min dan Recursive, lalu membuat embedding.</td></tr>
<tr><td>Kode 4 Baris 1-17</td><td>Embedding dan metadata disimpan ke file JSON dalam subfolder metode chunking pada `data/embeddings/`.</td></tr>
</tbody>
</table>

Tahap embedding tidak mengubah file chunk asli pada `data/chunked/`. Enrichment dan context prefix digunakan pada teks yang di-embed dan disimpan untuk kebutuhan downstream, terutama ChromaDB.

## 5.5 Implementasi Vector Database dan Retrieval

Vector database diimplementasikan menggunakan ChromaDB pada modul `src/chroma/`. Embedding JSON dari `data/embeddings/` dimuat ke persistent storage `data/chroma/`. Setiap metode chunking memiliki collection terpisah, yaitu `collection_element_based`, `collection_maxmin_semantic`, dan `collection_recursive`. Retrieval dilakukan dengan query embedding yang dikirim ke `collection.query()`, kemudian hasilnya dikembalikan sebagai daftar dictionary berisi `id`, `document`, `metadata`, dan `distance`.

Implementasi Vector Database dan Retrieval dapat dilihat pada Tabel 5.15 berikut.

Tabel 5.15 Kode sumber implementasi Vector Database dan Retrieval

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/chroma/client.py
2   persist_path = Path(persist_directory)
3   persist_path.mkdir(parents=True, exist_ok=True)
4   _recover_stale_sqlite_journal(persist_path)
5   
6   client = chromadb.PersistentClient(
7       path=str(persist_path.absolute())
8   )</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/chroma/loader.py
2   embeddings = np.array(embeddings_list, dtype=np.float32)
3   ids = []
4   documents = []
5   metadatas = []
6   
7   for chunk in chunks:
8       chunk_id = f"{file_path.stem}_{chunk.get('original_index', chunk.get('embedding_index', 0))}"
9       ids.append(chunk_id)
10      documents.append(chunk.get('text', ''))
11      chunk_metadata = chunk.get('metadata', {}).copy()
12      chunk_metadata['source_file'] = metadata.get('source_file', file_path.stem)
13      chunk_metadata['chunking_method'] = metadata.get('chunking_method', 'unknown')
14      metadatas.append(chunk_metadata)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/chroma/loader.py
2   success = batch_add_documents(
3       collection=collection,
4       ids=ids,
5       embeddings=embeddings,
6       documents=documents,
7       metadatas=metadatas,
8       batch_size=batch_size
9   )</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python">1   # Path: src/chroma/query.py
2   results = collection.query(
3       query_embeddings=[query_embedding.tolist()],
4       n_results=k,
5       where=filter
6   )
7   
8   documents = []
9   for i in range(len(results['documents'][0])):
10      doc = {
11          'id': results['ids'][0][i],
12          'document': results['documents'][0][i],
13          'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
14          'distance': results['distances'][0][i] if 'distances' in results else None
15      }
16      documents.append(doc)</code></pre></td>
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
<tr><td>Kode 1 Baris 1-8</td><td>ChromaDB diinisialisasi sebagai persistent client pada direktori `data/chroma/` sehingga data vektor dapat digunakan kembali setelah aplikasi berhenti.</td></tr>
<tr><td>Kode 2 Baris 1-14</td><td>File embedding JSON dibaca, embedding dikonversi ke `numpy`, lalu ID, dokumen, dan metadata chunk disiapkan untuk ChromaDB.</td></tr>
<tr><td>Kode 3 Baris 1-9</td><td>Data dimasukkan ke collection menggunakan `batch_add_documents()` dalam ukuran batch tertentu.</td></tr>
<tr><td>Kode 4 Baris 1-16</td><td>Fungsi `similarity_search()` melakukan pencarian berdasarkan query embedding dan mengubah output ChromaDB menjadi list dictionary yang digunakan oleh pipeline RAG.</td></tr>
</tbody>
</table>

Implementasi pemuatan embedding ke ChromaDB berada pada `src/chroma/loader.py`. File `scripts/load_embeddings_to_chroma.py` hanya berperan sebagai entry point yang memanggil fungsi loader tersebut, sedangkan logika penyimpanan dan retrieval tetap berada pada modul `src/chroma/`.

## 5.6 Implementasi RAG Pipeline dan Generator

Pipeline RAG diimplementasikan pada `src/rag/pipeline.py`, sedangkan generator diimplementasikan pada `src/rag/generator.py`. Alur runtime dimulai dari query pengguna atau pertanyaan QA gold, dilanjutkan dengan embedding query, retrieval chunk dari collection ChromaDB sesuai metode chunking, formatting konteks, dan generation answer. Pipeline ini juga dipanggil oleh `src/streamlit/rag_chat.py` untuk menjalankan chat dan evaluasi batch. Model generator yang digunakan oleh antarmuka tersebut diarahkan ke Qwen3-4B-Instruct-2507 sesuai konfigurasi auto-detect pada `rag_chat.py`.

Implementasi RAG Pipeline dan Generator dapat dilihat pada Tabel 5.17 berikut.

Tabel 5.17 Kode sumber implementasi RAG Pipeline dan Generator

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/rag/pipeline.py
2   COLLECTION_NAMES = {
3       "element_based":   "collection_element_based",
4       "maxmin_semantic": "collection_maxmin_semantic",
5       "recursive":       "collection_recursive",
6   }
7   
8   collection_name = COLLECTION_NAMES[chunking_method]
9   self.collection = get_or_create_collection(chroma_client, collection_name)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/rag/pipeline.py
2   query_embedding = self.embedder.embed(query)
3   query_vec = query_embedding[0]
4   
5   results = similarity_search(self.collection, query_vec, k=k)
6   return results</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/rag/pipeline.py
2   retrieved = self.retrieve(query, k=k)
3   contexts = [self._format_context(doc) for doc in retrieved]
4   
5   raw = self.generator.generate(query, contexts)
6   if isinstance(raw, tuple):
7       answer, thinking = raw
8   else:
9       answer, thinking = raw, ""
10  
11  return {
12      "query": query,
13      "answer": answer,
14      "thinking": thinking,
15      "retrieved_chunks": retrieved,
16      "chunking_method": self.chunking_method,
17      "num_chunks": len(retrieved),
18      "elapsed_seconds": elapsed,
19  }</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python">1   # Path: src/rag/generator.py
2   context_block = "\n\n".join(
3       f"[Konteks {i + 1}]\n{ctx.strip()}"
4       for i, ctx in enumerate(contexts)
5   )
6   
7   user_content = (
8       f"Konteks:\n{context_block}\n\n"
9       f"Pertanyaan: {query}\n\n"
10      f"Jawaban:"
11  )
12  
13  return [
14      {"role": "system", "content": self.system_prompt},
15      {"role": "user", "content": user_content},
16  ]</code></pre></td>
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
<tr><td>Kode 1 Baris 1-9</td><td>Pipeline memetakan nama metode chunking ke collection ChromaDB yang sesuai.</td></tr>
<tr><td>Kode 2 Baris 1-6</td><td>Fungsi `retrieve()` membuat embedding query lalu melakukan similarity search pada collection aktif.</td></tr>
<tr><td>Kode 3 Baris 1-19</td><td>Fungsi `run()` menjalankan alur lengkap retrieval, formatting konteks, pemanggilan generator, dan pengembalian hasil.</td></tr>
<tr><td>Kode 4 Baris 1-16</td><td>Generator membangun pesan chat berisi system prompt, konteks hasil retrieval, pertanyaan, dan instruksi jawaban.</td></tr>
</tbody>
</table>

Prompt generator mengarahkan model untuk menjawab berdasarkan konteks dan menyatakan bahwa informasi tidak memadai jika konteks tidak cukup. Bagian ini hanya menjelaskan mekanisme pembentukan jawaban, bukan kualitas jawaban yang dihasilkan.

## 5.7 Implementasi Ground Truth Retrieval dan Dataset Evaluasi

Dataset evaluasi dibangun dari QA gold, kandidat chunk, anotasi manual, dan konversi label ke JSON ground truth retrieval. Kandidat chunk dibuat oleh `scripts/build_candidates_v3.py` dengan membaca QA gold dan seluruh chunk JSON dari tiga metode. Kandidat tersebut kemudian diberi label manual melalui `src/streamlit/app.py`. Hasil anotasi disimpan sebagai `retrieval_labels_final.csv` dan `retrieval_labels_final.xlsx`. Setelah itu, `scripts/convert_ground_truth_to_json.py` mengubah label final menjadi `qa_pairs_binary.json` dengan skema binary relevance, yaitu label `0` sebagai tidak relevan dan label `>= 1` sebagai relevan.

Implementasi Ground Truth Retrieval dan Dataset Evaluasi dapat dilihat pada Tabel 5.19 berikut.

Tabel 5.19 Kode sumber implementasi Ground Truth Retrieval dan Dataset Evaluasi

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: scripts/build_candidates_v3.py
2   METHODS = ["element_based", "maxmin_semantic", "recursive"]
3   PRE_K_DEFAULT = 10
4   TOP_K_DEFAULT = 5
5   
6   def load_qa_gold():
7       df = pd.read_excel(str(QA_XLSX), sheet_name="qa_gold", dtype=str).fillna("")
8       rows = df.to_dict("records")
9       return rows
10  
11  def load_chunks(doc_id, method):
12      stem = DOC_MAP.get(doc_id)
13      fp = CHUNK_DIR / method / f"{stem}_chunks.json"
14      with open(fp, encoding="utf-8") as f:
15          return json.load(f)</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: scripts/build_candidates_v3.py
2   for method in METHODS:
3       candidates = build_candidates_for_group(qa, method, pre_k, top_k)
4   
5       if not candidates:
6           rows.append({
7               "query_id": qid,
8               "doc_id": doc_id,
9               "method": method,
10              "chunk_id": "",
11              "match_type": "not_found",
12              "suggested_label": "0",
13              "status": "needs_manual_validation",
14          })
15          continue</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/app.py
2   OUTPUT_XLSX = ROOT / "data/ground_truth/retrieval_labels_final.xlsx"
3   OUTPUT_CSV = ROOT / "data/ground_truth/retrieval_labels_final.csv"
4   
5   def apply_label(qid, method, chunk_id, label):
6       df = st.session_state.df
7       mask = (
8           (df["query_id"] == qid)
9           &amp; (df["method"] == method)
10          &amp; (df["chunk_id"] == chunk_id)
11      )
12      df.loc[mask, "label"] = label
13      df.loc[mask, "annotator"] = st.session_state.get("annotator_name", "")
14      st.session_state.df = df
15      save_data(df)</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python">1   # Path: scripts/convert_ground_truth_to_json.py
2   if label &lt; threshold:
3       continue
4   
5   pipeline_method = CSV_METHOD_TO_CODE[method_csv]
6   file_stem = DOC_TO_FILE_STEM[doc_id]
7   chroma_id = f"{file_stem}_{chunk_id_int}"
8   
9   result[q_id][pipeline_method].append(chroma_id)
10  
11  relevant_chunk_ids = {
12      m: query_labels.get(m, []) for m in ALL_METHODS
13  }</code></pre></td>
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
<tr><td>Kode 1 Baris 1-15</td><td>Script kandidat menetapkan tiga metode chunking, membaca QA gold, dan memuat chunk JSON sesuai `doc_id` dan metode.</td></tr>
<tr><td>Kode 2 Baris 1-15</td><td>Untuk setiap QA dan metode, kandidat dibangun dan disiapkan dalam format yang dapat divalidasi manual. Jika kandidat tidak ditemukan, baris tetap dibuat dengan status `needs_manual_validation`.</td></tr>
<tr><td>Kode 3 Baris 1-15</td><td>Aplikasi anotasi menyimpan label manual ke CSV dan XLSX final. Fungsi `apply_label()` memperbarui label berdasarkan kombinasi `query_id`, `method`, dan `chunk_id`.</td></tr>
<tr><td>Kode 4 Baris 1-13</td><td>Konverter membaca label final, memasukkan chunk dengan label minimal sesuai `relevance_threshold`, membentuk ID ChromaDB, dan menyusun `relevant_chunk_ids` per metode.</td></tr>
</tbody>
</table>

Ground truth retrieval yang digunakan oleh evaluasi akhir adalah `data/ground_truth/qa_pairs_binary.json`. File tersebut dibaca oleh evaluasi batch, bukan dibentuk ulang saat evaluasi dijalankan.

## 5.8 Implementasi Evaluasi Retrieval dan Generation

Evaluasi akhir diimplementasikan melalui tab Evaluasi Batch pada `src/streamlit/rag_chat.py`. Fitur ini membaca QA gold dari `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx` dan ground truth retrieval binary dari `data/ground_truth/qa_pairs_binary.json`. Evaluasi dijalankan untuk tiga metode chunking dan dapat menggunakan rentang top-k yang dipilih pada antarmuka. Proses evaluasi melakukan pre-compute query embedding, retrieval per metode, generation answer, perhitungan metrik retrieval dan generation, lalu menyimpan hasil ke CSV pada `results/final/generation/`. Implementasi evaluasi akhir tidak dijalankan dari `scripts/run_generation_eval.py` maupun `scripts/run_retrieval_eval.py`.

Implementasi evaluasi retrieval dan generation dapat dilihat pada Tabel 5.21 berikut.

Tabel 5.21 Kode sumber implementasi evaluasi retrieval dan generation

<table>
<thead>
<tr><th>No</th><th>Kode Sumber</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/rag_chat.py
2   EVAL_RESULTS_DIR = ROOT / "results" / "final" / "generation"
3   EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
4   
5   def _load_qa_gold():
6       qa_path = ROOT / "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx"
7       if qa_path.exists():
8           _QA_GOLD_DF = pd.read_excel(qa_path, sheet_name="qa_gold")
9       return _QA_GOLD_DF
10  
11  def _load_ground_truth():
12      gt_path = ROOT / "data/ground_truth/qa_pairs_binary.json"
13      if gt_path.exists():
14          with open(gt_path, encoding="utf-8") as f:
15              _GT_BINARY = json.load(f)
16      return _GT_BINARY</code></pre></td>
</tr>
<tr>
<td>2</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/rag_chat.py
2   top_k_min = st.number_input("Min Top-K", min_value=1, max_value=10, value=1)
3   top_k_max = st.number_input("Max Top-K", min_value=1, max_value=10, value=10)
4   
5   def _run_eval_and_save(qa_subset, mode_tag, top_k_range):
6       gt_data = _load_ground_truth()
7       gt_lookup = {item["id"]: item for item in gt_data}
8       min_k, max_k = top_k_range
9   
10      query_embeddings = {}
11      for _, qa_row in qa_subset.iterrows():
12          q_id = str(qa_row["query_id"])
13          question = str(qa_row["question"])
14          q_vec = pipeline.embedder.embed(question)[0]
15          query_embeddings[q_id] = (q_vec, True)</code></pre></td>
</tr>
<tr>
<td>3</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/rag_chat.py
2   for current_k in range(min_k, max_k + 1):
3       for _, qa_row in qa_subset.iterrows():
4           question = str(qa_row["question"])
5           gold_ans = str(qa_row["gold_answer"])
6           q_id = str(qa_row["query_id"])
7           gt_item = gt_lookup.get(q_id)
8           q_vec, embed_ok = query_embeddings.get(q_id, (None, False))
9   
10          for method in METHODS:
11              p = RAGPipeline(
12                  embedder=pipeline.embedder,
13                  generator=pipeline.generator,
14                  chroma_client=pipeline.chroma_client,
15                  chunking_method=method,
16                  top_k=current_k,
17              )
18              retrieved = p.retrieve_by_vector(q_vec, k=current_k) if embed_ok else p.retrieve(question, k=current_k)
19              retrieved_ids = [doc.get("id", "") for doc in retrieved]</code></pre></td>
</tr>
<tr>
<td>4</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/rag_chat.py
2   if rel_ids:
3       precision_val = compute_precision_at_k(retrieved_ids, rel_ids, current_k)
4       recall_val = compute_recall_at_k(retrieved_ids, rel_ids, current_k)
5       mrr_val = compute_mrr(retrieved_ids, rel_ids)
6   else:
7       precision_val = recall_val = mrr_val = "N/A"
8   
9   contexts = [p._format_context(doc) for doc in retrieved]
10  raw = pipeline.generator.generate(question, contexts)
11  gen_answer = raw[0] if isinstance(raw, tuple) else raw
12  bleu_val = compute_bleu(gen_answer, gold_ans)
13  rouge_val = compute_rouge(gen_answer, gold_ans, rouge_type="rougeL", mode="recall")</code></pre></td>
</tr>
<tr>
<td>5</td>
<td><pre><code class="language-python">1   # Path: src/streamlit/rag_chat.py
2   rows.append({
3       "query_id": q_id,
4       "method": METHOD_LABELS[method],
5       "question": question,
6       "gold_answer": gold_ans,
7       "generated_answer": gen_answer,
8       "precision_at_k": round(precision_val, 4) if isinstance(precision_val, (int, float)) else precision_val,
9       "recall_at_k": round(recall_val, 4) if isinstance(recall_val, (int, float)) else recall_val,
10      "mrr": round(mrr_val, 4) if isinstance(mrr_val, (int, float)) else mrr_val,
11      "bleu": round(bleu_val, 4) if isinstance(bleu_val, (int, float)) else bleu_val,
12      "rouge_l_recall": round(rouge_val, 4) if isinstance(rouge_val, (int, float)) else rouge_val,
13      "error": error_msg,
14      "hardware_info": hw_info_str,
15  })
16  
17  df_result = pd.DataFrame(rows)
18  save_path = EVAL_RESULTS_DIR / f"eval_{ts_wib}_{mode_tag}_top{current_k}.csv"
19  df_result.to_csv(save_path, index=False)</code></pre></td>
</tr>
<tr>
<td>6</td>
<td><pre><code class="language-python">1   # Path: src/evaluation/metrics.py
2   def compute_precision_at_k(
3       retrieved_ids: List[str],
4       relevant_ids: List[str],
5       k: int,
6   ) -&gt; float:
7       if k &lt;= 0:
8           return 0.0
9       top_k = retrieved_ids[:k]
10      relevant_set = set(relevant_ids)
11      hits = sum(1 for r_id in top_k if r_id in relevant_set)
12      return hits / k</code></pre></td>
</tr>
<tr>
<td>7</td>
<td><pre><code class="language-python">1   # Path: src/evaluation/metrics.py
2   def compute_recall_at_k(
3       retrieved_ids: List[str],
4       relevant_ids: List[str],
5       k: int,
6   ) -&gt; float:
7       if not relevant_ids:
8           return 0.0
9       top_k = retrieved_ids[:k]
10      relevant_set = set(relevant_ids)
11      hits = sum(1 for r_id in top_k if r_id in relevant_set)
12      return hits / len(relevant_ids)</code></pre></td>
</tr>
<tr>
<td>8</td>
<td><pre><code class="language-python">1   # Path: src/evaluation/metrics.py
2   def compute_mrr(
3       retrieved_ids: List[str],
4       relevant_ids: List[str],
5   ) -&gt; float:
6       relevant_set = set(relevant_ids)
7       for rank, r_id in enumerate(retrieved_ids, start=1):
8           if r_id in relevant_set:
9               return 1.0 / rank
10      return 0.0</code></pre></td>
</tr>
<tr>
<td>9</td>
<td><pre><code class="language-python">1   # Path: src/evaluation/metrics.py
2   def compute_bleu(response: str, reference: str) -&gt; float:
3       try:
4           from sacrebleu import corpus_bleu
5           result = corpus_bleu([response], [[reference]])
6           return result.score / 100.0
7       except Exception as e:
8           logger.error(f"compute_bleu error: {e}")
9           return 0.0</code></pre></td>
</tr>
<tr>
<td>10</td>
<td><pre><code class="language-python">1   # Path: src/evaluation/metrics.py
2   def compute_rouge(
3       response: str,
4       reference: str,
5       rouge_type: str = "rougeL",
6       mode: str = "recall",
7   ) -&gt; float:
8       from rouge_score import rouge_scorer
9       scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
10      scores = scorer.score(reference, response)
11      rouge_score = scores[rouge_type]
12      if mode == "precision":
13          return rouge_score.precision
14      elif mode == "recall":
15          return rouge_score.recall
16      elif mode == "fmeasure":
17          return rouge_score.fmeasure</code></pre></td>
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
<tr><td>Kode 1 Baris 1-16</td><td>Evaluasi batch menyiapkan direktori output final, membaca QA gold dari file Excel, dan membaca ground truth binary dari `qa_pairs_binary.json`.</td></tr>
<tr><td>Kode 2 Baris 1-15</td><td>Antarmuka menyediakan konfigurasi mode QA dan rentang top-k. Fungsi evaluasi melakukan pre-compute embedding query agar embedding tidak dihitung berulang untuk setiap metode dan top-k.</td></tr>
<tr><td>Kode 3 Baris 1-19</td><td>Evaluasi melakukan loop untuk setiap top-k, setiap QA, dan setiap metode chunking. Untuk setiap metode, pipeline baru dibuat dengan collection yang sesuai, lalu retrieval dilakukan menggunakan query vector yang sudah dihitung.</td></tr>
<tr><td>Kode 4 Baris 1-13</td><td>Metrik retrieval dihitung jika query memiliki chunk relevan pada ground truth. Setelah itu, konteks hasil retrieval dikirim ke generator untuk menghasilkan jawaban dan menghitung BLEU serta ROUGE-L Recall.</td></tr>
<tr><td>Kode 5 Baris 1-19</td><td>Hasil evaluasi disusun dalam row CSV dengan kolom query, metode, jawaban gold, jawaban generated, metrik retrieval, metrik generation, error, dan hardware info. File disimpan per top-k di `results/final/generation/`.</td></tr>
<tr><td>Kode 6 Baris 1-12</td><td>Fungsi `compute_precision_at_k()` mengambil hasil retrieval sampai cutoff `k`, menghitung jumlah chunk yang cocok dengan ground truth, lalu membagi jumlah hit dengan nilai `k`.</td></tr>
<tr><td>Kode 7 Baris 1-12</td><td>Fungsi `compute_recall_at_k()` menghitung jumlah chunk relevan yang ditemukan pada top-k dan membaginya dengan jumlah chunk relevan pada ground truth.</td></tr>
<tr><td>Kode 8 Baris 1-10</td><td>Fungsi `compute_mrr()` mencari posisi chunk relevan pertama pada hasil retrieval dan mengembalikan nilai reciprocal rank.</td></tr>
<tr><td>Kode 9 Baris 1-9</td><td>Fungsi `compute_bleu()` menggunakan `sacrebleu.corpus_bleu()` untuk menghitung skor BLEU dan mengonversi skala skor ke rentang 0 sampai 1.</td></tr>
<tr><td>Kode 10 Baris 1-17</td><td>Fungsi `compute_rouge()` menggunakan `rouge_scorer.RougeScorer` dan mengembalikan nilai ROUGE sesuai mode yang digunakan, yaitu precision, recall, atau f-measure.</td></tr>
</tbody>
</table>

Dengan implementasi tersebut, evaluasi akhir dilakukan dari tab Evaluasi Batch pada `rag_chat.py`. Output yang dihasilkan berupa file CSV per top-k. Bab ini tidak menyajikan nilai metrik atau interpretasi performa karena pembahasan hasil evaluasi ditempatkan pada Bab 6.
