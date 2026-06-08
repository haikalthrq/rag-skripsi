# CODEMAP - rag-skripsi

Terakhir diperbarui: 2026-06-08 WIB

Dokumen ini memetakan workspace `rag-skripsi` dari sudut pandang kode, data, entry point, dan artefak evaluasi. Gunakan ini sebagai peta cepat sebelum mengubah project.

## 1. Ringkasan Project

`rag-skripsi` adalah project skripsi untuk membangun dan mengevaluasi sistem Retrieval-Augmented Generation (RAG) berbasis dokumen statistik BPS Indonesia.

Tujuan utama:

- Membandingkan 3 metode chunking:
  - `element_based`
  - `maxmin_semantic`
  - `recursive`
- Menggunakan embedding Qwen3-Embedding-4B.
- Menggunakan generator Qwen3-4B-Instruct-2507.
- Menyimpan vektor di ChromaDB.
- Mengevaluasi retrieval dengan `Precision@k`, `Recall@k`, dan `MRR`.
- Mengevaluasi generation dengan `BLEU` dan `ROUGE-L`.

Skema evaluasi aktif:

- Binary relevance.
- Label `1` dan `2` lama dianggap relevan.
- Label `0` dianggap tidak relevan.
- Ground truth aktif: `data/ground_truth/qa_pairs_binary.json`.

## 2. Alur Sistem End-to-End

Alur data utama:

```text
data/raw/*.pdf
  -> src/preprocessing/
  -> data/cleaned/*.txt
  -> src/chunking/
  -> data/chunked/{element_based,maxmin_semantic,recursive}/*.json
  -> src/embedding/
  -> data/embeddings/*.json
  -> scripts/load_embeddings_to_chroma.py
  -> data/chroma/
  -> src/rag/pipeline.py
  -> src/streamlit/rag_chat.py atau scripts/run_generation_eval.py
  -> results/final/generation/*.csv
  -> notebooks/eval_visualization.ipynb
  -> results/final/figures/*
```

Alur ground truth retrieval:

```text
data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx
  + data/chunked/{method}/*.json
  -> scripts/build_candidates_v3.py
  -> data/ground_truth/archive/retrieval_relevant_chunks_candidate_v3_evidence_aware.*
  -> src/streamlit/app.py
  -> data/ground_truth/retrieval_labels_final.csv/.xlsx
  -> scripts/convert_ground_truth_to_json.py
  -> data/ground_truth/qa_pairs_binary.json
```

Alur runtime RAG:

```text
user query / QA gold selection
  -> QwenEmbedder.embed(query)
  -> ChromaDB similarity search
  -> retrieved chunks
  -> context formatter
  -> Qwen3 generator
  -> answer
  -> retrieval metrics + generation metrics
```

## 3. Top-Level Folder Map

### `src/`

Kode aplikasi utama. Ini sumber yang harus diprioritaskan saat mengubah logic.

Subfolder:

- `src/preprocessing/`: ekstraksi dan cleaning teks dari PDF.
- `src/chunking/`: implementasi 3 strategi chunking.
- `src/embedding/`: embedder, enrichment chunk, dan penyimpanan embedding.
- `src/chroma/`: client, loader, dan query ChromaDB.
- `src/rag/`: pipeline RAG dan generator.
- `src/evaluation/`: metrik dan evaluator.
- `src/streamlit/`: UI anotasi dan demo RAG chat.

### `scripts/`

Entry point CLI untuk job batch dan utilitas project.

File aktif:

- `build_candidates_v3.py`
- `convert_ground_truth_to_json.py`
- `download_embedding_model.py`
- `download_generator_model.py`
- `load_embeddings_to_chroma.py`
- `run_generation_eval.py`
- `run_retrieval_eval.py`

### `data/`

Data input, intermediate, dan ground truth.

Subfolder penting:

- `data/raw/`: 10 PDF BPS asli.
- `data/cleaned/`: teks hasil preprocessing.
- `data/chunked/`: chunk JSON per metode.
- `data/embeddings/`: embedding JSON.
- `data/chroma/`: ChromaDB persistent storage, biasanya ignored/symlink di environment cloud.
- `data/ground_truth/`: QA gold, label retrieval, dan JSON evaluasi aktif.

### `results/`

Output evaluasi, visualisasi, archive, dan chat history.

Subfolder penting:

- `results/final/generation/`: hasil evaluasi final Top-1 sampai Top-10.
- `results/final/figures/`: chart dan tabel final untuk analisis Bab 6.
- `results/visualizations/bab6/`: artefak tambahan Bab 6.
- `results/archive/`: hasil lama, strict lama, run GPU lama, dan baseline historis.
- `results/chat_history/`: riwayat chat Streamlit persisten.

### `notebooks/`

Notebook analisis dan visualisasi.

- `eval_visualization.ipynb`: notebook visualisasi aktif.
- `rag_inference.ipynb`: notebook inference manual.
- `_cache/data_eval.pkl`: cache data evaluasi visualisasi.

### `docs/`

Dokumentasi project, konteks percakapan, referensi metrik/model, dan dokumen skripsi.

File penting:

- `PROJECT_CONTEXT.txt`: konteks project paling lengkap dan terbaru.
- `Chat Context.txt`: ringkasan konteks chat.
- `SKRIPSI.md`: naskah/struktur skripsi.
- `CODEMAP.md`: file ini.

### `tests/`

Unit test aktif:

- `tests/test_evaluation.py`

Fokus test:

- BLEU
- ROUGE-L
- Precision@k
- Recall@k
- MRR
- summary aggregation
- QA gold loading

## 4. Source Code Map

### 4.1 Preprocessing

#### `src/preprocessing/pdf_extractor.py`

Fungsi utama:

- `_extract_page_hybrid(page)`
- `extract_text(pdf_path)`
- `extract_text_with_metadata(pdf_path)`

Tanggung jawab:

- Membaca PDF dengan PyMuPDF.
- Menghasilkan teks bersih per halaman.
- Mengatasi masalah PDF dua kolom dengan pendekatan hybrid/center-based extraction.

Output downstream:

- Dipakai oleh `src/preprocessing/pipeline.py`.
- Menghasilkan teks untuk `data/cleaned/`.

#### `src/preprocessing/text_cleaner.py`

Fungsi utama:

- `clean_text(text)`
- `clean_text_advanced(...)`
- `remove_headers_footers(...)`

Tanggung jawab:

- Normalisasi whitespace.
- Membersihkan noise ekstraksi PDF.
- Optional removal untuk pola header/footer.

#### `src/preprocessing/pipeline.py`

Fungsi utama:

- `setup_logging(...)`
- `get_pdf_files(...)`
- `process_single_pdf(...)`
- `run_preprocessing(...)`
- `run_preprocessing_single(...)`

Tanggung jawab:

- Orkestrasi PDF -> TXT.
- Menulis output ke `data/cleaned/`.

### 4.2 Chunking

#### `src/chunking/element_based.py`

Fungsi utama:

- `partition_document(...)`
- `categorize_element(...)`
- `merge_small_chunks_backward(...)`
- `convert_elements_to_chunks(...)`
- `save_chunks(...)`
- `process_single_pdf(...)`
- `run_element_based_chunking(...)`

Tanggung jawab:

- Menggunakan `unstructured.partition_pdf(strategy="hi_res")`.
- Menghasilkan chunk berbasis elemen PDF.
- Melakukan composite chunking, bukan 1 elemen = 1 chunk.
- Menyimpan metadata tabel seperti `text_as_html`, `page_numbers`, `section_title`, dan `source_file`.

Output:

- `data/chunked/element_based/*_chunks.json`

Catatan:

- OCR/table corruption pada beberapa chunk adalah keterbatasan ekstraksi PDF, bukan otomatis bug kode.

#### `src/chunking/maxmin_chunker.py`

Fungsi utama:

- `split_sentences(...)`
- `embed_sentences(...)`
- `apply_maxmin_chunking(...)`
- `convert_paragraphs_to_chunks(...)`
- `process_single_text(...)`
- `run_maxmin_chunking(...)`

Tanggung jawab:

- Custom MaxMin semantic chunking.
- Membuat chunk berdasarkan representasi embedding kalimat dan diversity.
- Dapat memakai GGUF atau HuggingFace embedding.

Output:

- `data/chunked/maxmin_semantic/*_chunks.json`

#### `src/chunking/recursive_split.py`

Fungsi utama:

- `create_text_splitter(...)`
- `run_recursive_splitter(...)`
- `convert_chunks_to_dict(...)`
- `process_single_text(...)`
- `run_recursive_chunking(...)`

Tanggung jawab:

- Menggunakan `RecursiveCharacterTextSplitter`.
- Membuat chunk fixed-ish dengan overlap.

Output:

- `data/chunked/recursive/*_chunks.json`

### 4.3 Embedding

#### `src/embedding/embedder.py`

Class dan fungsi utama:

- `QwenEmbedder`
- `initialize_gguf_embedder(...)`
- `initialize_hf_embedder(...)`

Tanggung jawab:

- Wrapper embedding Qwen3.
- Mendukung backend GGUF via `llama-cpp-python`.
- Mendukung backend HuggingFace via `sentence-transformers`.
- Normalisasi embedding untuk retrieval.

#### `src/embedding/io.py`

Fungsi utama:

- `load_chunks_from_json(...)`
- `_html_table_to_text(...)`
- `enrich_table_chunk_texts(...)`
- `clean_and_filter_chunks(...)`
- `save_embeddings(...)`
- `load_embeddings(...)`

Tanggung jawab:

- Membaca chunk JSON.
- Membersihkan chunk noise.
- Mengubah `text_as_html` tabel menjadi teks row-oriented.
- Menambahkan context prefix untuk chunk tabel.
- Menyimpan dan membaca embedding.

Catatan penting:

- `enrich_table_chunk_texts()` hanya memengaruhi embedding/ChromaDB, bukan file asli `data/chunked/`.

#### `src/embedding/embed_chunks.py`

Fungsi utama:

- `inject_context_prefix(...)`
- `embed_single_file(...)`
- `embed_all_chunks(...)`

Tanggung jawab:

- Batch embedding semua file chunk.
- Menulis output ke `data/embeddings/`.

### 4.4 ChromaDB

#### `src/chroma/client.py`

Fungsi utama:

- `initialize_chroma_client(...)`
- `get_or_create_collection(...)`
- `delete_collection(...)`
- `list_collections(...)`
- `get_collection_info(...)`
- `reset_collection(...)`

Tanggung jawab:

- Membuka persistent ChromaDB.
- Mengelola collection per chunking method.

Collection aktif:

- `collection_element_based`
- `collection_maxmin_semantic`
- `collection_recursive`

#### `src/chroma/loader.py`

Fungsi utama:

- `batch_add_documents(...)`
- `load_to_chroma(...)`
- `load_all_embeddings_to_chroma(...)`

Tanggung jawab:

- Membaca embedding JSON.
- Memasukkan documents, embeddings, metadatas, dan ids ke ChromaDB.

#### `src/chroma/query.py`

Class dan fungsi utama:

- `ChromaRetriever`
- `similarity_search(...)`
- `similarity_search_with_score(...)`
- `search_by_vector(...)`
- `get_documents_by_ids(...)`
- `filter_by_metadata(...)`

Tanggung jawab:

- Query ChromaDB berdasarkan embedding query.
- Mengembalikan list dict `{id, document, metadata, distance}`.

### 4.5 RAG Runtime

#### `src/rag/pipeline.py`

Class dan fungsi utama:

- `RAGPipeline`
- `build_pipeline(...)`

Tanggung jawab:

- Orkestrasi end-to-end:
  - embed query
  - retrieve ChromaDB
  - format context
  - generate answer
  - return result dict

Input penting:

- `chunking_method`
- `top_k`
- `embedder_mode`
- `generator_type`
- `chroma_path`

Output `RAGPipeline.run(...)`:

- `query`
- `answer`
- `thinking`
- `retrieved_chunks`
- `chunking_method`
- `num_chunks`
- `elapsed_seconds`

#### `src/rag/generator.py`

Class dan fungsi utama:

- `RAGGenerator`
- `HFRAGGenerator`
- `initialize_hf_generator(...)`
- `initialize_gguf_generator(...)`

Tanggung jawab:

- Membuat prompt RAG.
- Menjalankan generator GGUF atau HuggingFace.
- Menangani system prompt Bahasa Indonesia.
- Menangani output thinking jika model/backend mengembalikan thinking.

System prompt:

- Jawab berdasarkan konteks.
- Jika konteks tidak cukup, nyatakan tidak memiliki informasi memadai.
- Jawab dalam Bahasa Indonesia.

### 4.6 Evaluation

#### `src/evaluation/metrics.py`

Fungsi utama:

- `compute_precision_at_k(...)`
- `compute_recall_at_k(...)`
- `compute_mrr(...)`
- `compute_bleu(...)`
- `compute_rouge(...)`

Tanggung jawab:

- Implementasi metrik final.
- Retrieval metrics berbasis chunk ID.
- BLEU memakai `sacrebleu.corpus_bleu([response], [[reference]]) / 100`.
- ROUGE-L memakai `rouge_score.RougeScorer(..., use_stemmer=False)`.

Catatan penting:

- Jangan balik argumen BLEU.
- Jangan pakai `ragas`, `rank-eval`, atau `rapidfuzz`.

#### `src/evaluation/evaluator.py`

Class dan fungsi utama:

- `load_ground_truth(...)`
- `MethodResult`
- `RAGEvaluator`
- `build_evaluator(...)`

Tanggung jawab:

- Evaluasi retrieval per method.
- Optional generation metrics jika generator diberikan.
- Aggregate per-query metrics menjadi method-level metrics.

### 4.7 Streamlit Apps

#### `src/streamlit/app.py`

Fungsi utama:

- `load_data(...)`
- `save_data(...)`
- `highlight_excerpt(...)`
- `apply_label(...)`
- `render_sidebar(...)`
- `render_query_panel(...)`
- `render_chunk_card(...)`
- `main(...)`

Tanggung jawab:

- UI anotasi retrieval labels.
- Dipakai untuk menyelesaikan anotasi manual kandidat retrieval.
- Output aktif:
  - `data/ground_truth/retrieval_labels_final.csv`
  - `data/ground_truth/retrieval_labels_final.xlsx`

Status:

- Anotasi sudah selesai, tetapi app masih bisa dibuka ulang untuk audit.

#### `src/streamlit/rag_chat.py`

Fungsi utama:

- `_load_qa_gold()`
- `_load_ground_truth()`
- `_compute_chat_retrieval_metrics(...)`
- `load_pipeline()`
- `run_method(...)`
- `stream_answer(...)`
- `_render_history_turn(...)`

Tanggung jawab:

- Demo RAG interaktif.
- Input query berupa pilihan QA Gold, bukan text bebas.
- Mendukung single/compare mode.
- Menampilkan BLEU, ROUGE-L, Precision@k, Recall@k, dan MRR.
- Menyimpan chat history ke `results/chat_history/chat_history.jsonl`.

Default penting:

- `DEFAULT_TOP_K = 8`
- `DEFAULT_MAX_TOK = 16384`
- Ground truth: `qa_pairs_binary.json`

#### `src/streamlit/test_app.py`

Tanggung jawab:

- Test/diagnostic untuk Streamlit annotation app.
- Banyak test bersifat UI/data validation, bukan pytest utama project.

## 5. Scripts Map

### `scripts/build_candidates_v3.py`

Tujuan:

- Membuat kandidat chunk untuk anotasi manual.
- Menggunakan oracle pooling berbasis text/evidence matching.
- Tidak menggunakan ChromaDB untuk memilih kandidat.

Input:

- QA gold Excel.
- `data/chunked/{method}/*.json`

Output:

- Kandidat anotasi di archive ground truth.
- Summary/report kandidat.

Catatan:

- Script ini relevan jika ingin rebuild kandidat anotasi, bukan untuk inference biasa.

### `scripts/convert_ground_truth_to_json.py`

Tujuan:

- Mengonversi label final CSV/XLSX menjadi JSON evaluasi retrieval.

Input:

- QA gold.
- `retrieval_labels_final.csv` atau `.xlsx`

Output aktif:

- `data/ground_truth/qa_pairs_binary.json`

Command binary:

```bash
python scripts/convert_ground_truth_to_json.py --output data/ground_truth/qa_pairs_binary.json --relevance_threshold 1
```

### `scripts/load_embeddings_to_chroma.py`

Tujuan:

- Load embedding JSON ke ChromaDB.

Input:

- `data/embeddings/`

Output:

- `data/chroma/`

### `scripts/run_retrieval_eval.py`

Tujuan:

- Evaluasi retrieval tanpa generation.

Metrik:

- Precision@k
- Recall@k
- MRR

Input:

- `data/ground_truth/qa_pairs_binary.json`
- ChromaDB

### `scripts/run_generation_eval.py`

Tujuan:

- Evaluasi full RAG generation.
- Membaca QA gold.
- Retrieve top-k chunks.
- Generate answer.
- Hitung retrieval dan generation metrics.

Output final:

- `results/final/generation/eval_20260531_*_full_top{1..10}.csv`

Catatan:

- Output final saat ini sudah valid dan tidak perlu direrun kecuali ada perubahan metodologis.

### `scripts/download_embedding_model.py`

Tujuan:

- Download embedding model dari HuggingFace.

### `scripts/download_generator_model.py`

Tujuan:

- Download generator model dari HuggingFace.

## 6. Data Contract

### QA Gold

File:

- `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx`

Sheet:

- `qa_gold`

Kolom kunci:

- `query_id`
- `question`
- `gold_answer`
- `evidence_type`
- `table_id`
- `row_label`
- `column_label`
- `gold_value`
- `evidence_text`
- `evidence_anchor`

Catatan:

- Query ID tidak harus berurutan.
- Total QA aktif: 30.

### Retrieval Labels

File:

- `data/ground_truth/retrieval_labels_final.csv`
- `data/ground_truth/retrieval_labels_final.xlsx`

Kolom penting:

- `query_id`
- `method`
- `chunk_id`
- `label`

Label:

- `0`: tidak relevan.
- `1`: relevan dalam skema binary.
- Label lama `2` telah digabung ke relevan pada skema binary.

### Retrieval Ground Truth JSON

File:

- `data/ground_truth/qa_pairs_binary.json`

Format:

```json
[
  {
    "id": "Q001",
    "question": "...",
    "reference_answer": "...",
    "relevant_chunk_ids": {
      "element_based": ["..."],
      "maxmin_semantic": ["..."],
      "recursive": ["..."]
    }
  }
]
```

### Chunk JSON

Folder:

- `data/chunked/element_based/`
- `data/chunked/maxmin_semantic/`
- `data/chunked/recursive/`

Field umum:

- `chunk_id`
- `text`
- `metadata`

Metadata umum:

- `source_file`
- `page_numbers` atau `page_range`
- `section_title`
- `chunk_type`

Metadata khusus element-based table:

- `text_as_html`
- `element_types`

### Generation Eval CSV

Folder:

- `results/final/generation/`

Kolom final umum:

- `query_id`
- `method`
- `question`
- `gold_answer`
- `generated_answer`
- `precision_at_k`
- `recall_at_k`
- `mrr`
- `bleu`
- `rouge_l_recall`
- `error`
- `hardware_info`

Catatan:

- Ada 10 file final Top-1 sampai Top-10.
- Untuk analisis final Bab 6, banyak tabel memakai Top-5 sampai Top-10.

## 7. Results dan Artefak Aktif

### `results/final/generation/`

Berisi run final RTX 3090 2026-05-31.

File:

- `eval_20260531_181551_full_top1.csv`
- `eval_20260531_182241_full_top2.csv`
- `eval_20260531_183044_full_top3.csv`
- `eval_20260531_183822_full_top4.csv`
- `eval_20260531_184708_full_top5.csv`
- `eval_20260531_185558_full_top6.csv`
- `eval_20260531_190431_full_top7.csv`
- `eval_20260531_191252_full_top8.csv`
- `eval_20260531_192113_full_top9.csv`
- `eval_20260531_192940_full_top10.csv`

### `results/final/figures/`

Chart aktif:

- `chart1_precision_trend.png`
- `chart2_recall_trend.png`
- `chart3_mrr_trend.png`
- `chart4_bleu_topk.png`
- `chart5_rouge_l_topk.png`
- `chart6_ground_truth_evidence_types.png`
- `chart7_ground_truth_label_distribution.png`

Tabel aktif:

- `table1_precision_trend.csv`
- `table2_recall_trend.csv`
- `table3_mrr_trend.csv`
- `table4_bleu_topk.csv`
- `table5_rouge_l_topk.csv`
- `table6_ground_truth_evidence_types.csv`
- `table7_ground_truth_label_distribution.csv`
- `table_6_1_retrieval_summary.csv`
- `table_6_1_retrieval_summary.xlsx`
- `table_6_2_generation_summary.csv`
- `table_6_2_generation_summary.xlsx`
- `table_average_metrics_top5_top10.csv`
- `table_average_metrics_top5_top10.xlsx`
- `table_average_metrics_top5_top10.md`
- `audit_average_metrics_top5_top10.csv`

### `results/visualizations/bab6/`

Artefak tambahan Bab 6:

- `gambar_6_x_ringkasan_retrieval_top5_top10.png`
- `tabel_6_1_ringkasan_retrieval_top5_top10.csv`
- `tabel_6_1_ringkasan_retrieval_top5_top10.xlsx`

### `results/archive/`

Berisi hasil historis.

Jangan dipakai sebagai sumber angka final kecuali sedang audit historis.

Subfolder penting:

- `RTX3090_strict_archived/`: strict RTX 3090, valid tapi tidak dipakai untuk final binary.
- `former_final_generation/`: run lama RTX 5060 Ti, tidak valid untuk final karena bug metrik.
- `No GPU Info/`: baseline retrieval lama.
- `RTX 3090 24GB/`, `RTX 4050 6GB/`, `RTX 5060 Ti 16GB/`: run historis.

## 8. Notebook Map

### `notebooks/eval_visualization.ipynb`

Notebook visualisasi aktif.

Urutan chart saat ini:

1. Precision@k trend.
2. Recall@k trend.
3. MRR trend.
4. BLEU per method dan Top-k.
5. ROUGE-L per method dan Top-k.
6. Distribusi evidence type QA gold.
7. Distribusi label relevansi retrieval.

Cell tambahan:

- Tabel 6.1 Retrieval Summary Top-5 sampai Top-10.
- Tabel 6.2 Generation Summary Top-5 sampai Top-10.
- Tabel sintesis rerata lintas Top-k.

Jangan mengubah chart/table lain jika hanya diminta memperbaiki satu chart tertentu.

### `notebooks/rag_inference.ipynb`

Notebook inference manual.

Gunakan untuk eksplorasi, bukan sebagai entry point produksi.

## 9. Command Map

Run Streamlit annotation app:

```bash
streamlit run src/streamlit/app.py --server.port 8502
```

Run Streamlit RAG chat:

```bash
streamlit run src/streamlit/rag_chat.py --server.port 8501
```

Run test metrik:

```bash
python -m pytest tests/test_evaluation.py -v
```

Build kandidat anotasi:

```bash
python scripts/build_candidates_v3.py
```

Convert label ke binary ground truth JSON:

```bash
python scripts/convert_ground_truth_to_json.py --output data/ground_truth/qa_pairs_binary.json --relevance_threshold 1
```

Load embeddings ke ChromaDB:

```bash
python scripts/load_embeddings_to_chroma.py
```

Run retrieval evaluation:

```bash
python scripts/run_retrieval_eval.py --gt data/ground_truth/qa_pairs_binary.json --output results/retrieval_eval.csv --top_k 5
```

Run generation evaluation:

```bash
python scripts/run_generation_eval.py
python scripts/run_generation_eval.py --resume
python scripts/run_generation_eval.py --methods element_based --top_k 5
```

## 10. Dependency Boundaries

Core runtime:

- `torch`
- `transformers`
- `sentence-transformers`
- `chromadb`
- `llama-cpp-python`
- `numpy`

PDF/chunking:

- `PyMuPDF`
- `unstructured[pdf]`
- `pytesseract`
- `langchain-text-splitters`
- `nltk`
- `scikit-learn`

Evaluation:

- `sacrebleu`
- `rouge-score`

Data/IO:

- `pandas`
- `openpyxl`

UI:

- `streamlit`

Dev:

- `pytest`
- `black`
- `flake8`
- `mypy`
- `jupyter`

Do not reintroduce:

- `ragas`
- `rank-eval`
- `rapidfuzz`

## 11. Active vs Legacy Rules

Use these as active sources:

- `src/`
- `scripts/`
- `data/ground_truth/qa_pairs_binary.json`
- `data/ground_truth/retrieval_labels_final.csv/.xlsx`
- `data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx`
- `results/final/generation/`
- `results/final/figures/`
- `notebooks/eval_visualization.ipynb`

Treat these as historical/archive unless explicitly asked:

- `results/archive/`
- `data/ground_truth/archive/`
- `backup/`
- old notebooks not present in current tracked tree
- old chart/table filenames removed from active sequence

Do not use for final analysis:

- strict archived result as final score
- former RTX 5060 Ti final generation result
- top-1 sampai top-4 when the request explicitly says Top-5 sampai Top-10

## 12. Common Safe Change Points

If changing retrieval behavior:

- Check `src/rag/pipeline.py`.
- Check `src/chroma/query.py`.
- Check `src/evaluation/metrics.py`.
- Validate against `tests/test_evaluation.py`.

If changing generation behavior:

- Check `src/rag/generator.py`.
- Check `src/rag/pipeline.py`.
- Check `scripts/run_generation_eval.py`.
- Check `src/streamlit/rag_chat.py`.

If changing metrics:

- Change `src/evaluation/metrics.py`.
- Update `tests/test_evaluation.py`.
- Do not silently change historical result interpretation.

If changing ground truth:

- Update `data/ground_truth/retrieval_labels_final.csv/.xlsx`.
- Regenerate `qa_pairs_binary.json` with `scripts/convert_ground_truth_to_json.py`.
- Document any schema decision in `docs/PROJECT_CONTEXT.txt`.

If changing visualization:

- Prefer editing only `notebooks/eval_visualization.ipynb`.
- Keep output filenames in `results/final/figures/` aligned with chart/table numbering.
- Avoid touching unrelated chart cells.

## 13. Known Gotchas

- Query ID in QA gold is not sequential. This is normal.
- Missing retrieval metrics can occur when a query has no relevant GT chunk for a method.
- `DEFAULT_TOP_K` in chat app is 8, while many report tables compare Top-5 to Top-10.
- `max_tokens` is intentionally high at 16384.
- `components.html` should not be used in Streamlit; use `st.iframe(height=1)` pattern already present.
- `suggested_label` from candidate builder is not final label.
- `build_candidates_v3.py` is oracle pooling for annotation, not ChromaDB retrieval.
- `enrich_table_chunk_texts()` does not mutate `data/chunked/`.
- OCR/table corruption in element-based chunks can be extraction limitation.
- `results/final/generation/` is flat; there is no active strict/lenient subfolder.
- File names containing old "lenient" in archive/final context may be legacy naming, not active strict-vs-lenient logic.

## 14. Minimal Mental Model

For most tasks, think of the repo like this:

```text
Data:
  PDF -> cleaned text -> chunks -> embeddings -> ChromaDB

Runtime:
  selected QA/query -> embed -> retrieve -> generate -> metrics

Evaluation:
  qa_pairs_binary.json + final generation CSV -> charts/tables Bab 6

Final outputs:
  results/final/generation/
  results/final/figures/
  results/visualizations/bab6/
```

If a file is in `results/archive/` or `data/ground_truth/archive/`, do not use it as the final source unless the task is explicitly historical/audit-oriented.
