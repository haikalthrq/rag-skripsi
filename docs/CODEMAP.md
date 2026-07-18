# CODEMAP - rag-skripsi

Terakhir diperbarui: 2026-06-16 WIB

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
- Mengevaluasi retrieval dengan `Precision@k`, `Recall@k`, `MRR`, dan `F1@k`.
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
  -> notebooks/eval_visualization.ipynb atau notebooks/eval_analysis_top1_10.ipynb
  -> results/final/figures/* dan results/final/analysis10/*
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

- `results/final/generation/`: hasil evaluasi final Top-1 sampai Top-20.
- `results/final/analysis10/`: artefak analisis utama Bab 6 Top-1 sampai Top-10.
- `results/final/analysis/`: artefak audit/validasi tambahan Top-1 sampai Top-20.
- `results/final/figures/`: chart dan tabel visualisasi aktif Top-1 sampai Top-10.
- `results/final/figures20/`: chart dan tabel visualisasi tambahan Top-1 sampai Top-20.
- `results/visualizations/bab6/`: artefak tambahan Bab 6.
- `results/archive/`: hasil lama, strict lama, run GPU lama, dan baseline historis.
- `results/chat_history/`: riwayat chat Streamlit persisten.

### `notebooks/`

Notebook analisis dan visualisasi.

- `eval_visualization.ipynb`: notebook visualisasi aktif Top-1 sampai Top-10.
- `eval_analysis_top1_10.ipynb`: notebook analisis utama Bab 6 Top-1 sampai Top-10.
- `eval_visualization_top1-20.ipynb`: notebook visualisasi tambahan Top-1 sampai Top-20.
- `eval_analysis_top1_20.ipynb`: notebook audit analisis tambahan Top-1 sampai Top-20.
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
- F1@k
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
- `compute_f1_at_k(...)`
- `compute_bleu(...)`
- `compute_rouge(...)`

Tanggung jawab:

- Implementasi metrik final.
- Retrieval metrics berbasis chunk ID.
- F1@k dihitung dari Precision@k dan Recall@k.
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
- Menampilkan BLEU, ROUGE-L, Precision@k, Recall@k, MRR, dan F1@k.
- Menyimpan chat history ke `results/chat_history/chat_history.jsonl`.
- Mendukung batch eval Top-1 sampai Top-20 dengan skip otomatis jika file output sudah ada.

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
- Mengikuti skema binary dan metric helper yang sama dengan `rag_chat.py`.

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

- `results/final/generation/eval_*_top{1..20}.csv`

Catatan:

- Script sekarang menulis schema CSV yang sama dengan batch eval `rag_chat.py`:
  `query_id`, `method`, `question`, `gold_answer`, `generated_answer`,
  `precision_at_k`, `recall_at_k`, `mrr`, `f1_at_k`, `bleu`, `rouge_l_recall`,
  `error`, `hardware_info`.
- Output final Top-1 sampai Top-20 saat ini sudah valid dan tidak perlu direrun kecuali ada perubahan metodologis.

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
- `f1_at_k`
- `bleu`
- `rouge_l_recall`
- `error`
- `hardware_info`

Catatan:

- Ada 20 file final Top-1 sampai Top-20.
- Top-1 sampai Top-10 menjadi basis utama Bab 6.
- Top-11 sampai Top-20 menjadi validasi tambahan/lampiran.
- Nama file yang mengandung kata lama seperti `lenient` adalah legacy naming; isi final tetap binary.

## 7. Results dan Artefak Aktif

### `results/final/generation/`

Berisi hasil definitif valid dalam folder flat.

Isi:

- 20 file CSV `eval_*_top{1..20}.csv`.
- Top-1 sampai Top-10 berasal dari run RTX 3090 2026-05-31.
- Top-11 sampai Top-20 berasal dari batch lanjut 2026-06-13 via `src/streamlit/rag_chat.py`.
- Skema semua file final adalah binary relevance.

### `results/final/analysis10/`

Artefak analisis utama Bab 6 Top-1 sampai Top-10.

File penting:

- `top1_10_metrics_by_k.csv`
- `top1_10_metric_summary_by_method.csv`
- `top1_10_metric_winners.csv`
- `top1_10_overall_average.csv`
- `top1_10_audit_notes.md`
- `bab6_tables_top1_10.md`
- `bab6_data_notes_top1_10.md`
- `figures/*.png`

Catatan:

- Ini sumber utama untuk narasi Bab 6.
- Chart MRR analysis10 memakai lowest y-axis `0.15`.

### `results/final/analysis/`

Artefak audit/validasi tambahan Top-1 sampai Top-20.

File penting:

- `top1_20_metrics_by_k.csv`
- `top1_20_metric_summary_by_method.csv`
- `top1_20_metric_winners.csv`
- `top1_20_query_examples.csv`
- `top1_20_audit_notes.md`
- `bab6_tables_top1_20.md`
- `bab6_data_notes_top1_20.md`
- `figures/*.png`

Catatan:

- Gunakan sebagai lampiran atau validasi tambahan, bukan basis utama Bab 6.

### `results/final/figures/`

Chart aktif Top-1 sampai Top-10:

- `chart1_precision_trend.png`
- `chart2_recall_trend.png`
- `chart3_mrr_trend.png`
- `chart4_f1_trend.png`
- `chart5_bleu_topk.png`
- `chart6_rouge_l_topk.png`
- `chart7_ground_truth_evidence_types.png`
- `chart8_ground_truth_label_distribution.png`

Tabel aktif Top-1 sampai Top-10:

- `table1_precision_trend.csv`
- `table2_recall_trend.csv`
- `table3_mrr_trend.csv`
- `table4_f1_trend.csv`
- `table5_bleu_topk.csv`
- `table6_rouge_l_topk.csv`
- `table7_ground_truth_evidence_types.csv`
- `table8_ground_truth_label_distribution.csv`
- `table_6_1_retrieval_summary.csv`
- `table_6_1_retrieval_summary.xlsx`
- `table_6_2_generation_summary.csv`
- `table_6_2_generation_summary.xlsx`

### `results/final/figures20/`

Visualisasi dan tabel tambahan Top-1 sampai Top-20.

Isi umum:

- `chart1_precision_trend.png` sampai `chart8_ground_truth_label_distribution.png`
- `table1_precision_trend.csv` sampai `table8_ground_truth_label_distribution.csv`
- `table_6_1_retrieval_summary.csv/.xlsx`
- `table_6_2_generation_summary.csv/.xlsx`

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

Notebook visualisasi aktif Top-1 sampai Top-10.

Urutan chart saat ini:

1. Precision@k trend.
2. Recall@k trend.
3. MRR trend.
4. F1@k trend.
5. BLEU per method dan Top-k.
6. ROUGE-L per method dan Top-k.
7. Distribusi evidence type QA gold.
8. Distribusi label relevansi retrieval.

Cell tambahan:

- Tabel 6.1 Retrieval Summary Top-1 sampai Top-10.
- Tabel 6.2 Generation Summary Top-1 sampai Top-10.
- Tabel dan chart output ke `results/final/figures/`.

Jangan mengubah chart/table lain jika hanya diminta memperbaiki satu chart tertentu.

### `notebooks/eval_analysis_top1_10.ipynb`

Notebook analisis utama Bab 6.

Input:

- CSV final Top-1 sampai Top-10 dari `results/final/generation/`.

Output:

- `results/final/analysis10/top1_10_metrics_by_k.csv`
- `results/final/analysis10/top1_10_metric_summary_by_method.csv`
- `results/final/analysis10/top1_10_metric_winners.csv`
- `results/final/analysis10/top1_10_overall_average.csv`
- `results/final/analysis10/top1_10_audit_notes.md`
- `results/final/analysis10/bab6_tables_top1_10.md`
- `results/final/analysis10/bab6_data_notes_top1_10.md`
- `results/final/analysis10/figures/*.png`

Gunakan ini sebagai sumber utama narasi Bab 6.

### `notebooks/eval_visualization_top1-20.ipynb`

Notebook visualisasi tambahan Top-1 sampai Top-20.

Output:

- `results/final/figures20/`

Gunakan untuk validasi tambahan atau lampiran.

### `notebooks/eval_analysis_top1_20.ipynb`

Notebook audit analisis tambahan Top-1 sampai Top-20.

Output:

- `results/final/analysis/`
- `results/final/analysis/figures/`

Gunakan sebagai pembanding tambahan, bukan basis utama Bab 6.

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
- `results/final/analysis10/`
- `results/final/figures/`
- `notebooks/eval_visualization.ipynb`
- `notebooks/eval_analysis_top1_10.ipynb`

Use these as additional validation sources:

- `results/final/analysis/`
- `results/final/figures20/`
- `notebooks/eval_visualization_top1-20.ipynb`
- `notebooks/eval_analysis_top1_20.ipynb`

Treat these as historical/archive unless explicitly asked:

- `results/archive/`
- `data/ground_truth/archive/`
- `backup/`
- old notebooks not present in current tracked tree
- old chart/table filenames removed from active sequence

Do not use for final analysis:

- strict archived result as final score
- former RTX 5060 Ti final generation result
- Top-11 sampai Top-20 as the main Bab 6 basis; use it only as validation/lampiran

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

- Prefer editing only the relevant notebook:
  - `notebooks/eval_visualization.ipynb` for active Top-1 sampai Top-10 visualizations.
  - `notebooks/eval_analysis_top1_10.ipynb` for main Bab 6 analysis artifacts.
  - `notebooks/eval_visualization_top1-20.ipynb` or `notebooks/eval_analysis_top1_20.ipynb` for Top-1 sampai Top-20 validation artifacts.
- Keep output filenames in `results/final/figures/` aligned with chart/table numbering.
- Avoid touching unrelated chart cells.

## 13. Known Gotchas

- Query ID in QA gold is not sequential. This is normal.
- Missing retrieval metrics can occur when a query has no relevant GT chunk for a method.
- `DEFAULT_TOP_K` in chat app is 8, while Bab 6 active analysis compares Top-1 to Top-10.
- `max_tokens` is intentionally high at 16384.
- `components.html` should not be used in Streamlit; use `st.iframe(height=1)` pattern already present.
- `suggested_label` from candidate builder is not final label.
- `build_candidates_v3.py` is oracle pooling for annotation, not ChromaDB retrieval.
- `enrich_table_chunk_texts()` does not mutate `data/chunked/`.
- OCR/table corruption in element-based chunks can be extraction limitation.
- `results/final/generation/` is flat; there is no active strict/lenient subfolder.
- File names containing old "lenient" in archive/final context may be legacy naming, not active strict-vs-lenient logic.
- Top-1 sampai Top-10 is the main Bab 6 basis; Top-11 sampai Top-20 is validation/lampiran.
- `results/final/analysis10/` is the main analysis output; `results/final/analysis/` is the Top-20 audit output.

## 14. Minimal Mental Model

For most tasks, think of the repo like this:

```text
Data:
  PDF -> cleaned text -> chunks -> embeddings -> ChromaDB

Runtime:
  selected QA/query -> embed -> retrieve -> generate -> metrics

Evaluation:
  qa_pairs_binary.json + final generation CSV -> analysis10 + charts/tables Bab 6

Final outputs:
  results/final/generation/
  results/final/analysis10/
  results/final/figures/
  results/final/analysis/      (Top-20 validation)
  results/final/figures20/     (Top-20 validation)
  results/visualizations/bab6/
```

If a file is in `results/archive/` or `data/ground_truth/archive/`, do not use it as the final source unless the task is explicitly historical/audit-oriented.
