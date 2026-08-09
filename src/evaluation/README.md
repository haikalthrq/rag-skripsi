# Modul Evaluasi

Modul ini menyediakan metrik per-query dan orkestrasi evaluasi tiga metode
chunking: `element_based`, `maxmin_semantic`, dan `recursive`.

## Metrik

Metrik retrieval berbasis pencocokan chunk ID:

- Precision@k: jumlah ID relevan unik pada top-k dibagi `k`.
- Recall@k: jumlah ID relevan unik pada top-k dibagi jumlah `relevant_ids`.
- MRR: reciprocal rank dari ID relevan pertama dalam `retrieved_ids`.
- F1@k: harmonic mean dari Precision@k dan Recall@k.

Metrik generation membandingkan jawaban dengan reference answer:

- BLEU memakai `sacrebleu.corpus_bleu` dan dinormalisasi ke rentang 0-1.
- ROUGE-L memakai `rouge-score`; evaluasi utama menggunakan mode recall.

Jika dependency BLEU atau ROUGE tidak tersedia, `metrics.py` memakai fallback
Python sederhana. Fallback tidak ekuivalen dengan library utama, sehingga
backend metrik harus dicatat ketika membandingkan hasil antar-environment.
Schema CSV saat ini belum menyimpan nama backend metrik; pastikan dependency
`sacrebleu` dan `rouge-score` terpasang dan catat environment secara terpisah.

## Jalur Evaluasi

### `RAGEvaluator`

`RAGEvaluator.evaluate_method()` menghitung Precision@k, Recall@k, MRR, dan
F1@k jika query memiliki relevant chunk ID untuk metode tersebut. Jika generator
diberikan dan menghasilkan jawaban non-kosong, evaluator juga menghitung BLEU
dan ROUGE-L recall.

`RAGEvaluator.evaluate_all()` menjalankan evaluasi untuk semua atau sebagian
metode. Parameter `mrr_at_k` saat ini diterima tetapi tidak digunakan; MRR
memeriksa seluruh `retrieved_ids` yang diberikan, yang pada evaluator ini adalah
hasil retrieval sebanyak `top_k`.

### Script Standalone

- `scripts/run_retrieval_eval.py` menghasilkan Precision@k, Recall@k, dan MRR.
- `scripts/run_generation_eval.py` menghasilkan Precision@k, Recall@k, MRR,
  F1@k, BLEU, dan ROUGE-L recall untuk setiap method/top-k.
- Generation evaluation juga mencatat `retrieval_seconds`,
  `generation_seconds`, dan `total_response_seconds`. CUDA disinkronkan sebelum
  dan sesudah bagian yang diukur. Query embedding biasanya diprekomputasi dan
  tidak termasuk retrieval latency.
- Script generation menerima rentang Top-1 sampai Top-10 dan menulis satu CSV
  per nilai top-k.

## Aggregation

Semua nilai agregat adalah macro mean atas skor per-query, bukan micro metric
dari hit gabungan.

Pada `RAGEvaluator`, setiap metrik dirata-ratakan secara independen dari query
yang memiliki key metrik tersebut:

- Retrieval metric dan F1@k hanya memasukkan query dengan relevant chunk ID.
- BLEU dan ROUGE hanya memasukkan query dengan jawaban generation non-kosong.

Pada summary `run_generation_eval.py`, baris dikelompokkan berdasarkan metode
dan top-k. Setiap mean memakai nilai yang dapat dikonversi ke angka; `None`,
`N/A`, dan nilai non-numerik dikeluarkan secara independen untuk setiap metrik.
Akibatnya denominator Precision@k, BLEU, latency, dan metrik lain dapat berbeda.
Summary menyertakan `n_queries`, `n_success`, `n_retrieval_evaluated`, dan
`n_timed`. `n_retrieval_evaluated` adalah jumlah nilai Precision@k numerik,
sedangkan `n_timed` adalah jumlah nilai total response latency numerik. Counter
tersebut bukan denominator universal untuk semua kolom mean.

Untuk ketiga latency, summary generation melaporkan mean, median, dan sample
standard deviation. Standard deviation ditetapkan `0.0` jika hanya ada satu
nilai timing.

Summary `run_retrieval_eval.py` juga mengecualikan query tanpa relevant chunk ID
dan query yang gagal menghasilkan skor dari mean retrieval. Kolom
`n_queries_evaluated`, `missing_retrieval_gt`, dan `error_count` mencatat
cakupannya.

## Test

```bash
pytest tests/test_evaluation.py
```

Gunakan pytest agar semua test yang dapat ditemukan dijalankan. Eksekusi
langsung `python tests/test_evaluation.py` memakai daftar class manual dan dapat
memiliki cakupan berbeda.
