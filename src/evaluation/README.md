# Modul Evaluasi

Modul ini menyediakan Precision@k, Recall@k, MRR, F1@k, BLEU, ROUGE-L, dan
orkestrasi evaluasi tiga metode chunking.

## Backend Metrik

- BLEU utama memakai `sacrebleu`.
- ROUGE utama memakai `rouge-score`.
- Jika dependency tidak tersedia, kode memakai fallback Python sederhana.

Fallback bukan implementasi yang ekuivalen dengan library utama. Hasil dari
environment berbeda tidak boleh dibandingkan tanpa mencatat backend metrik yang
aktif.

## Aggregation

Denominator mean dapat berbeda per metrik:

- Retrieval metric hanya memasukkan query yang memiliki relevant chunk ID.
- BLEU dan ROUGE hanya memasukkan query yang berhasil menghasilkan jawaban.

Karena itu, simpan jumlah query yang benar-benar dievaluasi bersama nilai mean.

## Batasan API

- Parameter `mrr_at_k` pada `RAGEvaluator.evaluate_all()` belum digunakan.
- MRR saat ini memeriksa seluruh `retrieved_ids` yang diberikan evaluator.
- Standalone `scripts/run_generation_eval.py` belum menghasilkan `f1_at_k` dan
  hanya menerima Top-1 sampai Top-10.
- Output Streamlit dan standalone script belum memiliki schema identik.

## Test

Gunakan pytest agar seluruh test yang dapat ditemukan dijalankan:

```bash
pytest tests/test_evaluation.py
```

Eksekusi langsung `python tests/test_evaluation.py` memakai daftar class manual
dan tidak menjamin cakupan yang sama dengan pytest.
