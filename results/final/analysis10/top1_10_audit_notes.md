# Audit Analisis Top-1 sampai Top-10

## Sumber Data
- CSV final generation: `results/final/generation/`.
- File yang dibaca: `results/final/generation/eval_20260531_181551_full_top1.csv`, `results/final/generation/eval_20260531_182241_full_top2.csv`, `results/final/generation/eval_20260531_183044_full_top3.csv`, `results/final/generation/eval_20260531_183822_full_top4.csv`, `results/final/generation/eval_20260531_184708_full_top5.csv`, `results/final/generation/eval_20260531_185558_full_top6.csv`, `results/final/generation/eval_20260531_190431_full_top7.csv`, `results/final/generation/eval_20260531_191252_full_top8.csv`, `results/final/generation/eval_20260531_192113_full_top9.csv`, `results/final/generation/eval_20260531_192940_full_top10.csv`.
- Ground truth QA aktif: `data/ground_truth/qa_pairs_binary.json` (30 QA).
- Ground truth retrieval aktif: `data/ground_truth/retrieval_labels_final.csv` (416 anotasi relevansi chunk).

## Validasi Kelengkapan
- Rentang Top-k: Top-1 sampai Top-10.
- Jumlah baris agregat aktual: 30.
- Jumlah baris agregat ekspektasi: 30.
- Top-k lengkap 1 sampai 10: ya.
- Tiga metode lengkap: ya.
- Setiap kombinasi metode dan top-k berisi 30 QA: ya.

## Missing Value
- missing_precision_at_k: 40
- missing_recall_at_k: 40
- missing_mrr: 40
- missing_bleu: 0
- missing_rouge_l: 0

Missing retrieval metrics berulang muncul karena beberapa kombinasi query-method tidak memiliki ground truth retrieval relevan yang dapat dievaluasi pada CSV final. Nilai ini dilaporkan, bukan dihapus diam-diam.

## Catatan F1@k
- F1@k dihitung ulang dari rerata Precision@k dan Recall@k pada setiap kombinasi metode dan top-k.
- Kolom F1 lama pada CSV final tidak digunakan sebagai sumber utama perhitungan artefak ini.

## Catatan Top-11 sampai Top-20
- Bab 6 utama menggunakan Top-1 sampai Top-10.
- Top-11 sampai Top-20 diposisikan sebagai validasi tambahan/lampiran, bukan dasar utama pembahasan Bab 6.
- Pemenang rerata Top-1 sampai Top-10 konsisten dengan Top-1 sampai Top-20 untuk metrik: Precision@k, Recall@k, MRR, F1@k, BLEU, ROUGE-L.
