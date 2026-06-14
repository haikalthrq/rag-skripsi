# Tabel Bab 6 - Evaluasi Top-1 sampai Top-20

Sumber data: `results/final/generation/*.csv`, `data/ground_truth/qa_pairs_binary.json`, dan `data/ground_truth/retrieval_labels_final.csv`. F1@k pada tabel ini dihitung ulang dari Precision@k dan Recall@k agregat.

## Rerata Lintas Top-1 sampai Top-20

| Metode | Rerata Precision@k | Rerata Recall@k | Rerata MRR | Rerata F1@k | Rerata BLEU | Rerata ROUGE-L |
| --- | --- | --- | --- | --- | --- | --- |
| Element-Based | 0.1079 | 0.5112 | 0.3665 | 0.1617 | 0.1177 | 0.6948 |
| MaxMin Semantic | 0.1002 | 0.5303 | 0.3474 | 0.1519 | 0.1149 | 0.6522 |
| Recursive | 0.1239 | 0.4273 | 0.4327 | 0.1688 | 0.1403 | 0.6851 |

## Metode dengan Nilai Tertinggi per Metrik

| Metrik | Unggul Rerata Top-1..20 | Nilai Rerata | Menang Element | Menang MaxMin | Menang Recursive |
| --- | --- | --- | --- | --- | --- |
| precision_at_k | Recursive | 0.1239 | 2 | 0 | 18 |
| recall_at_k | MaxMin Semantic | 0.5303 | 4 | 16 | 0 |
| mrr | Recursive | 0.4327 | 0 | 0 | 20 |
| f1_at_k | Recursive | 0.1688 | 8 | 0 | 12 |
| bleu | Recursive | 0.1403 | 1 | 0 | 19 |
| rouge_l | Element-Based | 0.6948 | 16 | 0 | 4 |

## Titik Ringkas Top-1, Top-5, Top-10, Top-15, dan Top-20

| Top-k | Metode | Precision@k | Recall@k | MRR | F1@k | BLEU | ROUGE-L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Element-Based | 0.2857 | 0.1994 | 0.2857 | 0.2349 | 0.1542 | 0.6270 |
| 1 | MaxMin Semantic | 0.2069 | 0.1264 | 0.2069 | 0.1570 | 0.1252 | 0.6013 |
| 1 | Recursive | 0.3448 | 0.1488 | 0.3448 | 0.2079 | 0.1426 | 0.6406 |
| 5 | Element-Based | 0.1286 | 0.3923 | 0.3542 | 0.1937 | 0.1178 | 0.7005 |
| 5 | MaxMin Semantic | 0.1310 | 0.4466 | 0.3488 | 0.2026 | 0.1052 | 0.6369 |
| 5 | Recursive | 0.1448 | 0.3523 | 0.4293 | 0.2053 | 0.1440 | 0.6838 |
| 10 | Element-Based | 0.0893 | 0.5292 | 0.3786 | 0.1528 | 0.1079 | 0.6828 |
| 10 | MaxMin Semantic | 0.0828 | 0.5443 | 0.3619 | 0.1437 | 0.1150 | 0.6638 |
| 10 | Recursive | 0.0966 | 0.4379 | 0.4408 | 0.1582 | 0.1334 | 0.6938 |
| 15 | Element-Based | 0.0762 | 0.6286 | 0.3848 | 0.1359 | 0.1191 | 0.7009 |
| 15 | MaxMin Semantic | 0.0667 | 0.6592 | 0.3670 | 0.1211 | 0.1159 | 0.6625 |
| 15 | Recursive | 0.0759 | 0.4862 | 0.4439 | 0.1313 | 0.1365 | 0.6938 |
| 20 | Element-Based | 0.0589 | 0.6405 | 0.3848 | 0.1079 | 0.1148 | 0.7063 |
| 20 | MaxMin Semantic | 0.0552 | 0.6948 | 0.3670 | 0.1022 | 0.1151 | 0.6673 |
| 20 | Recursive | 0.0690 | 0.5851 | 0.4515 | 0.1234 | 0.1309 | 0.6893 |
