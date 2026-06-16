# Catatan Narasi Bab 6 - Top-1 sampai Top-10

## Ringkasan Angka Utama
Berdasarkan evaluasi Top-1 sampai Top-10, rerata metrik per metode adalah sebagai berikut:
| Metode           |   Precision@k |   Recall@k |   MRR |   F1@k |   BLEU |   ROUGE-L |
|:-----------------|--------------:|-----------:|------:|-------:|-------:|----------:|
| Element-Based    |         0.142 |      0.399 | 0.349 |  0.191 |  0.119 |     0.683 |
| Max-Min Semantic |         0.134 |      0.412 | 0.329 |  0.184 |  0.117 |     0.646 |
| Recursive        |         0.169 |      0.346 | 0.419 |  0.202 |  0.14  |     0.679 |

## Metode Unggul per Metrik
- Precision@k: Recursive dengan rerata 0.169; kemenangan per titik top-k: Element-Based: 0; Max-Min Semantic: 0; Recursive: 10.
- Recall@k: Max-Min Semantic dengan rerata 0.412; kemenangan per titik top-k: Element-Based: 2; Max-Min Semantic: 8; Recursive: 0.
- MRR: Recursive dengan rerata 0.419; kemenangan per titik top-k: Element-Based: 0; Max-Min Semantic: 0; Recursive: 10.
- F1@k: Recursive dengan rerata 0.202; kemenangan per titik top-k: Element-Based: 2; Max-Min Semantic: 0; Recursive: 8.
- BLEU: Recursive dengan rerata 0.140; kemenangan per titik top-k: Element-Based: 1; Max-Min Semantic: 0; Recursive: 9.
- ROUGE-L: Element-Based dengan rerata 0.683; kemenangan per titik top-k: Element-Based: 6; Max-Min Semantic: 0; Recursive: 4.

## Karakter Element-Based Chunking
Element-Based menunjukkan kecenderungan kuat pada ROUGE-L dengan rerata 0.683. Pada konfigurasi eksperimen ini, metode ini juga memiliki rerata Recall@k 0.399, tetapi tidak menjadi yang tertinggi dibanding Max-Min Semantic. Interpretasi yang dapat dipertanggungjawabkan: Element-Based cenderung menghasilkan jawaban dengan urutan informasi yang lebih dekat ke referensi, tetapi bukan yang paling kuat pada ketepatan retrieval atau posisi chunk relevan pertama.

## Karakter Max-Min Semantic Chunking
Max-Min Semantic menunjukkan kecenderungan unggul pada Recall@k dengan rerata 0.412. Ini berarti pada Top-1 sampai Top-10 metode ini lebih mampu menjangkau chunk relevan. Namun rerata Precision@k 0.134, MRR 0.329, dan F1@k 0.184 tidak menjadi yang tertinggi. Jadi klaim yang aman: metode ini cenderung kuat pada cakupan retrieval, bukan pada ketepatan atau ranking awal.

## Karakter Recursive Chunking
Recursive menunjukkan kecenderungan unggul pada Precision@k (0.169), MRR (0.419), F1@k (0.202), dan BLEU (0.140). Pada konfigurasi eksperimen ini, karakter utamanya adalah lebih presisi dan lebih stabil menempatkan chunk relevan awal, tetapi Recall@k dan ROUGE-L tidak menjadi yang tertinggi.

## Analisis Retrieval
Precision@k, MRR, dan F1@k lebih condong ke Recursive, sedangkan Recall@k lebih condong ke Max-Min Semantic. Ini menunjukkan trade-off yang jelas: Max-Min Semantic menjangkau lebih banyak chunk relevan, sementara Recursive lebih kuat pada ketepatan dan posisi relevansi awal. Klaim ini bersifat deskriptif karena tidak ada uji statistik.

## Analisis Generation
BLEU lebih tinggi pada Recursive, sedangkan ROUGE-L lebih tinggi pada Element-Based. Karena BLEU dan ROUGE-L mengukur aspek berbeda, hasil ini tidak boleh disederhanakan menjadi satu metode paling baik secara mutlak. Data hanya mendukung bahwa Recursive lebih kuat pada kemiripan leksikal, sedangkan Element-Based lebih kuat pada cakupan urutan informasi referensi.

## Implikasi terhadap Rumusan Masalah
- Rumusan masalah 1: Perbedaan metode chunking menghasilkan karakter retrieval yang berbeda pada Top-1 sampai Top-10.
- Rumusan masalah 2: Pengaruh terhadap generation tidak identik dengan retrieval; BLEU dan ROUGE-L memberi sinyal yang berbeda.
- Rumusan masalah 3: Perbandingan akhir perlu menekankan trade-off, bukan menyatakan satu metode terbaik mutlak.

## Catatan Keterbatasan
- Dataset evaluasi berisi 30 QA, sehingga generalisasi ke korpus lain belum bisa dipastikan.
- Tidak ada uji statistik; klaim inferensial tidak dibuat.
- Missing retrieval metrics dilaporkan dalam audit dan tidak dihapus diam-diam.
- Top-11 sampai Top-20 hanya validasi tambahan/lampiran; basis utama Bab 6 adalah Top-1 sampai Top-10.

## File Dasar Analisis
- `results/final/analysis10/top1_10_metrics_by_k.csv`
- `results/final/analysis10/top1_10_metric_summary_by_method.csv`
- `results/final/analysis10/top1_10_metric_winners.csv`
- `results/final/analysis10/top1_10_audit_notes.md`
