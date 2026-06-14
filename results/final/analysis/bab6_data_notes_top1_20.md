# Catatan Analisis Data Bab 6 - Top-1 sampai Top-20

## A. Ringkasan Angka Utama

Berdasarkan agregasi dari `results/final/generation/*.csv`, evaluasi memakai 20 nilai top-k, tiga metode chunking, dan 30 query per kombinasi method-top-k. Ground truth QA yang dipakai adalah `data/ground_truth/qa_pairs_binary.json` dengan 30 query, sedangkan label retrieval berasal dari `data/ground_truth/retrieval_labels_final.csv` dengan 416 baris anotasi.

| Metode | Rerata Precision@k | Rerata Recall@k | Rerata MRR | Rerata F1@k | Rerata BLEU | Rerata ROUGE-L |
| --- | --- | --- | --- | --- | --- | --- |
| Element-Based | 0.1079 | 0.5112 | 0.3665 | 0.1617 | 0.1177 | 0.6948 |
| MaxMin Semantic | 0.1002 | 0.5303 | 0.3474 | 0.1519 | 0.1149 | 0.6522 |
| Recursive | 0.1239 | 0.4273 | 0.4327 | 0.1688 | 0.1403 | 0.6851 |

## B. Tabel Metode Unggul per Metrik

| Metrik | Unggul Rerata Top-1..20 | Nilai Rerata | Menang Element | Menang MaxMin | Menang Recursive |
| --- | --- | --- | --- | --- | --- |
| precision_at_k | Recursive | 0.1239 | 2 | 0 | 18 |
| recall_at_k | MaxMin Semantic | 0.5303 | 4 | 16 | 0 |
| mrr | Recursive | 0.4327 | 0 | 0 | 20 |
| f1_at_k | Recursive | 0.1688 | 8 | 0 | 12 |
| bleu | Recursive | 0.1403 | 1 | 0 | 19 |
| rouge_l | Element-Based | 0.6948 | 16 | 0 | 4 |

Catatan kritis: istilah ?unggul? di sini berarti nilai rerata tertinggi pada konfigurasi eksperimen ini. Ini bukan klaim metode terbaik secara umum, karena tidak ada uji statistik, tidak ada variasi dataset, dan jumlah query terbatas pada 30 QA.

## C. Analisis Karakter Tiap Metode

Element-Based Chunking menunjukkan rerata ROUGE-L tertinggi, yaitu 0.6948, dan menang ROUGE-L pada 16 dari 20 titik top-k. Namun rerata Precision@k Element-Based hanya 0.1079 dan rerata MRR 0.3665. Artinya, pada data ini Element-Based cenderung menghasilkan jawaban dengan cakupan urutan referensi yang baik, tetapi tidak dominan pada ketepatan retrieval dan posisi chunk relevan pertama.

Max-Min Semantic Chunking menunjukkan rerata Recall@k tertinggi, yaitu 0.5303, dan menang Recall@k pada 16 dari 20 titik top-k. Ini menunjukkan kecenderungan Max-Min lebih mampu menjangkau chunk relevan ketika k diperbesar. Kelemahannya, rerata Precision@k 0.1002 dan MRR 0.3474 lebih rendah dibanding Recursive, sehingga jangkauan relevansi tidak otomatis berarti ranking awal lebih tajam.

Recursive Chunking menunjukkan rerata Precision@k 0.1239, MRR 0.4327, F1@k 0.1688, dan BLEU 0.1403 tertinggi pada ringkasan Top-1 sampai Top-20. Recursive menang MRR pada 20 dari 20 titik top-k. Data ini mendukung interpretasi bahwa Recursive lebih konsisten menempatkan chunk relevan pertama di ranking atas pada dataset ini. Klaim lebih jauh, misalnya ?metode paling efektif?, belum bisa dipertanggungjawabkan tanpa uji statistik dan dataset pembanding.

## D. Analisis Retrieval

Precision@k tertinggi secara rerata diperoleh Recursive (0.1239). Recall@k tertinggi diperoleh Max-Min Semantic (0.5303). MRR tertinggi diperoleh Recursive (0.4327). F1@k tertinggi diperoleh Recursive (0.1688). Pola ini menunjukkan trade-off: Max-Min lebih kuat pada cakupan chunk relevan, sedangkan Recursive lebih kuat pada ketepatan relatif dan ranking awal.

Data titik Top-20 memperlihatkan Recall@k Max-Min mencapai 0.6950, Element-Based 0.6400, dan Recursive 0.5850. Sebaliknya, pada Top-20 Precision@k turun menjadi 0.0550 untuk Max-Min, 0.0590 untuk Element-Based, dan 0.0690 untuk Recursive. Ini wajar secara metrik: ketika k bertambah, lebih banyak chunk masuk daftar sehingga recall cenderung naik, tetapi precision dapat turun karena denominator bertambah.

## E. Analisis Generation

BLEU tertinggi secara rerata diperoleh Recursive (0.1403), sedangkan ROUGE-L tertinggi diperoleh Element-Based (0.6948). Ini berarti kecenderungan generation tidak identik untuk semua metrik. BLEU lebih sensitif pada kecocokan n-gram leksikal, sedangkan ROUGE-L mengukur kemiripan urutan subsekuensi terpanjang. Karena itu, hasil BLEU dan ROUGE-L harus dibaca berdampingan, bukan dipaksa menjadi satu klaim tunggal.

Contoh query pada `results/final/analysis/top1_20_query_examples.csv` menunjukkan batasan ini. Pada Q005 Top-1, Element-Based memiliki Precision@k=0 dan MRR=0, tetapi BLEU=0.6998 dan ROUGE-L=0.8667 pada CSV Top-1. Ini memperlihatkan bahwa metrik generation dapat tinggi walaupun label retrieval Top-1 tidak menganggap chunk terambil sebagai relevan. Kesimpulan akademiknya: evaluasi retrieval dan generation mengukur aspek berbeda dan tidak boleh dipertukarkan.

## F. Implikasi terhadap Rumusan Masalah

Untuk rumusan masalah tentang pengaruh metode chunking terhadap retrieval, data mendukung bahwa perbedaan metode menghasilkan pola metrik berbeda: Max-Min unggul pada rerata Recall@k (0.5303), sedangkan Recursive unggul pada rerata Precision@k (0.1239), MRR (0.4327), dan F1@k (0.1688). Ini cukup untuk menyatakan adanya perbedaan kecenderungan kinerja pada konfigurasi eksperimen ini, bukan bukti generalisasi universal.

Untuk rumusan masalah tentang pengaruh metode chunking terhadap generation, Recursive unggul pada rerata BLEU (0.1403), sedangkan Element-Based unggul pada rerata ROUGE-L (0.6948). Jadi jawaban yang bertanggung jawab bukan ?satu metode menang mutlak?, melainkan ?metode berbeda unggul pada aspek generation yang berbeda?.

Untuk rumusan masalah perbandingan akhir, Recursive memiliki profil paling seimbang pada metrik retrieval gabungan karena unggul pada Precision@k, MRR, dan F1@k. Namun Max-Min tetap relevan bila prioritas analisis adalah coverage chunk relevan, dan Element-Based tetap kuat pada ROUGE-L. Pernyataan ini hanya berlaku untuk data `results/final/generation/*.csv` dan ground truth yang disebutkan di audit.

## G. Catatan Keterbatasan

Dataset evaluasi hanya berisi 30 query. Tidak ada uji statistik, sehingga pembahasan tidak boleh memakai klaim inferensial. Top-11 sampai Top-20 tersedia sebagai CSV final, tetapi interpretasi top-k besar harus hati-hati karena bertambahnya k secara mekanis dapat menaikkan recall dan menurunkan precision. Selain itu, contoh Q005 memperlihatkan bahwa label retrieval dan kualitas jawaban dapat berbeda arah; ini menuntut pembahasan Bab 6 yang memisahkan analisis retrieval dan generation.
