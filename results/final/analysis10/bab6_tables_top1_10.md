# Tabel Siap Tempel Bab 6 - Top-1 sampai Top-10



## Ringkasan Rerata Metrik per Metode

| Metode           |   Precision@k |   Recall@k |   MRR |   F1@k |   BLEU |   ROUGE-L |
|:-----------------|--------------:|-----------:|------:|-------:|-------:|----------:|
| Element-Based    |         0.142 |      0.399 | 0.349 |  0.191 |  0.119 |     0.683 |
| Max-Min Semantic |         0.134 |      0.412 | 0.329 |  0.184 |  0.117 |     0.646 |
| Recursive        |         0.169 |      0.346 | 0.419 |  0.202 |  0.14  |     0.679 |



## Metode Unggul per Metrik

| Metrik      | Metode Unggul Berdasarkan Rerata   |   Rerata Tertinggi | Jumlah Kemenangan pada 10 Top-k                      |
|:------------|:-----------------------------------|-------------------:|:-----------------------------------------------------|
| Precision@k | Recursive                          |              0.169 | Element-Based: 0; Max-Min Semantic: 0; Recursive: 10 |
| Recall@k    | Max-Min Semantic                   |              0.412 | Element-Based: 2; Max-Min Semantic: 8; Recursive: 0  |
| MRR         | Recursive                          |              0.419 | Element-Based: 0; Max-Min Semantic: 0; Recursive: 10 |
| F1@k        | Recursive                          |              0.202 | Element-Based: 2; Max-Min Semantic: 0; Recursive: 8  |
| BLEU        | Recursive                          |              0.14  | Element-Based: 1; Max-Min Semantic: 0; Recursive: 9  |
| ROUGE-L     | Element-Based                      |              0.683 | Element-Based: 6; Max-Min Semantic: 0; Recursive: 4  |



## Overall Average Deskriptif

| Metode           |   Overall Average |
|:-----------------|------------------:|
| Element-Based    |             0.314 |
| Max-Min Semantic |             0.304 |
| Recursive        |             0.326 |



## Nilai per Top-k dan Metode

|   Top-k | Metode           |   Precision@k |   Recall@k |   MRR |   F1@k |   BLEU |   ROUGE-L |
|--------:|:-----------------|--------------:|-----------:|------:|-------:|-------:|----------:|
|       1 | Element-Based    |         0.286 |      0.199 | 0.286 |  0.235 |  0.154 |     0.627 |
|       1 | Max-Min Semantic |         0.207 |      0.126 | 0.207 |  0.157 |  0.125 |     0.601 |
|       1 | Recursive        |         0.345 |      0.149 | 0.345 |  0.208 |  0.143 |     0.641 |
|       2 | Element-Based    |         0.179 |      0.244 | 0.304 |  0.206 |  0.13  |     0.692 |
|       2 | Max-Min Semantic |         0.207 |      0.267 | 0.293 |  0.233 |  0.122 |     0.624 |
|       2 | Recursive        |         0.276 |      0.26  | 0.397 |  0.268 |  0.142 |     0.673 |
|       3 | Element-Based    |         0.143 |      0.298 | 0.327 |  0.193 |  0.11  |     0.675 |
|       3 | Max-Min Semantic |         0.161 |      0.319 | 0.316 |  0.214 |  0.12  |     0.658 |
|       3 | Recursive        |         0.184 |      0.26  | 0.397 |  0.216 |  0.145 |     0.702 |
|       4 | Element-Based    |         0.152 |      0.374 | 0.354 |  0.216 |  0.113 |     0.664 |
|       4 | Max-Min Semantic |         0.147 |      0.395 | 0.342 |  0.214 |  0.118 |     0.661 |
|       4 | Recursive        |         0.164 |      0.329 | 0.422 |  0.219 |  0.145 |     0.678 |
|       5 | Element-Based    |         0.129 |      0.392 | 0.354 |  0.194 |  0.118 |     0.7   |
|       5 | Max-Min Semantic |         0.131 |      0.447 | 0.349 |  0.203 |  0.105 |     0.637 |
|       5 | Recursive        |         0.145 |      0.352 | 0.429 |  0.205 |  0.144 |     0.684 |
|       6 | Element-Based    |         0.125 |      0.452 | 0.366 |  0.196 |  0.102 |     0.693 |
|       6 | Max-Min Semantic |         0.121 |      0.493 | 0.355 |  0.194 |  0.113 |     0.646 |
|       6 | Recursive        |         0.144 |      0.414 | 0.441 |  0.213 |  0.139 |     0.682 |
|       7 | Element-Based    |         0.112 |      0.47  | 0.366 |  0.181 |  0.124 |     0.705 |
|       7 | Max-Min Semantic |         0.103 |      0.493 | 0.355 |  0.171 |  0.115 |     0.649 |
|       7 | Recursive        |         0.128 |      0.421 | 0.441 |  0.196 |  0.133 |     0.673 |
|       8 | Element-Based    |         0.107 |      0.517 | 0.375 |  0.178 |  0.116 |     0.699 |
|       8 | Max-Min Semantic |         0.095 |      0.51  | 0.355 |  0.16  |  0.123 |     0.657 |
|       8 | Recursive        |         0.112 |      0.421 | 0.441 |  0.177 |  0.132 |     0.686 |
|       9 | Element-Based    |         0.095 |      0.517 | 0.375 |  0.161 |  0.114 |     0.693 |
|       9 | Max-Min Semantic |         0.088 |      0.527 | 0.358 |  0.151 |  0.112 |     0.666 |
|       9 | Recursive        |         0.1   |      0.421 | 0.441 |  0.161 |  0.148 |     0.675 |
|      10 | Element-Based    |         0.089 |      0.529 | 0.379 |  0.153 |  0.108 |     0.683 |
|      10 | Max-Min Semantic |         0.083 |      0.544 | 0.362 |  0.144 |  0.115 |     0.664 |
|      10 | Recursive        |         0.097 |      0.438 | 0.441 |  0.158 |  0.133 |     0.694 |
