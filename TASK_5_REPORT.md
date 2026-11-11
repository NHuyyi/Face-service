| Chỉ số | CŨ: MobileFaceNet-192 (pipeline app) | CŨ + căn chỉnh landmark | MỚI: ArcFace-512 |
|---|---|---|---|
| Số ảnh / số người / số cặp | 82 / 15 / 3321 |  |  |
| Số cặp cùng người / khác người | 197 / 3124 |  |  |
| Kích thước vector | 192 | 192 | 512 |
| Ngưỡng mặc định | 0.75 | 0.75 | 0.40 |
| Accuracy tại ngưỡng mặc định | 94.279% | 97.019% | 100.000% |
| FAR tại ngưỡng mặc định | 0.000% | 0.000% | 0.000% |
| FRR tại ngưỡng mặc định | 96.447% | 50.254% | 0.000% |
| EER | 21.319% | 0.940% | 0.000% |
| AUC | 0.8753 | 0.9996 | 1.0000 |
| Accuracy tối ưu (ngưỡng tốt nhất) | 95.122% @ 0.55 | 99.789% @ 0.49 | 100.000% @ 0.46 |
| TAR @ FAR=1% | 32.487% | 99.492% | 100.000% |
| TAR @ FAR=0.1% | 14.213% | 96.954% | 100.000% |
| Rank-1 (nhận diện 1:N) | 79.268% | 100.000% | 100.000% |
| Similarity cùng người (TB ± SD) | 0.461 ± 0.176 | 0.736 ± 0.112 | 0.666 ± 0.082 |
| Similarity khác người (TB ± SD) | 0.181 ± 0.163 | 0.029 ± 0.144 | 0.012 ± 0.062 |
| d' (độ tách hai phân phối) | 1.65 | 5.48 | 8.98 |

Ảnh cần xoay để mặt thẳng: 0/82 (app Android cũ không xử lý được các ảnh này).
Ảnh bị loại (cả ba cột): 0