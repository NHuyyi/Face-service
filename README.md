# Face Recognition Service (RetinaFace + ArcFace 512-d)

Service Python/FastAPI thay thế engine MobileFaceNet-192 (TFLite) đang chạy trên Android trong
`FaceMatchHelper.java`. App Android giữ nguyên; service chạy trên server để các nhóm khác gọi chung.

| Bước | Thành phần |
|---|---|
| Phát hiện mặt + 5 landmark | **RetinaFace** (ONNX) – dự phòng: SCRFD |
| Xoay thẳng | EXIF → thử xoay 0/90/180/270° nếu không thấy mặt → căn theo 5 landmark (similarity transform) về 112×112 |
| Trích đặc trưng | **ArcFace** `w600k_r50.onnx` → vector **512** chiều, L2-normalize |
| Lưu trữ | SQLite (`float32` BLOB), tuỳ chọn TTL để lưu "tạm thời" |
| So khớp | **NumPy** cosine similarity (1:1 và 1:N bằng một phép nhân ma trận) |

## 1. Cài đặt

```bash
# 1. Clone repository (nếu bắt đầu từ mới)
git clone https://github.com/NHuyyi/Face-service.git && cd Face-service

# 2. Tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
# Trên Windows:
.\.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# 3. Tải tự động toàn bộ 3 model ONNX vào ./models
python scripts/download_models.py
```

Lệnh `python scripts/download_models.py` sẽ tự động tải và xác minh tính toàn vẹn (SHA-256) của cả 3 mô hình cần thiết:
- **`retinaface.onnx`** (RetinaFace ResNet50 – detector mặc định): Tải trực tiếp từ GitHub Releases của dự án.
- **`w600k_r50.onnx`** (ArcFace ResNet50 – trích xuất vector 512 chiều).
- **`det_10g.onnx`** (SCRFD-10G – detector dự phòng).

Nếu các file model đã tồn tại và mã băm SHA-256 hợp lệ, script sẽ tự động bỏ qua để tiết kiệm băng thông.

## 2. Chạy Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# hoặc: docker compose up --build
```
Swagger UI kiểm thử trực tiếp: http://localhost:8000/docs
Trong VS Code, bạn cũng có thể mở tab **Run and Debug** và chọn profile **`Face Service (RetinaFace)`** (hoặc **`Face Service (SCRFD)`**).

## 3. Quản lý Model RetinaFace & Export Thủ công (Tùy chọn)

Trong cài đặt thông thường, bạn **không cần export thủ công** vì script `python scripts/download_models.py` đã tự động tải bản ONNX ResNet50 chuẩn sẵn sàng hoạt động.

Nếu muốn tùy biến đường dẫn tải hoặc tự export từ checkpoint PyTorch:
- **Tải từ URL máy chủ mirror/dự phòng (`--retinaface-url`):**
  ```bash
  python scripts/download_models.py --retinaface-url https://mirror.example.com/models/retinaface.onnx
  ```
  > [!NOTE]
  > Tùy chọn `--retinaface-url` chỉ dùng để trỏ đến URL máy chủ mirror lưu trữ **chính xác cùng file nhị phân RetinaFace ResNet50 ONNX chuẩn**. File tải về **bắt buộc vẫn phải khớp đúng mã băm SHA-256 đã ghim** (`ed011b65dcc7e3f10f307c3d013bf2d0260ded5007314a96fb20339322463544`). Downloader không hỗ trợ các file weights ONNX tùy biến khác hash này nhằm ngăn chặn sai lệch kiến trúc hoặc suy giảm độ chính xác nhận diện.
- **Export thủ công từ mã nguồn gốc [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) (Tùy chọn nâng cao):**
  Nếu bạn tự huấn luyện hoặc xuất từ checkpoint PyTorch khác:
  ```bash
  git clone https://github.com/biubug6/Pytorch_Retinaface && cd Pytorch_Retinaface
  # Tải checkpoint weights (Resnet50_Final.pth)
  python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
  cp FaceDetector.onnx ../Face-service/models/retinaface.onnx
  ```
Code detector tự động nhận diện kích thước đầu vào cố định ($640 \times 640$) và ánh xạ 3 tensor output `loc / conf / landms` theo số chiều, tương thích hoàn toàn với kiến trúc ResNet50.

## 4. Quy trình theo yêu cầu

```bash
# Bước 1 – ảnh người thứ nhất: xoay thẳng → vector 512 → lưu DB
curl -F file=@a.jpg -F label=nguoi_1 http://localhost:8000/v1/faces
#  → {"id":"3f2c…","dim":512,"face":{"rotation_applied":0,"roll_degrees":-4.2,…}}

# Bước 2 – ảnh người thứ hai: vector 512 → so với vector đã lưu
curl -F file=@b.jpg http://localhost:8000/v1/faces/3f2c…/match
#  → {"result":"Match","match":1,"similarity":0.62,"threshold":0.4,…}
#  → {"result":"0 Match","match":0,"similarity":0.08,…}
```

| Endpoint | Mục đích |
|---|---|
| `POST /v1/faces` | Enroll: ảnh → vector 512 → lưu DB (`label`, `return_embedding` tuỳ chọn) |
| `POST /v1/faces/{id}/match` | So ảnh mới với vector đã lưu → `Match` / `0 Match` |
| `POST /v1/compare` | So trực tiếp 2 ảnh (`file_a`, `file_b`), không lưu |
| `POST /v1/identify` | 1:N – top-k giống nhất trong DB |
| `POST /v1/embeddings` | Chỉ trả vector 512 (cho nhóm khác tự xử lý) |
| `GET /v1/faces`, `DELETE /v1/faces/{id}`, `GET /health` | Quản lý / kiểm tra |

Lỗi: `422` không thấy mặt, `400` ảnh hỏng, `413` ảnh quá lớn, `404` id không tồn tại/hết hạn, `401` sai API key.

## 5. Thử nghiệm matching hàng loạt

```bash
# dataset/An/*.jpg, dataset/Binh/*.jpg, …
python scripts/evaluate.py dataset --threshold 0.40 --out report
```
Xuất `report/similarity_matrix.csv`, `report/pairs.csv`, in thống kê cùng người / khác người,
TP/FP/TN/FN, FAR/FRR và **ngưỡng gợi ý** theo dữ liệu của bạn.

## 6. Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DETECTOR_BACKEND` | `retinaface` | `retinaface` \| `scrfd` |
| `MATCH_THRESHOLD` | `0.40` | Ngưỡng cosine similarity (xem lưu ý dưới) |
| `DET_SCORE_THRESHOLD` / `MIN_FACE_SIZE` | `0.5` / `40` | Lọc mặt yếu / quá nhỏ (px) |
| `VECTOR_TTL_SECONDS` | `0` | >0: tự xoá vector sau N giây |
| `ORT_PROVIDERS` | `CPUExecutionProvider` | GPU: `CUDAExecutionProvider,CPUExecutionProvider` + `onnxruntime-gpu` |
| `API_KEY` | trống | Có giá trị → bắt buộc header `X-API-Key` |
| `DB_PATH`, `MODELS_DIR`, `MAX_UPLOAD_MB` | — | Đường dẫn DB / model, giới hạn ảnh |

**Lưu ý ngưỡng:** ngưỡng 0.75 của app Android là của MobileFaceNet, **không dùng lại** cho ArcFace.
Với ArcFace `w600k_r50`, cosine của cùng một người thường ≈ 0.4–0.8, người khác ≈ −0.1–0.25;
0.40 là điểm khởi đầu hợp lý, hãy hiệu chỉnh bằng `scripts/evaluate.py` trên ảnh thực tế của dự án.

## 7. Tích hợp với Android

Vector cũ (192-d) và mới (512-d) **không so sánh được với nhau** – khi chuyển sang service, cần
enroll lại ảnh sinh viên đã có (gửi ảnh lên `POST /v1/faces`, lưu `id` thay cho vector 192-d trong SQLite của app).
Phía Android chỉ cần gửi ảnh (multipart) và đọc `match`/`similarity` từ JSON.

## 2. Chạy Service

### Cách 1: Chạy bằng Docker
Tất cả cấu hình (TTL 300s, API Key, Model RetinaFace/SCRFD) đã được thiết lập sẵn trong `docker-compose.yml`:
```bash
docker compose up --build -d
```

### Cách 2: Chạy trực tiếp bằng Python Uvicorn (Phát triển & Test)

**Trên Windows (PowerShell):**
```powershell
$env:DETECTOR_BACKEND="retinaface"; $env:API_KEY="my-secret-key-2026"; $env:VECTOR_TTL_SECONDS="300"; python -m uvicorn app.main:app --port 8000 --reload
```

**Trên Linux / macOS:**
```bash
DETECTOR_BACKEND=retinaface API_KEY=my-secret-key-2026 VECTOR_TTL_SECONDS=300 uvicorn app.main:app --port 8000 --reload
```

> **Ghi chú các tham số cấu hình:**
> - `DETECTOR_BACKEND=retinaface`: Dùng model phát hiện RetinaFace (hoặc đổi thành `scrfd` nếu dùng SCRFD có sẵn trong `models/`).
> - `API_KEY=my-secret-key-2026`: Bật khoá bảo mật (yêu cầu truyền header `X-API-Key`).
> - `VECTOR_TTL_SECONDS=300`: Tự động giải phóng/xoá vector khỏi SQLite sau 5 phút (lưu tạm thời).


## 8. Test

```bash
pytest -q        # logic lõi + API (engine giả, không cần model)
```
