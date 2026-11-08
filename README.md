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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py          # tải ArcFace (+ SCRFD) vào ./models
```

## 2. Chạy

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# hoặc: docker compose up --build        (cần models/ đã tải)
```
Swagger: http://localhost:8000/docs

## 3. Model RetinaFace

`DETECTOR_BACKEND=retinaface` (mặc định) cần `models/retinaface.onnx`, loại export từ
[biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface):

```bash
git clone https://github.com/biubug6/Pytorch_Retinaface && cd Pytorch_Retinaface
# tải weights (mobilenet0.25_Final.pth hoặc Resnet50_Final.pth) theo hướng dẫn của repo
python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
cp FaceDetector.onnx ../face-service/models/retinaface.onnx
```
Code tự đọc kích thước đầu vào cố định của file ONNX (thường 640×640) và nhận ba output
`loc / conf / landms` theo số chiều, nên không phụ thuộc thứ tự output.
Muốn chạy ngay không cần export: `DETECTOR_BACKEND=scrfd` (dùng `det_10g.onnx` đã tải sẵn).

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
