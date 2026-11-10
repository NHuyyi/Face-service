# Face Recognition Service – Test Plan

---


### Chuẩn bị ảnh test

Đặt ảnh người thật vào `tests/assets/`:

| File | Mô tả |
|---|---|
| `person_a_1.jpg` | Người A – ảnh 1 (dùng để Enroll) |
| `person_a_2.jpg` | Người A – ảnh 2 (cùng người, góc khác / điều kiện khác) |
| `person_b_1.jpg` | Người B – ảnh khác người hoàn toàn |

> Script sẽ tự tạo ảnh dummy xám nếu chưa có, nhưng kết quả `Match/0 Match` sẽ không chính xác với ảnh dummy.

---

## Danh sách Test Case

### Luồng chính

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-01 | Health check | `GET /health` | — | HTTP 200, `status=ok`, `dim=512` |
| TC-02a | Enroll người A | `POST /v1/faces` | `person_a_1.jpg`, `label=person_a` | HTTP 200, có `id`, `dim=512` |
| TC-02b | Match cùng người | `POST /v1/faces/{id}/match` | `person_a_2.jpg` | `result=Match`, `match=1`, `similarity > 0.40` |
| TC-02c | Enroll + return_embedding | `POST /v1/faces` | `return_embedding=true` | HTTP 200, `embedding` length 512, L2-norm ≈ 1.0 |
| TC-03 | Match khác người | `POST /v1/faces/{id}/match` | `person_b_1.jpg` | `result=0 Match`, `match=0`, `similarity < 0.40` |

### Compare (không lưu DB)

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-04a | Compare cùng người | `POST /v1/compare` | `file_a=A1`, `file_b=A2` | `match=1`, có `face_a`, `face_b` |
| TC-04b | Compare khác người | `POST /v1/compare` | `file_a=A1`, `file_b=B1` | `match=0` |
| TC-04c | Compare ngưỡng tùy chỉnh | `POST /v1/compare` | `threshold=0.70` | `threshold=0.70` trong response |

### Embeddings

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-05 | Lấy vector | `POST /v1/embeddings` | `person_a_1.jpg` | HTTP 200, `dim=512`, array 512 phần tử, L2-norm ≈ 1 |

### Identify (1:N)

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-06a | Enroll người B | `POST /v1/faces` | `person_b_1.jpg` | HTTP 200 |
| TC-06b | Identify người A | `POST /v1/identify` | `person_a_1.jpg`, `top_k=5` | `candidates[0].label=person_a`, `match=1` |

### Quản lý DB

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-11 | Liệt kê faces | `GET /v1/faces` | `limit=10` | HTTP 200, array JSON, mỗi phần tử có `id,label,det_score,created_at` |
| TC-12 | Xóa face | `DELETE /v1/faces/{id}` | face_id hợp lệ | HTTP 200, `{"deleted": id}` |
| TC-12b | Xóa face lần 2 | `DELETE /v1/faces/{id}` | id đã xóa | HTTP 404 |

### Lỗi

| TC | Tên | Endpoint | Input | Kỳ vọng |
|---|---|---|---|---|
| TC-07 | 404 id không tồn tại | `POST /v1/faces/{id}/match` | id giả | HTTP 404, có `detail` |
| TC-07b | 404 DELETE id không tồn tại | `DELETE /v1/faces/{id}` | id giả | HTTP 404 |
| TC-08 | 422 không thấy mặt | `POST /v1/faces` | ảnh trắng 64×64 | HTTP 422, có `detail` |
| TC-09 | 400 file không hỗ trợ | `POST /v1/faces` | file `.txt` | HTTP 400 |
| TC-10 | 400 file rỗng | `POST /v1/faces` | file 0 byte | HTTP 400 |
| TC-11 | 413 ảnh quá lớn | `POST /v1/faces` | ảnh > 10MB | HTTP 413 |
| TC-13 | 401 sai API key | `POST /v1/faces` | `X-API-Key: wrong` | HTTP 401 _(chỉ khi `API_KEY` được cấu hình)_ |

---

## Cách chạy

### Tự động – curl script

**Linux / macOS:**
```bash
# Cài đặt: đặt ảnh thật vào tests/assets/
export BASE_URL=http://localhost:8000
export API_KEY=your-secret      # bỏ qua nếu không dùng
bash tests/qa_curl_test.sh
```

**Windows (PowerShell):**
```powershell
.\tests\qa_curl_test.ps1 -BaseUrl "http://localhost:8000" -ApiKey "your-secret"
```

Kết quả in ra màn hình dạng `[PASS]` / `[FAIL]` / `[SKIP]` với tóm tắt cuối.

### Tự động – Postman Collection Runner

1. Mở Postman → Import → chọn `tests/Face_Service_QA.postman_collection.json`
2. Đặt biến `base_url` = `http://localhost:8000`, `api_key` = _(để trống hoặc điền key)_
3. Trong mỗi request tab **Body** → cập nhật đường dẫn ảnh người thật
4. Click **Run Collection** → xem kết quả từng test case

### Tự động – pytest (unit + API)

```bash
pytest -q                              # tất cả
pytest tests/test_core.py -v           # chỉ logic lõi
pytest tests/test_api.py -v            # chỉ API (engine giả)
```

---

## Tiêu chí Pass/Fail

| Tiêu chí | Ngưỡng |
|---|---|
| Similarity cùng người | > 0.40 (ngưỡng mặc định) |
| Similarity khác người | < 0.40 |
| L2-norm của embedding | 1.000 ± 0.001 |
| Thời gian phản hồi `/v1/faces` | < 3 giây (CPU, ảnh ≤ 2MB) |
| Tất cả HTTP status code | Đúng như bảng test case |

---

