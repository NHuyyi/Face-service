# TÀI LIỆU BÀN GIAO: XUẤT VECTOR 512 & 192 (RETINAFACE & SCRFD)

> **Người thực hiện:**  phụ trách Trích xuất Vector (Task 3)  
> **Người nhận bàn giao:** Thành viên kiểm tra lại & Cặp phụ trách So trùng (Task 2)  
> **Trạng thái:** ✅ Đã hoàn thành 100% & Kiểm thử thành công

---

## 1. TỔNG QUAN KẾT QUẢ ĐÃ ĐẠT ĐƯỢC

1. **Tích hợp thành công mô hình MobileFaceNet (192-d):**
   * Đã tích hợp file `models/mobile_face_net.tflite` từ app Android vào service.
   * Xây dựng module trích xuất `app/engine/mobilefacenet.py` chuẩn hoá pixel đúng chuẩn Android `(pixel - 127.5) / 128.0` và L2-norm.
2. **Hỗ trợ đầy đủ 4 sự kết hợp:**
   * `RetinaFace + ArcFace-512`
   * `RetinaFace + MobileFaceNet-192`
   * `SCRFD + ArcFace-512`
   * `SCRFD + MobileFaceNet-192`
3. **Đã xuất sẵn toàn bộ dữ liệu Dataset (82 ảnh / 15 người):**
   * File dữ liệu: `report/dataset_vectors_all.json` (chứa đủ 164 bản ghi với cả 2 vector 512 và 192).

---

## 2. LẤY DỮ LIỆU ĐỂ SO TRÙNG (DÀNH CHO CẶP Task 2)

### Cách 1: Đọc trực tiếp từ file dữ liệu có sẵn 
File `report/dataset_vectors_all.json` đã được trích xuất sẵn từ toàn bộ dataset 15 người.

**Cấu trúc mỗi bản ghi trong file JSON:**
```json
{
  "person": "Person1",
  "filename": "AnhChinhDien.jpg",
  "relative_path": "Person1/AnhChinhDien.jpg",
  "detector": "retinaface",                 // "retinaface" hoặc "scrfd"
  "vector_512": [0.021, -0.045, ...],      // 512 số float (ArcFace)
  "vector_192": [0.112, 0.008, ...],       // 192 số float (MobileFaceNet)
  "dim_512": 512,
  "dim_192": 192,
  "bbox": [120, 85, 230, 260],
  "det_score": 0.998,
  "rotation_applied": 0
}
```

---

### Cách 2: Gọi qua API Real-time (`POST /v1/embeddings`)
Nếu muốn gửi ảnh mới lên để lấy vector theo thời gian thực:

* **Endpoint:** `POST /v1/embeddings?detector=retinaface` (hoặc `detector=scrfd`)
* **Headers:** `X-API-Key: <key>` (nếu có cấu hình API Key)
* **Body (Form-data):** `file: <file_ảnh>`
* **Response:**
```json
{
  "dim": 512,
  "embedding": [...],
  "dim_192": 192,
  "embedding_192": [...],
  "detector": "retinaface",
  "face": {
    "bbox": [120, 85, 230, 260],
    "det_score": 0.998,
    "rotation_applied": 0
  }
}
```

---

## 3. CÁCH CHẠY SCRIPT XUẤT VECTOR


```bash
# Xuất đồng thời cả RetinaFace và SCRFD ra file mới:
python scripts/export_vectors.py --detector both --out report/dataset_vectors_all.json
```

---

## 4. DANH SÁCH FILE ĐÃ BỔ SUNG / CẬP NHẬT
1. `models/mobile_face_net.tflite`: Mô hình MobileFaceNet của Android.
2. `app/engine/mobilefacenet.py`: Module trích xuất vector 192-d.
3. `app/engine/engine.py`: Nâng cấp `FaceEngine.extract_both()` hỗ trợ cả 2 vector và 2 detector.
4. `app/main.py` & `app/schemas.py`: Nâng cấp API `POST /v1/embeddings`.
5. `scripts/export_vectors.py`: Script quét dataset xuất hàng loạt.
6. `report/dataset_vectors_all.json`: File dữ liệu 164 bản ghi đã xuất sẵn.
7. `requirements.txt`: Đã bổ sung `ai-edge-litert>=2.0`.
8. `tests/test_core.py`: Bổ sung test tự động cho MobileFaceNet và `extract_both`.

