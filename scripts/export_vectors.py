"""Script trích xuất hàng loạt vector 512 (ArcFace) và 192 (MobileFaceNet) cho tập dataset.

Hỗ trợ cả 2 mô hình phát hiện khuôn mặt: RetinaFace và SCRFD.

Cách chạy:
    # 1. Xuất dữ liệu bằng RetinaFace:
    python scripts/export_vectors.py --detector retinaface --out exported_retinaface.json

    # 2. Xuất dữ liệu bằng SCRFD:
    python scripts/export_vectors.py --detector scrfd --out exported_scrfd.json

    # 3. Xuất đồng thời cả 2 detector :
    python scripts/export_vectors.py --detector both --out exported_vectors_all.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Thêm thư mục gốc vào path để import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.engine import FaceEngine, NoFaceError

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_dataset_images(dataset_dir: str):
    """Quét toàn bộ ảnh trong thư mục dataset."""
    items = []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {dataset_dir}")

    for person in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person)
        if os.path.isdir(pdir):
            for fn in sorted(os.listdir(pdir)):
                if fn.lower().endswith(EXTS):
                    items.append((person, fn, os.path.join(pdir, fn)))
    return items


def main():
    parser = argparse.ArgumentParser(description="Xuất vector 512 và 192 cho RetinaFace / SCRFD")
    parser.add_argument("--dataset", default="dataset", help="Đường dẫn thư mục dataset (mặc định: dataset)")
    parser.add_argument("--detector", choices=["retinaface", "scrfd", "both"], default="both",
                        help="Chọn detector: retinaface, scrfd hoặc both (mặc định: both)")
    parser.add_argument("--out", default="exported_vectors.json", help="Đường dẫn file JSON đầu ra")
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT VECTOR 512 & 192 CHO DATASET")
    print("=" * 65)
    print(f"📁 Thư mục dataset: {args.dataset}")
    print(f"🔍 Detector: {args.detector}")
    print(f"💾 File đầu ra: {args.out}")

    images = load_dataset_images(args.dataset)
    print(f"📸 Tìm thấy {len(images)} ảnh trong thư mục dataset.\n")

    settings = Settings()
    engine = FaceEngine(settings)

    detectors = ["retinaface", "scrfd"] if args.detector == "both" else [args.detector]

    results = []
    stats = {d: {"success": 0, "failed": 0} for d in detectors}

    t0 = time.time()

    for idx, (person, filename, filepath) in enumerate(images, 1):
        img = cv2.imread(filepath)
        if img is None:
            print(f"⚠️  [{idx}/{len(images)}] Không đọc được ảnh: {filepath}")
            continue

        for det in detectors:
            try:
                r = engine.extract_both(img, detector_backend=det)
                record = {
                    "person": person,
                    "filename": filename,
                    "relative_path": os.path.join(person, filename),
                    "detector": det,
                    "vector_512": r.embedding.tolist(),
                    "vector_192": r.embedding_192.tolist() if r.embedding_192 is not None else None,
                    "dim_512": len(r.embedding),
                    "dim_192": len(r.embedding_192) if r.embedding_192 is not None else 0,
                    "bbox": r.bbox,
                    "det_score": round(float(r.det_score), 4),
                    "rotation_applied": r.rotation_applied,
                    "roll_degrees": round(float(r.roll_degrees), 2),
                }
                results.append(record)
                stats[det]["success"] += 1
                status = f"✅ OK (512-d & 192-d, rot={r.rotation_applied}°)"
            except (NoFaceError, Exception) as e:
                stats[det]["failed"] += 1
                status = f"❌ Lỗi: {e}"

            print(f"[{idx:2d}/{len(images):2d}] [{det:10s}] {person:10s}/{filename:25s} -> {status}")

    total_time = time.time() - t0

    # Lưu ra file JSON
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(images),
            "total_records": len(results),
            "detectors": detectors,
            "models": {
                "512_dim": "ArcFace (w600k_r50.onnx)",
                "192_dim": "MobileFaceNet (mobile_face_net.tflite)"
            },
            "stats": stats,
            "elapsed_seconds": round(total_time, 2),
        },
        "data": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("📊 TỔNG KẾT QUÁ TRÌNH XUẤT VECTOR:")
    print("=" * 65)
    for det in detectors:
        s = stats[det]
        print(f" • Detector [{det}]: Thành công {s['success']}/{len(images)} ảnh ({s['failed']} ảnh không thấy mặt)")
    print(f"⏱️  Thời gian xử lý: {total_time:.2f} giây")
    print(f"🎉 Đã lưu thành công {len(results)} bản ghi vào: {args.out}")



if __name__ == "__main__":
    main()

