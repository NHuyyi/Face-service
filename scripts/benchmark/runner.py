"""WIDER FACE Benchmark Runner for RetinaFace vs SCRFD on CPU.

Thực hiện đo đạc độc lập tốc độ (latency theo từng giai đoạn: pre, infer, post, total)
và xuất file dự đoán phục vụ đánh giá độ chính xác (mAP Easy/Medium/Hard).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from typing import Any, Tuple, Dict, List, Optional

import cv2
import numpy as np

# Đảm bảo import được các module từ app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.engine.retinaface import MEAN_BGR, RetinaFaceDetector
from app.engine.scrfd import ScrfdDetector
from app.engine.types import DetectedFace
from app.engine.utils import letterbox
from scripts.benchmark.evaluator import evaluate_dataset


def warmup_detectors(rf: RetinaFaceDetector, sc: ScrfdDetector, iterations: int = 5) -> None:
    """Chạy khởi động (warmup) để ổn định ONNX Runtime threadpool và memory allocation."""
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(iterations):
        _ = rf.detect(dummy)
        _ = sc.detect(dummy)


def run_retinaface_timed(det: RetinaFaceDetector, img_bgr: np.ndarray) -> Tuple[List[DetectedFace], dict]:
    t0 = time.perf_counter_ns()
    canvas, scale = letterbox(img_bgr, (det.input_w, det.input_h), MEAN_BGR)
    blob = (canvas - np.asarray(MEAN_BGR, dtype=np.float32)).transpose(2, 0, 1)[None]
    blob_arr = np.ascontiguousarray(blob, dtype=np.float32)
    t1 = time.perf_counter_ns()

    outs = det.sess.run(None, {det.input_name: blob_arr})
    t2 = time.perf_counter_ns()

    faces = det.postprocess(outs, scale)
    t3 = time.perf_counter_ns()

    timing = {
        "pre_ms": (t1 - t0) / 1e6,
        "infer_ms": (t2 - t1) / 1e6,
        "post_ms": (t3 - t2) / 1e6,
        "total_ms": (t3 - t0) / 1e6,
    }
    return faces, timing


def run_scrfd_timed(det: ScrfdDetector, img_bgr: np.ndarray) -> Tuple[List[DetectedFace], dict]:
    t0 = time.perf_counter_ns()
    canvas, scale = letterbox(img_bgr, (det.size, det.size), (127.5, 127.5, 127.5))
    rgb = canvas[:, :, ::-1]
    blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
    blob_arr = np.ascontiguousarray(blob, dtype=np.float32)
    t1 = time.perf_counter_ns()

    outs = det.sess.run(None, {det.input_name: blob_arr})
    t2 = time.perf_counter_ns()

    faces = det.postprocess(outs, scale)
    t3 = time.perf_counter_ns()

    timing = {
        "pre_ms": (t1 - t0) / 1e6,
        "infer_ms": (t2 - t1) / 1e6,
        "post_ms": (t3 - t2) / 1e6,
        "total_ms": (t3 - t0) / 1e6,
    }
    return faces, timing


def write_prediction_file(out_file: str, img_name: str, faces: List[DetectedFace]) -> None:
    """Ghi file dự đoán theo format chính thức WIDER FACE.

    Line 1: img_name
    Line 2: N
    Line 3..N+2: x y w h score (sắp xếp giảm dần theo score)
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    sorted_faces = sorted(faces, key=lambda f: f.score, reverse=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"{img_name}\n")
        f.write(f"{len(sorted_faces)}\n")
        for face in sorted_faces:
            b = face.bbox
            x = float(b[0])
            y = float(b[1])
            w = max(0.1, float(b[2] - b[0]))
            h = max(0.1, float(b[3] - b[1]))
            s = float(face.score)
            f.write(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} {s:.6f}\n")


def compute_statistics(records: List[float]) -> dict:
    if not records:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(records, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WIDER FACE Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh chạy (vd: 20 cho smoke test)")
    parser.add_argument("--score-thr", type=float, default=0.02, help="Ngưỡng score cho prediction export (default: 0.02)")
    parser.add_argument("--nms-thr", type=float, default=0.4, help="Ngưỡng NMS (default: 0.4)")
    parser.add_argument("--dataset-root", default="datasets/wider_face", help="Đường dẫn gốc dataset")
    parser.add_argument("--manifest", default="datasets/wider_face/manifest.jsonl", help="Đường dẫn manifest.jsonl")
    parser.add_argument("--models-dir", default="models", help="Thư mục chứa model ONNX")
    parser.add_argument("--output-dir", default="artifacts/benchmark/widerface", help="Thư mục lưu artifacts benchmark")
    args = parser.parse_args()

    rf_model = os.path.join(args.models_dir, "retinaface.onnx")
    sc_model = os.path.join(args.models_dir, "det_10g.onnx")

    if not os.path.exists(rf_model):
        raise FileNotFoundError(f"RetinaFace model not found: {rf_model}")
    if not os.path.exists(sc_model):
        raise FileNotFoundError(f"SCRFD model not found: {sc_model}")

    print("=" * 70)
    print("KHỞI TẠO BENCHMARK RETINAFACE VS SCRFD")
    print(f"Device: CPU (CPUExecutionProvider)")
    print(f"RetinaFace Model: {rf_model}")
    print(f"SCRFD Model:      {sc_model}")
    print(f"Score Threshold:  {args.score_thr}")
    print(f"NMS Threshold:    {args.nms_thr}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    # 1. Khởi tạo detectors
    rf = RetinaFaceDetector(rf_model, providers=["CPUExecutionProvider"], score_thr=args.score_thr, nms_thr=args.nms_thr)
    sc = ScrfdDetector(sc_model, providers=["CPUExecutionProvider"], score_thr=args.score_thr, nms_thr=args.nms_thr)

    print("Đang khởi động (warmup 5 iterations)...")
    warmup_detectors(rf, sc, iterations=5)
    print("Khởi động hoàn tất.")

    # 2. Đọc manifest
    manifest_items = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    if args.limit is not None and args.limit > 0:
        manifest_items = manifest_items[:args.limit]
        print(f"Chạy giới hạn {len(manifest_items)} ảnh (Smoke Test Mode).")
    else:
        print(f"Chạy toàn bộ {len(manifest_items)} ảnh validation.")

    # 3. Chuẩn bị thư mục output
    pred_dir_rf = os.path.join(args.output_dir, "pred", "retinaface")
    pred_dir_sc = os.path.join(args.output_dir, "pred", "scrfd")
    os.makedirs(pred_dir_rf, exist_ok=True)
    os.makedirs(pred_dir_sc, exist_ok=True)

    timing_jsonl_path = os.path.join(args.output_dir, "timing.jsonl")
    timing_csv_path = os.path.join(args.output_dir, "timing.csv")

    rf_timings = {"pre": [], "infer": [], "post": [], "total": [], "faces": []}
    sc_timings = {"pre": [], "infer": [], "post": [], "total": [], "faces": []}

    csv_fields = [
        "image_id", "img_w", "img_h", "gt_faces",
        "rf_faces", "rf_pre_ms", "rf_infer_ms", "rf_post_ms", "rf_total_ms",
        "sc_faces", "sc_pre_ms", "sc_infer_ms", "sc_post_ms", "sc_total_ms"
    ]

    f_jsonl = open(timing_jsonl_path, "w", encoding="utf-8")
    f_csv = open(timing_csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
    csv_writer.writeheader()

    t_start_all = time.perf_counter()
    n_total = len(manifest_items)

    print(f"Bắt đầu xử lý tuần tự {n_total} ảnh...")

    try:
        for idx, item in enumerate(manifest_items):
            rel_path = item["image_id"]
            event_name = item["event_name"]
            file_base = item["file_name"].replace(".jpg", "")
            img_full_path = os.path.join(args.dataset_root, "images", rel_path)

            img = cv2.imread(img_full_path)
            if img is None:
                print(f"CẢNH BÁO: Không thể đọc ảnh {img_full_path}")
                continue

            h, w = img.shape[:2]
            gt_count = item["num_faces"]

            # Xen kẽ thứ tự chạy để tránh thiên lệch nhiệt độ / bộ nhớ đệm
            if idx % 2 == 0:
                faces_rf, t_rf = run_retinaface_timed(rf, img)
                faces_sc, t_sc = run_scrfd_timed(sc, img)
            else:
                faces_sc, t_sc = run_scrfd_timed(sc, img)
                faces_rf, t_rf = run_retinaface_timed(rf, img)

            # Giải phóng ngay ảnh khỏi RAM
            del img
            if (idx + 1) % 500 == 0:
                gc.collect()

            # Ghi file prediction
            out_file_rf = os.path.join(pred_dir_rf, event_name, f"{file_base}.txt")
            out_file_sc = os.path.join(pred_dir_sc, event_name, f"{file_base}.txt")
            write_prediction_file(out_file_rf, file_base, faces_rf)
            write_prediction_file(out_file_sc, file_base, faces_sc)

            # Ghi nhận số liệu
            rf_timings["pre"].append(t_rf["pre_ms"])
            rf_timings["infer"].append(t_rf["infer_ms"])
            rf_timings["post"].append(t_rf["post_ms"])
            rf_timings["total"].append(t_rf["total_ms"])
            rf_timings["faces"].append(len(faces_rf))

            sc_timings["pre"].append(t_sc["pre_ms"])
            sc_timings["infer"].append(t_sc["infer_ms"])
            sc_timings["post"].append(t_sc["post_ms"])
            sc_timings["total"].append(t_sc["total_ms"])
            sc_timings["faces"].append(len(faces_sc))

            record = {
                "image_id": rel_path,
                "img_w": w,
                "img_h": h,
                "gt_faces": gt_count,
                "rf_faces": len(faces_rf),
                "rf_pre_ms": round(t_rf["pre_ms"], 3),
                "rf_infer_ms": round(t_rf["infer_ms"], 3),
                "rf_post_ms": round(t_rf["post_ms"], 3),
                "rf_total_ms": round(t_rf["total_ms"], 3),
                "sc_faces": len(faces_sc),
                "sc_pre_ms": round(t_sc["pre_ms"], 3),
                "sc_infer_ms": round(t_sc["infer_ms"], 3),
                "sc_post_ms": round(t_sc["post_ms"], 3),
                "sc_total_ms": round(t_sc["total_ms"], 3),
            }
            f_jsonl.write(json.dumps(record) + "\n")
            csv_writer.writerow(record)

            if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
                elapsed = time.perf_counter() - t_start_all
                speed = (idx + 1) / elapsed
                remaining = (n_total - (idx + 1)) / speed if speed > 0 else 0
                avg_rf = np.mean(rf_timings["total"])
                avg_sc = np.mean(sc_timings["total"])
                print(
                    f"[{idx+1}/{n_total}] Đã chạy {elapsed:.1f}s | Còn lại ~{remaining:.1f}s | "
                    f"RF trung bình: {avg_rf:.1f}ms | SC trung bình: {avg_sc:.1f}ms"
                )

    finally:
        f_jsonl.close()
        f_csv.close()

    total_bench_time = time.perf_counter() - t_start_all
    print(f"\nHoàn tất xử lý ảnh! Tổng thời gian chạy: {total_bench_time:.2f}s ({total_bench_time/60:.2f} phút).")

    # 4. Thống kê tốc độ
    rf_stats = {
        "pre_ms": compute_statistics(rf_timings["pre"]),
        "infer_ms": compute_statistics(rf_timings["infer"]),
        "post_ms": compute_statistics(rf_timings["post"]),
        "total_ms": compute_statistics(rf_timings["total"]),
        "faces_detected": int(sum(rf_timings["faces"])),
        "mean_faces_per_img": float(np.mean(rf_timings["faces"])) if rf_timings["faces"] else 0.0,
        "sequential_fps": float(1000.0 / np.mean(rf_timings["total"])) if rf_timings["total"] else 0.0,
    }

    sc_stats = {
        "pre_ms": compute_statistics(sc_timings["pre"]),
        "infer_ms": compute_statistics(sc_timings["infer"]),
        "post_ms": compute_statistics(sc_timings["post"]),
        "total_ms": compute_statistics(sc_timings["total"]),
        "faces_detected": int(sum(sc_timings["faces"])),
        "mean_faces_per_img": float(np.mean(sc_timings["faces"])) if sc_timings["faces"] else 0.0,
        "sequential_fps": float(1000.0 / np.mean(sc_timings["total"])) if sc_timings["total"] else 0.0,
    }

    # 5. Đánh giá độ chính xác mAP WIDER FACE
    print("\nĐang tính toán mAP WIDER FACE (Easy, Medium, Hard) cho RetinaFace...")
    eval_manifest = args.manifest
    # Nếu chạy smoke test, tạo manifest tạm cho số lượng ảnh smoke test
    temp_manifest = None
    if args.limit is not None and args.limit > 0:
        temp_manifest = os.path.join(args.output_dir, "temp_manifest.jsonl")
        with open(temp_manifest, "w", encoding="utf-8") as f_tmp:
            for item in manifest_items:
                f_tmp.write(json.dumps(item) + "\n")
        eval_manifest = temp_manifest

    rf_eval, rf_curves = evaluate_dataset(pred_dir_rf, eval_manifest)
    print(f"RetinaFace -> Easy AP: {rf_eval['easy']['ap']:.4f} | Medium AP: {rf_eval['medium']['ap']:.4f} | Hard AP: {rf_eval['hard']['ap']:.4f}")

    print("Đang tính toán mAP WIDER FACE (Easy, Medium, Hard) cho SCRFD...")
    sc_eval, sc_curves = evaluate_dataset(pred_dir_sc, eval_manifest)
    print(f"SCRFD      -> Easy AP: {sc_eval['easy']['ap']:.4f} | Medium AP: {sc_eval['medium']['ap']:.4f} | Hard AP: {sc_eval['hard']['ap']:.4f}")

    if temp_manifest and os.path.exists(temp_manifest):
        try:
            os.remove(temp_manifest)
        except Exception:
            pass

    # 6. Tổng hợp summary.json
    summary = {
        "metadata": {
            "device": "CPU",
            "provider": "CPUExecutionProvider",
            "total_images_evaluated": len(manifest_items),
            "score_threshold": args.score_thr,
            "nms_threshold": args.nms_thr,
            "total_benchmark_seconds": round(total_bench_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "retinaface": {
            "model_path": rf_model,
            "input_resolution": [rf.input_h, rf.input_w],
            "metrics": rf_eval,
            "latency": rf_stats,
        },
        "scrfd": {
            "model_path": sc_model,
            "input_resolution": [sc.size, sc.size],
            "metrics": sc_eval,
            "latency": sc_stats,
        },
        "comparison": {
            "latency_ratio_rf_over_sc": round(rf_stats["total_ms"]["mean"] / max(1e-6, sc_stats["total_ms"]["mean"]), 2),
            "speedup_scrfd": f"{rf_stats['total_ms']['mean'] / max(1e-6, sc_stats['total_ms']['mean']):.2f}x faster",
            "ap_diff_easy": round(rf_eval["easy"]["ap"] - sc_eval["easy"]["ap"], 4),
            "ap_diff_medium": round(rf_eval["medium"]["ap"] - sc_eval["medium"]["ap"], 4),
            "ap_diff_hard": round(rf_eval["hard"]["ap"] - sc_eval["hard"]["ap"], 4),
        }
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2, ensure_ascii=False)

    curves_path = os.path.join(args.output_dir, "pr_curves.json")
    with open(curves_path, "w", encoding="utf-8") as f_crv:
        json.dump({"retinaface": rf_curves, "scrfd": sc_curves}, f_crv)

    print(f"\nĐã lưu kết quả tóm tắt benchmark vào: {summary_path}")
    print(f"Đã lưu đường cong PR vào: {curves_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
