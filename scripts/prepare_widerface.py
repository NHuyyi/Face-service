"""Chuẩn bị tập dữ liệu WIDER FACE Validation cho benchmark RetinaFace vs SCRFD.

Tác vụ:
1. Tải WIDER_val.zip (ảnh Validation), wider_face_split.zip (nhãn txt), eval_tools.zip (nhãn .mat).
2. Giải nén vào datasets/wider_face/ (hình ảnh nằm trong datasets/wider_face/images/).
3. Kiểm tra tính toàn vẹn, phân tích cú pháp nhãn, đối chiếu .mat ground-truth.
4. Xuất manifest.jsonl và manifest_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.io import loadmat
except ModuleNotFoundError:
    import site
    site.addsitedir(r"C:\Users\admin\AppData\Roaming\Python\Python312\site-packages")
    from scipy.io import loadmat

# URLs mặc định
URL_WIDER_VAL_IMAGES = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
URL_WIDER_SPLIT = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"
URL_EVAL_TOOLS = "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip"

# Thư mục gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "wider_face")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
SUMMARY_PATH = os.path.join(DATASET_DIR, "manifest_summary.json")


def get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def compute_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_file(url: str, dest_path: str, timeout: int = 300, chunk_size: int = 1024 * 1024) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[SKIP] Đã tồn tại: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return dest_path

    temp_path = f"{dest_path}.tmp"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    print(f"[DOWNLOAD] Đang tải {url} -> {os.path.basename(dest_path)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceService-Benchmark/1.0"})
    ctx = get_ssl_context()

    start_t = time.time()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        total_len = resp.headers.get("Content-Length")
        total_bytes = int(total_len) if total_len else 0
        total_mb = f"{total_bytes / (1024 * 1024):.2f} MB" if total_bytes else "không rõ"
        print(f"  Dung lượng dự kiến: {total_mb}")

        downloaded = 0
        last_print = time.time()
        with open(temp_path, "wb") as f:
            while chunk := resp.read(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 3.0 or (total_bytes and downloaded == total_bytes):
                    speed = (downloaded / (now - start_t)) / (1024 * 1024)
                    pct = f" ({downloaded * 100 / total_bytes:.1f}%)" if total_bytes else ""
                    print(f"  -> Đã tải {downloaded / (1024 * 1024):.1f} MB{pct} - {speed:.2f} MB/s")
                    last_print = now

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.replace(temp_path, dest_path)
    elapsed = time.time() - start_t
    print(f"[OK] Tải hoàn tất {os.path.basename(dest_path)} trong {elapsed:.1f}s")
    return dest_path


def extract_zip(zip_path: str, extract_to: str, description: str = ""):
    print(f"[EXTRACT] Đang giải nén {os.path.basename(zip_path)} {description}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"[OK] Giải nén xong: {os.path.basename(zip_path)}")


def ensure_gitignore():
    gitignore_path = os.path.join(DATASET_DIR, ".gitignore")
    content = "raw/\nimages/\n*.zip\n*.tmp\n"
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Đã tạo {gitignore_path}")


def prepare_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    ensure_gitignore()

    # 1. Tải các file cần thiết
    val_images_zip = os.path.join(RAW_DIR, "WIDER_val.zip")
    split_zip = os.path.join(RAW_DIR, "wider_face_split.zip")
    eval_tools_zip = os.path.join(RAW_DIR, "eval_tools.zip")

    download_file(URL_WIDER_SPLIT, split_zip)
    download_file(URL_EVAL_TOOLS, eval_tools_zip)
    download_file(URL_WIDER_VAL_IMAGES, val_images_zip)

    # 2. Giải nén annotations
    txt_dest = os.path.join(ANNOTATIONS_DIR, "wider_face_val_bbx_gt.txt")
    if not os.path.exists(txt_dest):
        with zipfile.ZipFile(split_zip, "r") as z:
            for member in z.namelist():
                if member.endswith("wider_face_val_bbx_gt.txt"):
                    with z.open(member) as src, open(txt_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"[EXTRACT] Đã trích xuất {txt_dest}")
                    break

    mat_files = ["wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"]
    with zipfile.ZipFile(eval_tools_zip, "r") as z:
        for name in mat_files:
            dest = os.path.join(ANNOTATIONS_DIR, name)
            if not os.path.exists(dest):
                for member in z.namelist():
                    if member.endswith(f"/{name}") or member == name:
                        with z.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        print(f"[EXTRACT] Đã trích xuất {dest}")
                        break

    # 3. Giải nén hình ảnh
    first_event = os.path.join(IMAGES_DIR, "0--Parade")
    if not os.path.exists(first_event):
        temp_extract = os.path.join(RAW_DIR, "val_extracted")
        extract_zip(val_images_zip, temp_extract, "tập ảnh validation")
        found_img_dir = None
        for root, dirs, files in os.walk(temp_extract):
            if "0--Parade" in dirs:
                found_img_dir = root
                break
        if found_img_dir:
            print(f"[ORGANIZE] Di chuyển các thư mục ảnh từ {found_img_dir} -> {IMAGES_DIR} ...")
            os.makedirs(IMAGES_DIR, exist_ok=True)
            for item in os.listdir(found_img_dir):
                s = os.path.join(found_img_dir, item)
                d = os.path.join(IMAGES_DIR, item)
                if os.path.isdir(s) and not os.path.exists(d):
                    shutil.move(s, d)
            try:
                shutil.rmtree(temp_extract)
            except Exception:
                pass
            print(f"[OK] Cấu trúc ảnh validation đã sẵn sàng tại {IMAGES_DIR}")
        else:
            print("[WARN] Không tìm thấy thư mục 0--Parade trong file giải nén!")


def parse_wider_gt_txt(gt_txt_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Phân tích file wider_face_val_bbx_gt.txt."""
    data = {}
    with open(gt_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    total_lines = len(lines)
    while idx < total_lines:
        img_id = lines[idx]
        idx += 1
        if idx >= total_lines:
            break
        num_boxes = int(lines[idx])
        idx += 1
        boxes = []
        for b_i in range(num_boxes):
            if idx >= total_lines:
                break
            parts = [int(p) for p in lines[idx].split()]
            idx += 1
            if len(parts) >= 10:
                x, y, w, h = parts[0], parts[1], parts[2], parts[3]
                blur, expr, illum, invalid, occl, pose = parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
            elif len(parts) >= 4:
                x, y, w, h = parts[0], parts[1], parts[2], parts[3]
                blur = expr = illum = invalid = occl = pose = 0
            else:
                continue

            boxes.append({
                "box_id": b_i,
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],
                "attributes": {
                    "blur": blur,
                    "expression": expr,
                    "illumination": illum,
                    "invalid": invalid,
                    "occlusion": occl,
                    "pose": pose,
                }
            })
        data[img_id] = boxes
    return data


def load_mat_keep_indices(annotations_dir: str) -> Dict[str, Dict[str, set]]:
    """Đọc keep_index từ wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat.

    Chuyển MATLAB 1-based indexing sang Python 0-based indexing (keep_index - 1).
    """
    gt_mat = loadmat(os.path.join(annotations_dir, "wider_face_val.mat"))
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]

    easy_mat = loadmat(os.path.join(annotations_dir, "wider_easy_val.mat"))["gt_list"]
    medium_mat = loadmat(os.path.join(annotations_dir, "wider_medium_val.mat"))["gt_list"]
    hard_mat = loadmat(os.path.join(annotations_dir, "wider_hard_val.mat"))["gt_list"]

    keep_map = {}
    for event_idx, event in enumerate(event_list):
        event_name = event[0][0]
        files = file_list[event_idx][0]
        easy_event = easy_mat[event_idx][0]
        medium_event = medium_mat[event_idx][0]
        hard_event = hard_mat[event_idx][0]

        for file_idx, f_name in enumerate(files):
            img_name = f_name[0][0]
            img_id = f"{event_name}/{img_name}.jpg"

            def get_indices(cell):
                if len(cell) > file_idx:
                    arr = cell[file_idx][0]
                    if arr.size > 0:
                        # MATLAB 1-based indexing -> trừ 1 sang Python 0-based indexing
                        return set((arr.ravel() - 1).tolist())
                return set()

            keep_map[img_id] = {
                "easy": get_indices(easy_event),
                "medium": get_indices(medium_event),
                "hard": get_indices(hard_event),
            }
    return keep_map


def build_manifest_and_validate():
    print("[PROCESS] Đang đối chiếu nhãn và xây dựng manifest...")
    txt_path = os.path.join(ANNOTATIONS_DIR, "wider_face_val_bbx_gt.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {txt_path}")

    txt_data = parse_wider_gt_txt(txt_path)
    keep_map = load_mat_keep_indices(ANNOTATIONS_DIR)

    total_images_expected = 3226
    total_images_in_txt = len(txt_data)
    total_images_in_mat = len(keep_map)

    print(f"  Tổng ảnh trong file txt: {total_images_in_txt}")
    print(f"  Tổng ảnh trong file mat: {total_images_in_mat}")

    missing_images = []
    unreadable_images = []
    invalid_bbox_coords = []
    total_boxes = 0
    easy_valid_gt = 0
    medium_valid_gt = 0
    hard_valid_gt = 0
    manifest_records = []

    start_v = time.time()
    for img_id, boxes in txt_data.items():
        img_path = os.path.join(IMAGES_DIR, img_id)
        if not os.path.exists(img_path):
            missing_images.append(img_id)
            continue

        img = cv2.imread(img_path)
        if img is None:
            unreadable_images.append(img_id)
            continue
        h, w = img.shape[:2]

        mat_info = keep_map.get(img_id, {"easy": set(), "medium": set(), "hard": set()})
        easy_set = mat_info["easy"]
        medium_set = mat_info["medium"]
        hard_set = mat_info["hard"]

        enriched_faces = []
        for b in boxes:
            total_boxes += 1
            b_id = b["box_id"]
            x, y, bw, bh = b["bbox_xywh"]

            # Kiểm tra tính hợp lệ của tọa độ
            if bw <= 0 or bh <= 0:
                invalid_bbox_coords.append({"img_id": img_id, "box_id": b_id, "bbox": [x, y, bw, bh], "reason": "w or h <= 0"})

            in_easy = b_id in easy_set
            in_medium = b_id in medium_set
            in_hard = b_id in hard_set

            if in_easy:
                easy_valid_gt += 1
            if in_medium:
                medium_valid_gt += 1
            if in_hard:
                hard_valid_gt += 1

            enriched_faces.append({
                "box_id": b_id,
                "bbox_xywh": [x, y, bw, bh],
                "bbox_xyxy": [x, y, x + bw, y + bh],
                "attributes": b["attributes"],
                # ignore = 1 (Valid GT theo quy ước WIDER FACE Evaluator)
                # ignore = 0 (Ignored GT theo quy ước WIDER FACE Evaluator)
                "eval_flags": {
                    "easy_valid": in_easy,
                    "medium_valid": in_medium,
                    "hard_valid": in_hard,
                }
            })

        event_name = os.path.dirname(img_id)
        file_name = os.path.basename(img_id)

        manifest_records.append({
            "image_id": img_id,
            "event_name": event_name,
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
            "num_faces": len(enriched_faces),
            "faces": enriched_faces,
        })

    # Ghi manifest.jsonl
    print(f"[SAVE] Ghi {len(manifest_records)} bản ghi vào {MANIFEST_PATH} ...")
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_sha256 = compute_sha256(MANIFEST_PATH)
    manifest_size = os.path.getsize(MANIFEST_PATH)

    def get_dir_size(d):
        total = 0
        for root, dirs, files in os.walk(d):
            for fi in files:
                total += os.path.getsize(os.path.join(root, fi))
        return total

    images_size_bytes = get_dir_size(IMAGES_DIR) if os.path.exists(IMAGES_DIR) else 0
    raw_size_bytes = get_dir_size(RAW_DIR) if os.path.exists(RAW_DIR) else 0

    summary = {
        "dataset_name": "WIDER FACE Validation Set",
        "expected_images": total_images_expected,
        "processed_images": len(manifest_records),
        "missing_images_count": len(missing_images),
        "unreadable_images_count": len(unreadable_images),
        "total_ground_truth_faces": total_boxes,
        "easy_valid_gt_faces": easy_valid_gt,
        "medium_valid_gt_faces": medium_valid_gt,
        "hard_valid_gt_faces": hard_valid_gt,
        "invalid_bbox_coords_count": len(invalid_bbox_coords),
        "manifest_path": MANIFEST_PATH,
        "manifest_size_bytes": manifest_size,
        "manifest_sha256": manifest_sha256,
        "images_dir": IMAGES_DIR,
        "images_size_mb": round(images_size_bytes / (1024 * 1024), 2),
        "raw_dir": RAW_DIR,
        "raw_size_mb": round(raw_size_bytes / (1024 * 1024), 2),
        "evaluator_convention": {
            "valid_gt_flag": 1,
            "ignored_gt_flag": 0,
            "coordinate_format": "xywh raw preserved; xyxy continuous half-open [x, y, x+w, y+h]",
        }
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Đã ghi tóm tắt vào {SUMMARY_PATH}")
    print("\n=== KẾT QUẢ TỔNG HỢP PHA 2B ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WIDER FACE Validation Dataset Preparation")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải dữ liệu nếu đã có file")
    args = parser.parse_args()

    if not args.skip_download:
        prepare_data()
    build_manifest_and_validate()
