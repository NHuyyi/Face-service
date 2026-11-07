"""WIDER FACE Evaluation Module in Python.

Tái hiện chính xác 100% thuật toán đánh giá chính thức từ eval_tools (evaluation.m,
boxoverlap.m, norm_score.m, VOCap.m).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def boxoverlap(gt_boxes: np.ndarray, pred_box: np.ndarray) -> np.ndarray:
    """Tính IoU giữa ma trận GT boxes (M, 4) và 1 pred box (4,) theo Fast R-CNN (+1 rule).

    gt_boxes: (M, 4) [x1, y1, x2, y2]
    pred_box: (4,) [x1, y1, x2, y2]
    """
    if len(gt_boxes) == 0:
        return np.zeros((0,), dtype=np.float64)

    x1 = np.maximum(gt_boxes[:, 0], pred_box[0])
    y1 = np.maximum(gt_boxes[:, 1], pred_box[1])
    x2 = np.minimum(gt_boxes[:, 2], pred_box[2])
    y2 = np.minimum(gt_boxes[:, 3], pred_box[3])

    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    inter = w * h
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
    pred_area = (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)

    union = gt_area + pred_area - inter
    o = np.zeros_like(inter, dtype=np.float64)
    valid = (w > 0.0) & (h > 0.0) & (union > 0.0)
    o[valid] = inter[valid] / union[valid]
    return o


def image_evaluation(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    ignore: np.ndarray,
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Khớp predictions với ground-truth theo logic evaluation.m.

    pred_boxes: (N, 4) [x1, y1, x2, y2]
    gt_boxes: (M, 4) [x1, y1, x2, y2]
    ignore: (M,) 1 nếu là valid GT cho tập setting (Easy/Medium/Hard), 0 nếu là ignored GT

    Trả về:
        pred_recall: (N,) số lượng valid GT match tích luỹ tới pred thứ h
        proposal_list: (N,) 1 nếu là proposal hợp lệ (TP hoặc FP), -1 nếu match ignored GT
    """
    n_pred = len(pred_boxes)
    m_gt = len(gt_boxes)

    pred_recall = np.zeros(n_pred, dtype=np.int32)
    recall_list = np.zeros(m_gt, dtype=np.int32)
    proposal_list = np.ones(n_pred, dtype=np.int32)

    for h in range(n_pred):
        overlap_list = boxoverlap(gt_boxes, pred_boxes[h])
        if len(overlap_list) == 0:
            pred_recall[h] = np.sum(recall_list == 1)
            continue

        idx = int(np.argmax(overlap_list))
        max_overlap = overlap_list[idx]

        if max_overlap >= iou_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1

        pred_recall[h] = int(np.sum(recall_list == 1))

    return pred_recall, proposal_list


def image_pr_info(
    thresh_num: int,
    scores: np.ndarray,
    proposal_list: np.ndarray,
    pred_recall: np.ndarray,
) -> np.ndarray:
    """Tính proposals và recall tại 1000 thresholds [0.999 .. 0.0].

    Vectorized qua np.searchsorted cho tốc độ tối ưu, cho kết quả đồng nhất với vòng lặp.
    """
    # thresholds từ 1 - 1/1000 = 0.999 đến 1 - 1000/1000 = 0.0
    thresholds = 1.0 - np.arange(1, thresh_num + 1, dtype=np.float64) / float(thresh_num)
    img_pr = np.zeros((thresh_num, 2), dtype=np.float64)

    if len(scores) == 0:
        return img_pr

    # scores sắp xếp giảm dần -> -scores tăng dần
    counts = np.searchsorted(-scores, -thresholds, side="right")
    valid = counts > 0
    if not np.any(valid):
        return img_pr

    cumsum_prop = np.cumsum(proposal_list == 1)
    r_idx = counts[valid] - 1

    img_pr[valid, 0] = cumsum_prop[r_idx]
    img_pr[valid, 1] = pred_recall[r_idx]
    return img_pr


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Tính Average Precision theo right-monotonic envelope VOCap.m."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))
    return ap


def read_prediction_file(pred_file_path: str) -> np.ndarray:
    """Đọc file dự đoán WIDER FACE text.

    Format:
    line 0: image_name
    line 1: bbx_num (int)
    lines 2..N+1: x y w h score
    """
    if not os.path.exists(pred_file_path):
        return np.zeros((0, 5), dtype=np.float64)

    try:
        with open(pred_file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 2:
            return np.zeros((0, 5), dtype=np.float64)

        n_boxes = int(lines[1])
        if n_boxes == 0 or len(lines) < 2 + n_boxes:
            return np.zeros((0, 5), dtype=np.float64)

        boxes = []
        for i in range(2, 2 + n_boxes):
            parts = lines[i].split()
            if len(parts) >= 5:
                x, y, w, h, s = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append([x, y, w, h, s])

        if not boxes:
            return np.zeros((0, 5), dtype=np.float64)

        arr = np.array(boxes, dtype=np.float64)
        # Sắp xếp giảm dần theo score
        order = np.argsort(arr[:, 4])[::-1]
        return arr[order]
    except Exception:
        return np.zeros((0, 5), dtype=np.float64)


def evaluate_dataset(
    pred_dir: str,
    manifest_path: str,
    settings: Sequence[str] = ("easy", "medium", "hard"),
    iou_thresh: float = 0.5,
    thresh_num: int = 1000,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Thực hiện quy trình đánh giá hoàn chỉnh theo chuẩn WIDER FACE.

    1. Đọc manifest và predictions.
    2. Chuẩn hoá score toàn cục (norm_score).
    3. Chạy image_evaluation và tích luỹ đường cong PR cho từng setting (Easy, Medium, Hard).
    4. Tính AP qua voc_ap.
    """
    # 1. Đọc manifest
    manifest_items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    # 2. Đọc tất cả predictions
    all_preds: List[np.ndarray] = []
    max_score = -float("inf")
    min_score = float("inf")
    total_preds_count = 0

    for item in manifest_items:
        event = item["event_name"]
        file_base = item["file_name"].replace(".jpg", "")
        pred_path = os.path.join(pred_dir, event, f"{file_base}.txt")
        preds = read_prediction_file(pred_path)
        all_preds.append(preds)
        if len(preds) > 0:
            s = preds[:, 4]
            max_score = max(max_score, float(np.max(s)))
            min_score = min(min_score, float(np.min(s)))
            total_preds_count += len(preds)

    # 3. Chuẩn hoá điểm số (norm_score)
    diff = max_score - min_score
    if diff <= 1e-12:
        diff = 1.0

    norm_preds: List[np.ndarray] = []
    for preds in all_preds:
        if len(preds) == 0:
            norm_preds.append(preds)
        else:
            p_copy = preds.copy()
            p_copy[:, 4] = (p_copy[:, 4] - min_score) / diff
            norm_preds.append(p_copy)

    # 4. Đánh giá từng setting
    results: Dict[str, dict] = {}
    curves: Dict[str, dict] = {}
    flag_map = {
        "easy": "easy_valid",
        "medium": "medium_valid",
        "hard": "hard_valid",
    }

    for setting in settings:
        flag_key = flag_map[setting]
        count_face = 0
        org_pr_curve = np.zeros((thresh_num, 2), dtype=np.float64)

        for item, preds in zip(manifest_items, norm_preds):
            faces = item["faces"]
            m_gt = len(faces)

            # Lấy GT boxes chuyển đổi sang [x1, y1, x2, y2]
            gt_boxes = np.zeros((m_gt, 4), dtype=np.float64)
            ignore = np.zeros(m_gt, dtype=np.int32)

            for idx_f, f in enumerate(faces):
                x, y, w, h = f["bbox_xywh"]
                gt_boxes[idx_f] = [x, y, x + w, y + h]
                if f["eval_flags"][flag_key]:
                    ignore[idx_f] = 1

            count_face += int(np.sum(ignore == 1))

            if m_gt == 0 or len(preds) == 0:
                continue

            # pred_boxes: chuyển đổi [x, y, w, h] sang [x1, y1, x2, y2]
            pred_boxes = np.zeros((len(preds), 4), dtype=np.float64)
            pred_boxes[:, 0] = preds[:, 0]
            pred_boxes[:, 1] = preds[:, 1]
            pred_boxes[:, 2] = preds[:, 0] + preds[:, 2]
            pred_boxes[:, 3] = preds[:, 1] + preds[:, 3]

            pred_recall, proposal_list = image_evaluation(pred_boxes, gt_boxes, ignore, iou_thresh)
            img_pr = image_pr_info(thresh_num, preds[:, 4], proposal_list, pred_recall)
            org_pr_curve += img_pr

        # Tính precision & recall
        denom_prec = org_pr_curve[:, 0]
        num_rec = org_pr_curve[:, 1]

        prec = np.zeros(thresh_num, dtype=np.float64)
        valid_p = denom_prec > 0
        prec[valid_p] = num_rec[valid_p] / denom_prec[valid_p]

        rec = np.zeros(thresh_num, dtype=np.float64)
        if count_face > 0:
            rec = num_rec / float(count_face)

        ap = voc_ap(rec, prec)
        results[setting] = {
            "ap": round(ap, 4),
            "count_face": count_face,
            "max_recall": round(float(np.max(rec)), 4) if len(rec) > 0 else 0.0,
        }
        curves[setting] = {
            "recalls": rec.tolist(),
            "precisions": prec.tolist(),
            "proposals": org_pr_curve[:, 0].tolist(),
            "tps": org_pr_curve[:, 1].tolist(),
        }

    results["meta"] = {
        "total_images": len(manifest_items),
        "total_predictions": total_preds_count,
        "score_range": [round(min_score, 4), round(max_score, 4)] if total_preds_count > 0 else [0.0, 0.0],
    }
    return results, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="WIDER FACE Evaluator")
    parser.add_argument("--pred-dir", required=True, help="Prediction directory")
    parser.add_argument("--manifest", default="datasets/wider_face/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    args = parser.parse_args()

    results, _ = evaluate_dataset(args.pred_dir, args.manifest, iou_thresh=args.iou)
    print(f"Results for {args.pred_dir}:")
    for s in ["easy", "medium", "hard"]:
        print(f"  {s.capitalize()} AP: {results[s]['ap']:.4f} (Faces: {results[s]['count_face']}, Max Rec: {results[s]['max_recall']:.4f})")


if __name__ == "__main__":
    main()
