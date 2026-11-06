from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list:
    """Non-maximum suppression thuần NumPy. boxes: (N,4) x1y1x2y2."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


def letterbox(img: np.ndarray, size: Tuple[int, int], fill: Tuple[float, float, float]):
    """Resize giữ tỉ lệ, đặt góc trên-trái, phần còn lại tô màu `fill`.

    size = (W, H). Trả về (canvas float32 HxWx3, scale).
    """
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.empty((th, tw, 3), dtype=np.float32)
    canvas[:] = np.asarray(fill, dtype=np.float32)
    canvas[:nh, :nw] = resized
    return canvas, scale
