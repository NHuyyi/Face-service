"""Căn thẳng khuôn mặt về chuẩn ArcFace 112x112 bằng 5 điểm landmark."""
from __future__ import annotations

import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ma trận biến đổi đồng dạng 2x3 (xoay + tỉ lệ đều + tịnh tiến) src -> dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    sign = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign[-1] = -1
    R = U @ np.diag(sign) @ Vt
    var_s = (s ** 2).sum() / len(src)
    scale = (S * sign).sum() / var_s
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3))
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def align_face(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Cắt + xoay thẳng mặt (mắt nằm ngang) -> ảnh BGR size x size."""
    template = ARCFACE_TEMPLATE * (size / 112.0)
    M = umeyama_similarity(kps, template)
    return cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)


def roll_angle_degrees(kps: np.ndarray) -> float:
    """Góc nghiêng của đường nối hai mắt (độ, dương = nghiêng xuống bên phải)."""
    dx, dy = kps[1] - kps[0]
    return float(np.degrees(np.arctan2(dy, dx)))
