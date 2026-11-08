"""Căn thẳng khuôn mặt về chuẩn ArcFace 112x112 bằng 5 điểm landmark."""
from __future__ import annotations

import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class AlignmentError(ValueError):
    """Invalid landmarks or a degenerate similarity transform."""


def _points(points: np.ndarray) -> np.ndarray:
    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AlignmentError("Landmark phải gồm 5 điểm (x, y) hữu hạn") from exc
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise AlignmentError("Landmark phải gồm 5 điểm (x, y) hữu hạn")
    centered = points - points.mean(axis=0)
    if (np.linalg.matrix_rank(centered) < 2
            or np.linalg.norm(points[1] - points[0]) < 1e-6):
        raise AlignmentError("Landmark suy biến: các điểm trùng/thẳng hàng hoặc hai mắt trùng nhau")
    return points


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ma trận biến đổi đồng dạng 2x3 (xoay + tỉ lệ đều + tịnh tiến) src -> dst."""
    src, dst = _points(src), _points(dst)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError as exc:
        raise AlignmentError("Không tính được phép biến đổi landmark") from exc
    if np.linalg.matrix_rank(cov) < 2:
        raise AlignmentError("Phép biến đổi landmark suy biến")
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
    if not np.isfinite(M).all() or scale <= 1e-12:
        raise AlignmentError("Phép biến đổi landmark suy biến")
    return M


def align_face(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Cắt + xoay thẳng mặt (mắt nằm ngang) -> ảnh BGR size x size."""
    if not isinstance(size, int) or size <= 0:
        raise AlignmentError("Kích thước ảnh căn chỉnh phải là số nguyên dương")
    template = ARCFACE_TEMPLATE * (size / 112.0)
    M = umeyama_similarity(kps, template)
    return cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)


def roll_angle_degrees(kps: np.ndarray) -> float:
    """Góc nghiêng của đường nối hai mắt (độ, dương = nghiêng xuống bên phải)."""
    kps = _points(kps)
    dx, dy = kps[1] - kps[0]
    return float(np.degrees(np.arctan2(dy, dx)))
