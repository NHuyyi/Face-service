"""Kiểm tra kích thước và tư thế khuôn mặt từ bbox + keypoints RetinaFace."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..config import Settings

# Đọc ngưỡng từ Settings một lần; hàm không giữ trạng thái.
_cfg = Settings()

MIN_FACE_PX: int = _cfg.min_face_px
MIN_EYE_DIST_PX: float = _cfg.min_eye_dist_px
MIN_DET_SCORE: float = _cfg.min_det_score
MAX_ROLL_DEG: float = _cfg.max_roll_deg
MAX_YAW_DEG: float = _cfg.max_yaw_deg
MIN_PITCH_RATIO: float = _cfg.min_pitch_ratio
MAX_PITCH_RATIO: float = _cfg.max_pitch_ratio


def _check(name: str, passed: bool, value: Optional[float], rule: str, code: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "rule": rule,
        "code": code,
    }


def check_face_size(
    bbox: np.ndarray | list,
    kps: np.ndarray,
    det_score: float,
) -> list[dict]:
    """Kiểm tra kích thước bbox, khoảng cách 2 mắt, và detection score.

    Parameters
    ----------
    bbox:      [x1, y1, x2, y2]
    kps:       (5, 2) — mắt trái, mắt phải, mũi, khoé miệng trái/phải
    det_score: float từ RetinaFace

    Returns
    -------
    list[dict] mỗi phần tử {"name", "passed", "value", "rule", "code"}
    """
    b = np.asarray(bbox, dtype=float)
    w = float(b[2] - b[0])
    h = float(b[3] - b[1])
    min_side = min(w, h)

    pts = np.asarray(kps, dtype=float)
    eye_dist = float(np.linalg.norm(pts[1] - pts[0]))

    score_val = float(det_score)

    return [
        _check(
            "face_min_side",
            min_side >= MIN_FACE_PX,
            min_side,
            f">= {MIN_FACE_PX} px",
            "FACE_TOO_SMALL",
        ),
        _check(
            "eye_distance",
            eye_dist >= MIN_EYE_DIST_PX,
            eye_dist,
            f">= {MIN_EYE_DIST_PX} px",
            "FACE_TOO_SMALL",
        ),
        _check(
            "det_score",
            score_val >= MIN_DET_SCORE,
            score_val,
            f">= {MIN_DET_SCORE}",
            "LOW_DET_SCORE",
        ),
    ]


def check_face_pose(kps: np.ndarray) -> list[dict]:
    """Ước tính roll/yaw/pitch từ 5 keypoints và kiểm tra giới hạn.

    kps thứ tự: mắt trái (0), mắt phải (1), mũi (2),
                khoé miệng trái (3), khoé miệng phải (4).

    Nếu 2 mắt trùng nhau → yaw/pitch = NaN, check FAIL (không crash).
    Giá trị NaN được đổi thành None trong dict (JSON-safe).

    Returns
    -------
    list[dict] mỗi phần tử {"name", "passed", "value", "rule", "code"}
    """
    pts = np.asarray(kps, dtype=float)   # (5, 2)

    le = pts[0]   # mắt trái
    re = pts[1]   # mắt phải
    nose = pts[2]
    ml = pts[3]   # khoé miệng trái
    mr = pts[4]   # khoé miệng phải

    # ---- roll -------------------------------------------------------
    dx = re[0] - le[0]
    dy = re[1] - le[1]
    roll_deg = math.degrees(math.atan2(dy, dx))

    # ---- xoay ngược -roll quanh trung điểm 2 mắt -------------------
    eye_dist = math.hypot(dx, dy)

    if eye_dist < 1e-6:
        # Hai mắt trùng → yaw/pitch không xác định
        nan_val: Optional[float] = None
        return [
            _check("roll", abs(roll_deg) <= MAX_ROLL_DEG,
                   _nan_to_none(roll_deg), f"|roll| <= {MAX_ROLL_DEG}°", "POSE_ROLL"),
            _check("yaw",  False, nan_val,
                   f"|yaw| <= {MAX_YAW_DEG}°", "POSE_YAW"),
            _check("pitch", False, nan_val,
                   f"{MIN_PITCH_RATIO} <= pitch <= {MAX_PITCH_RATIO}", "POSE_PITCH"),
        ]

    mid = (le + re) / 2.0
    cos_r = math.cos(math.radians(-roll_deg))
    sin_r = math.sin(math.radians(-roll_deg))

    def _rotate(p: np.ndarray) -> np.ndarray:
        d = p - mid
        return np.array([
            cos_r * d[0] - sin_r * d[1],
            sin_r * d[0] + cos_r * d[1],
        ])

    r_le   = _rotate(le)
    r_re   = _rotate(re)
    r_nose = _rotate(nose)
    r_ml   = _rotate(ml)
    r_mr   = _rotate(mr)

    # ---- yaw --------------------------------------------------------
    r_eye_span = r_re[0] - r_le[0]           # > 0 sau khi xoay thẳng
    if abs(r_eye_span) < 1e-6:
        yaw_deg_val: Optional[float] = None
        yaw_passed = False
    else:
        yaw_ratio = r_nose[0] / r_eye_span
        yaw_deg_val = math.degrees(math.atan(2.0 * yaw_ratio))
        yaw_passed = abs(yaw_deg_val) <= MAX_YAW_DEG

    # ---- pitch ------------------------------------------------------
    mouth_mid_y = (r_ml[1] + r_mr[1]) / 2.0
    if abs(mouth_mid_y) < 1e-6:
        pitch_val: Optional[float] = None
        pitch_passed = False
    else:
        pitch_val = r_nose[1] / mouth_mid_y
        pitch_passed = MIN_PITCH_RATIO <= pitch_val <= MAX_PITCH_RATIO

    return [
        _check("roll",  abs(roll_deg) <= MAX_ROLL_DEG,
               _nan_to_none(roll_deg),  f"|roll| <= {MAX_ROLL_DEG}°",   "POSE_ROLL"),
        _check("yaw",   yaw_passed,
               _nan_to_none(yaw_deg_val),  f"|yaw| <= {MAX_YAW_DEG}°",  "POSE_YAW"),
        _check("pitch", pitch_passed,
               _nan_to_none(pitch_val), f"{MIN_PITCH_RATIO} <= pitch <= {MAX_PITCH_RATIO}", "POSE_PITCH"),
    ]


def _nan_to_none(v: Optional[float]) -> Optional[float]:
    """Đổi float NaN → None để JSON serialize được."""
    if v is None:
        return None
    try:
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None
