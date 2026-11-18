"""Quality gates for decoded uint8 BGR images, before ArcFace inference."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ..engine.types import DetectedFace


from .image import validate_image
from .geometry import check_face_size, check_face_pose
from ..engine.align import align_face

class ImageQualityError(ValueError):
    """Readable image that does not meet face or image quality requirements."""


def check_face_count(faces: Sequence[DetectedFace]) -> int:
    """Require exactly one detected face, including detections below min size."""
    count = len(faces)
    if count != 1:
        raise ImageQualityError(f"Ảnh phải có đúng 1 khuôn mặt (phát hiện {count})")
    return count


def _gray(image: np.ndarray) -> np.ndarray:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ImageQualityError("Ảnh phải là mảng BGR uint8 không rỗng")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def check_blur(image: np.ndarray, threshold: float = 100.0) -> float:
    """Return Laplacian variance; reject scores below threshold (whole image)."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Ngưỡng độ mờ phải hữu hạn và >= 0")
    score = float(cv2.Laplacian(_gray(image), cv2.CV_64F).var())
    if score < threshold:
        raise ImageQualityError(f"Ảnh bị mờ (độ sắc nét {score:.2f} < {threshold:g})")
    return score


def check_brightness(image: np.ndarray, minimum: float = 40.0,
                     maximum: float = 220.0) -> float:
    """Return mean grayscale intensity; accept inclusive bounds on [0, 255]."""
    if not (0 <= minimum <= maximum <= 255):
        raise ValueError("Ngưỡng độ sáng phải thỏa 0 <= min <= max <= 255")
    score = float(cv2.mean(_gray(image))[0])
    if score < minimum:
        raise ImageQualityError(f"Ảnh quá tối (độ sáng {score:.2f} < {minimum:g})")
    if score > maximum:
        raise ImageQualityError(f"Ảnh quá sáng (độ sáng {score:.2f} > {maximum:g})")
    return score


# ---------------------------------------------------------------------------
# Điều phối tất cả kiểm tra — không gọi ArcFace, detector truyền qua tham số
# ---------------------------------------------------------------------------

def validate_face_quality(data: bytes, detector) -> dict[str, Any]:
    """Điều phối toàn bộ pipeline kiểm tra chất lượng ảnh khuôn mặt.

    Parameters
    ----------
    data:     raw bytes của file ảnh upload
    detector: instance detector (RetinaFaceDetector / ScrfdDetector) — không tải mô hình mới

    Returns
    -------
    dict với các khoá:
      "passed"  : bool
      "code"    : "OK" hoặc mã lỗi đầu tiên
      "reasons" : list mã của mọi check trượt
      "checks"  : list tất cả check dict
      "face"    : dict bbox/det_score/landmarks/pose hoặc None nếu fail sớm
      "aligned" : np.ndarray 112x112 BGR hoặc None
    """
    # Trả về sớm khi ảnh không decode được hoặc không tìm thấy mặt
    def _early_fail(code: str, message: str) -> dict[str, Any]:
        return {
            "passed": False,
            "code": code,
            "reasons": [code],
            "checks": [{"name": code.lower(), "passed": False,
                        "value": None, "rule": message, "code": code}],
            "face": None,
            "aligned": None,
        }

    # --- Bước 1: decode ảnh ---
    try:
        img = validate_image(data)
    except ValueError as exc:
        return _early_fail("INVALID_IMAGE", str(exc))

    # --- Bước 2: phát hiện mặt ---
    # Import cục bộ để tránh circular import (detection.py → quality.py → detection.py)
    from .detection import detect_face  # noqa: PLC0415
    try:
        faces = detect_face(img, detector)
    except ImageQualityError as exc:
        return _early_fail("NO_FACE", str(exc))

    # --- Bước 3: đếm mặt ---
    try:
        check_face_count(faces)
    except ImageQualityError as exc:
        # detect_face đã báo NO_FACE khi rỗng → tới đây chắc chắn là >1 mặt.
        return _early_fail("MULTIPLE_FACES", str(exc))

    face = faces[0]
    kps: np.ndarray = face.kps          # (5,2)
    bbox: np.ndarray = face.bbox        # (4,)
    det_score: float = face.score

    # --- Bước 4-7: gom tất cả check, không dừng giữa chừng ---
    all_checks: list[dict[str, Any]] = []

    # 4. Kích thước
    all_checks.extend(check_face_size(bbox, kps, det_score))

    # 5. Tư thế
    pose_checks = check_face_pose(kps)
    all_checks.extend(pose_checks)

    # Trích pose values để đưa vào face dict
    pose_vals: dict[str, Any] = {}
    for pc in pose_checks:
        if pc["name"] == "roll":
            pose_vals["roll_deg"] = pc["value"]
        elif pc["name"] == "yaw":
            pose_vals["yaw_deg"] = pc["value"]
        elif pc["name"] == "pitch":
            pose_vals["pitch_ratio"] = pc["value"]

    # 6. Align mặt → ảnh 112x112
    try:
        aligned = align_face(img, kps, size=112)
    except Exception:
        aligned = None

    # 7. Blur + Brightness trên ảnh aligned (hoặc ảnh gốc nếu align lỗi)
    check_img = aligned if aligned is not None else img
    # Lấy giá trị thật bằng ngưỡng "không bao giờ fail", rồi tự so với ngưỡng mặc định
    # → giá trị chính xác, không phải parse message lỗi.
    blur_min, bright_min, bright_max = 100.0, 40.0, 220.0
    blur_val = check_blur(check_img, threshold=0.0)
    bright_val = check_brightness(check_img, minimum=0.0, maximum=255.0)
    all_checks.append({
        "name": "blur", "passed": blur_val >= blur_min, "value": blur_val,
        "rule": f">= {blur_min:g} (Laplacian var)", "code": "BLURRY",
    })
    all_checks.append({
        "name": "brightness", "passed": bright_min <= bright_val <= bright_max,
        "value": bright_val, "rule": f"{bright_min:g} <= brightness <= {bright_max:g}",
        "code": "BAD_BRIGHTNESS",
    })

    # --- Tổng hợp kết quả ---
    reasons = [c["code"] for c in all_checks if not c["passed"]]
    passed = len(reasons) == 0
    first_code = reasons[0] if reasons else "OK"

    return {
        "passed": passed,
        "code": first_code,
        "reasons": reasons,
        "checks": all_checks,
        "face": {
            "bbox": [float(x) for x in bbox],
            "det_score": det_score,
            "landmarks": [[float(x), float(y)] for x, y in kps],
            "pose": pose_vals,
        },
        "aligned": aligned,
    }
