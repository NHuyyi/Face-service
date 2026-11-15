"""Quality gates for decoded uint8 BGR images, before ArcFace inference."""
from collections.abc import Sequence

import cv2
import numpy as np

from ..engine.types import DetectedFace


class ImageQualityError(ValueError):
    """Readable image that does not meet face or image quality requirements."""


def check_face_count(faces: Sequence[DetectedFace]) -> int:
    """Require exactly one detected face, including detections below min size."""
    count = int(np.asarray([face.bbox for face in faces]).shape[0])
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
