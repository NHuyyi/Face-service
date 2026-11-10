"""Shared byte decoder for API uploads and offline evaluation."""
from pathlib import Path

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode BGR pixels, applying EXIF (including reflection) exactly once.

    OpenCV IMREAD_COLOR honors orientation; do not additionally transpose EXIF
    or use IMREAD_UNCHANGED / IMREAD_IGNORE_ORIENTATION here.
    """
    if not data:
        raise ValueError("File ảnh rỗng")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Không đọc được ảnh (định dạng không hỗ trợ)") from exc
    if img is None:
        raise ValueError("Không đọc được ảnh (định dạng không hỗ trợ)")
    return img


def read_image(path: str | Path) -> np.ndarray:
    """Use the upload decoder for local files too (including Unicode paths)."""
    return decode_image(Path(path).read_bytes())
