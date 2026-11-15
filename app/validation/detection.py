"""Phát hiện khuôn mặt bằng RetinaFace ONNX có sẵn trong dự án."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import Settings
from ..engine.retinaface import RetinaFaceDetector
from ..engine.types import DetectedFace
from .quality import ImageQualityError, _gray


@lru_cache(maxsize=1)
def _default_detector() -> RetinaFaceDetector:
    settings = Settings()
    return RetinaFaceDetector(
        str(Path(settings.models_dir) / settings.retinaface_model),
        [provider.strip() for provider in settings.providers.split(",") if provider.strip()],
        settings.det_score_threshold, settings.det_nms_threshold, settings.det_input_size,
    )


def detect_face(image: np.ndarray, detector: RetinaFaceDetector | None = None) -> list[DetectedFace]:
    """Trả tất cả mặt phát hiện; báo ImageQualityError khi không thấy mặt.

    Mặc định tải RetinaFace một lần khi gọi lần đầu, không tải ArcFace.
    Có thể truyền detector đã khởi tạo để tái sử dụng session ONNX.
    Bbox/landmark thuộc ảnh đầu vào; hàm không thử xoay ảnh.
    """
    _gray(image)  # Validate BGR shape/dtype before calling ONNX.
    faces = (detector if detector is not None else _default_detector()).detect(image)
    if not faces:
        raise ImageQualityError("Không phát hiện được khuôn mặt trong ảnh")
    return faces
