"""Cấu hình service, đọc từ biến môi trường (xem README)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    models_dir: str = field(default_factory=lambda: _env("MODELS_DIR", "models"))
    # "retinaface" (mặc định, theo yêu cầu) hoặc "scrfd" (det_10g.onnx trong gói buffalo_l)
    detector_backend: str = field(default_factory=lambda: _env("DETECTOR_BACKEND", "retinaface").lower())
    retinaface_model: str = field(default_factory=lambda: _env("RETINAFACE_MODEL", "retinaface.onnx"))
    scrfd_model: str = field(default_factory=lambda: _env("SCRFD_MODEL", "det_10g.onnx"))
    arcface_model: str = field(default_factory=lambda: _env("ARCFACE_MODEL", "w600k_r50.onnx"))
    mobilefacenet_model: str = field(default_factory=lambda: _env("MOBILEFACENET_MODEL", "mobile_face_net.tflite"))

    # onnxruntime providers, ví dụ "CUDAExecutionProvider,CPUExecutionProvider"
    providers: str = field(default_factory=lambda: _env("ORT_PROVIDERS", "CPUExecutionProvider"))

    det_score_threshold: float = field(default_factory=lambda: float(_env("DET_SCORE_THRESHOLD", "0.5")))
    det_nms_threshold: float = field(default_factory=lambda: float(_env("DET_NMS_THRESHOLD", "0.4")))
    det_input_size: int = field(default_factory=lambda: int(_env("DET_INPUT_SIZE", "640")))
    min_face_size: int = field(default_factory=lambda: int(_env("MIN_FACE_SIZE", "40")))

    # Ngưỡng cosine similarity cho ArcFace w600k_r50. Hãy hiệu chỉnh bằng scripts/evaluate.py
    match_threshold: float = field(default_factory=lambda: float(_env("MATCH_THRESHOLD", "0.40")))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/faces.db"))
    # >0: tự xoá vector sau N giây (lưu "tạm thời"); 0: giữ đến khi xoá thủ công
    vector_ttl_seconds: int = field(default_factory=lambda: int(_env("VECTOR_TTL_SECONDS", "0")))

    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "10")))
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))  # rỗng = không bắt buộc

    # --- Ngưỡng kiểm tra chất lượng khuôn mặt (geometry + pose) ---
    min_face_px: int = field(default_factory=lambda: int(_env("MIN_FACE_PX", "64")))
    min_eye_dist_px: float = field(default_factory=lambda: float(_env("MIN_EYE_DIST_PX", "20")))
    min_det_score: float = field(default_factory=lambda: float(_env("MIN_DET_SCORE", "0.6")))
    max_roll_deg: float = field(default_factory=lambda: float(_env("MAX_ROLL_DEG", "45")))
    max_yaw_deg: float = field(default_factory=lambda: float(_env("MAX_YAW_DEG", "30")))
    min_pitch_ratio: float = field(default_factory=lambda: float(_env("MIN_PITCH_RATIO", "0.30")))
    max_pitch_ratio: float = field(default_factory=lambda: float(_env("MAX_PITCH_RATIO", "0.72")))
