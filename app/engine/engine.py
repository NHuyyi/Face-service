"""Pipeline: ảnh -> (EXIF + tự xoay) -> RetinaFace -> căn thẳng -> ArcFace 512-d."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..config import Settings
from .align import align_face, roll_angle_degrees
from .arcface import ArcFaceRecognizer
from .retinaface import RetinaFaceDetector
from .scrfd import ScrfdDetector
from .types import DetectedFace

# Thử lần lượt các hướng xoay nếu không thấy mặt (ảnh chụp ngang/ngược, mất EXIF)
_ROTATIONS = [
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
]


class NoFaceError(Exception):
    pass


@dataclass
class FaceResult:
    embedding: np.ndarray       # (512,) float32, L2-normalized
    bbox: List[float]           # ảnh sau EXIF + xoay thử, trước crop/alignment
    landmarks: List[List[float]]
    det_score: float
    face_count: int
    rotation_applied: int       # 0/90/180/270 chiều kim đồng hồ, sau EXIF
    roll_degrees: float         # góc đường nối hai mắt trước alignment, sau xoay thử
    aligned: Optional[np.ndarray] = None  # ảnh mặt 112x112 (BGR)


class FaceEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        providers = [p.strip() for p in settings.providers.split(",") if p.strip()]
        d = settings.models_dir
        if settings.detector_backend == "scrfd":
            self.detector = ScrfdDetector(os.path.join(d, settings.scrfd_model), providers,
                                          settings.det_score_threshold, settings.det_nms_threshold,
                                          settings.det_input_size)
        else:
            self.detector = RetinaFaceDetector(os.path.join(d, settings.retinaface_model), providers,
                                               settings.det_score_threshold, settings.det_nms_threshold,
                                               settings.det_input_size)
        self.recognizer = ArcFaceRecognizer(os.path.join(d, settings.arcface_model), providers)

    # -- phát hiện + xoay thẳng ảnh -----------------------------------------------------
    def _detect_upright(self, img: np.ndarray) -> tuple[np.ndarray, List[DetectedFace], int]:
        """First size-valid orientation, clockwise relative to EXIF-decoded pixels.

        Detect on each rotated candidate, so boxes/landmarks already refer to
        that candidate. Each rotation starts from img, never the last candidate.
        """
        for angle, code in _ROTATIONS:
            cand = img if code is None else cv2.rotate(img, code)
            faces = [f for f in self.detector.detect(cand) if f.size >= self.s.min_face_size]
            if faces:
                return cand, faces, angle
        raise NoFaceError("Không phát hiện được khuôn mặt hợp lệ trong ảnh")

    @staticmethod
    def _pick_main(faces: List[DetectedFace]) -> DetectedFace:
        # Chọn theo diện tích nhân điểm phát hiện ở hướng đầu tiên hợp lệ.
        return max(faces, key=lambda f: f.area * f.score)

    def extract(self, img_bgr: np.ndarray, keep_aligned: bool = False) -> FaceResult:
        upright, faces, angle = self._detect_upright(img_bgr)
        face = self._pick_main(faces)
        aligned = align_face(upright, face.kps, 112)
        emb = self.recognizer.embed(aligned)
        return FaceResult(
            embedding=emb,
            bbox=[float(x) for x in face.bbox],
            landmarks=[[float(x), float(y)] for x, y in face.kps],
            det_score=face.score,
            face_count=len(faces),
            rotation_applied=angle,
            roll_degrees=roll_angle_degrees(face.kps),
            aligned=aligned if keep_aligned else None,
        )
