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
from .mobilefacenet import MOBILEFACENET_DIM, MobileFaceNetRecognizer
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
    embedding_192: Optional[np.ndarray] = None  # (192,) float32, L2-normalized từ MobileFaceNet
    detector_name: str = "retinaface"


class FaceEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.providers = [p.strip() for p in settings.providers.split(",") if p.strip()]
        d = settings.models_dir
        self._detectors = {}

        # Khởi tạo detector mặc định (retinaface hoặc scrfd)
        self.detector = self._init_detector(settings.detector_backend)

        # Bộ trích xuất ArcFace (512 chiều)
        self.recognizer = ArcFaceRecognizer(os.path.join(d, settings.arcface_model), self.providers)

        # Bộ trích xuất MobileFaceNet (192 chiều)
        mfn_file = os.path.join(d, getattr(settings, "mobilefacenet_model", "mobile_face_net.tflite"))
        if os.path.exists(mfn_file):
            self.mobilefacenet = MobileFaceNetRecognizer(mfn_file)
        else:
            self.mobilefacenet = None

    def _init_detector(self, name: str):
        name = (name or "retinaface").lower()
        if name in self._detectors:
            return self._detectors[name]
        d = self.s.models_dir
        if name == "scrfd":
            det = ScrfdDetector(os.path.join(d, self.s.scrfd_model), self.providers,
                                self.s.det_score_threshold, self.s.det_nms_threshold,
                                self.s.det_input_size)
        else:
            det = RetinaFaceDetector(os.path.join(d, self.s.retinaface_model), self.providers,
                                     self.s.det_score_threshold, self.s.det_nms_threshold,
                                     self.s.det_input_size)
        self._detectors[name] = det
        return det

    # -- phát hiện + xoay thẳng ảnh -----------------------------------------------------
    def _detect_upright(self, img: np.ndarray, detector=None) -> tuple[np.ndarray, List[DetectedFace], int]:
        """First size-valid orientation, clockwise relative to EXIF-decoded pixels.

        Detect on each rotated candidate, so boxes/landmarks already refer to
        that candidate. Each rotation starts from img, never the last candidate.
        """
        det = detector or getattr(self, "detector", None)
        if det is None:
            det = self._init_detector(self.s.detector_backend)

        for angle, code in _ROTATIONS:
            cand = img if code is None else cv2.rotate(img, code)
            faces = [f for f in det.detect(cand) if f.size >= self.s.min_face_size]
            if faces:
                return cand, faces, angle
        raise NoFaceError("Không phát hiện được khuôn mặt hợp lệ trong ảnh")

    @staticmethod
    def _pick_main(faces: List[DetectedFace]) -> DetectedFace:
        # Chọn theo diện tích nhân điểm phát hiện ở hướng đầu tiên hợp lệ.
        return max(faces, key=lambda f: f.area * f.score)

    def extract(self, img_bgr: np.ndarray, keep_aligned: bool = False,
                detector_backend: Optional[str] = None,
                extract_192: bool = False) -> FaceResult:
        det = self._init_detector(detector_backend) if detector_backend else getattr(self, "detector", None)
        det_name = (detector_backend or getattr(self.s, "detector_backend", "retinaface")).lower()

        upright, faces, angle = self._detect_upright(img_bgr, detector=det)
        face = self._pick_main(faces)
        aligned = align_face(upright, face.kps, 112)
        emb = self.recognizer.embed(aligned)

        emb_192 = None
        if extract_192 and getattr(self, "mobilefacenet", None) is not None:
            emb_192 = self.mobilefacenet.embed(aligned)

        return FaceResult(
            embedding=emb,
            bbox=[float(x) for x in face.bbox],
            landmarks=[[float(x), float(y)] for x, y in face.kps],
            det_score=face.score,
            face_count=len(faces),
            rotation_applied=angle,
            roll_degrees=roll_angle_degrees(face.kps),
            aligned=aligned if keep_aligned else None,
            embedding_192=emb_192,
            detector_name=det_name,
        )

    def extract_both(self, img_bgr: np.ndarray, detector_backend: Optional[str] = None,
                     keep_aligned: bool = False) -> FaceResult:
        """Trích xuất đồng thời cả 2 vector 512 (ArcFace) và 192 (MobileFaceNet)."""
        return self.extract(img_bgr, keep_aligned=keep_aligned, detector_backend=detector_backend, extract_192=True)
