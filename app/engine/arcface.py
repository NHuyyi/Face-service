"""ArcFace (w600k_r50.onnx, InsightFace) -> vector 512 chiều đã L2-normalize."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from ..matching import EMBEDDING_DIM, l2_normalize


class ArcFaceRecognizer:
    def __init__(self, model_path: str, providers: Sequence[str]):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        # blobFromImage: (x - 127.5) / 127.5, swapRB=True (BGR -> RGB), NCHW
        blob = cv2.dnn.blobFromImage(aligned_bgr_112, 1.0 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        out = self.sess.run(None, {self.input_name: blob})[0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        if vec.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(f"Model trả về {vec.shape[0]} chiều, kỳ vọng {EMBEDDING_DIM}")
        return l2_normalize(vec)
