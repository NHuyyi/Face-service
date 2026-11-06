"""SCRFD detector (det_10g.onnx trong gói InsightFace buffalo_l) – phương án dự phòng.

SCRFD là thế hệ kế nhiệm RetinaFace của cùng nhóm InsightFace; gói này tải được bằng
scripts/download_models.py nên tiện để chạy thử ngay.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import onnxruntime as ort

from .types import DetectedFace
from .utils import letterbox, nms

STRIDES = [8, 16, 32]
NUM_ANCHORS = 2


class ScrfdDetector:
    def __init__(self, model_path: str, providers: Sequence[str], score_thr: float = 0.5,
                 nms_thr: float = 0.4, input_size: int = 640):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        self.size = input_size
        self.score_thr, self.nms_thr = score_thr, nms_thr
        self._centers: dict = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            ys, xs = np.mgrid[:h, :w]
            c = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._centers[key] = np.repeat(c, NUM_ANCHORS, axis=0)
        return self._centers[key]

    def detect(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        canvas, scale = letterbox(img_bgr, (self.size, self.size), (127.5, 127.5, 127.5))
        rgb = canvas[:, :, ::-1]
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.input_name: np.ascontiguousarray(blob, dtype=np.float32)})
        return self.postprocess(outs, scale)

    def postprocess(self, outs: Sequence[np.ndarray], scale: float) -> List[DetectedFace]:
        fmc = len(STRIDES)
        all_boxes, all_scores, all_kps = [], [], []
        for idx, stride in enumerate(STRIDES):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + fmc].reshape(-1, 4) * stride
            kps = outs[idx + 2 * fmc].reshape(-1, 10) * stride
            fh = fw = self.size // stride
            c = self._anchor_centers(fh, fw, stride)
            sel = scores >= self.score_thr
            if not sel.any():
                continue
            c, bbox, kps, scores = c[sel], bbox[sel], kps[sel], scores[sel]
            boxes = np.concatenate([c - bbox[:, :2], c + bbox[:, 2:]], axis=1)
            pts = np.stack([c + kps[:, 2 * k:2 * k + 2] for k in range(5)], axis=1)
            all_boxes.append(boxes); all_scores.append(scores); all_kps.append(pts)
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale
        keep = nms(boxes, scores, self.nms_thr)
        return [DetectedFace(boxes[i], float(scores[i]), kps[i]) for i in keep]
