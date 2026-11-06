"""RetinaFace detector (ONNX) – bản export từ biubug6/Pytorch_Retinaface.

Model đầu vào  : 1x3xHxW, BGR, trừ mean (104,117,123), không chia std.
Model đầu ra   : loc (1,N,4), conf (1,N,2 – đã softmax), landms (1,N,10).
Ba output được nhận diện theo kích thước chiều cuối nên không phụ thuộc thứ tự.
"""
from __future__ import annotations

from itertools import product
from math import ceil
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from .types import DetectedFace
from .utils import letterbox, nms

MEAN_BGR = (104.0, 117.0, 123.0)
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VARIANCES = (0.1, 0.2)


def make_priors(height: int, width: int) -> np.ndarray:
    """Sinh prior box (cx,cy,w,h) chuẩn hoá [0,1] – cùng thứ tự với PriorBox gốc."""
    anchors: List[float] = []
    for k, step in enumerate(STEPS):
        fh, fw = ceil(height / step), ceil(width / step)
        for i, j in product(range(fh), range(fw)):
            for ms in MIN_SIZES[k]:
                anchors += [(j + 0.5) * step / width, (i + 0.5) * step / height, ms / width, ms / height]
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 4)


def decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Trả về box x1y1x2y2 chuẩn hoá [0,1]."""
    centers = priors[:, :2] + loc[:, :2] * VARIANCES[0] * priors[:, 2:]
    wh = priors[:, 2:] * np.exp(loc[:, 2:] * VARIANCES[1])
    boxes = np.concatenate([centers - wh / 2, centers + wh / 2], axis=1)
    return boxes


def decode_landmarks(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Trả về (N,5,2) chuẩn hoá [0,1]."""
    pts = [priors[:, :2] + pre[:, 2 * k:2 * k + 2] * VARIANCES[0] * priors[:, 2:] for k in range(5)]
    return np.stack(pts, axis=1)


class RetinaFaceDetector:
    def __init__(self, model_path: str, providers: Sequence[str], score_thr: float = 0.5,
                 nms_thr: float = 0.4, input_size: int = 640):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        # Model export tĩnh -> dùng đúng kích thước đó; động -> dùng input_size (> 0)
        self.input_h = h if (isinstance(h, int) and h > 0) else input_size
        self.input_w = w if (isinstance(w, int) and w > 0) else input_size
        self.score_thr, self.nms_thr = score_thr, nms_thr
        self.priors = make_priors(self.input_h, self.input_w)

    def detect(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        if img_bgr is None or img_bgr.size == 0 or img_bgr.ndim != 3:
            return []
        canvas, scale = letterbox(img_bgr, (self.input_w, self.input_h), MEAN_BGR)
        blob = (canvas - np.asarray(MEAN_BGR, dtype=np.float32)).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.input_name: np.ascontiguousarray(blob, dtype=np.float32)})
        return self.postprocess(outs, scale)

    def postprocess(self, outs: Sequence[np.ndarray], scale: float) -> List[DetectedFace]:
        if not outs or scale <= 0:
            return []
        by_dim = {o.shape[-1]: o[0] for o in outs}
        if 4 not in by_dim or 2 not in by_dim or 10 not in by_dim:
            return []
        loc, conf, lm = by_dim[4], by_dim[2], by_dim[10]
        scores = conf[:, 1]
        mask = scores >= self.score_thr
        if not mask.any():
            return []
        priors = self.priors[mask]
        sc = scores[mask]
        wh = np.array([self.input_w, self.input_h] * 2, dtype=np.float32)
        boxes = decode_boxes(loc[mask], priors) * wh / scale
        kps = decode_landmarks(lm[mask], priors) * np.array([self.input_w, self.input_h], np.float32) / scale
        keep = nms(boxes, sc, self.nms_thr)
        return [DetectedFace(boxes[i], float(sc[i]), kps[i]) for i in keep]
