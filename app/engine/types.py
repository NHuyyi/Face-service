from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: np.ndarray       # (4,) x1,y1,x2,y2 theo toạ độ ảnh gốc
    score: float
    kps: np.ndarray        # (5,2): mắt trái, mắt phải, mũi, khoé miệng trái, khoé miệng phải

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))

    @property
    def size(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))
