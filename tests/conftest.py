import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.engine.align import ARCFACE_TEMPLATE
from app.engine.engine import FaceEngine
from app.engine.types import DetectedFace


@pytest.fixture
def asymmetric_image():
    img = np.zeros((160, 240, 3), np.uint8)
    img[:80, :120] = (20, 60, 220)
    img[:80, 120:] = (70, 180, 30)
    img[80:, :120] = (200, 30, 90)
    img[80:, 120:] = (80, 130, 230)
    return img


@pytest.fixture
def jpeg_with_orientation(asymmetric_image):
    """Real JPEG + minimal EXIF APP1; no Pillow or model dependency."""
    ok, encoded = cv2.imencode('.jpg', asymmetric_image)
    assert ok
    jpeg = encoded.tobytes()
    raw = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    def make(orientation=None):
        if orientation is None:
            return jpeg, raw.copy()
        tiff = (b'II' + struct.pack('<HIH', 42, 8, 1)
                + struct.pack('<HHIHHI', 0x0112, 3, 1, orientation, 0, 0))
        payload = b'Exif\x00\x00' + tiff
        tagged = jpeg[:2] + b'\xff\xe1' + struct.pack('>H', len(payload) + 2) + payload + jpeg[2:]
        # Independent pixel expectations for all eight EXIF orientations.
        expected = {
            1: raw, 2: raw[:, ::-1], 3: raw[::-1, ::-1], 4: raw[::-1],
            5: raw.transpose(1, 0, 2), 6: np.rot90(raw, -1),
            7: raw.transpose(1, 0, 2)[::-1, ::-1], 8: np.rot90(raw, 1),
        }[orientation]
        return tagged, expected.copy()

    return make


@pytest.fixture
def face_factory():
    def make(bbox=(20, 20, 110, 120), score=0.9, kps=None):
        return DetectedFace(np.array(bbox, np.float32), score,
                            ARCFACE_TEMPLATE.copy() if kps is None else kps)
    return make


@pytest.fixture
def engine_factory():
    def make(detect):
        class FakeDetector:
            def __init__(self):
                self.seen = []

            def detect(self, img):
                self.seen.append(img.copy())
                return detect(img)

        class FakeRecognizer:
            def __init__(self):
                self.seen = []

            def embed(self, aligned):
                self.seen.append(aligned.copy())
                return np.full(512, 1 / np.sqrt(512), np.float32)

        engine = FaceEngine.__new__(FaceEngine)
        engine.s = Settings(min_face_size=40)
        engine.detector, engine.recognizer = FakeDetector(), FakeRecognizer()
        return engine
    return make
