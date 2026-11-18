"""Integration tests cho POST /api/v1/faces/validate — không cần model ONNX thật.

Detector giả (FakeDetector) trả về kết quả cài sẵn theo kịch bản kiểm thử.
Toàn bộ pipeline validate_face_quality chạy thật; chỉ bước ONNX inference bị mock.
"""
from __future__ import annotations

import io
import math
import struct
from typing import List, Optional

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db import FaceStore  # noqa: E402
from app.engine import FaceResult, NoFaceError  # noqa: E402
from app.engine.align import ARCFACE_TEMPLATE  # noqa: E402
from app.engine.types import DetectedFace  # noqa: E402
from app.main import create_app  # noqa: E402
from app.matching import l2_normalize  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — tạo ảnh PNG bytes tổng hợp
# ---------------------------------------------------------------------------

TEMPLATE = ARCFACE_TEMPLATE  # (5,2) ArcFace chuẩn 112x112


def _png(value: int = 128, size: int = 160) -> bytes:
    """Ảnh màu đồng nhất BGR=value, kích thước size×size."""
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _blurry_png(size: int = 160) -> bytes:
    """Ảnh hoàn toàn đồng nhất (Laplacian variance = 0 → chắc chắn BLURRY)."""
    return _png(128, size)


def _dark_png() -> bytes:
    """Ảnh tối (giá trị pixel = 5 < ngưỡng 40)."""
    return _png(5)


def _bright_png() -> bytes:
    """Ảnh sáng (giá trị pixel = 250 > ngưỡng 220)."""
    return _png(250)


def _sharp_png(size: int = 160) -> bytes:
    """Ảnh chessboard sắc nét, Laplacian variance cao."""
    board = ((np.indices((size, size)).sum(axis=0) // 4 % 2) * 150 + 50).astype(np.uint8)
    bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def _mid_brightness_sharp_png(size: int = 160) -> bytes:
    """Ảnh sắc nét, độ sáng trung bình (~100), dùng cho ảnh PASS."""
    board = ((np.indices((size, size)).sum(axis=0) // 4 % 2) * 60 + 70).astype(np.uint8)
    bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Fake engine — dùng chung cho các API khác (không ảnh hưởng /validate)
# ---------------------------------------------------------------------------

class FakeEngine:
    """Engine giả: ảnh trắng = không mặt, còn lại trả embedding random."""

    def extract(self, img, keep_aligned=False):
        m = float(img.mean())
        if m > 250:
            raise NoFaceError("none")
        rng = np.random.RandomState(int(m))
        v = l2_normalize(rng.randn(512))
        return FaceResult(v, [0.0, 0.0, 1.0, 1.0], [[0.0, 0.0]] * 5, 0.99, 1, 0, 0.0)


# ---------------------------------------------------------------------------
# Fake detector — điều khiển kết quả detect() trong /validate pipeline
# ---------------------------------------------------------------------------

def _scale_kps(template: np.ndarray, scale: float = 1.0,
               offset: float = 0.0) -> np.ndarray:
    """Scale template keypoints cho các kịch bản bbox."""
    return (template * scale + offset).astype(np.float32)


class FakeDetector:
    """Detector mock; faces là list[DetectedFace] trả về khi gọi detect()."""

    def __init__(self, faces: List[DetectedFace]):
        self._faces = faces

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        return list(self._faces)


def _make_face(
    bbox=(10, 10, 110, 110),
    score: float = 0.95,
    kps: Optional[np.ndarray] = None,
) -> DetectedFace:
    if kps is None:
        # Scale template vào ảnh 160×160 (template gốc 112×112, scale ~1.4)
        kps = (TEMPLATE * 1.4).astype(np.float32)
    return DetectedFace(np.array(bbox, np.float32), score, kps)


# ---------------------------------------------------------------------------
# Fake engine với detector có thể thay thế, dùng cho /validate
# ---------------------------------------------------------------------------

class FakeEngineWithDetector:
    """FakeEngine nhưng để app.state.engine.detector trỏ vào FakeDetector."""

    def __init__(self, detector: FakeDetector):
        self.detector = detector

    def extract(self, img, keep_aligned=False):
        m = float(img.mean())
        if m > 250:
            raise NoFaceError("none")
        rng = np.random.RandomState(int(m))
        v = l2_normalize(rng.randn(512))
        return FaceResult(v, [0.0, 0.0, 1.0, 1.0], [[0.0, 0.0]] * 5, 0.99, 1, 0, 0.0)


def _client_with_detector(faces: List[DetectedFace]) -> TestClient:
    det = FakeDetector(faces)
    eng = FakeEngineWithDetector(det)
    app = create_app(Settings(), eng, FaceStore(":memory:"))
    # Dùng context manager để lifespan chạy và app.state được khởi tạo
    client = TestClient(app)
    client.__enter__()
    return client


# ---------------------------------------------------------------------------
# Helpers gọi API
# ---------------------------------------------------------------------------

def _post_validate(client: TestClient, data: bytes,
                   filename: str = "face.png",
                   content_type: str = "image/png",
                   return_aligned: bool = False) -> tuple[int, dict]:
    files = {"file": (filename, io.BytesIO(data), content_type)}
    form = {}
    if return_aligned:
        form["return_aligned"] = "true"
    r = client.post("/api/v1/faces/validate", files=files, data=form)
    return r.status_code, r.json()


# ---------------------------------------------------------------------------
# 1. Ảnh PASS — đủ tiêu chuẩn
# ---------------------------------------------------------------------------

class TestValidatePass:
    """Ảnh hợp lệ: 1 mặt, kích thước đủ, pose thẳng, sắc nét, độ sáng vừa."""

    def setup_method(self):
        # Tạo ảnh sắc nét, độ sáng trung bình
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(
            self.client, _mid_brightness_sharp_png()
        )

    def test_http_200(self):
        assert self.status == 200

    def test_success_true(self):
        assert self.body["success"] is True

    def test_code_ok(self):
        assert self.body["code"] == "OK"

    def test_no_reasons(self):
        assert self.body["quality"]["reasons"] == []

    def test_quality_passed(self):
        assert self.body["quality"]["passed"] is True

    def test_face_data_present(self):
        assert self.body["data"] is not None
        face = self.body["data"]["face"]
        assert "bbox" in face
        assert "det_score" in face
        assert "landmarks" in face
        assert "pose" in face

    def test_all_checks_passed(self):
        for c in self.body["quality"]["checks"]:
            assert c["passed"] is True, f"Check '{c['name']}' không PASS: {c}"

    def test_response_structure(self):
        """Đảm bảo đủ 5 trường cấp cao."""
        for key in ("success", "code", "message", "data", "quality"):
            assert key in self.body, f"Thiếu trường '{key}'"

    def test_message_ok(self):
        assert "đạt yêu cầu" in self.body["message"].lower() or self.body["message"] != ""


# ---------------------------------------------------------------------------
# 2. Không phát hiện khuôn mặt
# ---------------------------------------------------------------------------

class TestValidateNoFace:
    def setup_method(self):
        # Detector trả list rỗng → detect_face raises ImageQualityError
        self.client = _client_with_detector([])
        self.status, self.body = _post_validate(self.client, _png())

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_no_face(self):
        assert self.body["code"] == "NO_FACE"

    def test_data_none(self):
        assert self.body["data"] is None

    def test_reasons_contain_no_face(self):
        assert "NO_FACE" in self.body["quality"]["reasons"]


# ---------------------------------------------------------------------------
# 3. Nhiều khuôn mặt
# ---------------------------------------------------------------------------

class TestValidateMultipleFaces:
    def setup_method(self):
        faces = [
            _make_face(bbox=(10, 10, 80, 80)),
            _make_face(bbox=(90, 10, 160, 80)),
        ]
        self.client = _client_with_detector(faces)
        self.status, self.body = _post_validate(self.client, _png())

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_multiple_faces(self):
        assert self.body["code"] == "MULTIPLE_FACES"

    def test_data_none(self):
        assert self.body["data"] is None

    def test_message_mentions_multiple(self):
        assert "1" in self.body["message"] or "nhiều" in self.body["message"]


# ---------------------------------------------------------------------------
# 4. Ảnh bị mờ (blur < threshold)
# ---------------------------------------------------------------------------

class TestValidateBlurry:
    """Ảnh hoàn toàn đồng nhất → Laplacian variance = 0 → BLURRY."""

    def setup_method(self):
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        self.client = _client_with_detector([face])
        # _blurry_png tạo ảnh đồng nhất, nhưng độ sáng 128 (trung bình)
        # Laplacian variance ≈ 0 → chắc chắn fail blur
        # Tuy nhiên align_face sẽ warp ảnh → ảnh aligned vẫn uniform → var=0
        self.status, self.body = _post_validate(self.client, _blurry_png())

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_blurry(self):
        assert self.body["code"] == "BLURRY"

    def test_reasons_contain_blurry(self):
        assert "BLURRY" in self.body["quality"]["reasons"]

    def test_blur_check_failed(self):
        checks = {c["name"]: c for c in self.body["quality"]["checks"]}
        assert "blur" in checks
        assert checks["blur"]["passed"] is False


# ---------------------------------------------------------------------------
# 5. Ảnh quá tối
# ---------------------------------------------------------------------------

class TestValidateDark:
    def setup_method(self):
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(self.client, _dark_png())

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_bad_brightness(self):
        # Ảnh đồng nhất màu tối: vừa BLURRY (Laplacian var=0) vừa BAD_BRIGHTNESS.
        # Pipeline báo lỗi đầu tiên theo thứ tự checks (blur trước brightness),
        # nên code có thể là BLURRY hoặc BAD_BRIGHTNESS tuỳ ảnh.
        # Điều quan trọng là BAD_BRIGHTNESS có trong reasons.
        assert self.body["code"] in ("BAD_BRIGHTNESS", "BLURRY")

    def test_message_mentions_dark(self):
        msg = self.body["message"].lower()
        assert "tối" in msg or "sáng" in msg

    def test_reasons_contain_bad_brightness(self):
        assert "BAD_BRIGHTNESS" in self.body["quality"]["reasons"]


# ---------------------------------------------------------------------------
# 6. Ảnh quá sáng
# ---------------------------------------------------------------------------

class TestValidateBright:
    def setup_method(self):
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(self.client, _bright_png())

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_bad_brightness(self):
        # Ảnh đồng nhất màu tối: vừa BLURRY (Laplacian var=0) vừa BAD_BRIGHTNESS.
        # Pipeline báo lỗi đầu tiên theo thứ tự checks (blur trước brightness),
        # nên code có thể là BLURRY hoặc BAD_BRIGHTNESS tuỳ ảnh.
        # Điều quan trọng là BAD_BRIGHTNESS có trong reasons.
        assert self.body["code"] in ("BAD_BRIGHTNESS", "BLURRY")

    def test_reasons_contain_bad_brightness(self):
        assert "BAD_BRIGHTNESS" in self.body["quality"]["reasons"]


# ---------------------------------------------------------------------------
# 7. Khuôn mặt quá nhỏ (bbox < 64px)
# ---------------------------------------------------------------------------

class TestValidateFaceTooSmall:
    def setup_method(self):
        # bbox 50×50 → min_side=50 < MIN_FACE_PX=64 → FACE_TOO_SMALL
        face = _make_face(bbox=(10, 10, 60, 60), score=0.95)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(
            self.client, _mid_brightness_sharp_png()
        )

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_code_face_too_small(self):
        assert self.body["code"] == "FACE_TOO_SMALL"

    def test_reasons_contain_face_too_small(self):
        assert "FACE_TOO_SMALL" in self.body["quality"]["reasons"]

    def test_face_data_present(self):
        """face dict vẫn trả về dù fail size (fail sau bước 3)."""
        assert self.body["data"] is not None
        assert "face" in self.body["data"]


# ---------------------------------------------------------------------------
# 8. Khuôn mặt sai tư thế (yaw quá lớn)
# ---------------------------------------------------------------------------

class TestValidateBadPoseYaw:
    """Dịch mũi xa sang phải → yaw lớn → POSE_YAW fail."""

    def setup_method(self):
        kps = (TEMPLATE * 1.4).copy()
        kps[2, 0] += 25.0  # dịch mũi sang phải mạnh → yaw > 30°
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95, kps=kps)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(
            self.client, _mid_brightness_sharp_png()
        )

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_pose_yaw_in_reasons(self):
        assert "POSE_YAW" in self.body["quality"]["reasons"]

    def test_yaw_check_failed(self):
        checks = {c["name"]: c for c in self.body["quality"]["checks"]}
        assert "yaw" in checks
        assert checks["yaw"]["passed"] is False
        assert checks["yaw"]["code"] == "POSE_YAW"


# ---------------------------------------------------------------------------
# 9. Khuôn mặt sai tư thế (roll quá lớn)
# ---------------------------------------------------------------------------

class TestValidateBadPoseRoll:
    """Xoay keypoints 60° → |roll| > 45° → POSE_ROLL fail."""

    def setup_method(self):
        kps_base = (TEMPLATE * 1.4).copy()
        center = kps_base.mean(axis=0)
        angle = math.radians(60.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        d = kps_base - center
        rotated = np.stack(
            [cos_a * d[:, 0] - sin_a * d[:, 1],
             sin_a * d[:, 0] + cos_a * d[:, 1]], axis=1
        ) + center
        kps = rotated.astype(np.float32)
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95, kps=kps)
        self.client = _client_with_detector([face])
        self.status, self.body = _post_validate(
            self.client, _mid_brightness_sharp_png()
        )

    def test_http_422(self):
        assert self.status == 422

    def test_success_false(self):
        assert self.body["success"] is False

    def test_pose_roll_in_reasons(self):
        assert "POSE_ROLL" in self.body["quality"]["reasons"]


# ---------------------------------------------------------------------------
# 10. File ảnh lỗi / không đọc được
# ---------------------------------------------------------------------------

class TestValidateInvalidFile:
    def setup_method(self):
        # Dùng detector trả 1 mặt; ảnh lỗi sẽ bị bắt ở validate_image trước khi detect
        face = _make_face()
        self.client = _client_with_detector([face])

    def test_garbage_bytes(self):
        status, body = _post_validate(self.client, b"\x00\x01\x02\x03corrupt")
        assert status == 400
        assert body["success"] is False
        assert body["code"] == "INVALID_IMAGE"

    def test_empty_file(self):
        status, body = _post_validate(self.client, b"")
        assert status == 400
        assert body["success"] is False
        # File rỗng bị bắt bởi main.py trước khi vào validate_face_quality
        assert body["code"] == "INVALID_IMAGE"

    def test_data_none_on_invalid(self):
        _, body = _post_validate(self.client, b"garbage")
        assert body["data"] is None


# ---------------------------------------------------------------------------
# 11. Thiếu trường upload (không có field "file")
# ---------------------------------------------------------------------------

class TestValidateMissingField:
    def setup_method(self):
        self.client = _client_with_detector([_make_face()])

    def test_no_file_field(self):
        r = self.client.post("/api/v1/faces/validate")
        # FastAPI trả 422 Unprocessable Entity khi thiếu required field
        assert r.status_code == 422

    def test_no_file_field_json(self):
        r = self.client.post("/api/v1/faces/validate")
        body = r.json()
        # FastAPI default validation error response
        assert "detail" in body


# ---------------------------------------------------------------------------
# 12. File không đúng định dạng (txt, pdf giả)
# ---------------------------------------------------------------------------

class TestValidateWrongFormat:
    def setup_method(self):
        self.client = _client_with_detector([_make_face()])

    def test_text_file(self):
        status, body = _post_validate(
            self.client, b"hello world", filename="test.txt",
            content_type="text/plain"
        )
        assert status == 400
        assert body["success"] is False
        assert body["code"] == "INVALID_IMAGE"

    def test_fake_pdf(self):
        status, body = _post_validate(
            self.client, b"%PDF-1.4 fake", filename="fake.pdf",
            content_type="application/pdf"
        )
        assert status == 400
        assert body["success"] is False
        assert body["code"] == "INVALID_IMAGE"


# ---------------------------------------------------------------------------
# 13. PASS không được tiếp tục sang ArcFace embedding
# ---------------------------------------------------------------------------

class TestValidateDoesNotCallArcFace:
    """validate endpoint không được gọi engine.extract() (ArcFace inference)."""

    def test_extract_never_called(self):
        """
        Kiểm tra /validate không gọi engine.extract().
        FakeEngineWithDetector.extract() ghi nhận lời gọi; nếu được gọi → FAIL.
        """
        call_log = []

        class TrackingEngine(FakeEngineWithDetector):
            def extract(self, img, keep_aligned=False):
                call_log.append(img)
                return super().extract(img, keep_aligned)

        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        det = FakeDetector([face])
        eng = TrackingEngine(det)
        app = create_app(Settings(), eng, FaceStore(":memory:"))
        with TestClient(app) as c:
            files = {"file": ("face.png", io.BytesIO(_mid_brightness_sharp_png()), "image/png")}
            c.post("/api/v1/faces/validate", files=files)

        assert call_log == [], (
            f"engine.extract() bị gọi {len(call_log)} lần — /validate không được tạo embedding!"
        )


# ---------------------------------------------------------------------------
# 14. Fail sớm không tiếp tục kiểm tra sau
# ---------------------------------------------------------------------------

class TestValidateEarlyFailDoesNotRunQualityChecks:
    """Khi NO_FACE, checks chỉ có 1 entry và không có size/pose/blur/brightness."""

    def test_no_face_has_no_quality_checks(self):
        client = _client_with_detector([])
        status, body = _post_validate(client, _png())
        assert status == 422
        checks = body["quality"]["checks"]
        # Fail sớm: chỉ có 1 check entry (no_face)
        assert len(checks) == 1
        assert checks[0]["code"] == "NO_FACE"

    def test_multiple_faces_has_no_quality_checks(self):
        faces = [_make_face(bbox=(10, 10, 80, 80)), _make_face(bbox=(90, 10, 160, 80))]
        client = _client_with_detector(faces)
        status, body = _post_validate(client, _png())
        assert status == 422
        checks = body["quality"]["checks"]
        assert len(checks) == 1
        assert checks[0]["code"] == "MULTIPLE_FACES"


# ---------------------------------------------------------------------------
# 15. return_aligned=True → trả aligned_jpeg_b64 khi PASS
# ---------------------------------------------------------------------------

class TestValidateReturnAligned:
    def test_pass_returns_aligned_b64(self):
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        client = _client_with_detector([face])
        status, body = _post_validate(
            client, _mid_brightness_sharp_png(), return_aligned=True
        )
        assert status == 200
        assert body["success"] is True
        # aligned_jpeg_b64 phải có và không rỗng
        assert body["data"].get("aligned_jpeg_b64") is not None
        assert len(body["data"]["aligned_jpeg_b64"]) > 10

    def test_fail_no_aligned_b64(self):
        """FAIL sớm (NO_FACE) → data=None → không có aligned."""
        client = _client_with_detector([])
        status, body = _post_validate(client, _png(), return_aligned=True)
        assert status == 422
        assert body["data"] is None


# ---------------------------------------------------------------------------
# 16. Kiểm tra cấu trúc checks dict đầy đủ 5 trường
# ---------------------------------------------------------------------------

class TestValidateChecksStructure:
    def test_checks_have_required_keys(self):
        face = _make_face(bbox=(10, 10, 60, 60), score=0.95)  # too small → FAIL
        client = _client_with_detector([face])
        _, body = _post_validate(client, _mid_brightness_sharp_png())
        for c in body["quality"]["checks"]:
            for key in ("name", "passed", "value", "rule", "code"):
                assert key in c, f"Check thiếu trường '{key}': {c}"
            assert isinstance(c["passed"], bool)
            assert isinstance(c["rule"], str)
            assert isinstance(c["code"], str)


# ---------------------------------------------------------------------------
# 17. HTTP status code cho các loại lỗi
# ---------------------------------------------------------------------------

class TestValidateHttpStatusCodes:
    """Kiểm tra HTTP status code theo loại lỗi."""

    def test_invalid_image_400(self):
        client = _client_with_detector([_make_face()])
        status, body = _post_validate(client, b"not an image")
        assert status == 400

    def test_no_face_422(self):
        client = _client_with_detector([])
        status, _ = _post_validate(client, _png())
        assert status == 422

    def test_multiple_faces_422(self):
        faces = [_make_face(bbox=(10, 10, 80, 80)), _make_face(bbox=(90, 10, 160, 80))]
        client = _client_with_detector(faces)
        status, _ = _post_validate(client, _png())
        assert status == 422

    def test_pass_200(self):
        face = _make_face(bbox=(10, 10, 150, 150), score=0.95)
        client = _client_with_detector([face])
        status, _ = _post_validate(client, _mid_brightness_sharp_png())
        assert status == 200


# ---------------------------------------------------------------------------
# 18. API key — /validate phải được bảo vệ giống các endpoint khác
# ---------------------------------------------------------------------------

class TestValidateApiKey:
    def _client(self) -> TestClient:
        det = FakeDetector([_make_face(bbox=(10, 10, 150, 150), score=0.95)])
        app = create_app(Settings(api_key="secret"), FakeEngineWithDetector(det),
                         FaceStore(":memory:"))
        client = TestClient(app)
        client.__enter__()
        return client

    def test_missing_key_401(self):
        r = self._client().post(
            "/api/v1/faces/validate",
            files={"file": ("face.png", io.BytesIO(_mid_brightness_sharp_png()), "image/png")},
        )
        assert r.status_code == 401

    def test_valid_key_200(self):
        r = self._client().post(
            "/api/v1/faces/validate",
            files={"file": ("face.png", io.BytesIO(_mid_brightness_sharp_png()), "image/png")},
            headers={"X-API-Key": "secret"},
        )
        assert r.status_code == 200
        assert r.json()["code"] == "OK"
