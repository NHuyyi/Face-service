"""Tests cho geometry.py — không cần model ONNX."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.validation.geometry import check_face_pose, check_face_size


# ---------------------------------------------------------------------------
# Template ArcFace chuẩn 112x112
# ---------------------------------------------------------------------------
TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # mắt trái (0)
        [73.5318, 51.5014],   # mắt phải (1)
        [56.0252, 71.7366],   # mũi (2)
        [41.5493, 92.3655],   # khoé miệng trái (3)
        [70.7299, 92.2041],   # khoé miệng phải (4)
    ],
    dtype=np.float32,
)


def _rotate_kps(kps: np.ndarray, angle_deg: float) -> np.ndarray:
    """Xoay toàn bộ điểm quanh tâm bounding box của chúng."""
    center = kps.mean(axis=0)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    d = kps - center
    rotated = np.stack(
        [cos_a * d[:, 0] - sin_a * d[:, 1],
         sin_a * d[:, 0] + cos_a * d[:, 1]],
        axis=1,
    )
    return rotated + center


def _by_name(checks: list[dict], name: str) -> dict:
    return next(c for c in checks if c["name"] == name)


# ---------------------------------------------------------------------------
# check_face_pose
# ---------------------------------------------------------------------------

class TestCheckFacePoseTemplate:
    """Template ArcFace nhìn thẳng → tất cả check PASS."""

    def setup_method(self):
        self.checks = check_face_pose(TEMPLATE)

    def test_all_pass(self):
        assert all(c["passed"] for c in self.checks), self.checks

    def test_roll_near_zero(self):
        roll = _by_name(self.checks, "roll")["value"]
        assert abs(roll) < 2.0, f"roll={roll}"

    def test_yaw_near_zero(self):
        yaw = _by_name(self.checks, "yaw")["value"]
        assert abs(yaw) < 5.0, f"yaw={yaw}"

    def test_pitch_around_half(self):
        pitch = _by_name(self.checks, "pitch")["value"]
        assert 0.30 <= pitch <= 0.72, f"pitch={pitch}"
        # nhìn thẳng ≈ 0.5
        assert abs(pitch - 0.5) < 0.15, f"pitch={pitch}"


class TestCheckFacePoseRotated25:
    """Xoay template 25° → roll ≈ 25, yaw/pitch không đổi đáng kể, tất cả vẫn PASS."""

    def setup_method(self):
        kps_rot = _rotate_kps(TEMPLATE, 25.0)
        self.checks = check_face_pose(kps_rot)

    def test_roll_near_25(self):
        roll = _by_name(self.checks, "roll")["value"]
        assert abs(roll - 25.0) < 3.0, f"roll={roll}"

    def test_yaw_still_small(self):
        yaw = _by_name(self.checks, "yaw")["value"]
        assert abs(yaw) < 10.0, f"yaw={yaw}"

    def test_all_pass(self):
        # |25| <= 45 → roll PASS; yaw/pitch gần không đổi → PASS
        assert all(c["passed"] for c in self.checks), self.checks


class TestCheckFacePoseYawFail:
    """Dịch mũi sang phải 12px → POSE_YAW fail."""

    def setup_method(self):
        kps = TEMPLATE.copy()
        kps[2, 0] += 12.0   # dịch mũi sang phải
        self.checks = check_face_pose(kps)

    def test_yaw_check_fails(self):
        yaw = _by_name(self.checks, "yaw")
        assert not yaw["passed"], f"yaw={yaw['value']}"
        assert yaw["code"] == "POSE_YAW"

    def test_roll_still_passes(self):
        roll = _by_name(self.checks, "roll")
        assert roll["passed"]


class TestCheckFacePoseDegenerateEyes:
    """2 mắt trùng nhau → không crash, yaw và pitch fail."""

    def test_no_crash(self):
        kps = TEMPLATE.copy()
        kps[1] = kps[0]     # mắt phải = mắt trái
        checks = check_face_pose(kps)
        assert isinstance(checks, list)
        assert len(checks) == 3

    def test_yaw_pitch_fail(self):
        kps = TEMPLATE.copy()
        kps[1] = kps[0]
        checks = check_face_pose(kps)
        yaw = _by_name(checks, "yaw")
        pitch = _by_name(checks, "pitch")
        assert not yaw["passed"]
        assert not pitch["passed"]

    def test_nan_becomes_none(self):
        """Value của yaw/pitch phải là None (không phải NaN) khi mắt trùng."""
        kps = TEMPLATE.copy()
        kps[1] = kps[0]
        checks = check_face_pose(kps)
        yaw = _by_name(checks, "yaw")
        pitch = _by_name(checks, "pitch")
        assert yaw["value"] is None
        assert pitch["value"] is None


# ---------------------------------------------------------------------------
# check_face_size
# ---------------------------------------------------------------------------

class TestCheckFaceSizePass:
    """bbox đủ lớn, eye_dist đủ, score cao → tất cả PASS."""

    def test_all_pass(self):
        bbox = np.array([0, 0, 100, 100], dtype=float)
        checks = check_face_size(bbox, TEMPLATE, det_score=0.95)
        assert all(c["passed"] for c in checks), checks


class TestCheckFaceSizeTooSmall:
    """bbox 40x40 → FACE_TOO_SMALL fail (min_side < 64)."""

    def setup_method(self):
        bbox = np.array([0, 0, 40, 40], dtype=float)
        self.checks = check_face_size(bbox, TEMPLATE, det_score=0.95)

    def test_min_side_fails(self):
        c = next(c for c in self.checks if c["name"] == "face_min_side")
        assert not c["passed"]
        assert c["code"] == "FACE_TOO_SMALL"

    def test_value_is_40(self):
        c = next(c for c in self.checks if c["name"] == "face_min_side")
        assert abs(c["value"] - 40.0) < 1e-6


class TestCheckFaceSizeLowScore:
    """det_score 0.3 → LOW_DET_SCORE fail."""

    def setup_method(self):
        bbox = np.array([0, 0, 100, 100], dtype=float)
        self.checks = check_face_size(bbox, TEMPLATE, det_score=0.3)

    def test_score_fails(self):
        c = next(c for c in self.checks if c["name"] == "det_score")
        assert not c["passed"]
        assert c["code"] == "LOW_DET_SCORE"

    def test_value_is_0_3(self):
        c = next(c for c in self.checks if c["name"] == "det_score")
        assert abs(c["value"] - 0.3) < 1e-9


class TestCheckFaceSizeSmallEyeDist:
    """Khoảng cách 2 mắt < 20px → FACE_TOO_SMALL (eye_distance check)."""

    def test_eye_dist_fails(self):
        kps = TEMPLATE.copy()
        # Đặt 2 mắt rất gần nhau (cách 5px)
        kps[0] = [50.0, 56.0]
        kps[1] = [55.0, 56.0]
        bbox = np.array([0, 0, 100, 100], dtype=float)
        checks = check_face_size(bbox, kps, det_score=0.95)
        c = next(c for c in checks if c["name"] == "eye_distance")
        assert not c["passed"]
        assert c["code"] == "FACE_TOO_SMALL"


# ---------------------------------------------------------------------------
# Kiểm tra format dict trả về
# ---------------------------------------------------------------------------

class TestCheckReturnFormat:
    """Mọi check phải có đủ 5 khoá chuẩn."""

    def test_pose_format(self):
        for c in check_face_pose(TEMPLATE):
            assert set(c.keys()) == {"name", "passed", "value", "rule", "code"}
            assert isinstance(c["passed"], bool)
            assert isinstance(c["rule"], str)
            assert isinstance(c["code"], str)

    def test_size_format(self):
        bbox = np.array([0, 0, 100, 100], dtype=float)
        for c in check_face_size(bbox, TEMPLATE, det_score=0.9):
            assert set(c.keys()) == {"name", "passed", "value", "rule", "code"}
            assert isinstance(c["passed"], bool)
