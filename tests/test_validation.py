import cv2
import numpy as np
import pytest

from app.validation import (
    ImageQualityError, validate_image, detect_face, check_face_count,
    check_blur, check_brightness,
)


@pytest.mark.parametrize("source", [b"", b"invalid", None])
def test_invalid_image(source):
    with pytest.raises(ValueError):
        validate_image(source)


def test_decode_bytes_and_path(tmp_path):
    img = np.full((24, 24, 3), 128, np.uint8)
    ok, data = cv2.imencode(".png", img)
    assert ok
    path = tmp_path / "ảnh.png"
    path.write_bytes(data.tobytes())
    np.testing.assert_array_equal(validate_image(path), img)
    np.testing.assert_array_equal(validate_image(data.tobytes()), img)
    with pytest.raises(ValueError):
        validate_image(tmp_path / "missing.png")


@pytest.mark.parametrize("count", [0, 1, 2])
def test_detection_and_count(count, face_factory):
    faces = [face_factory() for _ in range(count)]

    class Detector:
        def detect(self, image):
            return faces

    img = np.full((64, 64, 3), 128, np.uint8)
    if count == 0:
        with pytest.raises(ImageQualityError, match="Không phát hiện"):
            detect_face(img, Detector())
    else:
        assert detect_face(img, Detector()) is faces
    if count == 1:
        assert check_face_count(faces) == 1
    else:
        with pytest.raises(ImageQualityError):
            check_face_count(faces)


def test_blur():
    gray = ((np.indices((128, 128)).sum(axis=0) // 4 % 2) * 150 + 50).astype(np.uint8)
    sharp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    score = check_blur(sharp)
    assert check_blur(sharp, threshold=score) == score
    with pytest.raises(ImageQualityError, match="mờ"):
        check_blur(cv2.GaussianBlur(sharp, (21, 21), 5))


@pytest.mark.parametrize("value", [0, 39, 40, 128, 220, 221, 255])
def test_brightness(value):
    image = np.full((32, 32, 3), value, np.uint8)
    if 40 <= value <= 220:
        assert check_brightness(image) == value
    else:
        with pytest.raises(ImageQualityError, match="tối|sáng"):
            check_brightness(image)


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), np.uint8),
                                   np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3))])
def test_invalid_pixels(image):
    for check in (check_blur, check_brightness, detect_face):
        with pytest.raises(ImageQualityError):
            check(image)


def test_invalid_thresholds():
    image = np.full((32, 32, 3), 128, np.uint8)
    for threshold in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            check_blur(image, threshold)
    for minimum, maximum in ((-1, 220), (40, 256), (220, 40), (float("nan"), 220)):
        with pytest.raises(ValueError):
            check_brightness(image, minimum, maximum)
