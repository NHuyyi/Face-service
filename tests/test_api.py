"""Test API bằng engine giả (không cần model ONNX)."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db import FaceStore  # noqa: E402
from app.engine import FaceResult, NoFaceError  # noqa: E402
from app.main import create_app  # noqa: E402
from app.matching import l2_normalize  # noqa: E402


class FakeEngine:
    """Màu trung bình của ảnh quyết định 'danh tính'; ảnh trắng = không có mặt."""
    def extract(self, img, keep_aligned=False):
        m = float(img.mean())
        if m > 250:
            raise NoFaceError("none")
        rng = np.random.RandomState(int(m))
        v = l2_normalize(rng.randn(512))
        return FaceResult(v, [0, 0, 1, 1], [[0, 0]] * 5, 0.99, 1, 0, 0.0)


def png(value):
    ok, buf = cv2.imencode(".png", np.full((64, 64, 3), value, np.uint8))
    return buf.tobytes()


@pytest.fixture()
def client():
    app = create_app(Settings(match_threshold=0.4), FakeEngine(), FaceStore(":memory:"))
    with TestClient(app) as c:
        yield c


def test_flow_enroll_then_match(client):
    r = client.post("/v1/faces", files={"file": ("a.png", png(10), "image/png")}, data={"label": "an"})
    assert r.status_code == 200 and r.json()["dim"] == 512
    fid = r.json()["id"]
    same = client.post(f"/v1/faces/{fid}/match", files={"file": ("b.png", png(10), "image/png")}).json()
    assert same["result"] == "Match" and same["match"] == 1 and same["similarity"] > 0.99
    diff = client.post(f"/v1/faces/{fid}/match", files={"file": ("c.png", png(99), "image/png")}).json()
    assert diff["result"] == "0 Match" and diff["match"] == 0


def test_compare_identify_errors(client):
    r = client.post("/v1/compare", files={"file_a": ("a.png", png(5), "image/png"),
                                          "file_b": ("b.png", png(5), "image/png")})
    assert r.json()["match"] == 1
    client.post("/v1/faces", files={"file": ("a.png", png(5), "image/png")}, data={"label": "an"})
    client.post("/v1/faces", files={"file": ("a.png", png(7), "image/png")}, data={"label": "binh"})
    ident = client.post("/v1/identify", files={"file": ("q.png", png(7), "image/png")}).json()
    assert ident["candidates"][0]["label"] == "binh" and ident["match"] == 1
    assert client.post("/v1/faces", files={"file": ("w.png", png(255), "image/png")}).status_code == 422
    assert client.post("/v1/faces", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/v1/faces/nope/match", files={"file": ("a.png", png(5), "image/png")}).status_code == 404


@pytest.mark.parametrize('orientation', [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_api_exif_pixels_and_face_coordinates(orientation, jpeg_with_orientation, face_factory, engine_factory):
    data, expected = jpeg_with_orientation(orientation)
    face = face_factory()
    eng = engine_factory(lambda img: [face] if np.array_equal(img, expected) else [])
    with TestClient(create_app(Settings(), eng, FaceStore(':memory:'))) as c:
        response = c.post('/v1/embeddings', files={'file': ('exif.jpg', data, 'image/jpeg')})
    assert response.status_code == 200
    info = response.json()['face']
    assert info['rotation_applied'] == 0  # EXIF is not counted as trial rotation.
    assert info['face_count'] == 1
    np.testing.assert_array_equal(info['bbox'], face.bbox)
    np.testing.assert_array_equal(info['landmarks'], face.kps)
    np.testing.assert_array_equal(eng.detector.seen[0], expected)
    assert eng.recognizer.seen[0].shape == (112, 112, 3)


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_api_missing_exif_rotation(angle, asymmetric_image, face_factory, engine_factory):
    source = np.rot90(asymmetric_image, angle // 90).copy()
    ok, data = cv2.imencode('.png', source)
    assert ok
    face = face_factory()
    eng = engine_factory(lambda img: [face] if np.array_equal(img, asymmetric_image) else [])
    with TestClient(create_app(Settings(), eng, FaceStore(':memory:'))) as c:
        response = c.post('/v1/embeddings', files={'file': ('no-exif.png', data.tobytes(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['face']['rotation_applied'] == angle
    np.testing.assert_array_equal(response.json()['face']['landmarks'], face.kps)


def test_api_trial_rotation_is_relative_to_exif(jpeg_with_orientation, face_factory, engine_factory):
    data, decoded = jpeg_with_orientation(6)  # EXIF itself applies 90 clockwise.
    target = np.rot90(decoded, -1).copy()     # Detector needs a further 90.
    main, other = face_factory(), face_factory((0, 0, 40, 40))
    eng = engine_factory(lambda img: [main, other] if np.array_equal(img, target) else [])
    with TestClient(create_app(Settings(), eng, FaceStore(':memory:'))) as c:
        response = c.post('/v1/embeddings', files={'file': ('a.jpg', data, 'image/jpeg')})
    assert response.status_code == 200
    info = response.json()['face']
    assert info['rotation_applied'] == 90 and info['face_count'] == 2
    np.testing.assert_array_equal(info['bbox'], main.bbox)
    np.testing.assert_array_equal(eng.detector.seen[-1], target)


@pytest.mark.parametrize('kind', ['none', 'small', 'invalid_landmarks'])
def test_real_engine_api_returns_422(kind, face_factory, engine_factory):
    faces = {
        'none': [],
        'small': [face_factory((0, 0, 39, 60))],
        'invalid_landmarks': [face_factory(kps=np.zeros((5, 2)))],
    }[kind]
    eng = engine_factory(lambda img: faces)
    with TestClient(create_app(Settings(), eng, FaceStore(':memory:'))) as c:
        response = c.post('/v1/embeddings', files={'file': ('a.png', png(10), 'image/png')})
    assert response.status_code == 422
    assert response.json()['detail']
    assert len(eng.detector.seen) == (1 if kind == 'invalid_landmarks' else 4)
    assert eng.recognizer.seen == []
