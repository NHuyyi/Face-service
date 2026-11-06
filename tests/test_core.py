import numpy as np
import pytest

from app.db import FaceStore
from app.engine.align import ARCFACE_TEMPLATE, align_face, roll_angle_degrees, umeyama_similarity
from app.engine.retinaface import RetinaFaceDetector, decode_boxes, decode_landmarks, make_priors
from app.engine.utils import nms
from app.matching import batch_cosine_similarity, cosine_distance, cosine_similarity, is_match, l2_normalize


def test_cosine_basic():
    a = np.random.RandomState(0).randn(512).astype(np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(a, 5 * a) == pytest.approx(1.0, abs=1e-6)  # bất biến theo độ dài
    assert cosine_similarity(a, np.zeros(512)) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity(a, a[:100])


def test_batch_equals_pairwise():
    rng = np.random.RandomState(1)
    g, q = rng.randn(50, 512).astype(np.float32), rng.randn(512).astype(np.float32)
    batch = batch_cosine_similarity(q, g)
    for i in range(50):
        assert batch[i] == pytest.approx(cosine_similarity(q, g[i]), abs=1e-5)
    assert batch_cosine_similarity(q, np.zeros((0, 512), np.float32)).shape == (0,)


def test_threshold():
    assert is_match(0.5, 0.4) and not is_match(0.3, 0.4)


def test_store_roundtrip_and_gallery():
    st = FaceStore(":memory:")
    v = l2_normalize(np.random.RandomState(2).randn(512))
    rec = st.add(v, "an", 0.9)
    got = st.get(rec.id)
    assert got.label == "an" and np.allclose(got.embedding, v)
    meta, mat = st.gallery()
    assert mat.shape == (1, 512) and meta[0][0] == rec.id
    with pytest.raises(ValueError):
        st.add(np.zeros(128))
    assert st.delete(rec.id) and st.get(rec.id) is None


def test_store_ttl(monkeypatch):
    import app.db as dbmod
    st = FaceStore(":memory:", ttl_seconds=10)
    rec = st.add(np.ones(512))
    assert st.get(rec.id) is not None
    t0 = dbmod.time.time()
    monkeypatch.setattr(dbmod.time, "time", lambda: t0 + 11)
    assert st.get(rec.id) is None


def test_umeyama_recovers_rotation_and_scale():
    ang, s, t = np.radians(25), 1.7, np.array([10.0, -4.0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = (s * (R @ ARCFACE_TEMPLATE.T)).T + t       # mặt bị xoay 25°, phóng 1.7x, dịch chuyển
    M = umeyama_similarity(src, ARCFACE_TEMPLATE)    # map ngược về template
    mapped = (M[:, :2] @ src.T).T + M[:, 2]
    assert np.abs(mapped - ARCFACE_TEMPLATE).max() < 1e-3


def test_align_output_and_roll():
    img = np.random.RandomState(3).randint(0, 255, (300, 300, 3), np.uint8)
    kps = ARCFACE_TEMPLATE * 2 + 50
    assert align_face(img, kps).shape == (112, 112, 3)
    assert roll_angle_degrees(kps) == pytest.approx(0, abs=1.0)
    tilted = kps.copy(); tilted[1, 1] += 40
    assert roll_angle_degrees(tilted) > 10


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.4) == [0, 2]


def test_retinaface_decode_zero_offsets_gives_prior():
    p = make_priors(640, 640)
    assert p.shape == (16800, 4)  # 80*80*2 + 40*40*2 + 20*20*2 (kích thước chuẩn của RetinaFace 640)
    box = decode_boxes(np.zeros((len(p), 4), np.float32), p)[100]
    cx, cy, w, h = p[100]
    assert np.allclose(box, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    lm = decode_landmarks(np.zeros((len(p), 10), np.float32), p)
    assert np.allclose(lm[100], [[cx, cy]] * 5)


def test_retinaface_postprocess_rescales_to_original():
    det = RetinaFaceDetector.__new__(RetinaFaceDetector)
    det.input_w = det.input_h = 640
    det.score_thr, det.nms_thr = 0.5, 0.4
    det.priors = make_priors(640, 640)
    n = len(det.priors)
    loc, lm = np.zeros((1, n, 4), np.float32), np.zeros((1, n, 10), np.float32)
    conf = np.zeros((1, n, 2), np.float32); conf[..., 0] = 1.0
    idx = 200
    conf[0, idx] = [0.01, 0.99]
    out = det.postprocess([loc, conf, lm], scale=0.5)   # ảnh gốc lớn gấp 2 lần
    assert len(out) == 1 and out[0].score == pytest.approx(0.99)
    cx, cy, w, h = det.priors[idx]
    assert out[0].bbox[2] - out[0].bbox[0] == pytest.approx(w * 640 / 0.5, rel=1e-4)
    assert out[0].kps.shape == (5, 2)


def test_engine_tries_rotations_and_scrfd_decode():
    from app.config import Settings
    from app.engine.engine import FaceEngine, NoFaceError
    from app.engine.scrfd import ScrfdDetector
    from app.engine.types import DetectedFace

    class FakeDet:  # "thấy mặt" chỉ khi ảnh là dạng đứng (cao > rộng)
        def detect(self, img):
            if img.shape[0] > img.shape[1]:
                return [DetectedFace(np.array([10, 10, 110, 110.]), 0.9, ARCFACE_TEMPLATE.copy())]
            return []

    class FakeRec:
        def embed(self, a):
            return l2_normalize(np.ones(512))

    eng = FaceEngine.__new__(FaceEngine)
    eng.s, eng.detector, eng.recognizer = Settings(), FakeDet(), FakeRec()
    r = eng.extract(np.zeros((200, 400, 3), np.uint8), keep_aligned=True)  # ảnh nằm ngang
    assert r.rotation_applied == 90 and r.aligned.shape == (112, 112, 3) and r.embedding.shape == (512,)
    eng.detector = type("D", (), {"detect": lambda self, i: []})()
    with pytest.raises(NoFaceError):
        eng.extract(np.zeros((200, 400, 3), np.uint8))

    # SCRFD: 1 anchor có điểm cao ở stride 8, ô (row=3,col=5), anchor 0
    det = ScrfdDetector.__new__(ScrfdDetector)
    det.size, det.score_thr, det.nms_thr, det._centers = 640, 0.5, 0.4, {}
    outs = []
    for stride in (8, 16, 32):
        n = (640 // stride) ** 2 * 2
        outs.append(np.zeros((n, 1), np.float32))
    outs[0][(3 * 80 + 5) * 2, 0] = 0.95
    for stride in (8, 16, 32):
        n = (640 // stride) ** 2 * 2
        b = np.zeros((n, 4), np.float32); b[:] = 4  # khoảng cách 4*stride mỗi phía
        outs.append(b)
    for stride in (8, 16, 32):
        outs.append(np.zeros(((640 // stride) ** 2 * 2, 10), np.float32))
    res = det.postprocess(outs, scale=1.0)
    assert len(res) == 1
    cx, cy = 5 * 8, 3 * 8
    assert np.allclose(res[0].bbox, [cx - 32, cy - 32, cx + 32, cy + 32])
