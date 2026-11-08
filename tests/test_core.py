import numpy as np
import pytest

from app.db import FaceStore
from app.engine.align import ARCFACE_TEMPLATE, align_face, roll_angle_degrees, umeyama_similarity
from app.engine.retinaface import RetinaFaceDetector, decode_boxes, decode_landmarks, make_priors
from app.engine.utils import nms
from app.matching import batch_cosine_similarity, cosine_distance, cosine_similarity, is_match, l2_normalize
from app.engine.align import AlignmentError
from app.engine.engine import NoFaceError
from app.engine.image import decode_image, read_image


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


def test_retinaface_dimension_handling():
    """Kiểm tra xử lý kích thước đầu vào động hoặc không hợp lệ (-1, 0, string, None)."""
    class FakeInput:
        def __init__(self, h, w):
            self.name = "input0"
            self.shape = [1, 3, h, w]

    class FakeSession:
        def __init__(self, h, w):
            self._inp = FakeInput(h, w)
        def get_inputs(self):
            return [self._inp]

    # 1. Kích thước tĩnh dương: 640x640
    det = RetinaFaceDetector.__new__(RetinaFaceDetector)
    det.sess = FakeSession(640, 640)
    inp = det.sess.get_inputs()[0]
    h, w = inp.shape[2], inp.shape[3]
    det.input_h = h if (isinstance(h, int) and h > 0) else 640
    det.input_w = w if (isinstance(w, int) and w > 0) else 640
    assert det.input_h == 640 and det.input_w == 640

    # 2. Kích thước âm (-1): fallback về input_size
    det_neg = RetinaFaceDetector.__new__(RetinaFaceDetector)
    det_neg.sess = FakeSession(-1, -1)
    inp = det_neg.sess.get_inputs()[0]
    h, w = inp.shape[2], inp.shape[3]
    det_neg.input_h = h if (isinstance(h, int) and h > 0) else 640
    det_neg.input_w = w if (isinstance(w, int) and w > 0) else 640
    assert det_neg.input_h == 640 and det_neg.input_w == 640

    # 3. Kích thước chuỗi động ('?'): fallback về input_size
    det_dyn = RetinaFaceDetector.__new__(RetinaFaceDetector)
    det_dyn.sess = FakeSession("?", "?")
    inp = det_dyn.sess.get_inputs()[0]
    h, w = inp.shape[2], inp.shape[3]
    det_dyn.input_h = h if (isinstance(h, int) and h > 0) else 640
    det_dyn.input_w = w if (isinstance(w, int) and w > 0) else 640
    assert det_dyn.input_h == 640 and det_dyn.input_w == 640


def test_retinaface_threshold_boundary_and_empty_behavior():
    """Kiểm tra biên ngưỡng (>=), output rỗng, ảnh rỗng và thiếu tensor."""
    det = RetinaFaceDetector.__new__(RetinaFaceDetector)
    det.input_w = det.input_h = 640
    det.score_thr, det.nms_thr = 0.5, 0.4
    det.priors = make_priors(640, 640)
    n = len(det.priors)

    loc, lm = np.zeros((1, n, 4), np.float32), np.zeros((1, n, 10), np.float32)
    conf = np.zeros((1, n, 2), np.float32); conf[..., 0] = 1.0

    # Điểm đúng bằng ngưỡng 0.5 -> phải được giữ lại với toán tử >=
    conf[0, 10] = [0.5, 0.5]
    out = det.postprocess([loc, conf, lm], scale=1.0)
    assert len(out) == 1 and out[0].score == pytest.approx(0.5)

    # Điểm dưới ngưỡng 0.5 -> rỗng
    conf[0, 10] = [0.51, 0.49]
    assert len(det.postprocess([loc, conf, lm], scale=1.0)) == 0

    # Outs rỗng hoặc scale <= 0
    assert len(det.postprocess([], scale=1.0)) == 0
    assert len(det.postprocess([loc, conf, lm], scale=0.0)) == 0

    # Thiếu outputs
    assert len(det.postprocess([loc], scale=1.0)) == 0

    # Ảnh rỗng trong detect
    assert len(det.detect(None)) == 0
    assert len(det.detect(np.zeros((0, 0, 3), np.uint8))) == 0


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_upright_pixels_coordinates_and_compensation(angle, asymmetric_image, face_factory, engine_factory):
    # Input is rotated counterclockwise; compensation must be clockwise.
    source = np.rot90(asymmetric_image, angle // 90).copy()
    face = face_factory()
    eng = engine_factory(lambda img: [face] if np.array_equal(img, asymmetric_image) else [])
    r = eng.extract(source, keep_aligned=True)
    assert r.rotation_applied == angle
    assert len(eng.detector.seen) == angle // 90 + 1
    for i, seen in enumerate(eng.detector.seen):
        np.testing.assert_array_equal(seen, np.rot90(source, -i))
    np.testing.assert_array_equal(r.bbox, face.bbox)
    np.testing.assert_array_equal(r.landmarks, face.kps)
    assert np.all(np.array(r.landmarks) < [240, 160])
    assert r.face_count == 1
    assert r.aligned.shape == (112, 112, 3)
    np.testing.assert_array_equal(r.aligned, align_face(asymmetric_image, face.kps))
    np.testing.assert_array_equal(eng.recognizer.seen[0], r.aligned)
    assert r.roll_degrees == pytest.approx(roll_angle_degrees(face.kps))


def test_multiface_uses_area_times_score_and_filters_size(asymmetric_image, face_factory, engine_factory):
    small = face_factory((0, 0, 39, 150), 1.0)
    large = face_factory((0, 0, 100, 100), 0.5)  # 5000
    main = face_factory((10, 10, 90, 90), 0.9)   # 5760
    boundary = face_factory((0, 0, 40, 40), 0.9)
    eng = engine_factory(lambda img: [small, large, main, boundary])
    r = eng.extract(asymmetric_image)
    assert r.face_count == 3 and r.rotation_applied == 0
    assert r.det_score == main.score
    np.testing.assert_array_equal(r.bbox, main.bbox)
    assert len(eng.detector.seen) == 1 and r.aligned is None


def test_small_face_does_not_stop_orientation_search(asymmetric_image, face_factory, engine_factory):
    small, valid = face_factory((0, 0, 39, 100)), face_factory()
    second = face_factory((0, 0, 40, 40))
    eng = engine_factory(lambda img: [valid, second, small]
                         if np.array_equal(img, asymmetric_image) else [small])
    r = eng.extract(np.rot90(asymmetric_image).copy())
    assert r.rotation_applied == 90 and r.face_count == 2
    np.testing.assert_array_equal(r.bbox, valid.bbox)


@pytest.mark.parametrize('too_small', [False, True])
def test_all_orientations_without_valid_face(too_small, asymmetric_image, face_factory, engine_factory):
    faces = [face_factory((0, 0, 100, 39))] if too_small else []
    eng = engine_factory(lambda img: faces)
    with pytest.raises(NoFaceError):
        eng.extract(asymmetric_image)
    assert len(eng.detector.seen) == 4
    assert eng.recognizer.seen == []


@pytest.mark.parametrize('orientation', [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_decode_exif_exactly_once_and_local_file(orientation, jpeg_with_orientation, tmp_path):
    data, expected = jpeg_with_orientation(orientation)
    np.testing.assert_array_equal(decode_image(data), expected)
    path = tmp_path / 'ảnh.jpg'
    path.write_bytes(data)
    np.testing.assert_array_equal(read_image(path), expected)


@pytest.mark.parametrize('data', [b'', b'not an image'])
def test_decode_invalid(data):
    with pytest.raises(ValueError):
        decode_image(data)


@pytest.mark.parametrize('mirrored', [False, True])
def test_alignment_preserves_image_handedness(mirrored):
    # Exact template -> identity. Asymmetric pixels reveal accidental flipping.
    img = np.random.RandomState(12).randint(0, 256, (112, 112, 3), np.uint8)
    if mirrored:
        img = img[:, ::-1].copy()
    np.testing.assert_array_equal(align_face(img, ARCFACE_TEMPLATE), img)
    matrix = umeyama_similarity(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
    assert np.linalg.det(matrix[:, :2]) > 0


def test_alignment_maps_tilted_landmark_pixels():
    import cv2
    angle = np.radians(23)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    kps = 1.6 * ARCFACE_TEMPLATE @ rotation.T + [90, 30]
    image = np.zeros((300, 300, 3), np.uint8)
    for i, point in enumerate(kps):
        cv2.circle(image, tuple(np.rint(point).astype(int)), 4, (30 + 40*i, 255, 100), -1)
    aligned = align_face(image, kps)
    assert aligned.shape == (112, 112, 3)
    for point in ARCFACE_TEMPLATE:
        x, y = np.rint(point).astype(int)
        assert aligned[y-2:y+3, x-2:x+3, 1].max() > 200
    assert roll_angle_degrees(kps) == pytest.approx(23 + roll_angle_degrees(ARCFACE_TEMPLATE))


def test_alignment_of_mirrored_landmarks_keeps_reflection():
    import cv2
    # Mirror the pixels and reorder paired landmarks into image-left/right order,
    # as a detector does; alignment must not undo the selfie's reflection.
    img = np.zeros((224, 224, 3), np.uint8)
    kps = ARCFACE_TEMPLATE * 2
    for i, point in enumerate(kps):
        cv2.circle(img, tuple(np.rint(point).astype(int)), 7, (20 + i*50, 200, 80), -1)
    mirrored = img[:, ::-1].copy()
    mirror_kps = kps.copy()
    mirror_kps[:, 0] = 223 - mirror_kps[:, 0]
    mirror_kps = mirror_kps[[1, 0, 2, 4, 3]]
    matrix = umeyama_similarity(mirror_kps, ARCFACE_TEMPLATE)
    assert np.linalg.det(matrix[:, :2]) > 0
    aligned = align_face(mirrored, mirror_kps)
    # Image-left eye now has the original image-right eye's marker color.
    left, right = np.rint(ARCFACE_TEMPLATE[:2]).astype(int)
    assert aligned[left[1], left[0], 0] > aligned[right[1], right[0], 0] + 30


@pytest.mark.parametrize('kps', [
    np.zeros((5, 2)), np.ones((4, 2)), np.full((5, 2), np.nan),
    np.full((5, 2), np.inf), np.column_stack([np.arange(5), np.arange(5)]),
    np.vstack([ARCFACE_TEMPLATE[0], ARCFACE_TEMPLATE[0], ARCFACE_TEMPLATE[2:]]),
])
def test_invalid_landmarks_fail_clearly(kps, asymmetric_image, face_factory, engine_factory):
    eng = engine_factory(lambda img: [face_factory(kps=kps)])
    with pytest.raises(AlignmentError, match='Landmark'):
        eng.extract(asymmetric_image)
    assert eng.recognizer.seen == []
