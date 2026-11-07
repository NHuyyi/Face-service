"""Unit tests cho scripts/download_models.py (sử dụng mock, không tải thật)."""
import hashlib
import io
import os
import urllib.error
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.download_models import (BUFFALO_MODELS, MODEL_HASHES, compute_sha256,
                                     download_all, download_buffalo_models, download_file,
                                     download_retinaface_model, verify_model_file)


def _hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().lower()


class MockResponse:
    def __init__(self, data: bytes, headers: dict = None):
        self._bio = io.BytesIO(data)
        self.headers = headers or {"Content-Length": str(len(data))}

    def read(self, chunk_size: int = -1):
        return self._bio.read(chunk_size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_compute_sha256(tmp_path):
    f = tmp_path / "sample.bin"
    content = b"Antigravity Face Recognition 2026"
    f.write_bytes(content)
    assert compute_sha256(str(f)) == _hash(content)


def test_verify_model_file(tmp_path):
    f = tmp_path / "model.onnx"
    expected = _hash(b"model-weights-content")
    assert not verify_model_file(str(f), expected)  # Chưa tồn tại

    f.write_bytes(b"")
    assert not verify_model_file(str(f), expected)  # File rỗng

    f.write_bytes(b"wrong-content")
    assert not verify_model_file(str(f), expected)  # Sai nội dung

    f.write_bytes(b"model-weights-content")
    assert verify_model_file(str(f), expected)      # Hợp lệ


def test_download_file_success(tmp_path):
    dest = tmp_path / "test_model.onnx"
    data = b"VALID_ONNX_MODEL_BINARY_STREAM"
    sha = _hash(data)

    with patch("urllib.request.urlopen", return_value=MockResponse(data)):
        result = download_file("http://example.com/model.onnx", str(dest), expected_sha=sha)

    assert result == str(dest)
    assert dest.exists()
    assert dest.read_bytes() == data
    assert not (tmp_path / "test_model.onnx.tmp").exists()


def test_download_file_invalid_checksum(tmp_path):
    dest = tmp_path / "corrupted_model.onnx"
    data = b"ACTUAL_DATA_RECEIVED"
    wrong_sha = _hash(b"DIFFERENT_DATA")

    with patch("urllib.request.urlopen", return_value=MockResponse(data)):
        with pytest.raises(ValueError, match="Mã băm SHA-256 không khớp"):
            download_file("http://example.com/bad.onnx", str(dest), expected_sha=wrong_sha)

    # File đích không được tạo
    assert not dest.exists()
    # File tạm phải được dọn dẹp
    assert not (tmp_path / "corrupted_model.onnx.tmp").exists()


def test_download_file_network_failure(tmp_path):
    dest = tmp_path / "network_fail.onnx"
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "1000"}
    mock_resp.read.side_effect = urllib.error.URLError("Network dropped")

    mock_urlopen = MagicMock()
    mock_urlopen.__enter__.return_value = mock_resp
    mock_urlopen.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_urlopen):
        with pytest.raises(urllib.error.URLError):
            download_file("http://example.com/fail.onnx", str(dest), expected_sha="fake_hash")

    assert not dest.exists()
    assert not (tmp_path / "network_fail.onnx.tmp").exists()


def test_retinaface_skip_when_valid(tmp_path, monkeypatch):
    dummy_data = b"RETINAFACE_CACHED_BYTES"
    sha = _hash(dummy_data)
    monkeypatch.setitem(MODEL_HASHES, "retinaface.onnx", sha)

    retina_file = tmp_path / "retinaface.onnx"
    retina_file.write_bytes(dummy_data)

    mock_urlopen = MagicMock()
    with patch("urllib.request.urlopen", mock_urlopen):
        download_retinaface_model(str(tmp_path))

    # Không gọi download khi file đã tồn tại và hợp lệ
    mock_urlopen.assert_not_called()


def test_retinaface_redownload_when_corrupted(tmp_path, monkeypatch):
    valid_data = b"RETINAFACE_FRESH_DOWNLOAD"
    valid_sha = _hash(valid_data)
    monkeypatch.setitem(MODEL_HASHES, "retinaface.onnx", valid_sha)

    retina_file = tmp_path / "retinaface.onnx"
    retina_file.write_bytes(b"CORRUPTED_OLD_FILE")

    with patch("urllib.request.urlopen", return_value=MockResponse(valid_data)):
        download_retinaface_model(str(tmp_path))

    assert retina_file.read_bytes() == valid_data


def test_buffalo_models_skip_when_both_valid(tmp_path, monkeypatch):
    data_rec = b"ARCFACE_DATA"
    data_det = b"SCRFD_DATA"
    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(data_rec))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(data_det))

    (tmp_path / "w600k_r50.onnx").write_bytes(data_rec)
    (tmp_path / "det_10g.onnx").write_bytes(data_det)

    mock_urlopen = MagicMock()
    with patch("urllib.request.urlopen", mock_urlopen):
        download_buffalo_models(str(tmp_path))

    mock_urlopen.assert_not_called()


def test_buffalo_models_extract_and_verify(tmp_path, monkeypatch):
    data_rec = b"ARCFACE_CONTENT"
    data_det = b"SCRFD_CONTENT"
    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(data_rec))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(data_det))

    # Tạo zip giả lập gói buffalo_l
    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w") as z:
        z.writestr("buffalo_l/w600k_r50.onnx", data_rec)
        z.writestr("buffalo_l/det_10g.onnx", data_det)
        z.writestr("buffalo_l/other.onnx", b"OTHER")
    zip_bytes = zip_bio.getvalue()

    with patch("urllib.request.urlopen", return_value=MockResponse(zip_bytes)):
        download_buffalo_models(str(tmp_path))

    assert (tmp_path / "w600k_r50.onnx").read_bytes() == data_rec
    assert (tmp_path / "det_10g.onnx").read_bytes() == data_det
    assert not (tmp_path / "buffalo_l.zip.tmp").exists()


def test_buffalo_models_corrupt_entry_fails(tmp_path, monkeypatch):
    data_rec = b"CORRUPTED_ARCFACE_CONTENT"
    data_det = b"SCRFD_CONTENT"
    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(b"EXPECTED_DIFFERENT_HASH"))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(data_det))

    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w") as z:
        z.writestr("w600k_r50.onnx", data_rec)
        z.writestr("det_10g.onnx", data_det)
    zip_bytes = zip_bio.getvalue()

    with patch("urllib.request.urlopen", return_value=MockResponse(zip_bytes)):
        with pytest.raises(ValueError, match="Mã băm SHA-256 không khớp"):
            download_buffalo_models(str(tmp_path))

    # File không được lưu vào đường dẫn cuối
    assert not (tmp_path / "w600k_r50.onnx").exists()
    assert not (tmp_path / "w600k_r50.onnx.tmp").exists()
    assert not (tmp_path / "buffalo_l.zip.tmp").exists()


def test_forced_download_failure_preserves_existing_valid_model(tmp_path, monkeypatch):
    """Xác minh rằng khi download bị force mà gặp lỗi mạng hoặc sai SHA-256, model hợp lệ sẵn có vẫn không đổi."""
    initial_retina = b"ORIGINAL_VALID_RETINAFACE_MODEL"
    retina_sha = _hash(initial_retina)
    monkeypatch.setitem(MODEL_HASHES, "retinaface.onnx", retina_sha)

    retina_file = tmp_path / "retinaface.onnx"
    retina_file.write_bytes(initial_retina)

    # 1. Thất bại do lỗi mạng khi force=True
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "1000"}
    mock_resp.read.side_effect = urllib.error.URLError("Connection reset by peer")
    mock_urlopen = MagicMock()
    mock_urlopen.__enter__.return_value = mock_resp
    mock_urlopen.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_urlopen):
        with pytest.raises(urllib.error.URLError):
            download_retinaface_model(str(tmp_path), force=True)

    # Model cũ phải được bảo toàn nguyên vẹn 100%
    assert retina_file.exists()
    assert retina_file.read_bytes() == initial_retina
    assert not (tmp_path / "retinaface.onnx.tmp").exists()

    # 2. Thất bại do sai SHA-256 khi force=True
    corrupted_data = b"CORRUPTED_NEW_DOWNLOAD_BYTES"
    with patch("urllib.request.urlopen", return_value=MockResponse(corrupted_data)):
        with pytest.raises(ValueError, match="Mã băm SHA-256 không khớp"):
            download_retinaface_model(str(tmp_path), force=True)

    # Model cũ vẫn phải được bảo toàn nguyên vẹn 100%
    assert retina_file.exists()
    assert retina_file.read_bytes() == initial_retina
    assert not (tmp_path / "retinaface.onnx.tmp").exists()

    # 3. Với Buffalo models: file sẵn có hợp lệ không bị ghi đè/xóa khi giải nén lỗi
    rec_orig = b"ORIGINAL_VALID_ARCFACE"
    det_orig = b"ORIGINAL_VALID_SCRFD"
    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(rec_orig))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(det_orig))

    rec_file = tmp_path / "w600k_r50.onnx"
    det_file = tmp_path / "det_10g.onnx"
    rec_file.write_bytes(rec_orig)
    det_file.write_bytes(det_orig)

    # Zip trả về có entry det_10g bị hỏng
    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w") as z:
        z.writestr("w600k_r50.onnx", rec_orig)
        z.writestr("det_10g.onnx", b"CORRUPTED_SCRFD_PAYLOAD")
    bad_zip_bytes = zip_bio.getvalue()

    with patch("urllib.request.urlopen", return_value=MockResponse(bad_zip_bytes)):
        with pytest.raises(ValueError, match="Mã băm SHA-256 không khớp"):
            download_buffalo_models(str(tmp_path), force=True)

    # Cả hai file đích hợp lệ ban đầu vẫn giữ nguyên
    assert rec_file.read_bytes() == rec_orig
    assert det_file.read_bytes() == det_orig
    assert not (tmp_path / "w600k_r50.onnx.tmp").exists()
    assert not (tmp_path / "det_10g.onnx.tmp").exists()
    assert not (tmp_path / "buffalo_l.zip.tmp").exists()


def test_force_flag_triggers_redownload_of_valid_models(tmp_path, monkeypatch):
    """Xác minh rằng force=True sẽ không bỏ qua mà thực hiện tải lại các model dù đã tồn tại hợp lệ."""
    data_rec = b"ARCFACE_OLD"
    data_det = b"SCRFD_OLD"
    data_ret = b"RETINAFACE_OLD"

    # Mới hơn sau khi tải
    new_rec = b"ARCFACE_NEW"
    new_det = b"SCRFD_NEW"
    new_ret = b"RETINAFACE_NEW"

    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(data_rec))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(data_det))
    monkeypatch.setitem(MODEL_HASHES, "retinaface.onnx", _hash(data_ret))

    (tmp_path / "w600k_r50.onnx").write_bytes(data_rec)
    (tmp_path / "det_10g.onnx").write_bytes(data_det)
    (tmp_path / "retinaface.onnx").write_bytes(data_ret)

    # 1. Khi force=False -> không được gọi urlopen
    mock_urlopen = MagicMock()
    with patch("urllib.request.urlopen", mock_urlopen):
        download_all(str(tmp_path), force=False)
    mock_urlopen.assert_not_called()

    # 2. Khi force=True -> phải gọi tải lại
    monkeypatch.setitem(MODEL_HASHES, "w600k_r50.onnx", _hash(new_rec))
    monkeypatch.setitem(MODEL_HASHES, "det_10g.onnx", _hash(new_det))
    monkeypatch.setitem(MODEL_HASHES, "retinaface.onnx", _hash(new_ret))

    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w") as z:
        z.writestr("w600k_r50.onnx", new_rec)
        z.writestr("det_10g.onnx", new_det)
    zip_bytes = zip_bio.getvalue()

    def fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "buffalo" in url:
            return MockResponse(zip_bytes)
        return MockResponse(new_ret)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen) as mock_dl:
        download_all(str(tmp_path), force=True)
        assert mock_dl.call_count == 2

    # Các file đã được cập nhật nội dung mới
    assert (tmp_path / "w600k_r50.onnx").read_bytes() == new_rec
    assert (tmp_path / "det_10g.onnx").read_bytes() == new_det
    assert (tmp_path / "retinaface.onnx").read_bytes() == new_ret
