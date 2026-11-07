"""Tải và kiểm tra tính toàn vẹn các model ONNX cần thiết vào ./models.

Lệnh mặc định:
  python scripts/download_models.py

Tải hoặc xác minh cả 3 model với mã băm SHA-256:
  - w600k_r50.onnx : ArcFace (ResNet50, WebFace600K) -> vector 512 chiều
  - det_10g.onnx   : SCRFD-10G (detector dự phòng, DETECTOR_BACKEND=scrfd)
  - retinaface.onnx: RetinaFace ResNet50 (detector mặc định, DETECTOR_BACKEND=retinaface)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from typing import Dict, Optional

# URLs mặc định
DEFAULT_BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DEFAULT_RETINAFACE_URL = "https://github.com/NHuyyi/Face-service/releases/download/models-v1.0/retinaface.onnx"

# Mã băm SHA-256 chuẩn của các mô hình
MODEL_HASHES: Dict[str, str] = {
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "retinaface.onnx": "ed011b65dcc7e3f10f307c3d013bf2d0260ded5007314a96fb20339322463544",
}

BUFFALO_MODELS = ("w600k_r50.onnx", "det_10g.onnx")


def get_ssl_context() -> ssl.SSLContext:
    """Tạo SSL context có CA certs từ certifi (nếu có) để tránh lỗi cert trên Windows."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def compute_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Tính mã băm SHA-256 của file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_model_file(path: str, expected_sha: str) -> bool:
    """Kiểm tra file tồn tại và mã băm SHA-256 có khớp với kỳ vọng hay không."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    return compute_sha256(path) == expected_sha.lower()


def download_file(
    url: str,
    dest_path: str,
    expected_sha: Optional[str] = None,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Tải file từ URL về file tạm, kiểm tra SHA-256 trước khi ghi đè vào đích."""
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)

    temp_path = f"{dest_path}.tmp"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    print(f"Tải {url} -> {os.path.basename(dest_path)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceService-Downloader/1.0"})
    ctx = get_ssl_context()
    hasher = hashlib.sha256() if expected_sha else None

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            total = resp.headers.get("Content-Length")
            total_mb = f"{int(total) / (1024 * 1024):.1f} MB" if total else "không rõ"
            print(f"  Dung lượng: {total_mb}")
            downloaded = 0
            with open(temp_path, "wb") as f:
                while chunk := resp.read(chunk_size):
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    downloaded += len(chunk)

        if expected_sha:
            actual_sha = hasher.hexdigest().lower()
            if actual_sha != expected_sha.lower():
                raise ValueError(
                    f"Mã băm SHA-256 không khớp cho {os.path.basename(dest_path)}:\n"
                    f"  Nhận:   {actual_sha}\n"
                    f"  Kỳ vọng: {expected_sha.lower()}"
                )

        # Chuyển nguyên tử file tạm sang đích
        os.replace(temp_path, dest_path)
        print(f"  [OK] Đã xác minh SHA-256 và lưu vào: {dest_path}")
        return dest_path

    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def download_buffalo_models(
    out_dir: str,
    buffalo_url: str = DEFAULT_BUFFALO_URL,
    force: bool = False,
    timeout: int = 180,
) -> None:
    """Tải và giải nén w600k_r50.onnx và det_10g.onnx từ gói buffalo_l."""
    needed = []
    for model_name in BUFFALO_MODELS:
        target_path = os.path.join(out_dir, model_name)
        expected_sha = MODEL_HASHES[model_name]
        if not force and verify_model_file(target_path, expected_sha):
            print(f"  [Bỏ qua] {model_name} đã tồn tại và hợp lệ (SHA-256 khớp).")
        else:
            needed.append(model_name)

    if not needed:
        return

    print(f"\nCần tải các model InsightFace buffalo_l: {', '.join(needed)}")
    os.makedirs(out_dir, exist_ok=True)
    temp_zip = os.path.join(out_dir, "buffalo_l.zip.tmp")
    created_tmps = []

    try:
        download_file(buffalo_url, temp_zip, expected_sha=None, timeout=timeout)
        with zipfile.ZipFile(temp_zip, "r") as z:
            name_map = {os.path.basename(n): n for n in z.namelist()}
            for model_name in needed:
                if model_name not in name_map:
                    raise KeyError(f"Không tìm thấy {model_name} trong gói zip {buffalo_url}")
                zip_entry = name_map[model_name]
                target_path = os.path.join(out_dir, model_name)
                temp_extract = f"{target_path}.tmp"
                created_tmps.append(temp_extract)

                print(f"  Giải nén {model_name} ...")
                try:
                    with z.open(zip_entry) as src, open(temp_extract, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    expected_sha = MODEL_HASHES[model_name]
                    actual_sha = compute_sha256(temp_extract)
                    if actual_sha != expected_sha.lower():
                        raise ValueError(
                            f"Mã băm SHA-256 không khớp cho {model_name}:\n"
                            f"  Nhận:   {actual_sha}\n"
                            f"  Kỳ vọng: {expected_sha.lower()}"
                        )

                    os.replace(temp_extract, target_path)
                    created_tmps.remove(temp_extract)
                    print(f"  [OK] Đã xác minh SHA-256 và lưu: {target_path}")
                except Exception:
                    if os.path.exists(temp_extract):
                        try:
                            os.remove(temp_extract)
                        except OSError:
                            pass
                    raise

    finally:
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except OSError:
                pass
        for tmp in created_tmps:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def download_retinaface_model(
    out_dir: str,
    retinaface_url: str = DEFAULT_RETINAFACE_URL,
    force: bool = False,
    timeout: int = 180,
) -> None:
    """Tải và xác minh RetinaFace ResNet50 (models/retinaface.onnx)."""
    target_path = os.path.join(out_dir, "retinaface.onnx")
    expected_sha = MODEL_HASHES["retinaface.onnx"]

    if not force and verify_model_file(target_path, expected_sha):
        print(f"  [Bỏ qua] retinaface.onnx đã tồn tại và hợp lệ (SHA-256 khớp).")
        return

    print(f"\nCần tải RetinaFace ResNet50 ONNX:")
    url = retinaface_url.strip() if retinaface_url else DEFAULT_RETINAFACE_URL
    download_file(url, target_path, expected_sha=expected_sha, timeout=timeout)


def download_all(
    out_dir: str = "models",
    retinaface_url: str = DEFAULT_RETINAFACE_URL,
    buffalo_url: str = DEFAULT_BUFFALO_URL,
    force: bool = False,
    timeout: int = 180,
) -> None:
    """Tải và xác minh tất cả 3 model cần thiết cho dự án."""
    os.makedirs(out_dir, exist_ok=True)
    print("=== KIỂM TRA & TẢI MÔ HÌNH KHUÔN MẶT (ONNX) ===")
    download_buffalo_models(out_dir, buffalo_url=buffalo_url, force=force, timeout=timeout)
    download_retinaface_model(out_dir, retinaface_url=retinaface_url, force=force, timeout=timeout)
    print("\n[HOÀN TẤT] Toàn bộ 3 mô hình đã sẵn sàng trong thư mục:", os.path.abspath(out_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tải và kiểm tra tính toàn vẹn các model ONNX (ArcFace, SCRFD, RetinaFace).")
    ap.add_argument("--out", default="models", help="Thư mục lưu model (mặc định: models)")
    ap.add_argument(
        "--retinaface-url",
        default="",
        help=f"URL file retinaface.onnx (mặc định: {DEFAULT_RETINAFACE_URL})",
    )
    ap.add_argument(
        "--buffalo-url",
        default="",
        help=f"URL gói buffalo_l.zip (mặc định: {DEFAULT_BUFFALO_URL})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Tải lại toàn bộ mô hình kể cả khi file đã tồn tại",
    )
    a = ap.parse_args()

    r_url = a.retinaface_url.strip() or DEFAULT_RETINAFACE_URL
    b_url = a.buffalo_url.strip() or DEFAULT_BUFFALO_URL

    try:
        download_all(out_dir=a.out, retinaface_url=r_url, buffalo_url=b_url, force=a.force)
    except Exception as e:
        print(f"\n[LỖI] Quá trình tải model thất bại: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
