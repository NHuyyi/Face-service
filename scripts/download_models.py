"""Tải model ONNX cần thiết vào ./models (chạy 1 lần, cần Internet).

  python scripts/download_models.py

Gói InsightFace `buffalo_l` chứa:
  - w600k_r50.onnx : ArcFace (ResNet50, huấn luyện WebFace600K) -> vector 512 chiều
  - det_10g.onnx   : SCRFD-10G (detector phương án dự phòng, DETECTOR_BACKEND=scrfd)

RetinaFace (mặc định): đặt file ONNX vào models/retinaface.onnx (xem README mục 3),
hoặc truyền --retinaface-url nếu bạn có nơi lưu file ONNX của mình.
"""
import argparse
import io
import os
import urllib.request
import zipfile

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
WANTED = {"w600k_r50.onnx", "det_10g.onnx"}


def fetch(url: str) -> bytes:
    print(f"Tải {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--retinaface-url", default="", help="URL tới file retinaface .onnx (tuỳ chọn)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(fetch(BUFFALO_URL))) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in WANTED:
                with open(os.path.join(a.out, base), "wb") as f:
                    f.write(z.read(name))
                print("  ->", os.path.join(a.out, base))

    if a.retinaface_url:
        with open(os.path.join(a.out, "retinaface.onnx"), "wb") as f:
            f.write(fetch(a.retinaface_url))
        print("  ->", os.path.join(a.out, "retinaface.onnx"))
    elif not os.path.exists(os.path.join(a.out, "retinaface.onnx")):
        print("\n[!] Chưa có models/retinaface.onnx. Xem README mục 3 để export, "
              "hoặc chạy tạm với DETECTOR_BACKEND=scrfd.")


if __name__ == "__main__":
    main()
