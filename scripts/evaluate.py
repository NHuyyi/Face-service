"""Thử nghiệm matching hàng loạt: cho một thư mục ảnh, so khớp mọi cặp và báo cáo kết quả.

Cấu trúc thư mục (mỗi thư mục con = một người):
    dataset/
      An/   a1.jpg a2.jpg
      Binh/ b1.jpg b2.jpg

    python scripts/evaluate.py dataset --threshold 0.40 --out report

Kết quả:
  - report/similarity_matrix.csv : ma trận cosine similarity giữa mọi ảnh
  - report/pairs.csv             : từng cặp, nhãn đúng (cùng người?), similarity, Match/0 Match
  - in ra thống kê genuine/impostor, TP/FP/TN/FN, FAR/FRR và ngưỡng gợi ý.
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import Settings  # noqa: E402
from app.engine import FaceEngine, NoFaceError  # noqa: E402
from app.engine.align import AlignmentError  # noqa: E402
from app.engine.image import read_image  # noqa: E402
from app.matching import batch_cosine_similarity  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_dataset(root):
    items = []
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        for fn in sorted(os.listdir(pdir)):
            if fn.lower().endswith(EXTS):
                items.append((person, os.path.join(pdir, fn)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="report")
    a = ap.parse_args()

    cfg = Settings()
    thr = cfg.match_threshold if a.threshold is None else a.threshold
    eng = FaceEngine(cfg)
    os.makedirs(a.out, exist_ok=True)

    names, labels, vecs = [], [], []
    for person, path in load_dataset(a.dataset):
        try:
            img = read_image(path)
        except (OSError, ValueError):
            print(f"[bỏ qua] không đọc được {path}")
            continue
        try:
            r = eng.extract(img)
        except NoFaceError:
            print(f"[bỏ qua] không thấy mặt: {path}")
            continue
        except AlignmentError as e:
            print(f"[bỏ qua] không căn chỉnh được {path}: {e}")
            continue
        names.append(os.path.relpath(path, a.dataset)); labels.append(person); vecs.append(r.embedding)
        print(f"ok  {names[-1]:40s} rot={r.rotation_applied:3d} roll={r.roll_degrees:6.1f} score={r.det_score:.2f}")

    n = len(vecs)
    if n < 2:
        sys.exit("Cần ít nhất 2 ảnh hợp lệ")
    M = np.stack(vecs)
    S = np.stack([batch_cosine_similarity(M[i], M) for i in range(n)])  # NxN

    with open(os.path.join(a.out, "similarity_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + names)
        for i in range(n):
            w.writerow([names[i]] + [f"{S[i, j]:.4f}" for j in range(n)])

    pairs, same, sims = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        s = float(S[i, j]); is_same = labels[i] == labels[j]
        pairs.append((names[i], names[j], is_same, s)); same.append(is_same); sims.append(s)
    same, sims = np.array(same), np.array(sims)

    with open(os.path.join(a.out, "pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_a", "image_b", "same_person", "similarity", "result", "correct"])
        for x, y, sp, s in pairs:
            ok = s >= thr
            w.writerow([x, y, int(sp), f"{s:.4f}", "Match" if ok else "0 Match", int(ok == sp)])

    pred = sims >= thr
    tp = int((pred & same).sum()); fp = int((pred & ~same).sum())
    tn = int((~pred & ~same).sum()); fn = int((~pred & same).sum())
    print(f"\n{n} ảnh, {len(sims)} cặp ({int(same.sum())} cùng người, {int((~same).sum())} khác người)")
    if same.any():
        print(f"Cùng người : mean={sims[same].mean():.3f} min={sims[same].min():.3f}")
    if (~same).any():
        print(f"Khác người : mean={sims[~same].mean():.3f} max={sims[~same].max():.3f}")
    print(f"\nNgưỡng {thr:.2f}: TP={tp} FP={fp} TN={tn} FN={fn}  "
          f"accuracy={(tp + tn) / len(sims):.3f}  "
          f"FAR={fp / max(1, fp + tn):.3f}  FRR={fn / max(1, fn + tp):.3f}")

    if same.any() and (~same).any():
        cand = np.linspace(-0.2, 0.9, 111)
        accs = [((sims >= t) == same).mean() for t in cand]
        best = cand[int(np.argmax(accs))]
        print(f"Ngưỡng gợi ý theo dữ liệu này: {best:.2f} (accuracy {max(accs):.3f})")

    if n <= 12:
        print("\nMa trận similarity:")
        print(" " * 14 + " ".join(f"{i:6d}" for i in range(n)))
        for i in range(n):
            print(f"{i:2d} {names[i][:10]:11s}" + " ".join(f"{S[i, j]:6.2f}" for j in range(n)))
    print(f"\nĐã ghi báo cáo vào {a.out}/")


if __name__ == "__main__":
    main()
