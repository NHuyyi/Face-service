"""So sánh giải thuật CŨ (MobileFaceNet-192, app Android) với MỚI (RetinaFace + ArcFace-512)
trên CÙNG một dataset, CÙNG các cặp ảnh, CÙNG bộ chỉ số.

Đặt file này vào Face-service/scripts/ rồi chạy từ thư mục gốc repo:

    pip install ai-edge-litert matplotlib
    python scripts/compare_old_vs_new.py dataset \
        --tflite path/to/mobile_face_net.tflite --out report_compare

Dataset (giống scripts/evaluate.py, mỗi thư mục con = một người):
    dataset/An/*.jpg   dataset/Binh/*.jpg ...

Pipeline CŨ được tái hiện đúng theo FaceMatchHelper.java + FaceCaptureActivity.java:
    cắt khung mặt sát (không padding, không căn landmark) -> resize 112x112 (méo nếu khung
    không vuông) -> RGB -> (x - 127.5) / 128 -> MobileFaceNet -> vector 192 -> cosine.
    Ngưỡng của app: similarity > 0.75.
Pipeline MỚI: engine của Face-service (xoay thẳng, căn 5 landmark, ArcFace-512), ngưỡng 0.40.

Báo cáo gồm 3 cột để tách đóng góp của PIPELINE (cắt/căn mặt) và của MODEL:
    (1) CŨ        : MobileFaceNet-192 + pipeline của app (cắt sát, không căn chỉnh)
    (2) CŨ+CĂN    : MobileFaceNet-192 + căn 5 landmark như pipeline mới (chỉ khác model so với cột 3)
    (3) MỚI       : RetinaFace + ArcFace-512 (Face-service)
So (1) với (3) = hiệu quả của cả hệ thống; so (2) với (3) = hiệu quả riêng của model.

Lưu ý công bằng: cả ba cột dùng CÙNG khung mặt/landmark do RetinaFace tìm ra (app Android dùng
ML Kit nên khung có thể lệch nhẹ so với cột (1) thực tế).
"""
import argparse
import csv
import os
import sys

import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------------------------------------------------------- metrics
def cosine_matrix(vecs):
    v = np.asarray(vecs, dtype=np.float64)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v @ v.T


def pair_arrays(S, labels):
    iu, ju = np.triu_indices(len(labels), k=1)
    labels = np.asarray(labels)
    return iu, ju, S[iu, ju], labels[iu] == labels[ju]


def far_frr(sims, same, thr, strict=False):
    pred = sims > thr if strict else sims >= thr
    tp = int((pred & same).sum()); fp = int((pred & ~same).sum())
    tn = int((~pred & ~same).sum()); fn = int((~pred & same).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                acc=(tp + tn) / max(1, len(sims)),
                far=fp / max(1, fp + tn), frr=fn / max(1, fn + tp))


def roc_stats(sims, same):
    gen, imp = np.sort(sims[same]), np.sort(sims[~same])
    ng, ni = len(gen), len(imp)
    cand = np.unique(sims)
    frr = np.searchsorted(gen, cand, "left") / ng
    far = 1.0 - np.searchsorted(imp, cand, "left") / ni
    k = int(np.argmin(np.abs(far - frr)))
    lo, hi = np.searchsorted(imp, gen, "left"), np.searchsorted(imp, gen, "right")
    auc = float((lo + 0.5 * (hi - lo)).sum() / (ng * ni))
    acc = np.array([(((sims >= t) == same).mean()) for t in cand])
    out = dict(eer=float((far[k] + frr[k]) / 2), eer_thr=float(cand[k]), auc=auc,
               best_acc=float(acc.max()), best_acc_thr=float(cand[int(acc.argmax())]),
               gen_mean=float(gen.mean()), gen_std=float(gen.std()),
               imp_mean=float(imp.mean()), imp_std=float(imp.std()))
    out["dprime"] = (out["gen_mean"] - out["imp_mean"]) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12)
    for f in (0.01, 0.001):
        t = np.quantile(imp, 1 - f)
        out[f"tar@far{f}"] = float((gen > t).mean())
    return out, (far, frr, cand)


def rank1(S, labels):
    labels = np.asarray(labels)
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    counts = {l: int((labels == l).sum()) for l in set(labels)}
    ok = np.array([counts[l] >= 2 for l in labels])
    nn = S.argmax(axis=1)
    return float((labels[nn] == labels)[ok].mean())


# ----------------------------------------------------------------------------- old model
def load_tflite(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert tuple(inp["shape"]) == (1, 112, 112, 3), f"input shape lạ: {inp['shape']}"
    assert out["shape"][-1] == 192, f"output không phải 192 chiều: {out['shape']}"
    return it, inp["index"], out["index"]


def old_embedding(model, img_bgr, bbox):
    import cv2
    it, i_in, i_out = model
    h, w = img_bgr.shape[:2]
    x = max(0, int(bbox[0])); y = max(0, int(bbox[1]))
    cw = min(int(bbox[2] - bbox[0]), w - x); ch = min(int(bbox[3] - bbox[1]), h - y)  # như Android
    if cw <= 0 or ch <= 0:
        return None
    face = cv2.resize(img_bgr[y:y + ch, x:x + cw], (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    x_in = ((rgb - 127.5) / 128.0)[None]
    it.set_tensor(i_in, x_in)
    it.invoke()
    return it.get_tensor(i_out)[0].astype(np.float32)


def old_embedding_aligned(model, aligned_bgr):
    it, i_in, i_out = model
    import cv2
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    it.set_tensor(i_in, ((rgb - 127.5) / 128.0)[None])
    it.invoke()
    return it.get_tensor(i_out)[0].astype(np.float32)


# ----------------------------------------------------------------------------- main
def load_dataset(root):
    items = []
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if os.path.isdir(pdir):
            for fn in sorted(os.listdir(pdir)):
                if fn.lower().endswith(EXTS):
                    items.append((person, os.path.join(pdir, fn)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--tflite", required=True, help="app/src/main/assets/mobile_face_net.tflite")
    ap.add_argument("--out", default="report_compare")
    ap.add_argument("--thr-old", type=float, default=0.75)
    ap.add_argument("--thr-new", type=float, default=0.40)
    a = ap.parse_args()

    import cv2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.config import Settings
    from app.engine import FaceEngine, NoFaceError
    from app.engine.align import align_face

    eng = FaceEngine(Settings())
    old = load_tflite(a.tflite)
    os.makedirs(a.out, exist_ok=True)

    names, labels, v_old, v_oal, v_new, rotated = [], [], [], [], [], 0
    skipped = []
    for person, path in load_dataset(a.dataset):
        img = cv2.imread(path)
        if img is None:
            skipped.append((path, "không đọc được")); continue
        try:
            upright, faces, angle = eng._detect_upright(img)
        except NoFaceError:
            skipped.append((path, "không thấy mặt")); continue
        face = eng._pick_main(faces)
        aligned = align_face(upright, face.kps, 112)
        e_new = eng.recognizer.embed(aligned)
        e_oal = old_embedding_aligned(old, aligned)
        e_old = old_embedding(old, upright, face.bbox)
        if e_old is None:
            skipped.append((path, "khung mặt lỗi")); continue
        names.append(os.path.relpath(path, a.dataset)); labels.append(person)
        v_old.append(e_old); v_oal.append(e_oal); v_new.append(e_new); rotated += int(angle != 0)
        print(f"ok {names[-1]:40s} rot={angle:3d}")

    n = len(names)
    if n < 2 or len(set(labels)) < 2:
        sys.exit("Cần ít nhất 2 người và 2 ảnh hợp lệ")

    M = [  # (nhãn cột, vector, ngưỡng mặc định, so sánh '>' như Android?, kích thước vector)
        ("CŨ: MobileFaceNet-192 (pipeline app)", v_old, a.thr_old, True, 192),
        ("CŨ + căn chỉnh landmark", v_oal, a.thr_old, True, 192),
        ("MỚI: ArcFace-512", v_new, a.thr_new, False, 512),
    ]
    Ss = [cosine_matrix(v) for _, v, _, _, _ in M]
    iu, ju, _, same = pair_arrays(Ss[0], labels)
    sims = [S[iu, ju] for S in Ss]
    if not same.any() or same.all():
        sys.exit("Cần cả cặp cùng người lẫn khác người")

    with open(os.path.join(a.out, "pairs_both.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "same_person", "sim_old", "sim_old_aligned", "sim_new",
                    "correct_old", "correct_old_aligned", "correct_new"])
        for k in range(len(iu)):
            ok = [int(((s[k] > t) if st else (s[k] >= t)) == same[k])
                  for s, (_, _, t, st, _) in zip(sims, M)]
            w.writerow([names[iu[k]], names[ju[k]], int(same[k])] + [f"{s[k]:.4f}" for s in sims] + ok)

    R = [roc_stats(s, same) for s in sims]
    C = [far_frr(s, same, t, strict=st) for s, (_, _, t, st, _) in zip(sims, M)]
    RK = [rank1(S, labels) for S in Ss]
    ms = lambda r, p: f"{r[p + '_mean']:.3f} ± {r[p + '_std']:.3f}"
    rows = [
        ("Số ảnh / số người / số cặp", f"{n} / {len(set(labels))} / {len(iu)}", "", ""),
        ("Số cặp cùng người / khác người", f"{int(same.sum())} / {int((~same).sum())}", "", ""),
        ("Kích thước vector", *[str(m[4]) for m in M]),
        ("Ngưỡng mặc định", *[f"{m[2]:.2f}" for m in M]),
        ("Accuracy tại ngưỡng mặc định", *[f"{c['acc']:.3%}" for c in C]),
        ("FAR tại ngưỡng mặc định", *[f"{c['far']:.3%}" for c in C]),
        ("FRR tại ngưỡng mặc định", *[f"{c['frr']:.3%}" for c in C]),
        ("EER", *[f"{r['eer']:.3%}" for r, _ in R]),
        ("AUC", *[f"{r['auc']:.4f}" for r, _ in R]),
        ("Accuracy tối ưu (ngưỡng tốt nhất)", *[f"{r['best_acc']:.3%} @ {r['best_acc_thr']:.2f}" for r, _ in R]),
        ("TAR @ FAR=1%", *[f"{r['tar@far0.01']:.3%}" for r, _ in R]),
        ("TAR @ FAR=0.1%", *[f"{r['tar@far0.001']:.3%}" for r, _ in R]),
        ("Rank-1 (nhận diện 1:N)", *[f"{x:.3%}" for x in RK]),
        ("Similarity cùng người (TB ± SD)", *[ms(r, "gen") for r, _ in R]),
        ("Similarity khác người (TB ± SD)", *[ms(r, "imp") for r, _ in R]),
        ("d' (độ tách hai phân phối)", *[f"{r['dprime']:.2f}" for r, _ in R]),
    ]
    heads = ["Chỉ số"] + [m[0] for m in M]
    with open(os.path.join(a.out, "comparison.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(heads); w.writerows(rows)

    md = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", f"Ảnh cần xoay để mặt thẳng: {rotated}/{n} (app Android cũ không xử lý được các ảnh này).",
           f"Ảnh bị loại (cả ba cột): {len(skipped)}"]
    md += [f"- {p}: {why}" for p, why in skipped]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.5))
        for (r, (far, frr, _)), m in zip(R, M):
            ax[0].plot(far, 1 - frr, label=m[0])
        ax[0].set(xscale="log", xlabel="FAR", ylabel="TAR", title="ROC (trục X log)")
        ax[0].legend(fontsize=7)
        for ax_i, s, m in zip(ax[1:], sims, M):
            ax_i.hist(s[same], bins=40, alpha=.6, label="Cùng người")
            ax_i.hist(s[~same], bins=40, alpha=.6, label="Khác người")
            ax_i.axvline(m[2], color="k", ls="--", label=f"ngưỡng {m[2]}")
            ax_i.set(title=m[0], xlabel="cosine"); ax_i.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "roc_and_hist.png"), dpi=150)
    except ImportError:
        print("(bỏ qua biểu đồ: chưa cài matplotlib)")
    print(f"\nĐã ghi vào {a.out}/ : comparison.csv, summary.md, pairs_both.csv, roc_and_hist.png")


if __name__ == "__main__":
    main()
