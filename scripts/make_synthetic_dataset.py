#!/usr/bin/env python3
"""Sinh bộ dữ liệu giả lập cùng cấu trúc IITB-Corridor để KIỂM TRA NHANH pipeline
(không dùng cho kết quả đồ án).

  Train/<id>/<id>.avi : người đi bộ chậm qua hành lang (bình thường)
  Test/<id>/<id>.avi  : có đoạn chạy nhanh / di chuyển hỗn loạn (bất thường)
  Test/<id>/<id>.npy  : nhãn 0/1 cho từng khung hình

    python scripts/make_synthetic_dataset.py --out data/synthetic --n-train 12 --n-test 6
    python scripts/run_pipeline.py --config configs/synthetic.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def background(h, w, rng):
    bg = np.full((h, w), 110, np.float32)
    bg[: h // 3] = 150                                     # tường
    bg[h // 3: h // 3 + 3] = 60                            # chân tường
    noise = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 3)
    return bg + noise


def draw_person(img, cx, cy, scale, shade):
    cv2.ellipse(img, (int(cx), int(cy)), (int(6 * scale), int(16 * scale)), 0, 0, 360, shade, -1)
    cv2.circle(img, (int(cx), int(cy - 20 * scale)), int(5 * scale), shade + 30, -1)


def make_video(path, n_frames, abnormal_ranges, rng, w=320, h=180, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), isColor=True)
    bg = background(h, w, rng)
    people = [{"x": rng.uniform(0, w), "y": rng.uniform(h * 0.5, h * 0.9),
               "vx": rng.choice([-1, 1]) * rng.uniform(0.6, 1.2), "vy": 0.0,
               "s": rng.uniform(0.9, 1.3), "c": int(rng.uniform(20, 70))} for _ in range(rng.integers(2, 5))]
    labels = np.zeros(n_frames, np.uint8)
    for a, b in abnormal_ranges:
        labels[a:b] = 1
    for t in range(n_frames):
        img = bg.copy()
        ab = labels[t] == 1
        for p in people:
            if ab:      # chạy nhanh, đổi hướng hỗn loạn
                p["vx"] = np.clip(p["vx"] + rng.normal(0, 1.5), -7, 7)
                p["vy"] = np.clip(p["vy"] + rng.normal(0, 1.0), -4, 4)
            else:       # đi bộ đều, quay về vận tốc bình thường
                sgn = np.sign(p["vx"]) or 1.0
                p["vx"] = 0.8 * p["vx"] + 0.2 * sgn * 1.0
                p["vy"] *= 0.8
            p["x"] = (p["x"] + p["vx"]) % w
            p["y"] = float(np.clip(p["y"] + p["vy"], h * 0.45, h * 0.95))
            draw_person(img, p["x"], p["y"], p["s"], p["c"])
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    for i in range(a.n_train):
        vid = f"{209 + i:06d}"
        make_video(root / "Train" / vid / f"{vid}.avi", a.frames, [], rng)
    for i in range(a.n_test):
        vid = f"{1 + i:06d}"
        s = int(rng.integers(a.frames // 4, a.frames // 2))
        lab = make_video(root / "Test" / vid / f"{vid}.avi", a.frames, [(s, s + a.frames // 3)], rng)
        np.save(root / "Test" / vid / f"{vid}.npy", lab)
    print(f"Đã tạo {a.n_train} video train, {a.n_test} video test tại {root}")


if __name__ == "__main__":
    main()
