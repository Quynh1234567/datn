#!/usr/bin/env python3
"""Chuẩn bị thư mục data/UNM để TEST NGUYÊN VIDEO – giống hệt cách đọc tập Test IITB.

Không cắt, không chép, không chia đoạn: pipeline đọc TRỰC TIẾP từng video đầy đủ
trong data/UNM (kể cả file .mov). Script chỉ sinh thêm nhãn + cấu hình:

  data/UNM/
  ├── Abnormal Crowds/  *.mov *.avi  → nhãn mọi khung = 1
  │   └── Crowd-Activity-All.avi    → (UMN gốc) nhãn TỪNG KHUNG lấy từ dòng chữ
  │                                    "Abnormal Crowd Activity" in trên video
  └── Normal Crowds/    *.mov *.avi  → nhãn mọi khung = 0

  Sinh ra:
  data/UNM_labels/<tên video>.npy   nhãn 0/1 cho từng khung (đủ độ dài video)
  data/UNM_labels/test.txt          loại: UMN / Abnormal / Normal
  configs/unm_full.yaml             test_dir = data/UNM, che dòng chữ của file UMN gốc

Mọi tham số tiền xử lý / dòng quang học / đặc trưng được GIỮ NGUYÊN như mô hình IITB
(file cấu hình chỉ ghi đè đường dẫn dữ liệu). Dòng chữ trên Crowd-Activity-All.avi
được che bằng ô xám cố định lúc đọc → không tạo dòng quang học giả, mô hình không
"đọc" được nhãn.

    python scripts/13_prepare_unm_full.py
    python scripts/12_cross_dataset_eval.py --umn-config configs/unm_full.yaml --jobs 8
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
UMN_FILE = "crowd-activity-all.avi"
BANNER_PROBE = (slice(10, 28), slice(8, 168))
BANNER_BOX = [0, 34, 0, 178]                 # y0, y1, x0, x1 trên khung gốc 320×240


def umn_banner_labels(path: Path) -> np.ndarray:
    """Nhãn từng khung của UMN gốc: 1 khi có dòng chữ trắng-đỏ ở góc trên-trái."""
    cap = cv2.VideoCapture(str(path))
    flags = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = f[BANNER_PROBE].astype(np.int32)
        white = ((r[..., 0] > 200) & (r[..., 1] > 200) & (r[..., 2] > 200)).mean()
        red = ((r[..., 2] > 150) & (r[..., 1] < 90) & (r[..., 0] < 90)).mean()
        flags.append(white > 0.4 and red > 0.05)
    cap.release()
    out = np.asarray(flags, np.uint8)
    return (np.convolve(out, np.ones(5), "same") >= 3).astype(np.uint8)


def count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/UNM", help="thư mục video (mặc định data/UNM)")
    ap.add_argument("--labels-dir", default="data/UNM_labels")
    ap.add_argument("--name", default="unm_full", help="tên cấu hình → configs/<ten>.yaml")
    a = ap.parse_args()

    root = Path(a.root) if Path(a.root).is_absolute() else ROOT / a.root
    lab_dir = Path(a.labels_dir) if Path(a.labels_dir).is_absolute() else ROOT / a.labels_dir
    if not root.is_dir():
        sys.exit(f"Không thấy thư mục {root}")
    lab_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    print(f"Thư mục: {root} – {len(videos)} video")

    cls_lines, counters, seen_stems = [], {"UMN": 0, "Abnormal": 0, "Normal": 0}, set()
    for v in videos:
        rel_dir = v.parent.relative_to(root).as_posix().lower()
        if v.stem in seen_stems:
            print(f"  ⚠ trùng tên {v.stem} – đổi tên một file để nhãn không bị ghi đè; bỏ qua {v}")
            continue
        if v.name.lower() == UMN_FILE:
            kind, lab = "UMN", umn_banner_labels(v)
        elif "abnormal" in rel_dir or "anomal" in rel_dir:
            kind, lab = "Abnormal", np.ones(count_frames(v), np.uint8)
        elif "normal" in rel_dir:
            kind, lab = "Normal", np.zeros(count_frames(v), np.uint8)
        else:
            print(f"  ⚠ {v.relative_to(root)} không nằm trong thư mục Normal/Abnormal – bỏ qua")
            continue
        if len(lab) == 0:
            print(f"  ⚠ không đọc được {v.name} – bỏ qua")
            continue
        seen_stems.add(v.stem)
        counters[kind] += 1
        np.save(lab_dir / f"{v.stem}.npy", lab)
        cls_lines.append(f"{kind}_{counters[kind]}\t{v.stem}")
        print(f"  {kind:8s} {v.relative_to(root).as_posix():45s} {len(lab):6d} khung, "
              f"bất thường {lab.mean():6.1%}")

    (lab_dir / "test.txt").write_text("\n".join(cls_lines) + "\n", encoding="utf-8")
    rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()  # noqa: E731
    cfg = f"""# {a.name} – sinh bởi scripts/13_prepare_unm_full.py
# Test NGUYÊN VIDEO trong {rel(root)} bằng mô hình đã huấn luyện (giữ nguyên tham số đặc trưng IITB).
project:
  output_dir: outputs_{a.name}
data:
  train_dir: {rel(lab_dir)}/_khong_co_train     # không huấn luyện trên bộ này
  test_dir: {rel(root)}
  annotation_dir: {rel(lab_dir)}
  test_class_file: {rel(lab_dir)}/test.txt
  val_video_ratio: 0.0
  video_extensions: [".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"]
preprocess:
  mask_regions:                 # che dòng chữ nhãn in sẵn trên video UMN gốc
    - pattern: "crowd-activity-all"
      box: {BANNER_BOX}
"""
    cfg_path = ROOT / "configs" / f"{a.name}.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    print(f"Nhãn → {rel(lab_dir)} | cấu hình → configs/{a.name}.yaml "
          f"({counters['UMN']} UMN, {counters['Abnormal']} Abnormal, {counters['Normal']} Normal)")


if __name__ == "__main__":
    main()
