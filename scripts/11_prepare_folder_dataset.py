#!/usr/bin/env python3
"""Chuẩn bị bộ dữ liệu dạng HAI THƯ MỤC: video bình thường / video bất thường.

Ví dụ cấu trúc (UMN / UCF Web "Abnormal Crowd"):
    UMN/
    ├── Normal Crowds/    *.mov | *.avi | *.mp4 ...   (toàn bộ video bình thường)
    └── Abnormal Crowds/  *.mov ...                    (video có hành vi bất thường)

Bộ dạng này chỉ có NHÃN CẢ VIDEO, không có nhãn từng khung. Script:
  1. Chia video Normal thành Train / Test (cố định theo seed).
     --split video : mỗi video Normal thuộc HẲN train hoặc test (mặc định – công bằng
                     khi các video quay ở nhiều cảnh khác nhau)
     --split time  : mỗi video Normal cắt theo thời gian, phần đầu train, phần sau test
                     (chỉ dùng khi mọi video cùng MỘT camera/cảnh)
  2. Toàn bộ video Abnormal vào Test, nhãn mọi khung = 1 (trừ khi có --intervals).
     Video Normal ở Test có nhãn mọi khung = 0.
  3. Ghi data/<ten>/{Train,Test}, nhãn .npy, test.txt (Normal_k / Abnormal_k)
     và configs/<ten>.yaml. Bước 03 sẽ in thêm AUC MỨC VIDEO (thước đo đúng cho bộ này).

Nếu TẤT CẢ video nằm chung một thư mục (không có 2 thư mục con), dùng file nhãn
CSV "video,label" (label = normal/abnormal hoặc 0/1) qua --labels. Chạy lần đầu
không có --labels, script sẽ tạo sẵn file mẫu <root>/labels.csv để bạn điền.

Nếu biết chính xác khung bất thường trong từng video Abnormal, đưa vào CSV
"video,start,end" (cùng định dạng scripts/08_annotate_videos.py) qua --intervals.

    python scripts/11_prepare_folder_dataset.py                     # mặc định --root data/UNM
    python scripts/11_prepare_folder_dataset.py --root "D:/New folder/ocsvm_anomaly_fpga/data/UNM"
    python scripts/run_pipeline.py --config configs/umn_folder.yaml
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
DEFAULT_ROOT = ROOT / "data" / "UNM"
# File UMN gốc chứa CẢ đoạn bình thường lẫn bất thường (nhãn từng khung) → không được coi
# cả file là "bất thường"; xử lý riêng bằng scripts/10_prepare_umn.py
ALWAYS_EXCLUDE = {"crowd-activity-all.avi"}


EXCLUDED_SEEN: list[Path] = []


def list_videos(d: Path) -> list[Path]:
    out = []
    for p in sorted(d.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in EXTS):
            continue
        if p.name.lower() in ALWAYS_EXCLUDE:
            EXCLUDED_SEEN.append(p)
            continue
        out.append(p)
    return out


def pick_dir(root: Path, explicit: str | None, want_abnormal: bool) -> Path | None:
    """Thư mục con có chữ 'abnormal'/'anomal' (bất thường) hoặc 'normal' nhưng không 'abnormal'."""
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else root / p
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        n = d.name.lower()
        is_abn = "abnormal" in n or "anomal" in n
        if want_abnormal and is_abn:
            return d
        if not want_abnormal and "normal" in n and not is_abn:
            return d
    return None


def split_by_label_file(root: Path, labels: str | None) -> tuple[list[Path], list[Path]]:
    """Mọi video chung một thư mục: đọc nhãn từ CSV; chưa có CSV thì tạo file mẫu rồi dừng."""
    videos = list_videos(root)
    if not videos:
        sys.exit(f"Không có video trong {root}")
    if labels is None:
        tpl = root / "labels.csv"
        if not tpl.exists():
            lines = ["video,label"] + [f"{v.relative_to(root).as_posix()},TODO" for v in videos]
            tpl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sys.exit(f"Mọi video nằm chung một thư mục → cần biết video nào bình thường/bất thường.\n"
                 f"Đã tạo file mẫu {tpl} ({len(videos)} video). Điền cột label = normal hoặc abnormal "
                 f"(hoặc 0/1) rồi chạy lại với:  --labels \"{tpl}\"")
    lab = {}
    with open(labels, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() == "video":
                continue
            v = row[1].strip().lower()
            if v in ("normal", "0", "n"):
                lab[Path(row[0].strip()).as_posix()] = 0
            elif v in ("abnormal", "1", "a", "anomaly"):
                lab[Path(row[0].strip()).as_posix()] = 1
    normals, abnormals, missing = [], [], []
    for v in videos:
        rel = v.relative_to(root).as_posix()
        key = rel if rel in lab else (v.name if v.name in lab else v.stem if v.stem in lab else None)
        if key is None:
            missing.append(rel)
        elif lab[key] == 0:
            normals.append(v)
        else:
            abnormals.append(v)
    if missing:
        print(f"  ⚠ {len(missing)} video chưa có nhãn (TODO) – bỏ qua: {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}")
    print(f"Theo {labels}: {len(normals)} video bình thường, {len(abnormals)} video bất thường")
    return normals, abnormals


def probe(path: Path) -> tuple[int, float, int, int]:
    """Đếm số khung THỰC (đọc hết video) – metadata của .mov đôi khi sai."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, fps, w, h


def cut_video(src: Path, dst: Path, start: int, end: int):
    """Ghi khung [start, end) của src ra dst (.avi MJPG)."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vw, i = None, 0
    while True:
        ok, f = cap.read()
        if not ok or i >= end:
            break
        if i >= start:
            if vw is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (f.shape[1], f.shape[0]))
            vw.write(f)
        i += 1
    cap.release()
    if vw is not None:
        vw.release()


def read_intervals(path: Path | None) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if path is None:
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip() and row[0].strip().lower() != "video":
                try:
                    out[Path(row[0].strip()).stem].append((int(float(row[1])), int(float(row[2]))))
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="thư mục chứa 2 thư mục con Normal / Abnormal (mặc định data/UNM)")
    ap.add_argument("--name", default="umn_folder", help="tên bộ dữ liệu → data/<ten>, configs/<ten>.yaml")
    ap.add_argument("--normal-dir", default=None, help="mặc định: thư mục con có chữ 'normal' (không phải 'abnormal')")
    ap.add_argument("--abnormal-dir", default=None, help="mặc định: thư mục con có chữ 'abnormal'")
    ap.add_argument("--split", choices=["video", "time"], default="video")
    ap.add_argument("--train-ratio", type=float, default=0.6,
                    help="video: tỉ lệ số video Normal dùng train | time: tỉ lệ thời lượng đầu mỗi video")
    ap.add_argument("--labels", default=None,
                    help="CSV video,label (normal/abnormal hoặc 0/1) khi mọi video nằm chung một thư mục")
    ap.add_argument("--intervals", default=None, help="CSV video,start,end cho khung bất thường (tùy chọn)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = Path(a.root)
    root = root if root.is_absolute() else ROOT / root
    print(f"Thư mục dữ liệu: {root}")
    if not root.is_dir():
        sys.exit(f"Không thấy thư mục {root}")
    abn_dir = pick_dir(root, a.abnormal_dir, True)
    nor_dir = pick_dir(root, a.normal_dir, False)
    if abn_dir is not None and nor_dir is not None and not a.labels:
        normals, abnormals = list_videos(nor_dir), list_videos(abn_dir)
        print(f"Normal  : {nor_dir}  → {len(normals)} video")
        print(f"Abnormal: {abn_dir}  → {len(abnormals)} video")
    else:
        normals, abnormals = split_by_label_file(root, a.labels)
    for p in EXCLUDED_SEEN:
        print(f"  ↷ bỏ qua {p.name}: đây là bộ UMN gốc (có cả đoạn bình thường) – "
              f"dùng scripts/10_prepare_umn.py cho file này")
    if not normals or not abnormals:
        sys.exit("Cần ít nhất 1 video ở mỗi thư mục")

    out = ROOT / "data" / a.name
    if out.exists():
        shutil.rmtree(out)
    (out / "Train").mkdir(parents=True)
    (out / "Test").mkdir(parents=True)
    intervals = read_intervals(Path(a.intervals) if a.intervals else None)
    rng = np.random.default_rng(a.seed)
    cls_lines, rows, fps_seen = [], [], []

    def add_test(src: Path, label: int, tag: str, k: int, start: int = 0, end: int | None = None):
        n, fps, w, h = probe(src)
        if n == 0:
            print(f"  ⚠ không đọc được {src.name} (OpenCV) – chuyển sang .avi bằng ffmpeg rồi chạy lại")
            return
        end = n if end is None else end
        stem = f"{tag.lower()}{k:02d}_{src.stem}".replace(" ", "_")
        dst = out / "Test" / f"{stem}.avi"
        cut_video(src, dst, start, end)
        m = end - start
        lab = np.full(m, label, np.uint8)
        if label == 1 and src.stem in intervals:          # có khoảng bất thường chính xác
            lab[:] = 0
            for s, e in intervals[src.stem]:
                lab[max(0, s - start):max(0, min(m, e + 1 - start))] = 1
        np.save(out / "Test" / f"{stem}.npy", lab)
        cls_lines.append(f"{tag}_{k} {stem}")
        rows.append(f"  TEST  {tag:8s} {src.name:30s} {m:5d} khung  {w}×{h} @ {fps:.0f} fps  bất thường {lab.mean():.0%}")
        fps_seen.append(fps)

    if a.split == "video":
        order = rng.permutation(len(normals))
        n_tr = max(1, min(len(normals) - 1, int(round(len(normals) * a.train_ratio))))
        tr_idx, te_idx = set(order[:n_tr].tolist()), order[n_tr:].tolist()
        for i in sorted(tr_idx):
            src = normals[i]
            n, fps, w, h = probe(src)
            if n == 0:
                print(f"  ⚠ không đọc được {src.name} – bỏ qua")
                continue
            cut_video(src, out / "Train" / f"normal_{src.stem}.avi".replace(" ", "_"), 0, n)
            rows.append(f"  TRAIN Normal   {src.name:30s} {n:5d} khung  {w}×{h} @ {fps:.0f} fps")
            fps_seen.append(fps)
        for k, i in enumerate(te_idx, 1):
            add_test(normals[i], 0, "Normal", k)
    else:
        for k, src in enumerate(normals, 1):
            n, fps, w, h = probe(src)
            if n == 0:
                continue
            cut = int(n * a.train_ratio)
            cut_video(src, out / "Train" / f"normal_{src.stem}.avi".replace(" ", "_"), 0, cut)
            rows.append(f"  TRAIN Normal   {src.name:30s} khung 0–{cut - 1}")
            add_test(src, 0, "Normal", k, cut, n)
    for k, src in enumerate(abnormals, 1):
        add_test(src, 1, "Abnormal", k)

    print("\n".join(rows))
    (out / "test.txt").write_text("\n".join(cls_lines) + "\n", encoding="utf-8")
    fps = float(np.median(fps_seen)) if fps_seen else 25.0
    step = max(1, int(round(fps / 12.5)))                  # giữ khoảng ~0,08 s giữa 2 khung như IITB
    cfg = f"""# {a.name} – sinh bởi scripts/11_prepare_folder_dataset.py (nhãn mức video)
project:
  output_dir: outputs_{a.name}
data:
  train_dir: data/{a.name}/Train
  test_dir: data/{a.name}/Test
  annotation_dir: data/{a.name}/Test
  test_class_file: data/{a.name}/test.txt
  val_video_ratio: 0.0
preprocess:
  frame_step: {step}                  # fps trung vị {fps:.0f} → ~{fps / step:.1f} khung/s
"""
    (ROOT / "configs" / f"{a.name}.yaml").write_text(cfg, encoding="utf-8")
    print(f"Đã ghi data/{a.name}/ và configs/{a.name}.yaml (frame_step = {step})")
    if a.split == "video" and len(normals) < 4:
        print("  ⚠ Rất ít video Normal – cân nhắc --split time nếu các video cùng một camera")


if __name__ == "__main__":
    main()
