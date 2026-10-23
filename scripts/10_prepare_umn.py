#!/usr/bin/env python3
"""Chuẩn bị bộ UMN (file gốc Crowd-Activity-All.avi) cho pipeline của project.

File UMN gốc là MỘT video 320×240, 30 fps, ghép liền 11 đoạn của 3 cảnh
(bãi cỏ, trong nhà, quảng trường). Mỗi đoạn bắt đầu bình thường rồi chuyển sang
cảnh mọi người hoảng loạn bỏ chạy; khi bất thường, video có in sẵn dòng chữ
"Abnormal Crowd Activity" ở góc trên-trái.

Script tự động:
  1. Phát hiện dòng chữ trên từng khung → nhãn bất thường (0/1) theo khung.
  2. Phát hiện ranh giới 11 đoạn (cắt cảnh + thời điểm dòng chữ tắt).
  3. CHE dòng chữ ở MỌI khung bằng nền tĩnh của đoạn → dòng chữ không tạo
     ra dòng quang học giả và mô hình không "nhìn" được nhãn.
  4. UMN không có cách chia chính thức → mỗi đoạn được tách theo thời gian:
       Train: `train_ratio` đầu tiên của phần BÌNH THƯỜNG
       Test : toàn bộ phần còn lại của đoạn (bình thường + bất thường)
     → không khung nào vừa nằm trong train vừa nằm trong test.
  5. Ghi data/UMN_prepared/{Train,Test}/*.avi, nhãn Test/*.npy, test.txt (loại = cảnh)
     và configs/umn.yaml.

Dòng chữ của UMN xuất hiện TRỄ hơn thời điểm đám đông bắt đầu chạy 0–1,6 s.
Tùy chọn --refine-onset dời điểm bắt đầu bất thường về lúc năng lượng chuyển
động tăng vọt (mặc định TẮT, dùng đúng nhãn gốc của bộ dữ liệu).

    python scripts/10_prepare_umn.py          # tự tìm data/UNM/**/Crowd-Activity-All.avi
    python scripts/10_prepare_umn.py --video "D:/New folder/ocsvm_anomaly_fpga/data/UNM/Abnormal Crowds/Crowd-Activity-All.avi"
    python scripts/run_pipeline.py --config configs/umn.yaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Vùng chữ "Abnormal Crowd Activity" (tọa độ trên khung gốc 320×240)
BANNER_PROBE = (slice(10, 28), slice(8, 168))      # vùng đo để phát hiện chữ
BANNER_MASK = (slice(0, 34), slice(0, 178))        # vùng che (rộng hơn một chút)
SCENE_NAMES = {0: "Lawn", 1: "Indoor", 2: "Plaza"}
UMN_FILE = "Crowd-Activity-All.avi"
DEFAULT_SEARCH = [ROOT / "data" / "UNM", ROOT / "data"]     # nơi tự tìm file UMN gốc


def locate_umn(explicit: str | None) -> Path:
    """Đường dẫn file UMN gốc: --video nếu có, nếu không tìm trong data/UNM rồi data/."""
    if explicit:
        p = Path(explicit)
        p = p if p.is_absolute() else ROOT / p
        if not p.is_file():
            sys.exit(f"Không thấy file: {p}")
        return p
    for base in DEFAULT_SEARCH:
        if base.is_dir():
            hits = sorted(q for q in base.rglob("*") if q.is_file() and q.name.lower() == UMN_FILE.lower())
            if hits:
                return hits[0]
    sys.exit(f"Không tìm thấy {UMN_FILE} trong {', '.join(str(b) for b in DEFAULT_SEARCH)} – dùng --video")


def read_all(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def banner_flags(frames) -> np.ndarray:
    """1 nếu khung có dòng chữ: nền trắng + chữ đỏ trong vùng góc trên-trái."""
    out = np.zeros(len(frames), np.uint8)
    for i, f in enumerate(frames):
        r = f[BANNER_PROBE].astype(np.int32)
        white = ((r[..., 0] > 200) & (r[..., 1] > 200) & (r[..., 2] > 200)).mean()
        red = ((r[..., 2] > 150) & (r[..., 1] < 90) & (r[..., 0] < 90)).mean()
        out[i] = white > 0.4 and red > 0.05
    # lọc nhiễu: bỏ các đoạn bật/tắt ngắn hơn 3 khung
    k = np.convolve(out, np.ones(5), "same")
    return (k >= 3).astype(np.uint8)


def motion_energy(frames) -> np.ndarray:
    e = np.zeros(len(frames))
    prev = None
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            e[i] = np.abs(g - prev).mean()
        prev = g
    return e


def find_clips(flags, energy, scene_jump=40.0, merge_gap=25):
    """Ranh giới đoạn = cắt cảnh (năng lượng nhảy vọt) ∪ thời điểm dòng chữ tắt."""
    cuts = [int(i) for i in np.flatnonzero(energy > scene_jump)]
    offs = [int(i) + 1 for i in np.flatnonzero(np.diff(flags.astype(int)) == -1)]
    starts = sorted(set([0] + cuts))
    for o in offs:                     # chữ tắt ngay trước một lần cắt cảnh → dùng điểm cắt cảnh
        if not any(0 <= c - o <= merge_gap for c in cuts) and o < len(flags):
            starts.append(o)
    starts = sorted(set(starts))
    clips = [(s, e) for s, e in zip(starts, starts[1:] + [len(flags)]) if e - s > 30]
    scene_of, scene = [], 0
    for s, _ in clips:
        if s in cuts:
            scene += 1
        scene_of.append(scene)
    return clips, scene_of


def refine_onset(energy, s, a, k_base=1.8, smooth=9):
    """Dời điểm bắt đầu bất thường a về lúc năng lượng chuyển động vượt k_base × mức nền."""
    sm = np.convolve(energy, np.ones(smooth) / smooth, "same")
    base = np.median(sm[s + 5:max(s + 6, a - 60)])
    k = a
    while k > s + 10 and sm[k - 1] > k_base * base:
        k -= 1
    return k


def write_video(path: Path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None,
                    help="đường dẫn Crowd-Activity-All.avi (mặc định: tự tìm trong data/UNM)")
    ap.add_argument("--out", default="data/UMN_prepared",
                    help="thư mục ghi dữ liệu đã chuẩn bị (KHÔNG ghi đè thư mục video gốc data/UNM)")
    ap.add_argument("--train-ratio", type=float, default=0.5,
                    help="tỉ lệ đầu của phần bình thường mỗi đoạn dùng để huấn luyện")
    ap.add_argument("--per-scene", action="store_true",
                    help="tạo thêm 3 bộ con data/UMN_lawn|indoor|plaza + configs/umn_<cảnh>.yaml "
                         "(mỗi cảnh một mô hình – cách đánh giá UMN phổ biến trong các bài báo)")
    ap.add_argument("--refine-onset", action="store_true",
                    help="dời điểm bắt đầu bất thường về lúc chuyển động tăng vọt")
    a = ap.parse_args()

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    video = locate_umn(a.video)
    print(f"File UMN: {video}")
    frames, fps = read_all(video)
    print(f"Đọc {len(frames)} khung, {frames[0].shape[1]}×{frames[0].shape[0]}, {fps:.0f} fps")
    flags = banner_flags(frames)
    energy = motion_energy(frames)
    clips, scene_of = find_clips(flags, energy)
    print(f"Tìm thấy {len(clips)} đoạn" + ("" if len(clips) == 11 else "  ⚠ UMN chuẩn có 11 đoạn – kiểm tra lại"))

    subsets = [("", None)]
    if a.per_scene:
        subsets += [("_" + SCENE_NAMES[i].lower(), i) for i in sorted(set(scene_of))]
    for suffix, only in subsets:
        build(frames, fps, flags, energy, clips, scene_of, a,
              Path(str(out) + suffix), "umn" + suffix, only)


def build(frames, fps, flags, energy, clips, scene_of, a, out, cfg_name, only_scene):
    lines, cls_lines = [], []
    n_train = n_test = n_ab = 0
    for k, ((s, e), sc) in enumerate(zip(clips, scene_of), 1):
        if only_scene is not None and sc != only_scene:
            continue
        lab = flags[s:e].copy()
        ab = np.flatnonzero(lab)
        if len(ab) == 0:
            print(f"  ⚠ đoạn {k} không có khung bất thường – bỏ qua")
            continue
        onset = s + int(ab[0])
        if a.refine_onset:
            new = refine_onset(energy, s, onset)
            lab[new - s:onset - s] = 1
            onset = new
        # che dòng chữ bằng nền tĩnh lấy từ khung đầu đoạn (chắc chắn không có chữ)
        patch = frames[s][BANNER_MASK].copy()
        clip = []
        for f in frames[s:e]:
            g = f.copy()
            g[BANNER_MASK] = patch
            clip.append(g)
        split = int((onset - s) * a.train_ratio)
        name = f"clip{k:02d}_{SCENE_NAMES.get(sc, f'scene{sc}').lower()}"
        write_video(out / "Train" / f"{name}_train.avi", clip[:split], fps)
        write_video(out / "Test" / f"{name}_test.avi", clip[split:], fps)
        np.save(out / "Test" / f"{name}_test.npy", lab[split:].astype(np.uint8))
        cls_lines.append(f"{SCENE_NAMES.get(sc, 'Scene')}_{k} {name}_test")
        n_train += split
        n_test += e - s - split
        n_ab += int(lab[split:].sum())
        lines.append(f"  {name:20s} khung {s:5d}–{e - 1:5d} | bất thường {onset:5d}–{s + int(ab[-1]):5d} "
                     f"| train {split:4d} khung | test {e - s - split:4d} khung")
    print(f"--- {out.name} ---")
    print("\n".join(lines))
    (out / "test.txt").write_text("\n".join(cls_lines) + "\n", encoding="utf-8")
    print(f"Train {n_train} khung | Test {n_test} khung ({n_ab} bất thường, {n_ab / max(n_test, 1):.1%})")

    rel = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out.as_posix()
    cfg = f"""# UMN – sinh bởi scripts/10_prepare_umn.py
project:
  output_dir: outputs_{cfg_name}
data:
  train_dir: {rel}/Train
  test_dir: {rel}/Test
  annotation_dir: {rel}/Test
  test_class_file: {rel}/test.txt
  val_video_ratio: 0.0           # ít video train ngắn → ngưỡng lấy từ điểm số train
preprocess:
  frame_step: 2                  # UMN 30 fps → 15 fps
"""
    cfg_path = ROOT / "configs" / f"{cfg_name}.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    print(f"Đã ghi {out} và {cfg_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()