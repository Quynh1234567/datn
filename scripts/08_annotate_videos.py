#!/usr/bin/env python3
"""Công cụ gán nhãn khung hình bất thường cho video Test (khi chưa có nhãn chính thức).

Mỗi video sinh ra:
  <stem>.npy            mảng uint8 0/1, độ dài = số khung thực tế (1 = bất thường)
  <stem>_segments.txt   các đoạn bất thường "start end" (dễ đọc / sửa tay)
đặt cạnh file video (hoặc trong --out-dir) – đúng định dạng pipeline đọc được.

CHẾ ĐỘ 1 – Giao diện xem video (OpenCV):
    python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test

  Phím tắt:
    SPACE  phát / dừng              d / a   tiến / lùi 1 khung
    l / j  tiến / lùi 1 giây         L / J   tiến / lùi 10 giây
    [      đánh dấu BẮT ĐẦU đoạn bất thường tại khung hiện tại
    ]      đánh dấu KẾT THÚC đoạn (đoạn được thêm vào danh sách)
    u      xóa đoạn vừa thêm         c       xóa tất cả đoạn của video
    + / -  tăng / giảm tốc độ phát
    s      LƯU và sang video tiếp    n       bỏ qua (không lưu)
    p      quay lại video trước      q / ESC thoát (hỏi lưu nếu có thay đổi)
  Video đã có nhãn sẽ được bỏ qua (dùng --redo để sửa lại; nhãn cũ được nạp sẵn).

CHẾ ĐỘ 0 – Tạo file CSV mẫu từ test.txt (phân loại video của IITB-Corridor):
    python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test \
        --make-template data/IITB-Corridor/test.txt --out-csv segments.csv
  → video "Normal" điền sẵn (không có đoạn bất thường), video bất thường ghi TODO để bạn điền.
  Trong chế độ giao diện, thêm --class-file test.txt để hiện loại hành vi và bỏ qua video Normal.

CHẾ ĐỘ 2 – Từ file CSV (ghi lại các đoạn khi xem bằng VLC/MPC, nhanh nhất):
    python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test --from-csv segments.csv

  segments.csv – mỗi dòng: video,start,end  (đơn vị khung hình; thêm --seconds nếu dùng giây)
      000001,120,310
      000001,540,700
      000002,0,0          ← video không có bất thường (start=end=0 với --empty-ok) hoặc bỏ trống
  Video không xuất hiện trong CSV, hoặc còn ghi TODO, sẽ KHÔNG được tạo nhãn.
  Cột thứ 4 trở đi (vd tên loại hành vi) được bỏ qua.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXTS = {".avi", ".mp4", ".mkv", ".mov"}


# ----------------------------------------------------------------- tiện ích chung
def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def label_paths(video: Path, out_dir: Path | None) -> tuple[Path, Path]:
    d = out_dir if out_dir else video.parent
    return d / f"{video.stem}.npy", d / f"{video.stem}_segments.txt"


def segments_to_mask(segments: list[tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, np.uint8)
    for s, e in segments:
        s, e = max(0, min(s, e)), min(n - 1, max(s, e))
        mask[s:e + 1] = 1
    return mask


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def save_labels(video: Path, mask: np.ndarray, out_dir: Path | None) -> Path:
    npy, txt = label_paths(video, out_dir)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, mask.astype(np.uint8))
    segs = mask_to_segments(mask)
    txt.write_text(f"# {video.name}: {len(mask)} khung, {int(mask.sum())} khung bất thường\n"
                   + "".join(f"{s} {e}\n" for s, e in segs), encoding="utf-8")
    return npy


def count_frames(video: Path) -> tuple[int, float]:
    """Đếm số khung THỰC TẾ (CAP_PROP_FRAME_COUNT của .avi đôi khi sai)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, fps


# ----------------------------------------------------------------- chế độ CSV
def from_csv(videos: list[Path], csv_path: Path, out_dir: Path | None, seconds: bool) -> None:
    segs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    pending: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for ln, row in enumerate(csv.reader(f), 1):
            if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "video":
                continue
            name = Path(row[0].strip()).stem
            a = row[1].strip() if len(row) > 1 else ""
            b = row[2].strip() if len(row) > 2 else ""
            if not a and not b:
                segs.setdefault(name, [])                  # video không có bất thường
                continue
            try:
                segs[name].append((float(a), float(b)))
            except ValueError:                             # TODO / ô chưa điền
                pending.add(name)
    for name in pending:                                   # chưa điền xong → không tạo nhãn
        segs.pop(name, None)
    if pending:
        print(f"… {len(pending)} video còn TODO/chưa điền – bỏ qua: "
              f"{', '.join(sorted(pending)[:8])}{' ...' if len(pending) > 8 else ''}")
    by_stem = {v.stem: v for v in videos}
    missing = [k for k in segs if k not in by_stem]
    if missing:
        print(f"⚠ Không tìm thấy video cho: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    for stem, s in sorted(segs.items()):
        if stem not in by_stem:
            continue
        n, fps = count_frames(by_stem[stem])
        frames = [(int(round(a * fps)), int(round(b * fps))) if seconds else (int(a), int(b)) for a, b in s]
        bad = [(a, b) for a, b in frames if b < a or a >= n]
        if bad:
            print(f"⚠ {stem}: đoạn không hợp lệ {bad} (video có {n} khung, đánh số 0..{n - 1})")
        mask = segments_to_mask(frames, n)
        save_labels(by_stem[stem], mask, out_dir)
        print(f"✔ {stem}: {n} khung, {int(mask.sum())} bất thường ({mask.mean():.1%})")
    print(f"Đã tạo nhãn cho {sum(1 for k in segs if k in by_stem)}/{len(videos)} video")


# ----------------------------------------------------------------- tạo CSV mẫu
def read_classes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        p = line.split()
        if len(p) >= 2:
            name = p[0]
            tail = name.rsplit("_", 1)
            out[Path(p[1]).stem] = tail[0] if len(tail) == 2 and tail[1].isdigit() else name
    return out


def make_template(videos: list[Path], class_file: Path, out_csv: Path) -> None:
    classes = read_classes(class_file)
    stems = {v.stem for v in videos}
    lines = ["video,start,end,class"]
    n_todo = 0
    for stem, cls in sorted(classes.items(), key=lambda kv: (kv[1] != "Normal", kv[1], kv[0])):
        if cls == "Normal":
            lines.append(f"{stem},,,Normal")
        else:
            lines.append(f"{stem},TODO,TODO,{cls}")
            n_todo += 1
    missing_vid = sorted(set(classes) - stems)
    no_class = sorted(stems - set(classes))
    for s_ in no_class:
        lines.append(f"{s_},TODO,TODO,?")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã tạo {out_csv}: {len(classes)} video trong test.txt, {n_todo} video bất thường cần điền.")
    print("Mỗi đoạn bất thường một dòng; nhiều đoạn → lặp lại tên video trên nhiều dòng.")
    if missing_vid:
        print(f"⚠ test.txt có {len(missing_vid)} mã không thấy video: {', '.join(missing_vid[:8])}")
    if no_class:
        print(f"⚠ {len(no_class)} video không có trong test.txt: {', '.join(no_class[:8])}")


# ----------------------------------------------------------------- chế độ giao diện
class Annotator:
    WIN = "Gan nhan bat thuong"

    def __init__(self, video: Path, width: int, out_dir: Path | None, cls: str = ""):
        self.video = video
        self.cls = cls
        self.out_dir = out_dir
        self.frames, self.fps = self._preload(width)
        self.n = len(self.frames)
        self.segments: list[tuple[int, int]] = []
        npy, _ = label_paths(video, out_dir)
        if npy.exists():                               # nạp nhãn cũ để sửa
            old = np.load(npy)
            self.segments = mask_to_segments(old[: self.n])
        self.pos, self.playing, self.speed, self.start_mark = 0, False, 1.0, None
        self.dirty = False

    def _preload(self, width: int):
        cap = cv2.VideoCapture(str(self.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
            if len(frames) % 200 == 0:
                print(f"\r  đang nạp {self.video.name}: {len(frames)}/{total}", end="", flush=True)
        cap.release()
        print(f"\r  đã nạp {self.video.name}: {len(frames)} khung @ {fps:.1f} fps" + " " * 10)
        return frames, fps

    def mask(self) -> np.ndarray:
        return segments_to_mask(self.segments, self.n)

    def render(self) -> np.ndarray:
        img = cv2.imdecode(self.frames[self.pos], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        bar_h = 46
        canvas = np.zeros((h + bar_h + 28, w, 3), np.uint8)
        canvas[:h] = img
        m = self.mask()
        abnormal_now = bool(m[self.pos])
        # thanh thời gian: xám = bình thường, đỏ = bất thường, vàng = đoạn đang mở
        y0 = h + 6
        cv2.rectangle(canvas, (0, y0), (w - 1, y0 + 18), (90, 90, 90), -1)
        for s, e in self.segments:
            cv2.rectangle(canvas, (int(s * w / self.n), y0), (int((e + 1) * w / self.n), y0 + 18), (0, 0, 220), -1)
        if self.start_mark is not None:
            a, b = sorted((self.start_mark, self.pos))
            cv2.rectangle(canvas, (int(a * w / self.n), y0), (int((b + 1) * w / self.n), y0 + 18), (0, 220, 220), -1)
        x = int(self.pos * w / self.n)
        cv2.line(canvas, (x, y0 - 4), (x, y0 + 22), (255, 255, 255), 2)
        t = self.pos / self.fps
        info = (f"{self.video.stem} {self.cls}  khung {self.pos}/{self.n - 1}  {int(t // 60):02d}:{t % 60:05.2f}  "
                f"x{self.speed:g}  {'PLAY' if self.playing else 'PAUSE'}  doan={len(self.segments)}"
                + ("  [DANG MO DOAN]" if self.start_mark is not None else ""))
        cv2.putText(canvas, info, (6, h + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        color = (0, 0, 255) if abnormal_now else (0, 200, 0)
        cv2.putText(canvas, "BAT THUONG" if abnormal_now else "binh thuong", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 3 if abnormal_now else 1)
        return canvas

    def seek(self, delta: int):
        self.pos = int(np.clip(self.pos + delta, 0, self.n - 1))

    def run(self) -> str:
        """Trả về: 'save' | 'skip' | 'prev' | 'quit'."""
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        sec = int(round(self.fps))
        while True:
            cv2.imshow(self.WIN, self.render())
            delay = max(1, int(1000 / (self.fps * self.speed))) if self.playing else 0
            k = cv2.waitKey(delay) & 0xFF
            if k == 255:                                   # không bấm phím (đang phát)
                if self.playing:
                    if self.pos < self.n - 1:
                        self.pos += 1
                    else:
                        self.playing = False
                continue
            if k == 27:
                return "quit"
            ch = chr(k) if 32 <= k < 127 else ""
            if not ch:
                continue
            if ch == " ":
                self.playing = not self.playing
            elif ch == "d":
                self.seek(1)
            elif ch == "a":
                self.seek(-1)
            elif ch == "l":
                self.seek(sec)
            elif ch == "j":
                self.seek(-sec)
            elif ch == "L":
                self.seek(10 * sec)
            elif ch == "J":
                self.seek(-10 * sec)
            elif ch in "+=":
                self.speed = min(8.0, self.speed * 2)
            elif ch == "-":
                self.speed = max(0.125, self.speed / 2)
            elif ch == "[":
                self.start_mark = self.pos
            elif ch == "]":
                if self.start_mark is not None:
                    a, b = sorted((self.start_mark, self.pos))
                    self.segments.append((a, b))
                    self.segments = mask_to_segments(self.mask())    # gộp các đoạn chồng nhau
                    self.start_mark, self.dirty = None, True
            elif ch == "u":
                if self.segments:
                    self.segments.pop()
                    self.dirty = True
            elif ch == "c":
                self.segments, self.start_mark, self.dirty = [], None, True
            elif ch == "s":
                if self.start_mark is not None:
                    print("  ⚠ Còn đoạn đang mở (đã bấm [ mà chưa bấm ]) – bấm ] hoặc u trước khi lưu")
                    continue
                return "save"
            elif ch == "n":
                return "skip"
            elif ch == "p":
                return "prev"
            elif ch == "q":
                return "quit"


def gui(videos: list[Path], out_dir: Path | None, width: int, redo: bool,
        classes: dict[str, str] | None = None) -> None:
    classes = classes or {}
    todo = [v for v in videos if (redo or not label_paths(v, out_dir)[0].exists())
            and classes.get(v.stem) != "Normal"]
    done0 = len(videos) - len(todo)
    print(f"{len(videos)} video, {done0} đã có nhãn, còn {len(todo)} cần gán")
    i = 0
    while 0 <= i < len(todo):
        v = todo[i]
        print(f"[{done0 + i + 1}/{len(videos)}] {v}")
        ann = Annotator(v, width, out_dir, classes.get(v.stem, ""))
        action = ann.run()
        if action == "save" or (action == "quit" and ann.dirty and _ask_save()):
            m = ann.mask()
            save_labels(v, m, out_dir)
            print(f"  ✔ lưu {v.stem}: {int(m.sum())}/{ann.n} khung bất thường ({m.mean():.1%})")
        if action == "quit":
            break
        i = i - 1 if action == "prev" else i + 1
        i = max(i, 0)
    cv2.destroyAllWindows()
    n_done = sum(label_paths(v, out_dir)[0].exists() for v in videos)
    print(f"Hoàn thành {n_done}/{len(videos)} video có nhãn")


def _ask_save() -> bool:
    try:
        return input("  Lưu thay đổi của video hiện tại? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


# ----------------------------------------------------------------- tóm tắt
def summary(videos: list[Path], out_dir: Path | None) -> None:
    tot, ab, have = 0, 0, 0
    for v in videos:
        npy = label_paths(v, out_dir)[0]
        if npy.exists():
            m = np.load(npy)
            tot, ab, have = tot + len(m), ab + int(m.sum()), have + 1
    print(f"Nhãn: {have}/{len(videos)} video | {tot} khung | {ab} khung bất thường "
          f"({ab / max(tot, 1):.1%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="nơi lưu nhãn (mặc định: cạnh video)")
    ap.add_argument("--from-csv", default=None, help="tạo nhãn hàng loạt từ CSV video,start,end")
    ap.add_argument("--seconds", action="store_true", help="start/end trong CSV tính bằng giây")
    ap.add_argument("--width", type=int, default=640, help="độ rộng hiển thị")
    ap.add_argument("--redo", action="store_true", help="mở lại cả video đã có nhãn")
    ap.add_argument("--summary", action="store_true", help="chỉ in thống kê nhãn hiện có")
    ap.add_argument("--make-template", default=None, metavar="TEST_TXT", help="tạo CSV mẫu từ test.txt")
    ap.add_argument("--out-csv", default="segments.csv")
    ap.add_argument("--class-file", default=None, help="test.txt – hiện loại hành vi, bỏ qua video Normal")
    a = ap.parse_args()
    videos = find_videos(Path(a.video_dir))
    if not videos:
        sys.exit(f"Không có video trong {a.video_dir}")
    out_dir = Path(a.out_dir) if a.out_dir else None
    if a.make_template:
        make_template(videos, Path(a.make_template), Path(a.out_csv))
    elif a.summary:
        summary(videos, out_dir)
    elif a.from_csv:
        from_csv(videos, Path(a.from_csv), out_dir, a.seconds)
        summary(videos, out_dir)
    else:
        gui(videos, out_dir, a.width, a.redo, read_classes(Path(a.class_file)) if a.class_file else None)
        summary(videos, out_dir)


if __name__ == "__main__":
    main()
