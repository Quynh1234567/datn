"""Tìm video IITB-Corridor, đọc nhãn khung hình và chia tập train/val/test.

Cấu trúc IITB-Corridor thường gặp sau khi giải nén:
    Train/000209/000209.avi ...          (208 video chỉ chứa hành vi bình thường)
    Test/000001/000001.avi ...           (video kiểm thử có hành vi bất thường)
    Nhãn: mảng 0/1 theo từng khung hình (.npy) – tên file trùng tên video.
Vì các bản phát hành có thể khác nhau, mọi thứ đều được tìm đệ quy và
các mẫu tên file nhãn được cấu hình trong `data.annotation_patterns`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VideoItem:
    path: Path
    split: str                               # train | val | test
    labels_path: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def uid(self) -> str:
        """Định danh duy nhất (thư mục cha + tên) – dùng làm tên file cache."""
        return f"{self.path.parent.name}__{self.path.stem}" if self.path.parent.name != self.path.stem \
            else self.path.stem


def find_videos(root: str | Path, extensions=(".avi", ".mp4")) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_annotation(stem: str, ann_root: str | Path | None, patterns: list[str],
                    video_path: Path | None = None) -> Path | None:
    """Tìm file nhãn theo các mẫu tên; ưu tiên thư mục chứa video rồi tới ann_root."""
    candidates_dirs = []
    if video_path is not None:
        candidates_dirs.append(video_path.parent)
    if ann_root is not None and Path(ann_root).exists():
        candidates_dirs.append(Path(ann_root))
    for d in candidates_dirs:
        for pat in patterns:
            name = pat.format(stem=stem)
            direct = d / name
            if direct.is_file():
                return direct
    if ann_root is not None and Path(ann_root).exists():
        for pat in patterns:
            hits = list(Path(ann_root).rglob(pat.format(stem=stem)))
            if hits:
                return hits[0]
    return None


def _intervals_to_mask(intervals: np.ndarray, n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=np.uint8)
    for s, e in intervals.reshape(-1, 2).astype(int):
        mask[max(0, s): min(n_frames, e + 1)] = 1
    return mask


def load_frame_labels(path: str | Path, n_frames: int | None = None) -> np.ndarray:
    """Trả về mảng nhãn 0/1 cho từng khung hình (1 = bất thường).

    Hỗ trợ:
      .npy / .mat : mảng 0/1 theo khung, hoặc mảng (k, 2) các khoảng [start, end]
      .txt / .csv : mỗi dòng "start end" (khoảng khung bất thường, tính cả hai đầu)
                    hoặc một dãy 0/1 theo khung; dòng bắt đầu bằng '#' bị bỏ qua.
    File khoảng rỗng (không dòng nào) = video không có khung bất thường.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".npy", ".mat"):
        if suffix == ".npy":
            arr = np.asarray(np.load(path, allow_pickle=True))
        else:
            from scipy.io import loadmat
            mat = loadmat(path)
            arr = np.asarray(next(v for k, v in mat.items()
                                  if not k.startswith("__") and isinstance(v, np.ndarray)))
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] != 1 or \
                (arr.ndim == 2 and arr.shape == (1, 2) and arr.max() > 1):
            if n_frames is None:
                n_frames = int(arr.max()) + 1
            return _intervals_to_mask(arr, n_frames)
        return (arr.reshape(-1) > 0).astype(np.uint8)

    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#")[0].replace(",", " ").strip()
        if line:
            rows.append([float(t) for t in line.split()])
    if not rows:                                         # không có đoạn bất thường
        if n_frames is None:
            raise ValueError(f"{path}: file rỗng – cần biết số khung của video")
        return np.zeros(n_frames, np.uint8)
    if all(len(r) == 2 for r in rows):                   # các khoảng start end
        arr = np.array(rows)
        if n_frames is None:
            n_frames = int(arr.max()) + 1
        return _intervals_to_mask(arr, n_frames)
    flat = np.array([v for r in rows for v in r])        # dãy 0/1 theo khung
    return (flat > 0).astype(np.uint8)


def load_test_classes(path: str | Path | None) -> dict[str, str]:
    """Đọc test.txt dạng "CarryObject_3 000209" → {"000209": "CarryObject"}."""
    if path is None or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(None, 1)            # tên video có thể chứa dấu cách
        if len(parts) >= 2:
            name, vid = parts[0], parts[1].strip()
            cls = name.rsplit("_", 1)[0] if "_" in name and name.rsplit("_", 1)[1].isdigit() else name
            suffix = Path(vid).suffix.lower()
            out[Path(vid).stem if suffix in (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
                else vid] = cls
    return out


def build_splits(cfg, rng: np.random.Generator, resolve) -> dict[str, list[VideoItem]]:
    """Chia: train (≈90% video Train), val (≈10% video Train), test (video Test có nhãn)."""
    exts = cfg.data.video_extensions
    train_videos = find_videos(resolve(cfg.data.train_dir), exts)
    test_videos = find_videos(resolve(cfg.data.test_dir), exts) if cfg.data.test_dir else []
    if cfg.data.max_videos:
        train_videos = train_videos[: cfg.data.max_videos]
        test_videos = test_videos[: cfg.data.max_videos]

    idx = rng.permutation(len(train_videos))
    n_val = int(round(len(train_videos) * cfg.data.val_video_ratio))
    val_idx = set(idx[:n_val].tolist())
    splits: dict[str, list[VideoItem]] = {"train": [], "val": [], "test": []}
    for i, p in enumerate(train_videos):
        splits["val" if i in val_idx else "train"].append(VideoItem(p, "val" if i in val_idx else "train"))

    ann_root = resolve(cfg.data.annotation_dir) if cfg.data.annotation_dir else None
    for p in test_videos:
        lab = find_annotation(p.stem, ann_root, cfg.data.annotation_patterns, p)
        splits["test"].append(VideoItem(p, "test", labels_path=lab))
    return splits
