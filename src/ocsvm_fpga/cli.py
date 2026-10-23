"""Tiện ích dòng lệnh dùng chung cho các script trong scripts/."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config, resolve_path
from .data.dataset import VideoItem, build_splits, find_annotation
from .utils import get_logger, load_json, save_json, set_seed


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="file YAML ghi đè configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="ghi đè nhanh, vd: --set model.nu=0.02 features.feature_set=F4")
    return ap


def setup(args, step: str):
    cfg = load_config(args.config, args.set)
    out = resolve_path(cfg.project.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=out / "logs" / f"{step}.log")
    rng = set_seed(int(cfg.project.seed))
    return cfg, out, logger, rng


def get_splits(cfg, out: Path, rng: np.random.Generator, rebuild: bool = False) -> dict[str, list[VideoItem]]:
    """Chia tập một lần rồi lưu outputs/splits.json để mọi bước dùng cùng một cách chia."""
    path = out / "splits.json"
    if path.exists() and not rebuild:
        raw = load_json(path)
        splits = {k: [VideoItem(Path(d["path"]), k, Path(d["labels"]) if d["labels"] else None)
                      for d in v] for k, v in raw.items()}
        # Nhãn có thể được tạo SAU khi chia tập (vd bằng 08_annotate_videos.py)
        # → luôn tìm lại file nhãn cho video Test chưa có / có đường dẫn cũ không còn tồn tại.
        ann = resolve_path(cfg.data.annotation_dir) if cfg.data.annotation_dir else None
        for it in splits.get("test", []):
            if it.labels_path is None or not it.labels_path.exists():
                it.labels_path = find_annotation(it.path.stem, ann, cfg.data.annotation_patterns, it.path)
        return splits
    splits = build_splits(cfg, rng, resolve_path)
    save_json({k: [{"path": str(i.path), "labels": str(i.labels_path) if i.labels_path else None}
                   for i in v] for k, v in splits.items()}, path)
    return splits
