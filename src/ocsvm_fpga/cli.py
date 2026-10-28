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


def expected_feature_dim(cfg) -> int:
    """Số chiều véc-tơ đặc trưng mà cấu hình hiện tại sẽ tạo ra."""
    f = cfg.features
    if f.get("descriptor", "covariance") == "grid_flow":
        from .features.grid_flow import grid_dim
        return grid_dim(f, int(cfg.preprocess.height), int(cfg.preprocess.width))
    from .features.covariance import descriptor_dim
    return descriptor_dim(f.feature_set)


def load_detector_checked(cfg, out: Path):
    """Nạp outputs/models/detector.joblib và dừng với hướng dẫn rõ ràng nếu số chiều của mô hình
    không khớp cấu hình hiện tại (vd đổi grid_diff / biên lưới mà chưa huấn luyện lại)."""
    import sys
    from .models.base import AnomalyDetector
    path = out / "models" / "detector.joblib"
    if not path.exists():
        sys.exit(f"Chưa có mô hình {path} – chạy scripts/02_train.py trước")
    det = AnomalyDetector.load(path)
    got, want = int(det.model.support_vectors.shape[1]), expected_feature_dim(cfg)
    if got != want:
        sys.exit(f"Mô hình đã lưu có {got} chiều nhưng cấu hình hiện tại tạo véc-tơ {want} chiều.\n"
                 "→ Huấn luyện lại với CÙNG cấu hình trước, ví dụ:\n"
                 "   python scripts/02_train.py --set <các tham số --set y hệt lệnh này>\n"
                 "hoặc bỏ các --set đặc trưng để dùng đúng cấu hình đã huấn luyện mô hình.")
    return det
