#!/usr/bin/env python3
"""BƯỚC 0 – Kiểm tra dữ liệu: đếm video, độ phân giải, fps, số khung, file nhãn.

    python scripts/00_inspect_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.data.dataset import load_frame_labels  # noqa: E402
from ocsvm_fpga.data.video_reader import video_info  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rebuild-splits", action="store_true", help="chia lại train/val/test")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "00_inspect")
    splits = get_splits(cfg, out, rng, rebuild=args.rebuild_splits)

    report = {}
    for name, items in splits.items():
        infos = [video_info(i.path) for i in items]
        bad = [str(i.path) for i, inf in zip(items, infos) if not inf["opened"]]
        frames = np.array([inf["frames"] for inf in infos]) if infos else np.zeros(0)
        rep = {
            "videos": len(items),
            "unreadable": bad,
            "total_frames": int(frames.sum()),
            "resolutions": sorted({f"{i['width']}x{i['height']}" for i in infos}),
            "fps": sorted({round(i["fps"], 2) for i in infos}),
        }
        if name == "test":
            with_lab = [i for i in items if i.labels_path]
            rep["with_labels"] = len(with_lab)
            ab = [load_frame_labels(i.labels_path, inf["frames"]).mean()
                  for i, inf in zip(items, infos) if i.labels_path]
            rep["mean_abnormal_ratio"] = float(np.mean(ab)) if ab else None
            if items and not with_lab:
                log.warning("Không tìm thấy file nhãn nào cho tập Test – kiểm tra "
                            "data.annotation_dir và data.annotation_patterns")
        report[name] = rep
        log.info(f"[{name:5s}] {rep['videos']} video, {rep['total_frames']} khung, "
                 f"{rep['resolutions']} @ {rep['fps']} fps")
    est = report["train"]["total_frames"] // max(1, cfg.preprocess.frame_step)
    log.info(f"Ước tính ~{est} cặp khung huấn luyện với frame_step={cfg.preprocess.frame_step}")
    save_json(report, out / "reports" / "dataset_report.json")
    if report["train"]["videos"] == 0:
        log.error(f"Không có video trong {cfg.data.train_dir}. Sửa data.train_dir trong config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
