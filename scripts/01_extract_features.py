#!/usr/bin/env python3
"""BƯỚC 1→4 – Video → khung xám → Horn–Schunck → F(x,y) → ma trận hiệp biến C → véc-tơ.

Kết quả cache trong outputs/features/<hash cấu hình>/<split>/<video>.npz
(chạy lại sẽ bỏ qua video đã xong).

    python scripts/01_extract_features.py --jobs 8
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.features.covariance import descriptor_dim, feature_dim  # noqa: E402
from ocsvm_fpga.features.pipeline import extract_split, feature_signature, load_split  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--jobs", type=int, default=None, help="số tiến trình (mặc định project.n_jobs)")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "01_extract")
    splits = get_splits(cfg, out, rng)
    jobs = args.jobs or int(cfg.project.n_jobs)
    fs = cfg.features.feature_set
    if cfg.features.get("descriptor", "covariance") == "grid_flow":
        from ocsvm_fpga.features.grid_flow import grid_dim
        f = cfg.features
        log.info(f"Đặc trưng lưới ({f.get('grid_stats', 'basic')}{', diff' if f.get('grid_diff') else ''}) → véc-tơ "
                 f"{grid_dim(f, int(cfg.preprocess.height), int(cfg.preprocess.width))} chiều | cache theo khối 10×10 | "
                 f"dòng quang học = {cfg.optical_flow.get('method', 'hs')} | cache = {feature_signature(cfg)}")
    else:
        log.info(f"Tập đặc trưng {fs}: d = {feature_dim(fs)} → C {feature_dim(fs)}×{feature_dim(fs)} "
                 f"→ véc-tơ {descriptor_dim(fs)} chiều | cache = {feature_signature(cfg)}")
    for name in args.splits:
        items = splits.get(name, [])
        if not items:
            log.info(f"[{name}] không có video – bỏ qua")
            continue
        t0 = time.time()
        log.info(f"[{name}] trích xuất {len(items)} video với {jobs} tiến trình ...")
        extract_split(items, cfg, out, jobs, log)
        data = load_split(items, cfg, out)
        msg = f"[{name}] {len(data['X'])} véc-tơ, {time.time() - t0:.0f}s"
        if data["y"] is not None:
            msg += f", tỉ lệ khung bất thường = {data['y'].mean():.3f}"
        log.info(msg)


if __name__ == "__main__":
    main()
