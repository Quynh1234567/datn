#!/usr/bin/env python3
"""BƯỚC 5 – Huấn luyện One-Class SVM trên véc-tơ hiệp biến của video bình thường.

    python scripts/02_train.py
    python scripts/02_train.py --set model.nu=0.02 model.max_support_vectors=128
    python scripts/02_train.py --set tuning.enabled=true        # tìm (ν, γ) tự động
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.models.training import calibrate_threshold, fit_detector, tune_hyperparams  # noqa: E402
from ocsvm_fpga.utils import save_json, timer  # noqa: E402


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "02_train")
    splits = get_splits(cfg, out, rng)
    train = load_split(splits["train"], cfg, out)
    val = load_split(splits["val"], cfg, out)
    test = load_split(splits["test"], cfg, out)
    if len(train["X"]) == 0:
        log.error("Chưa có đặc trưng huấn luyện – chạy scripts/01_extract_features.py trước.")
        sys.exit(1)
    log.info(f"Train: {train['X'].shape} | Val: {val['X'].shape} | Test: {test['X'].shape}")

    nu, gamma_mult, tuning_rows, held_out = None, 1.0, None, None
    if cfg.tuning.enabled:
        if test["y"] is None or len(np.unique(test["y"])) < 2:
            log.warning("tuning.enabled nhưng tập Test không có nhãn hai lớp – bỏ qua tìm tham số")
        else:
            log.info("Tìm siêu tham số trên phần validation của tập Test ...")
            best, tuning_rows, report_part = tune_hyperparams(train["X"], test, cfg, rng, log)
            nu, gamma_mult = best["nu"], best["gamma_mult"]
            held_out = np.setdiff1d(np.unique(test["video_id"]), np.unique(report_part["video_id"]))
            log.info(f"→ Chọn ν={nu}, γ×{gamma_mult} (AUC val = {best['val_auc']:.4f})")

    with timer(log, "Huấn luyện"):
        det, info = fit_detector(train["X"], cfg, rng, nu, gamma_mult, log)
    thr = calibrate_threshold(det, cfg, val, log)
    det.meta["tuning_val_video_ids"] = held_out.tolist() if held_out is not None else []

    tr_scores = det.score(train["X"], train["video_id"])
    report = {
        "model": det.meta | {"threshold": thr},
        "n_support_vectors": det.model.n_sv,
        "descriptor_dim": int(train["X"].shape[1]),
        "train_false_alarm_rate": float(np.mean(tr_scores > thr)),
        "tuning": tuning_rows,
    }
    if len(val["X"]):
        report["val_false_alarm_rate"] = float(np.mean(det.score(val["X"], val["video_id"]) > thr))
    det.save(out / "models" / "detector.joblib")
    save_json(report, out / "reports" / "train_report.json")
    log.info(f"SV = {det.model.n_sv}, γ = {det.model.gamma:.5g}, ρ = {det.model.rho:.5g}, "
             f"ngưỡng = {thr:.5g}, báo động giả train = {report['train_false_alarm_rate']:.3%}"
             + (f", val = {report['val_false_alarm_rate']:.3%}" if "val_false_alarm_rate" in report else ""))
    log.info(f"Đã lưu mô hình → {out / 'models' / 'detector.joblib'}")


if __name__ == "__main__":
    main()
