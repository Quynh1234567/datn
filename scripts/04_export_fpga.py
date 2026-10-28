#!/usr/bin/env python3
"""BƯỚC 7 – Lượng tử hóa điểm cố định, kiểm chứng Golden Model, xuất file cho FPGA/ESP32.

    python scripts/04_export_fpga.py
    python scripts/04_export_fpga.py --set fpga.data_frac=11 fpga.exp_lut_addr_bits=11
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup, load_detector_checked  # noqa: E402
from ocsvm_fpga.export.fpga_exporter import export_all  # noqa: E402
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--max-verify", type=int, default=20000, help="số mẫu dùng để đo sai số lượng tử")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "04_export")
    det = load_detector_checked(cfg, out)
    splits = get_splits(cfg, out, rng)
    from ocsvm_fpga.evaluation.events import alarm_params, alarm_params_need_normal
    if alarm_params_need_normal(det, cfg):                   # mô hình cũ / đổi cấu hình cảnh báo
        val = load_split(splits["val"], cfg, out)
        det.meta["alarm"] = alarm_params(det, cfg, det.score(val["X"], val["video_id"]) if len(val["X"]) else None)
    a = det.meta["alarm"]
    log.info(f"Cảnh báo có trễ (ghi vào ocsvm_params.h): bật > {a['on']:.4f}, tắt < {a['off']:.4f}, "
             f"giữ {a['hold']} cặp khung")

    data = load_split(splits["test"], cfg, out)
    src = "test"
    if len(data["X"]) == 0:
        data, src = load_split(splits["val"], cfg, out), "val"
    if len(data["X"]) == 0:
        data, src = load_split(splits["train"], cfg, out), "train"
    x, y = data["X"], data["y"]
    if len(x) > args.max_verify:
        sel = np.sort(rng.choice(len(x), args.max_verify, replace=False))
        x, y = x[sel], (y[sel] if y is not None else None)

    fx = FixedPointOCSVM(det, cfg.fpga)
    for w in fx.check_ranges():
        log.warning(w)
    log.info(f"Lượng tử hóa: SV {fx.sv_q.shape}, γ_q = {fx.gamma_q} (FRAC {fx.p.gamma_frac}), "
             f"LUT_SHIFT = {fx.p.lut_shift}, ρ_q = {fx.rho_q}, THR_q = {fx.thr_q}")

    # ---------------------------------------------- so sánh float ↔ fixed-point
    f_float = det.decision(x)
    xq, sat = fx.quantize_input(x)
    f_q = fx.decision_q(xq)
    f_fix = fx.decision_float_equiv(f_q)
    lab_float = (det.model.score_from_decision(f_float) > det.threshold).astype(np.uint8)
    lab_fix = fx.anomaly_q(f_q)
    err = np.abs(f_fix - f_float)
    rep = {
        "verify_source": src, "n_samples": int(len(x)),
        "input_saturated_elements": int(sat),
        "decision_abs_err_max": float(err.max()), "decision_abs_err_mean": float(err.mean()),
        "decision_rel_err_mean": float(err.mean() / (np.abs(f_float).mean() + 1e-12)),
        "label_agreement_per_frame": float(np.mean(lab_float == lab_fix)),
        "quant": fx.summary(),
    }
    if y is not None and len(np.unique(y)) == 2:
        rep["auc_float"] = float(roc_auc_score(y, det.model.score_from_decision(f_float)))
        rep["auc_fixed"] = float(roc_auc_score(y, det.model.score_from_decision(f_fix)))
    log.info(f"Sai số |f_fixed − f_float|: max = {rep['decision_abs_err_max']:.3e}, "
             f"TB = {rep['decision_abs_err_mean']:.3e} ({rep['decision_rel_err_mean']:.2%}) | "
             f"nhãn trùng khớp = {rep['label_agreement_per_frame']:.4%}"
             + (f" | AUC float/fixed = {rep['auc_float']:.4f}/{rep['auc_fixed']:.4f}" if "auc_fixed" in rep else ""))
    if sat:
        lo, hi = -(1 << (fx.p.data_width - 1)), (1 << (fx.p.data_width - 1)) - 1
        row_sat = ((xq <= lo) | (xq >= hi)).any(1)
        normal = (y == 0) if y is not None else np.ones(len(x), bool)
        rate_n = float(row_sat[normal].mean()) if normal.any() else 0.0
        rep["input_saturated_rows_normal_ratio"] = rate_n
        what = "BÌNH THƯỜNG" if y is not None else "(chưa có nhãn – gồm cả khung bất thường)"
        msg = (f"{sat} phần tử đầu vào bị bão hòa; tỉ lệ véc-tơ {what} bị bão hòa = {rate_n:.2%}. "
               "Bão hòa ở khung bất thường là vô hại (kernel ≈ 0)")
        (log.warning if rate_n > 0.01 and y is not None else log.info)(msg + ("; nên giảm data_frac" if rate_n > 0.01 else ""))

    fpga_dir = out / "fpga"
    rep["export"] = export_all(fx, cfg, fpga_dir, data["X"], data["y"], rng, log)
    save_json(rep, out / "reports" / "quant_report.json")


if __name__ == "__main__":
    main()
