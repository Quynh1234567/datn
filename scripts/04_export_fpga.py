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
    hw = cfg.features.get("descriptor", "covariance") == "grid_flow_hw"
    q_hw = None
    if hw:
        # Đường số nguyên của FPGA: đặc trưng Q5.10 (đã là số nguyên) → z = (g − mean_q)·inv_q >> INV_FRAC
        from ocsvm_fpga.hwmodel import hs_q as HQ
        sc = HQ.quantize_scaler(det.scaler.mean, det.scaler.scale)
        lo_d, hi_d = -(1 << (HQ.DEFAULT.data_w - 1)), (1 << (HQ.DEFAULT.data_w - 1)) - 1

        def q_hw(x_raw):
            g = np.round(np.atleast_2d(x_raw) * (1 << HQ.DEFAULT.data_frac)).astype(np.int64)   # = Xq chính xác
            z = HQ.standardize_q(g, sc)
            return z, int(np.sum((z <= lo_d) | (z >= hi_d)))
        log.info(f"grid_flow_hw: chuẩn hóa số nguyên INV_FRAC = {sc.inv_frac}, mean bão hòa {sc.n_sat_mean}")
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
    if hw:                                    # phần cứng thật dùng z số nguyên, không phải z số thực
        xq_hw, _ = q_hw(x)
        f_q_hw = fx.decision_q(xq_hw)
        lab_hw = fx.anomaly_q(f_q_hw)
        z_lsb = np.abs(xq_hw - xq)
    err = np.abs(f_fix - f_float)
    rep = {
        "verify_source": src, "n_samples": int(len(x)),
        "input_saturated_elements": int(sat),
        "decision_abs_err_max": float(err.max()), "decision_abs_err_mean": float(err.mean()),
        "decision_rel_err_mean": float(err.mean() / (np.abs(f_float).mean() + 1e-12)),
        "label_agreement_per_frame": float(np.mean(lab_float == lab_fix)),
        "quant": fx.summary(),
    }
    if hw:
        rep["label_agreement_hw_path"] = float(np.mean(lab_float == lab_hw))
        rep["z_int_vs_float_lsb_max"] = int(z_lsb.max())
        rep["z_int_vs_float_lsb_mean"] = float(z_lsb.mean())
    if y is not None and len(np.unique(y)) == 2:
        rep["auc_float"] = float(roc_auc_score(y, det.model.score_from_decision(f_float)))
        rep["auc_fixed"] = float(roc_auc_score(y, det.model.score_from_decision(f_fix)))
        if hw:
            rep["auc_hw_path"] = float(roc_auc_score(y, det.model.score_from_decision(fx.decision_float_equiv(f_q_hw))))
    log.info(f"Sai số |f_fixed − f_float|: max = {rep['decision_abs_err_max']:.3e}, "
             f"TB = {rep['decision_abs_err_mean']:.3e} ({rep['decision_rel_err_mean']:.2%}) | "
             f"nhãn trùng khớp = {rep['label_agreement_per_frame']:.4%}"
             + (f" | AUC float/fixed = {rep['auc_float']:.4f}/{rep['auc_fixed']:.4f}" if "auc_fixed" in rep else ""))
    if hw:
        log.info(f"Đường số nguyên FPGA (z_q): lệch z số thực max {rep['z_int_vs_float_lsb_max']} LSB, "
                 f"TB {rep['z_int_vs_float_lsb_mean']:.2f} | nhãn trùng với mô hình số thực = "
                 f"{rep['label_agreement_hw_path']:.4%}" + (f" | AUC = {rep['auc_hw_path']:.4f}" if "auc_hw_path" in rep else ""))
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
    rep["export"] = export_all(fx, cfg, fpga_dir, data["X"], data["y"], rng, log, quantize_fn=q_hw)
    if hw:
        rep["hw_vectors"] = export_hw_vectors(cfg, out, splits, det, fx, sc, fpga_dir / "hs", log)
    save_json(rep, out / "reports" / "quant_report.json")


def export_hw_vectors(cfg, out, splits, det, fx, sc, vec_dir, log, n_cases=8, max_frames=400):
    """Véc-tơ kiểm thử TỪNG TẦNG cho khối trích đặc trưng (ảnh → … → z → f_q) từ khung thật của tập test,
    cùng hs_params.vh, log_lut.mem, scaler_*_q.mem – xem src/ocsvm_fpga/hwmodel/hs_export.py."""
    from ocsvm_fpga.features.pipeline import load_sensor_frames
    from ocsvm_fpga.hwmodel import hs_export as HX
    from ocsvm_fpga.hwmodel import hs_q as HQ
    from ocsvm_fpga.hwmodel import preproc_q as PQ
    items = splits.get("test") or splits.get("val") or []
    if not items:
        log.warning("Không có video test/val để xuất véc-tơ khối trích đặc trưng")
        return None
    item = items[0]
    frames, fidx = load_sensor_frames(item, cfg, out)
    dark = float(cfg.preprocess.get("skip_dark_below", 0) or 0)
    cand, prev = [], None
    for i, raw in zip(fidx[:max_frames], frames[:max_frames]):
        g = PQ.gauss5_q(np.ascontiguousarray(raw, np.uint8))
        if dark > 0 and PQ.frame_sum(g) < dark * g.size:
            prev = None
            continue
        if prev is not None:
            cand.append((prev[0], prev[1], i, g))
        prev = (i, g)
    if not cand:
        return None
    lut = HQ.log_lut()
    motion = [float(HQ.features_q(a, b, lut=lut).mag.mean()) for _, a, _, b in cand]
    order = np.argsort(motion)
    pick = sorted(set([int(order[0])] + [int(k) for k in order[::-1][:n_cases - 1]]))
    pairs = [(cand[k][1], cand[k][3]) for k in pick]
    names = [f"{item.path.stem} khung {cand[k][0]}→{cand[k][2]}" for k in pick]

    def ocsvm(z):
        fq = fx.decision_q(np.atleast_2d(z))
        return int(fq[0]), int(fx.anomaly_q(fq)[0])
    rows = HX.export_all(pairs, vec_dir, scaler=sc, ocsvm=ocsvm, names=names)
    log.info(f"Đã xuất {len(rows)} ca véc-tơ khối trích đặc trưng → {vec_dir} (hs_params.vh, mem/, case_XX/)")
    return {"dir": str(vec_dir), "cases": rows}


if __name__ == "__main__":
    main()
