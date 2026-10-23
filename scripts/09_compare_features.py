#!/usr/bin/env python3
"""THÍ NGHIỆM SO SÁNH ĐẶC TRƯNG – cùng dữ liệu, cùng cách chia tập, cùng OC-SVM,
cùng quy tắc chọn ngưỡng; CHỈ khác bước biến dòng quang học thành véc-tơ.

Biến thể có sẵn:
  cov        ma trận hiệp biến + Log-Euclidean (phương pháp chính, cấu hình hiện tại)
  grid       lưới 6×8, mỗi ô [u, v, |v|]                 – Horn–Schunck (144 chiều)
  grid_rich  lưới 6×8, mỗi ô [u, v, |v|, max|v|, std|v|] – Horn–Schunck (240 chiều)
  grid_fb    lưới 6×8 basic nhưng dòng quang học Farneback (gần với repo gốc nhất)
  Tự định nghĩa:  "ten:khoa=gia_tri;khoa=gia_tri"   vd  "grid4x4:features.descriptor=grid_flow;features.grid_rows=4;features.grid_cols=4"

Ví dụ:
  python scripts/09_compare_features.py --jobs 8 --set model.nu=0.02 model.max_support_vectors=128
  python scripts/09_compare_features.py --variants cov grid --set model.nu=0.02 model.max_support_vectors=128

Kết quả: outputs/experiments/
  comparison.md / comparison.csv   bảng so sánh (đưa thẳng vào báo cáo)
  roc_compare.png                  đường ROC chồng nhau
  per_class_compare.png            AUC theo loại hành vi
  <bien_the>/report.json, detector.joblib
Đặc trưng được cache như các bước khác: biến thể "cov" dùng lại cache đã có;
mỗi biến thể lưới chỉ phải trích xuất MỘT lần (nhanh hơn nhờ cache khung hình).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ocsvm_fpga.cli import get_splits  # noqa: E402
from ocsvm_fpga.config import load_config, resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.data.video_reader import smooth_frame  # noqa: E402
from ocsvm_fpga.evaluation.metrics import bootstrap_auc_ci, frame_metrics, per_class_auc  # noqa: E402
from ocsvm_fpga.features.covariance import covariance_descriptor  # noqa: E402
from ocsvm_fpga.features.grid_flow import grid_flow_descriptor  # noqa: E402
from ocsvm_fpga.features.optical_flow import compute_flow  # noqa: E402
from ocsvm_fpga.features.pipeline import (descriptor_kind, extract_split, feature_signature,  # noqa: E402
                                          load_sensor_frames, load_split)
from ocsvm_fpga.models.training import calibrate_threshold, fit_detector  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402
from ocsvm_fpga.utils import get_logger, save_json, set_seed  # noqa: E402

BUILTIN = {
    "cov": [],
    "grid": ["features.descriptor=grid_flow", "features.grid_stats=basic"],
    "grid_rich": ["features.descriptor=grid_flow", "features.grid_stats=rich"],
    "grid_fb": ["features.descriptor=grid_flow", "features.grid_stats=basic", "optical_flow.method=farneback"],
}
LABELS = {
    "cov": "Hiệp biến (HS + Log-Euclidean)",
    "grid": "Lưới 6×8 basic (HS)",
    "grid_rich": "Lưới 6×8 rich (HS)",
    "grid_fb": "Lưới 6×8 basic (Farneback)",
}
FPGA_CLOCK_MHZ = 100.0


def parse_variant(spec: str) -> tuple[str, list[str]]:
    if spec in BUILTIN:
        return spec, BUILTIN[spec]
    if ":" in spec:
        name, body = spec.split(":", 1)
        return name, [kv for kv in body.split(";") if kv.strip()]
    raise SystemExit(f"Biến thể không hợp lệ: {spec} (có sẵn: {', '.join(BUILTIN)})")


def feature_cost_ms(cfg, items, out_root, n_pairs: int = 40) -> tuple[float, float]:
    """Thời gian/cặp khung trên PC: (dòng quang học + đặc trưng, riêng bước đặc trưng)."""
    if not items:
        return float("nan"), float("nan")
    frames, _ = load_sensor_frames(items[0], cfg, out_root)
    frames = [smooth_frame(f, cfg.preprocess) for f in frames[1:n_pairs + 2]]
    t_all, t_desc = 0.0, 0.0
    f = cfg.features
    for a, b in zip(frames[:-1], frames[1:]):
        t0 = time.perf_counter()
        u, v = compute_flow(a, b, cfg.optical_flow)
        t1 = time.perf_counter()
        if descriptor_kind(cfg) == "grid_flow":
            grid_flow_descriptor(u, v, f.grid_rows, f.grid_cols, f.grid_stats)
        else:
            covariance_descriptor(u, v, f)
        t2 = time.perf_counter()
        t_all += t2 - t0
        t_desc += t2 - t1
    n = max(1, len(frames) - 1)
    return 1000 * t_all / n, 1000 * t_desc / n


def run_variant(name, overrides, args, log, exp_dir):
    cfg = load_config(args.config, list(args.set) + ["tuning.enabled=false"] + overrides)
    out = resolve_path(cfg.project.output_dir)
    rng = set_seed(int(cfg.project.seed))
    splits = get_splits(cfg, out, rng)                     # cùng splits.json cho mọi biến thể
    log.info(f"[{name}] cache đặc trưng = {feature_signature(cfg)}")

    t0 = time.time()
    for sp in ("train", "val", "test"):
        extract_split(splits[sp], cfg, out, args.jobs, log)
    t_extract = time.time() - t0
    train, val, test = (load_split(splits[s], cfg, out) for s in ("train", "val", "test"))
    if len(train["X"]) == 0 or test["y"] is None:
        raise SystemExit(f"[{name}] thiếu đặc trưng train hoặc nhãn test")
    dim = int(train["X"].shape[1])

    det, info = fit_detector(train["X"], cfg, np.random.default_rng(int(cfg.project.seed)), logger=log)
    thr = calibrate_threshold(det, cfg, val, log)
    x, y, vid = test["X"], test["y"], test["video_id"]
    s_raw = det.score(x, vid, smooth=False)
    s = det.score(x, vid, smooth=True)
    m = frame_metrics(y, s, thr)
    ci = bootstrap_auc_ci(y, s, vid, n_boot=args.bootstrap, seed=int(cfg.project.seed))
    classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
    stems = [it.path.stem for it in splits["test"]]
    cls = np.array([classes.get(stems[v], "?") for v in vid]) if classes else np.array(["?"] * len(vid))
    pc = per_class_auc(y, s, cls) if classes else {}
    far_val = float(np.mean(det.score(val["X"], val["video_id"]) > thr)) if len(val["X"]) else float("nan")

    # ---- kiểm chứng điểm cố định trên một tập con (bit-accurate như FPGA)
    fx = FixedPointOCSVM(det, cfg.fpga)
    sub = np.sort(np.random.default_rng(0).choice(len(x), min(args.fixed_samples, len(x)), replace=False))
    xq, n_sat = fx.quantize_input(x[sub])
    f_fix = fx.decision_float_equiv(fx.decision_q(xq))
    s_fix = det.model.score_from_decision(f_fix)
    auc_fix = float(roc_auc_score(y[sub], s_fix)) if len(np.unique(y[sub])) == 2 else float("nan")
    auc_float_sub = float(roc_auc_score(y[sub], s_raw[sub])) if len(np.unique(y[sub])) == 2 else float("nan")
    lim = (1 << (cfg.fpga.data_width - 1)) - 1
    sat_rows = float(np.mean((np.abs(xq) >= lim).any(1)))

    # ---- chi phí phần cứng ước tính
    n_sv = det.model.n_sv
    sv_kb = n_sv * dim * cfg.fpga.data_width / 8 / 1024
    macs = n_sv * dim
    lat_seq_us = macs / FPGA_CLOCK_MHZ                # 1 bộ nhân, 1 MAC/chu kỳ
    lat_par_us = n_sv / FPGA_CLOCK_MHZ                # DIM bộ nhân song song, 1 SV/chu kỳ
    ms_all, ms_desc = feature_cost_ms(cfg, splits["test"], out)

    rep = {
        "variant": name, "label": LABELS.get(name, name), "overrides": overrides,
        "data_train_dir": cfg.data.train_dir, "data_test_dir": cfg.data.test_dir,
        "descriptor": descriptor_kind(cfg), "flow": cfg.optical_flow.get("method", "hs"),
        "dim": dim, "n_train": int(len(train["X"])), "n_test": int(len(x)),
        "n_sv": n_sv, "gamma": det.model.gamma, "nu": det.meta.get("nu"), "threshold": thr,
        "auc": m.get("auc"), "auc_ci95": ci, "auc_raw": float(roc_auc_score(y, s_raw)),
        "eer": m.get("eer"), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "far_val": far_val, "per_class_auc": pc,
        "auc_fixed_subset": auc_fix, "auc_float_subset": auc_float_sub,
        "input_saturated_rows": sat_rows, "input_saturated_elems": int(n_sat),
        "sv_mem_kb": sv_kb, "mac_per_frame": macs, "latency_seq_us": lat_seq_us,
        "latency_par_us": lat_par_us, "dsp_parallel": dim,
        "pc_ms_per_pair": ms_all, "pc_ms_descriptor": ms_desc, "extract_seconds": t_extract,
        "train_info": {k: v for k, v in info.items() if k != "history"},
    }
    vdir = exp_dir / name
    vdir.mkdir(parents=True, exist_ok=True)
    det.save(vdir / "detector.joblib")
    save_json(rep, vdir / "report.json")
    fpr, tpr, _ = roc_curve(y, s)
    log.info(f"[{name}] D={dim} SV={n_sv} AUC={rep['auc']:.4f} (95% CI {ci[0]:.3f}–{ci[1]:.3f}) "
             f"EER={rep['eer']:.4f} | AUC fixed={auc_fix:.4f} | SV {sv_kb:.1f} KB, {macs} MAC/khung")
    return rep, (fpr, tpr)


def write_outputs(reps, rocs, exp_dir, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("Biến thể", "label"), ("Số chiều", "dim"), ("SV", "n_sv"), ("AUC", "auc"),
            ("AUC thô", "auc_raw"), ("EER", "eer"), ("Báo động giả val", "far_val"),
            ("AUC float/fixed (tập con, thô)", "auc_float_fixed"), ("Véc-tơ bão hòa", "input_saturated_rows"),
            ("Bộ nhớ SV (KB)", "sv_mem_kb"), ("MAC/khung", "mac_per_frame"), ("Bộ nhân song song", "dsp_parallel"),
            ("Trễ tuần tự (µs)", "latency_seq_us"), ("Trễ song song (µs)", "latency_par_us"),
            ("PC ms/cặp", "pc_ms_per_pair")]

    for r in reps:
        r["auc_float_fixed"] = f"{r['auc_float_subset']:.4f}/{r['auc_fixed_subset']:.4f}"

    def fmt(k, v):
        if isinstance(v, float):
            if k in ("far_val", "input_saturated_rows"):
                return f"{v:.2%}"
            return f"{v:.4f}" if k.startswith("auc") or k == "eer" else f"{v:.1f}"
        return str(v)

    with open(exp_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols] + ["AUC CI95 thấp", "AUC CI95 cao"])
        for r in reps:
            w.writerow([r[k] for _, k in cols] + list(r["auc_ci95"]))

    lines = ["# So sánh đặc trưng – cùng OC-SVM, cùng dữ liệu", "",
             f"Dữ liệu: `{reps[0]['data_train_dir']}` (train) – `{reps[0]['data_test_dir']}` (test), "
             f"{reps[0]['n_train']} véc-tơ train, {reps[0]['n_test']} véc-tơ test.", "",
             "Điều kiện giống nhau cho mọi biến thể: cùng cách chia train/val/test (`outputs/splits.json`), "
             "cùng tiền xử lý (160×120, `frame_step`, bỏ khung đen), cùng siêu tham số OC-SVM (ν, ngân sách SV, "
             "γ = `scale` tính trên dữ liệu của từng biến thể), cùng quy tắc ngưỡng và làm mượt thời gian. "
             "Khác biệt duy nhất: cách biến dòng quang học thành véc-tơ đặc trưng (và thuật toán dòng quang học "
             "ở biến thể Farneback).", "",
             "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in reps:
        lines.append("| " + " | ".join(fmt(k, r[k]) for _, k in cols) + " |")
    lines += ["", "Khoảng tin cậy 95% của AUC (bootstrap theo video):", ""]
    for r in reps:
        lines.append(f"- {r['label']}: {r['auc']:.4f} [{r['auc_ci95'][0]:.4f}, {r['auc_ci95'][1]:.4f}]")
    classes = sorted({c for r in reps for c in r["per_class_auc"]})
    if classes:
        lines += ["", "## AUC theo loại hành vi", "",
                  "| Loại | " + " | ".join(r["label"] for r in reps) + " |",
                  "|---|" + "---|" * len(reps)]
        for c in classes:
            vals = [r["per_class_auc"].get(c, float("nan")) for r in reps]
            best = int(np.nanargmax(vals))
            cells = [f"**{v:.4f}**" if i == best else f"{v:.4f}" for i, v in enumerate(vals)]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Ghi chú: trễ FPGA ước tính ở 100 MHz – tuần tự = 1 bộ nhân (SV×D chu kỳ), "
              "song song = D bộ nhân (SV chu kỳ). AUC fixed-point tính trên tập con test bằng "
              "Golden Model bit-accurate. Thời gian PC đo trên máy chạy thí nghiệm."]
    (exp_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for r, (fpr, tpr) in zip(reps, rocs):
        ax.plot(fpr, tpr, lw=1.8, label=f"{r['label']} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – so sánh đặc trưng")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "roc_compare.png", dpi=150)
    plt.close(fig)

    if classes:
        fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.9), 4))
        wbar = 0.8 / len(reps)
        xs = np.arange(len(classes))
        for i, r in enumerate(reps):
            ax.bar(xs + i * wbar, [r["per_class_auc"].get(c, np.nan) for c in classes], wbar, label=r["label"])
        ax.axhline(0.5, c="gray", ls="--", lw=1)
        ax.set_xticks(xs + wbar * (len(reps) - 1) / 2, classes, rotation=35, ha="right")
        ax.set(ylabel="AUC", ylim=(0.3, 1.0), title="AUC theo loại hành vi")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(exp_dir / "per_class_compare.png", dpi=150)
        plt.close(fig)
    log.info(f"Bảng so sánh → {exp_dir / 'comparison.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="tham số CHUNG cho mọi biến thể, vd model.nu=0.02 model.max_support_vectors=128")
    ap.add_argument("--variants", nargs="+", default=["cov", "grid", "grid_rich", "grid_fb"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=200, help="số lần bootstrap cho khoảng tin cậy AUC")
    ap.add_argument("--fixed-samples", type=int, default=20000, help="số mẫu test để đo AUC fixed-point")
    args = ap.parse_args()

    base = load_config(args.config, args.set)
    exp_dir = resolve_path(base.project.output_dir) / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=exp_dir / "compare.log")
    reps, rocs = [], []
    for spec in args.variants:
        name, ov = parse_variant(spec)
        log.info(f"===== Biến thể {name}: {ov or '(cấu hình hiện tại)'} =====")
        rep, roc = run_variant(name, ov, args, log, exp_dir)
        reps.append(rep)
        rocs.append(roc)
    write_outputs(reps, rocs, exp_dir, log)


if __name__ == "__main__":
    main()
