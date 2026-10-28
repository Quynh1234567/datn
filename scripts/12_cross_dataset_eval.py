#!/usr/bin/env python3
"""ĐÁNH GIÁ CHÉO: mô hình IITB-Corridor áp nguyên cho 3 cảnh UMN – chạy MỘT lần.

Không huấn luyện lại trên UMN. Mô hình IITB (scaler + OC-SVM + ngưỡng trong
outputs/models/detector.joblib) được đánh giá trên:

  IITB → IITB            Test IITB (bỏ các video đã dùng chọn tham số – giống bước 03)
  IITB → UMN (Lawn)      phần Test UMN của cảnh bãi cỏ
  IITB → UMN (Indoor)    phần Test UMN của cảnh trong nhà
  IITB → UMN (Plaza)     phần Test UMN của cảnh quảng trường
  IITB → UMN (gộp 3 cảnh) cả 3 cảnh cùng lúc (tắt bằng --no-combined)

Mỗi kịch bản có đủ chỉ số như bước 03: AUC, AUC thô, EER, Precision, Recall, F1,
báo động giả, khoảng tin cậy 95% (bootstrap theo video). Đặc trưng UMN được trích
MỘT lần bằng đúng cấu hình của mô hình IITB; 3 cảnh được tách theo test.txt do
scripts/10_prepare_umn.py sinh ra (kết quả trùng với chạy riêng từng cảnh vì mô hình
không đổi).

    python scripts/10_prepare_umn.py
    python scripts/12_cross_dataset_eval.py --jobs 8

Kết quả: outputs/cross_dataset/
  comparison.md / .csv     bảng chỉ số IITB + từng cảnh UMN
  metrics_compare.png      AUC, 1−EER, Precision, Recall, F1 – tất cả kịch bản trên MỘT biểu đồ
  auc_compare.png          AUC kèm khoảng tin cậy 95%
  roc_compare.png          các đường ROC chồng nhau
  per_class_iitb.png       AUC theo loại hành vi IITB
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ocsvm_fpga.cli import get_splits  # noqa: E402
from ocsvm_fpga.config import load_config, resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.evaluation.metrics import bootstrap_auc_ci, frame_metrics  # noqa: E402
from ocsvm_fpga.features.pipeline import extract_split, feature_signature, load_split  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.utils import get_logger, save_json, set_seed  # noqa: E402

DATA_KEYS = ("train_dir", "test_dir", "annotation_dir", "test_class_file", "val_video_ratio")
SCENE_ORDER = ("Lawn", "Indoor", "Plaza")
COLORS = {"IITB": "#1f77b4", "Lawn": "#2ca02c", "Indoor": "#d62728", "Plaza": "#ff7f0e", "UMN": "#7f7f7f"}


# ----------------------------------------------------------------------------- cấu hình & dữ liệu
def umn_cfg(base_cfg_path, umn_yaml: Path):
    """Cấu hình đặc trưng của mô hình IITB + đường dẫn dữ liệu UMN (file sinh bởi bước 10)."""
    if not umn_yaml.exists():
        raise SystemExit(f"Không thấy {umn_yaml} – chạy trước: python scripts/10_prepare_umn.py --per-scene")
    raw = yaml.safe_load(umn_yaml.read_text(encoding="utf-8")) or {}
    cfg = load_config(base_cfg_path)
    for k in DATA_KEYS:
        if k in raw.get("data", {}):
            cfg["data"][k] = raw["data"][k]
    if "video_extensions" in raw.get("data", {}):
        cfg["data"]["video_extensions"] = raw["data"]["video_extensions"]
    if "mask_regions" in raw.get("preprocess", {}):             # cấu hình của 13_prepare_unm_full.py
        cfg["preprocess"]["mask_regions"] = raw["preprocess"]["mask_regions"]
    cfg["project"]["output_dir"] = raw.get("project", {}).get("output_dir", "outputs_umn")
    test_dir = resolve_path(cfg.data.test_dir)
    if not test_dir.is_dir() or not any(p.is_file() for p in test_dir.rglob("*.*")):
        raise SystemExit(f"Thư mục {test_dir} trống – chạy lại scripts/10_prepare_umn.py")
    return cfg


def check_model(det, cfg, log):
    m = det.meta
    want = {"descriptor": cfg.features.get("descriptor", "covariance"),
            "frame_step": cfg.preprocess.frame_step, "flow": cfg.optical_flow.get("method", "hs")}
    for k, v in want.items():
        if k in m and m[k] != v:
            log.warning(f"Mô hình được huấn luyện với {k}={m[k]} nhưng configs/default.yaml đang là {v} – "
                        "sửa default.yaml về đúng cấu hình đã huấn luyện, nếu không đặc trưng sẽ không khớp")


def prepare_umn(cfg, jobs, log):
    """Trích đặc trưng Train/Test của một bộ UMN; trả (train, test, items_test, classes)."""
    out = resolve_path(cfg.project.output_dir)
    sp = get_splits(cfg, out, set_seed(int(cfg.project.seed)))
    log.info(f"  {cfg.data.test_dir}: {len(sp['train'])} video train, {len(sp['test'])} video test "
             f"(cache {feature_signature(cfg)})")
    if sp["test"]:                                   # chỉ cần Test – không huấn luyện lại trên UMN
        extract_split(sp["test"], cfg, out, jobs, log)
    train = None
    test = load_split(sp["test"], cfg, out)
    if test["y"] is None:
        raise SystemExit(f"Thiếu nhãn test trong {cfg.data.annotation_dir}")
    return train, test, sp["test"], load_test_classes(resolve_path(cfg.data.get("test_class_file")))


# ----------------------------------------------------------------------------- chỉ số
def summarize(name, y, s, raw, vid, cls, thr, n_sv, dim, n_boot):
    met = frame_metrics(y, s, thr)
    pc = {}
    has_normal = "Normal" in set(cls.tolist())
    for c in sorted(set(cls.tolist()) - {"?", "Normal"}):
        # IITB (có video Normal): loại + video Normal – giống bước 03; UMN (cảnh): khung trong cảnh đó
        m = (cls == c) | (cls == "Normal") if has_normal else (cls == c)
        if len(np.unique(y[m])) == 2:
            pc[c] = float(roc_auc_score(y[m], s[m]))
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "scenario": name, "n_videos": int(len(np.unique(vid))), "n_pairs": int(len(y)),
        "abnormal_ratio": float(y.mean()), "auc": met.get("auc"), "auc_raw": float(roc_auc_score(y, raw)),
        "auc_ci95": bootstrap_auc_ci(y, s, vid, n_boot=n_boot), "eer": met.get("eer"),
        "precision": met["precision"], "recall": met["recall"], "f1": met["f1"],
        "far_test_normal": float(np.mean(s[y == 0] > thr)), "per_class_auc": pc,
        "n_sv": n_sv, "dim": dim,
    }, (fpr, tpr)


def log_line(r, log):
    log.info(f"[{r['scenario']}] AUC={r['auc']:.4f} [{r['auc_ci95'][0]:.3f}–{r['auc_ci95'][1]:.3f}] "
             f"EER={r['eer']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
             f"FAR(khung bình thường)={r['far_test_normal']:.2%}")


def eval_detector(name, det, data, items, classes, log, n_boot, exclude=()):
    m = ~np.isin(data["video_id"], list(exclude))
    x, y, vid = data["X"][m], data["y"][m], data["video_id"][m]
    if x.shape[1] != det.model.support_vectors.shape[1]:
        raise SystemExit(f"[{name}] số chiều đặc trưng {x.shape[1]} ≠ mô hình {det.model.support_vectors.shape[1]}")
    raw = det.score(x, vid, smooth=False)
    s = det.score(x, vid, smooth=True)
    stems = [it.path.stem for it in items]
    cls = np.array([classes.get(stems[v], "?") for v in vid])
    rep, roc = summarize(name, y, s, raw, vid, cls, det.threshold, det.model.n_sv, int(x.shape[1]), n_boot)
    log_line(rep, log)
    return rep, roc


def eval_subset(name, det, x, y, vid, cls, log, n_boot):
    raw = det.score(x, vid, smooth=False)
    s = det.score(x, vid, smooth=True)
    rep, roc = summarize(name, y, s, raw, vid, cls, det.threshold, det.model.n_sv, int(x.shape[1]), n_boot)
    log_line(rep, log)
    return rep, roc


def color_of(name):
    for k, c in COLORS.items():
        if k in name and not (k == "IITB" and "UMN" in name):
            return c
    return "#7f7f7f"


# ----------------------------------------------------------------------------- xuất kết quả
def write_outputs(reps, rocs, iitb_rep, out_dir: Path, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("Kịch bản", "scenario"), ("Video test", "n_videos"), ("Cặp khung", "n_pairs"),
            ("Tỉ lệ bất thường", "abnormal_ratio"), ("AUC", "auc"), ("AUC thô", "auc_raw"), ("EER", "eer"),
            ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
            ("Báo động giả (khung bình thường)", "far_test_normal")]

    def fmt(k, v):
        if isinstance(v, float):
            return f"{v:.2%}" if k in ("abnormal_ratio", "far_test_normal") else f"{v:.4f}"
        return str(v)

    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols] + ["AUC CI95 thấp", "AUC CI95 cao"])
        for r in reps:
            w.writerow([r[k] for _, k in cols] + list(r["auc_ci95"]))
    lines = ["# Đánh giá chéo – mô hình IITB-Corridor áp nguyên cho UMN (không huấn luyện lại)", "",
             f"Mô hình: {iitb_rep['n_sv']} SV × {iitb_rep['dim']} chiều, cùng scaler và ngưỡng cho mọi kịch bản.", "",
             "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(k, r[k]) for _, k in cols) + " |" for r in reps]
    lines += ["", "Khoảng tin cậy 95% AUC (bootstrap theo video):", ""]
    lines += [f"- {r['scenario']}: {r['auc']:.4f} [{r['auc_ci95'][0]:.4f}, {r['auc_ci95'][1]:.4f}]" for r in reps]
    if iitb_rep["per_class_auc"]:
        lines += ["", "## AUC theo loại hành vi IITB (IITB → IITB)", "", "| Loại | AUC |", "|---|---|"]
        lines += [f"| {c} | {v:.4f} |" for c, v in sorted(iitb_rep["per_class_auc"].items(), key=lambda kv: -kv[1])]
    lines += ["", "Ghi chú: AUC/EER không phụ thuộc ngưỡng. Precision/Recall/F1/báo động giả dùng nguyên "
              "ngưỡng của mô hình IITB. Cảnh UMN có ít video (2–6 đoạn) nên khoảng tin cậy rộng."]
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- ROC
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for r, (fpr, tpr) in zip(reps, rocs):
        ax.plot(fpr, tpr, lw=2, c=color_of(r["scenario"]), label=f"{r['scenario']} (AUC {r['auc']:.3f})",
                ls="--" if "gộp" in r["scenario"] else "-")
    ax.plot([0, 1], [0, 1], ":", c="gray", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – mô hình IITB trên IITB và UMN")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_compare.png", dpi=150)
    plt.close(fig)

    # --- Biểu đồ chung các chỉ số
    keys = [("AUC", "auc"), ("1 − EER", None), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(reps) + 5), 4.9))
    w = 0.85 / len(reps)
    xs = np.arange(len(keys))
    for i, r in enumerate(reps):
        vals = [1 - r["eer"] if k is None else r[k] for _, k in keys]
        bars = ax.bar(xs + i * w, vals, w, label=r["scenario"], color=color_of(r["scenario"]),
                      hatch="//" if "gộp" in r["scenario"] else None, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs + w * (len(reps) - 1) / 2, [n for n, _ in keys])
    ax.set(ylim=(0, 1.12), ylabel="Giá trị", title="Mô hình IITB-Corridor: IITB và từng cảnh UMN")
    ax.legend(fontsize=8, ncol=min(len(reps), 5), loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_compare.png", dpi=150)
    plt.close(fig)

    # --- AUC kèm khoảng tin cậy
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["scenario"] for r in reps]
    aucs = np.array([r["auc"] for r in reps])
    lo = aucs - np.array([r["auc_ci95"][0] for r in reps])
    hi = np.array([r["auc_ci95"][1] for r in reps]) - aucs
    ax.bar(range(len(reps)), aucs, color=[color_of(n) for n in names], yerr=[lo, hi], capsize=4)
    for i, v in enumerate(aucs):
        ax.text(i, v + hi[i] + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, c="gray", ls="--", lw=1)
    ax.set_xticks(range(len(reps)), [n.replace("IITB → ", "") for n in names], rotation=15)
    ax.set(ylim=(0.3, 1.1), ylabel="AUC", title="AUC (khoảng tin cậy 95%) – mô hình IITB")
    fig.tight_layout()
    fig.savefig(out_dir / "auc_compare.png", dpi=150)
    plt.close(fig)

    if iitb_rep["per_class_auc"]:
        items = sorted(iitb_rep["per_class_auc"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS["IITB"])
        ax.axvline(0.5, c="gray", ls="--", lw=1)
        ax.set(xlim=(0.3, 1.0), xlabel="AUC", title="AUC theo loại hành vi – IITB → IITB")
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_iitb.png", dpi=150)
        plt.close(fig)
    log.info(f"Kết quả → {out_dir}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="cấu hình IITB (mặc định configs/default.yaml)")
    ap.add_argument("--umn-config", default="configs/umn.yaml", help="file do 10_prepare_umn.py sinh ra (gộp 3 cảnh)")
    ap.add_argument("--no-combined", action="store_true", help="không thêm cột 'gộp 3 cảnh'")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args()

    # 1. Mô hình IITB
    iitb_cfg = load_config(a.config)
    iitb_out = resolve_path(iitb_cfg.project.output_dir)
    out_dir = iitb_out / "cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out_dir / "cross.log")
    model_path = iitb_out / "models" / "detector.joblib"
    if not model_path.is_file():
        raise SystemExit(f"Không tìm thấy mô hình IITB: {model_path} – chạy pipeline IITB trước")
    det = AnomalyDetector.load(model_path)
    check_model(det, iitb_cfg, log)
    reps, rocs = [], []

    # 2. IITB → IITB
    log.info("========== IITB → IITB ==========")
    sp = get_splits(iitb_cfg, iitb_out, set_seed(int(iitb_cfg.project.seed)))
    test = load_split(sp["test"], iitb_cfg, iitb_out)
    if len(test["X"]) == 0 or test["y"] is None:
        raise SystemExit("Chưa có đặc trưng hoặc nhãn Test IITB – chạy pipeline IITB trước")
    held = det.meta.get("tuning_val_video_ids", [])
    m = ~np.isin(test["video_id"], held)
    iitb_cls = load_test_classes(resolve_path(iitb_cfg.data.get("test_class_file")))
    stems = [it.path.stem for it in sp["test"]]
    cls = np.array([iitb_cls.get(stems[v], "?") for v in test["video_id"]])
    iitb_rep, roc = eval_subset("IITB → IITB", det, test["X"][m], test["y"][m], test["video_id"][m],
                                cls[m], log, a.bootstrap)
    reps.append(iitb_rep)
    rocs.append(roc)

    # 3. IITB → UMN: trích đặc trưng MỘT lần, tách 3 cảnh theo test.txt
    log.info("========== IITB → UMN (3 cảnh) ==========")
    ucfg = umn_cfg(a.config, resolve_path(a.umn_config))
    _, utest, uitems, ucls_map = prepare_umn(ucfg, a.jobs, log)
    if utest["X"].shape[1] != det.model.support_vectors.shape[1]:
        raise SystemExit(f"Số chiều đặc trưng UMN {utest['X'].shape[1]} ≠ mô hình "
                         f"{det.model.support_vectors.shape[1]} – kiểm tra configs/default.yaml")
    ustems = [it.path.stem for it in uitems]
    ucls = np.array([ucls_map.get(ustems[v], "?") for v in utest["video_id"]])
    scenes = [s for s in SCENE_ORDER if s in set(ucls.tolist())]
    if not scenes:
        raise SystemExit("Không xác định được cảnh trong test.txt – chạy lại scripts/10_prepare_umn.py")
    for sc in scenes:
        mm = ucls == sc
        if len(np.unique(utest["y"][mm])) < 2:
            log.warning(f"Cảnh {sc} không có đủ hai lớp – bỏ qua")
            continue
        r, roc = eval_subset(f"IITB → UMN ({sc})", det, utest["X"][mm], utest["y"][mm],
                             utest["video_id"][mm], ucls[mm], log, a.bootstrap)
        reps.append(r)
        rocs.append(roc)
    if not a.no_combined:
        r, roc = eval_subset("IITB → UMN (gộp 3 cảnh)", det, utest["X"], utest["y"], utest["video_id"],
                             ucls, log, a.bootstrap)
        reps.append(r)
        rocs.append(roc)

    # 4. Xuất kết quả
    save_json(reps, out_dir / "report.json")
    write_outputs(reps, rocs, iitb_rep, out_dir, log)


if __name__ == "__main__":
    main()
