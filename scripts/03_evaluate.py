#!/usr/bin/env python3
"""BƯỚC 6 – Đánh giá mô hình số thực: AUC/EER/F1 mức khung hình + biểu đồ.

    python scripts/03_evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.config import resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.evaluation.metrics import (frame_metrics, per_video_auc, plot_roc,  # noqa: E402
                                           plot_score_hist, plot_timelines)
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "03_evaluate")
    det = AnomalyDetector.load(out / "models" / "detector.joblib")
    splits = get_splits(cfg, out, rng)
    val = load_split(splits["val"], cfg, out)
    test = load_split(splits["test"], cfg, out)
    fig_dir = out / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = {"threshold": det.threshold, "smoothing": det.smoothing}

    hist = {}
    if len(val["X"]):
        sv = det.score(val["X"], val["video_id"])
        report["val_false_alarm_rate"] = float(np.mean(sv > det.threshold))
        hist["val (bình thường)"] = sv

    if test.get("n_unlabeled_dropped"):
        log.warning(f"Bỏ {test['n_unlabeled_dropped']} video Test chưa có nhãn khỏi phần đánh giá")
    if len(test["X"]) and test["y"] is not None:
        held = det.meta.get("tuning_val_video_ids", [])
        m = ~np.isin(test["video_id"], held)
        if held:
            log.info(f"Loại {len(held)} video đã dùng để tìm tham số khỏi tập báo cáo")
        x, y, vid = test["X"][m], test["y"][m], test["video_id"][m]
        raw = det.score(x, vid, smooth=False)
        smooth = det.score(x, vid, smooth=True)
        report["test_raw"] = frame_metrics(y, raw, det.threshold)
        report["test_smoothed"] = frame_metrics(y, smooth, det.threshold)
        report["test_mean_per_video_auc"] = per_video_auc(y, smooth, vid)
        hist["test – bình thường"] = smooth[y == 0]
        hist["test – bất thường"] = smooth[y == 1]
        if len(np.unique(y)) == 2:
            plot_roc(y, smooth, fig_dir / "roc_test.png")
        # ---- AUC theo từng loại hành vi (test.txt) – mỗi loại gồm video của loại đó
        #      cộng tất cả video "Normal" để luôn có đủ hai lớp
        classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
        if classes:
            stems = [it.path.stem for it in splits["test"]]
            cls = np.array([classes.get(stems[v], "?") for v in vid])
            per = {}
            for c in sorted(set(cls) - {"Normal", "?"}):
                mk = (cls == c) | (cls == "Normal")
                if len(np.unique(y[mk])) == 2:
                    per[c] = {"auc": float(roc_auc_score(y[mk], smooth[mk])),
                              "videos": int(len(np.unique(vid[cls == c]))),
                              "recall": float(np.mean(smooth[(cls == c) & (y == 1)] > det.threshold))}
            if per:
                report["per_class"] = per
                log.info("AUC theo loại hành vi (loại + video Normal):")
                for c, r in sorted(per.items(), key=lambda kv: -kv[1]["auc"]):
                    log.info(f"   {c:<16} AUC = {r['auc']:.4f}  recall = {r['recall']:.2%}  ({r['videos']} video)")
        plot_timelines(smooth, y, vid, det.threshold, fig_dir / "timelines_test.png")
        r = report["test_smoothed"]
        n_vid = len(np.unique(vid))
        if len(np.unique(y)) < 2:
            only = "bình thường" if y.max() == 0 else "bất thường"
            log.warning(f"{n_vid} video Test có nhãn nhưng TẤT CẢ khung đều {only} → không tính được "
                        "AUC/EER/Precision/Recall (cần cả hai lớp). Hãy gán nhãn thêm video có hành vi "
                        "bất thường (scripts/08_annotate_videos.py).")
            if only == "bình thường":
                log.info(f"Tỉ lệ báo động giả trên {n_vid} video Test bình thường = {np.mean(smooth > det.threshold):.3%}")
            save_json(report, out / "reports" / "eval_report.json")
            return
        log.info(f"Đánh giá trên {n_vid} video Test có nhãn ({int(y.sum())}/{len(y)} cặp khung bất thường)")
        log.info(f"TEST (làm mượt k={det.smoothing}): AUC = {r.get('auc', float('nan')):.4f} | "
                 f"EER = {r.get('eer', float('nan')):.4f} | P = {r['precision']:.3f} "
                 f"R = {r['recall']:.3f} F1 = {r['f1']:.3f} | AUC thô = "
                 f"{report['test_raw'].get('auc', float('nan')):.4f}")
    else:
        log.warning("Không có tập Test có nhãn – chỉ báo cáo tỉ lệ báo động giả trên validation.")
    if hist:
        plot_score_hist(hist, det.threshold, fig_dir / "score_hist.png")
    if "val_false_alarm_rate" in report:
        log.info(f"Tỉ lệ báo động giả (val bình thường) = {report['val_false_alarm_rate']:.3%}")
    save_json(report, out / "reports" / "eval_report.json")
    log.info(f"Báo cáo → {out / 'reports'}")


if __name__ == "__main__":
    main()
