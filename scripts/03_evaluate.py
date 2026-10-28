#!/usr/bin/env python3
"""BƯỚC 6 – Đánh giá mô hình số thực: AUC/EER/F1 mức khung hình + biểu đồ.

    python scripts/03_evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup, load_detector_checked  # noqa: E402
from ocsvm_fpga.config import resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.evaluation.metrics import (frame_metrics, per_video_auc, plot_roc,  # noqa: E402
                                           plot_score_hist, plot_timelines, video_level_auc)
from ocsvm_fpga.evaluation.events import (alarm_params, alarm_settings, event_metrics,  # noqa: E402
                                          frame_metrics_binary, hysteresis_per_video)
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def evaluate_events(det, cfg, val, test, m, smooth, y, vid, splits, log):
    """(b) Đánh giá theo SỰ KIỆN, so sánh cảnh báo thường và cảnh báo có trễ (a)."""
    sv = det.score(val["X"], val["video_id"]) if len(val["X"]) else smooth[y == 0]
    ap = alarm_params(det, cfg, sv)
    fps = alarm_settings(cfg)["video_fps"]
    fidx = test["frame_idx"][m]
    classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
    stems = [it.path.stem for it in splits["test"]]
    cls_by_vid = {int(v): classes.get(stems[v], "?") for v in np.unique(vid)} if classes else None
    plain = (smooth > ap["on"]).astype(np.uint8)
    held = hysteresis_per_video(smooth, vid, ap["on"], ap["off"], ap["hold"])
    out = {"params": ap, "fps": fps}
    hold_s = ap["hold"] * cfg.preprocess.frame_step / fps
    log.info(f"Cảnh báo theo SỰ KIỆN – bật khi điểm > {ap['on']:.4f}; có trễ: tắt khi < {ap['off']:.4f} "
             f"liên tục {ap['hold']} cặp khung (≈ {hold_s:.1f} s)")
    for name, al in (("không trễ", plain), ("có trễ", held)):
        ev = event_metrics(y, al, vid, fidx, fps, cls_by_vid)
        fm = frame_metrics_binary(y, al)
        if len(val["X"]):
            av = (sv > ap["on"]).astype(np.uint8) if name == "không trễ" else \
                hysteresis_per_video(sv, val["video_id"], ap["on"], ap["off"], ap["hold"])
            evv = event_metrics(np.zeros(len(sv), np.uint8), av, val["video_id"], val["frame_idx"], fps)
            ev["val_false_alarms_per_hour"] = evv["false_alarms_per_hour"]
        out["hysteresis" if name == "có trễ" else "plain"] = {"events": {k: v for k, v in ev.items()}, "frames": fm}
        log.info(f"  {name:<9}: phát hiện {ev['detected']}/{ev['events']} sự kiện ({ev['event_recall']:.1%}) | "
                 f"trễ trung vị {ev['delay_median_s']:.2f} s | phủ TB {ev['coverage_mean']:.0%} sự kiện | "
                 f"báo giả {ev['false_alarms_per_hour']:.1f} lần/giờ (Test)"
                 + (f", {ev['val_false_alarms_per_hour']:.1f} lần/giờ (val)" if "val_false_alarms_per_hour" in ev else "")
                 + f" | khung: P = {fm['precision']:.3f} R = {fm['recall']:.3f} F1 = {fm['f1']:.3f}")
    per = out["hysteresis"]["events"].get("per_class")
    if per:
        log.info("  Theo loại (có trễ): " + " | ".join(
            f"{c} {r['recall']:.0%}" + (f" ({r['delay_median_s']:.1f}s)" if r['recall'] else "")
            for c, r in sorted(per.items(), key=lambda kv: -kv[1]["recall"])))
    return out


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "03_evaluate")
    det = load_detector_checked(cfg, out)
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
        vl = video_level_auc(y, smooth, vid)
        if vl is not None:
            report["video_level_auc"] = {"auc": vl[0], "normal_videos": vl[1], "abnormal_videos": vl[2]}
            log.info(f"AUC mức video (điểm video = trung bình khung): {vl[0]:.4f} "
                     f"({vl[1]} video bình thường, {vl[2]} video bất thường)")
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
        report["events"] = evaluate_events(det, cfg, val, test, m, smooth, y, vid, splits, log)
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
