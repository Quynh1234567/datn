#!/usr/bin/env python3
"""GIẢM BÁO ĐỘNG GIẢ – thử nhiều cấu hình trên đặc trưng ĐÃ CÓ (không trích xuất lại).

Hai nguyên nhân báo nhầm trên video IITB bình thường (xem phân tích 04/11/2026):
  1. Ô ít người qua (cột phải: bụi cây) có σ rất nhỏ → người đi qua làm z tăng 2–10 lần → điểm chạm đáy.
     Sửa: mức sàn σ  (model.scaler_std_floor_rel = k → σ ≥ k × trung vị σ).
  2. Nhiều người đi sát nhau → nhiều chiều z cùng lớn → khoảng cách RBF cộng dồn.
     Sửa: kernel rộng hơn (model.gamma nhỏ hơn).
  + Một kết quả vượt ngưỡng là bật báo động → đòi hỏi vượt liên tục (alarm.on_pairs).

Mỗi biến thể: huấn luyện lại (chuẩn hóa + OC-SVM + ngưỡng, giống 02_train) rồi đo trên Test như 03_evaluate:
AUC khung, phát hiện sự kiện, báo giả / giờ, % thời gian báo động trên video Normal. KHÔNG ghi đè
mô hình đang dùng – chỉ ghi báo cáo. Chọn xong thì chạy 02 → 03 → 04 với --set của biến thể đó.

    python scripts/31_tune_false_alarm.py --config configs/hw.yaml
    python scripts/31_tune_false_alarm.py --config configs/hw.yaml --floors 0 0.5 1 --gammas 1 0.5 \
        --on-pairs 1 3 5 --videos 000225 000239 000255 000299 000338 000343

Kết quả: <output_dir>/reports/false_alarm_tuning.md / .json
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.config import resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.evaluation.events import alarm_params, alarm_settings, event_metrics, hysteresis_per_video  # noqa: E402
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.models.training import calibrate_threshold, fit_detector  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0],
                    help="model.scaler_std_floor_rel cần thử (0 = như cũ)")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 0.5],
                    help="hệ số nhân với model.gamma hiện tại (1 = như cũ)")
    ap.add_argument("--on-pairs", type=int, nargs="+", default=[1, 3, 5],
                    help="alarm.on_pairs cần thử (1 = như cũ); không cần huấn luyện lại")
    ap.add_argument("--videos", nargs="*", default=[],
                    help="tên video Test cần báo riêng %% thời gian báo động (vd 000225 000239)")
    ap.add_argument("--max-recall-drop", type=float, default=0.03,
                    help="chỉ đề xuất biến thể có tỉ lệ phát hiện sự kiện giảm không quá mức này so với bản gốc")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "31_tune_false_alarm")
    splits = get_splits(cfg, out, rng)
    train = load_split(splits["train"], cfg, out)
    val = load_split(splits["val"], cfg, out)
    test = load_split(splits["test"], cfg, out)
    if not len(train["X"]) or test.get("y") is None:
        sys.exit("Cần cache đặc trưng train/val/test có nhãn – chạy 01_extract_features.py trước")
    x, y, vid, fidx = test["X"], test["y"], test["video_id"], test["frame_idx"]
    stems = [it.path.stem for it in splits["test"]]
    classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
    cls_by_vid = {int(v): classes.get(stems[v], "?") for v in np.unique(vid)} if classes else None
    normal_mask = np.array([cls_by_vid.get(int(v)) == "Normal" for v in vid]) if cls_by_vid else np.zeros(len(vid), bool)
    want = {s: i for i, s in enumerate(stems) if s in set(args.videos)}
    missing = sorted(set(args.videos) - set(want))
    if missing:
        log.warning(f"Không có trong tập Test (hoặc chưa có cache): {missing}")
    fps = alarm_settings(cfg)["video_fps"]
    g0 = cfg.model.gamma
    log.info(f"Train {train['X'].shape} · Val {val['X'].shape} · Test {x.shape} · "
             f"{int(normal_mask.sum())} cặp thuộc video Normal · γ gốc = {g0}")

    rows = []
    for floor, gm in itertools.product(args.floors, args.gammas):
        cfg["model"]["scaler_std_floor_rel"] = float(floor)
        t0 = time.time()
        det, _ = fit_detector(train["X"], cfg, np.random.default_rng(int(cfg.project.seed)), None, gm, None)
        calibrate_threshold(det, cfg, val, None)
        s = det.score(x, vid, smooth=True)
        auc = float(roc_auc_score(y, s))
        sv = det.score(val["X"], val["video_id"]) if len(val["X"]) else s[y == 0]
        for k in args.on_pairs:
            cfg["alarm"]["on_pairs"] = int(k)
            a = alarm_params(det, cfg, sv)
            al = hysteresis_per_video(s, vid, a["on"], a["off"], a["hold"], a["on_pairs"])
            ev = event_metrics(y, al, vid, fidx, fps, cls_by_vid)
            r = {"floor_rel": floor, "gamma_mult": gm, "gamma": float(det.model.gamma), "on_pairs": int(k),
                 "n_sv": int(det.model.n_sv), "n_floored": int(det.meta.get("scaler_n_floored", 0)),
                 "auc": auc, "event_recall": ev["event_recall"], "delay_median_s": ev["delay_median_s"],
                 "fa_per_hour": ev["false_alarms_per_hour"],
                 "alarm_pct_normal_videos": float(al[normal_mask].mean()) if normal_mask.any() else float("nan"),
                 "alarm_pct_normal_frames": float(al[y == 0].mean()),
                 "per_video": {st: float(al[vid == i].mean()) for st, i in want.items() if (vid == i).any()}}
            rows.append(r)
            log.info(f"σ-sàn {floor:<4} γ×{gm:<4} bật sau {k} cặp | SV {r['n_sv']:3d} | AUC {auc:.4f} | "
                     f"phát hiện {r['event_recall']:.1%} (trễ {r['delay_median_s']:.2f} s) | báo giả {r['fa_per_hour']:.1f}/giờ | "
                     f"báo động trên video Normal {r['alarm_pct_normal_videos']:.1%}"
                     + "".join(f" | {st} {p:.1%}" for st, p in r["per_video"].items()))
        log.info(f"   ({time.time() - t0:.0f} s)")
    cfg["alarm"]["on_pairs"] = 1

    base = next(r for r in rows if r["floor_rel"] == args.floors[0] and r["gamma_mult"] == args.gammas[0]
                and r["on_pairs"] == args.on_pairs[0])
    ok = [r for r in rows if r["event_recall"] >= base["event_recall"] - args.max_recall_drop
          and r["n_sv"] <= int(cfg.model.max_support_vectors)]
    best = min(ok, key=lambda r: (r["fa_per_hour"], -r["auc"])) if ok else None

    cols = [("σ-sàn", "floor_rel", "{:g}"), ("γ", "gamma", "{:.4g}"), ("bật sau (cặp)", "on_pairs", "{}"),
            ("SV", "n_sv", "{}"), ("AUC", "auc", "{:.4f}"), ("Phát hiện sự kiện", "event_recall", "{:.1%}"),
            ("Trễ (s)", "delay_median_s", "{:.2f}"), ("Báo giả / giờ", "fa_per_hour", "{:.1f}"),
            ("% báo động – video Normal", "alarm_pct_normal_videos", "{:.1%}")]
    vids = list(want)
    lines = ["# Giảm báo động giả – so sánh cấu hình", "",
             f"Cấu hình: `{args.config}` · Test {len(np.unique(vid))} video, {len(x)} cặp · "
             f"bản gốc = dòng đầu tiên (σ-sàn {args.floors[0]:g}, γ×{args.gammas[0]:g}, bật sau {args.on_pairs[0]} cặp).", "",
             "| " + " | ".join(c for c, _, _ in cols) + "".join(f" | {v}" for v in vids) + " |",
             "|" + "---|" * (len(cols) + len(vids))]
    for r in sorted(rows, key=lambda r: (r["floor_rel"], -r["gamma_mult"], r["on_pairs"])):
        mark = " **(đề xuất)**" if r is best else ""
        lines.append("| " + " | ".join(f.format(r[k]) for _, k, f in cols) +
                     "".join(f" | {r['per_video'].get(v, float('nan')):.1%}" for v in vids) + f" |{mark}")
    if best:
        cmd = (f"--set model.scaler_std_floor_rel={best['floor_rel']:g} model.gamma={best['gamma']:.6g} "
               f"alarm.on_pairs={best['on_pairs']}")
        lines += ["", f"**Đề xuất:** σ-sàn {best['floor_rel']:g}, γ = {best['gamma']:.4g}, bật sau {best['on_pairs']} cặp – "
                  f"báo giả {base['fa_per_hour']:.1f} → {best['fa_per_hour']:.1f} lần/giờ, phát hiện "
                  f"{base['event_recall']:.1%} → {best['event_recall']:.1%}, AUC {base['auc']:.4f} → {best['auc']:.4f}.",
                  "", "Ghi vào `configs/hw.yaml` (mục model / alarm) hoặc dùng:", "",
                  f"    python scripts/02_train.py --config {args.config} {cmd}",
                  f"    python scripts/03_evaluate.py --config {args.config} {cmd}",
                  f"    python scripts/04_export_fpga.py --config {args.config} {cmd}"]
        log.info(f"ĐỀ XUẤT: {cmd}  (báo giả {base['fa_per_hour']:.1f} → {best['fa_per_hour']:.1f}/giờ, "
                 f"phát hiện {base['event_recall']:.1%} → {best['event_recall']:.1%})")
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "false_alarm_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json({"rows": rows, "best": best, "base": base}, rep_dir / "false_alarm_tuning.json")
    log.info(f"Báo cáo → {rep_dir / 'false_alarm_tuning.md'}")


if __name__ == "__main__":
    main()
