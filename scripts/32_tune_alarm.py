#!/usr/bin/env python3
"""CHỈNH LOGIC CẢNH BÁO trên MÔ HÌNH ĐANG DÙNG (không train lại, không đổi FPGA) – chạy vài giây.

Quét các tham số chỉ nằm ở khâu cảnh báo:
  alarm.on_percentile  ngưỡng BẬT = phân vị (100 − p)% điểm bình thường (val); 2 = ngưỡng mô hình hiện tại
  alarm.on_pairs       bật khi vượt ngưỡng LIÊN TỤC N cặp
  alarm.off_percentile ngưỡng TẮT = phân vị p% điểm bình thường (cao hơn → tắt sớm hơn)
  alarm.hold_seconds   phải xuống dưới ngưỡng tắt bao lâu mới tắt

Đo trên Test: phát hiện sự kiện, trễ, báo giả / giờ, % thời gian báo động trên video Normal (+ từng video
--videos). In thêm ngưỡng f_q tương ứng (để đặt --thr cho màn hình demo 27_demo_monitor.py).

    python scripts/32_tune_alarm.py --config configs/hw.yaml --videos 000225 000239 000255 000299 000338 000343

Kết quả: <output_dir>/reports/alarm_tuning.md / .json
"""
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, load_detector_checked, setup  # noqa: E402
from ocsvm_fpga.config import resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import load_test_classes  # noqa: E402
from ocsvm_fpga.evaluation.events import alarm_params, event_metrics, hysteresis_per_video  # noqa: E402
from ocsvm_fpga.features.pipeline import load_split  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--on-pcts", type=float, nargs="+", default=[2.0, 1.5, 1.0, 0.5])
    ap.add_argument("--on-pairs", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--off-pcts", type=float, nargs="+", default=[85.0, 95.0, 98.0])
    ap.add_argument("--holds", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--max-recall-drop", type=float, default=0.05,
                    help="chỉ đề xuất cấu hình có tỉ lệ phát hiện giảm không quá mức này so với bản gốc")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "32_tune_alarm")
    det = load_detector_checked(cfg, out)
    splits = get_splits(cfg, out, rng)
    val = load_split(splits["val"], cfg, out)
    test = load_split(splits["test"], cfg, out)
    vid, y, fidx = test["video_id"], test["y"], test["frame_idx"]
    s = det.score(test["X"], vid)
    sv = det.score(val["X"], val["video_id"])
    fps = float((cfg.get("alarm") or {}).get("video_fps", 25.0))
    stems = [it.path.stem for it in splits["test"]]
    classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
    cls_by_vid = {int(v): classes.get(stems[v], "?") for v in np.unique(vid)} if classes else None
    normal = np.array([cls_by_vid.get(int(v)) == "Normal" for v in vid]) if cls_by_vid else np.zeros(len(vid), bool)
    want = {st: i for i, st in enumerate(stems) if st in set(args.videos)}
    # hệ số đổi điểm → f_q (FPGA): f_q = −score × RHO_Q / ρ
    scale = None
    vh = out / "fpga" / "ocsvm_params.vh"
    if vh.exists():
        m = re.search(r"RHO_Q\s*=\s*(-?)(?:\d+'s?d)?(\d+)", vh.read_text(encoding="utf-8", errors="ignore"))
        if m and det.model.rho:
            scale = int(m.group(1) + m.group(2)) / float(det.model.rho)

    rows = []
    for onp, k, offp, hs in itertools.product(args.on_pcts, args.on_pairs, args.off_pcts, args.holds):
        cfg["alarm"]["on_percentile"] = onp
        cfg["alarm"]["on_pairs"] = k
        cfg["alarm"]["off_percentile"] = offp
        cfg["alarm"]["hold_seconds"] = hs
        a = alarm_params(det, cfg, sv)
        al = hysteresis_per_video(s, vid, a["on"], a["off"], a["hold"], a["on_pairs"])
        ev = event_metrics(y, al, vid, fidx, fps, cls_by_vid)
        r = {"on_percentile": onp, "on_pairs": k, "off_percentile": offp, "hold_seconds": hs,
             "on": a["on"], "off": a["off"], "on_fq": None if scale is None else int(round(-a["on"] * scale)),
             "event_recall": ev["event_recall"], "detected": ev["detected"], "events": ev["events"],
             "delay_median_s": ev["delay_median_s"], "fa_per_hour": ev["false_alarms_per_hour"],
             "alarm_pct_normal_videos": float(al[normal].mean()) if normal.any() else float("nan"),
             "per_video": {st: float(al[vid == i].mean()) for st, i in want.items() if (vid == i).any()}}
        rows.append(r)
    base = next(r for r in rows if r["on_percentile"] == 2.0 and r["on_pairs"] == 1 and r["off_percentile"] == 85.0
                and r["hold_seconds"] == 1.0) if any(r["on_percentile"] == 2.0 for r in rows) else rows[0]
    ok = [r for r in rows if r["event_recall"] >= base["event_recall"] - args.max_recall_drop]
    best = min(ok, key=lambda r: (r["fa_per_hour"], r["alarm_pct_normal_videos"]))
    vids = list(want)
    hdr = ["ngưỡng bật (phân vị)", "bật sau", "ngưỡng tắt (phân vị)", "giữ (s)", "on (điểm)", "on (f_q)",
           "phát hiện", "trễ (s)", "báo giả/giờ", "% báo động video Normal"] + vids
    lines = ["# Chỉnh logic cảnh báo – mô hình giữ nguyên", "",
             f"Mô hình: {det.model.n_sv} SV · ngưỡng mô hình {det.threshold:.5f} · Test {len(np.unique(vid))} video. "
             "Bản gốc = bật ở phân vị 2 %, ngay lần đầu, tắt ở phân vị 85 %, giữ 1 s.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in sorted(rows, key=lambda r: (r["fa_per_hour"])):
        lines.append("| " + " | ".join([f"{100 - r['on_percentile']:g} %", str(r["on_pairs"]), f"{r['off_percentile']:g} %",
                                         f"{r['hold_seconds']:g}", f"{r['on']:.4f}", str(r["on_fq"]),
                                         f"{r['detected']}/{r['events']} ({r['event_recall']:.1%})",
                                         f"{r['delay_median_s']:.2f}", f"{r['fa_per_hour']:.1f}",
                                         f"{r['alarm_pct_normal_videos']:.1%}"]
                                        + [f"{r['per_video'].get(v, float('nan')):.1%}" for v in vids])
                     + (" | **đề xuất**" if r is best else " |"))
    yaml_txt = (f"alarm:\n  on_percentile: {best['on_percentile']:g}\n  on_pairs: {best['on_pairs']}\n"
                f"  off_percentile: {best['off_percentile']:g}\n  hold_seconds: {best['hold_seconds']:g}")
    lines += ["", f"**Đề xuất** (phát hiện giảm ≤ {args.max_recall_drop:.0%}): báo giả {base['fa_per_hour']:.1f} → "
              f"{best['fa_per_hour']:.1f}/giờ, phát hiện {base['detected']} → {best['detected']}/{best['events']}, "
              f"% báo động video Normal {base['alarm_pct_normal_videos']:.1%} → {best['alarm_pct_normal_videos']:.1%}.",
              "", "Ghi vào configs/hw.yaml:", "", "```yaml", yaml_txt, "```"]
    if best["on_fq"] is not None:
        lines += ["", f"Màn hình demo (27_demo_monitor.py / demo1_launcher.json): `--thr {best['on_fq']}`"]
    rep = out / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "alarm_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json({"rows": rows, "best": best, "base": base}, rep / "alarm_tuning.json")
    log.info(f"Bản gốc : báo giả {base['fa_per_hour']:.1f}/giờ, phát hiện {base['detected']}/{base['events']}, "
             f"video Normal {base['alarm_pct_normal_videos']:.1%}")
    log.info(f"Đề xuất : {yaml_txt.replace(chr(10), ' ')} → báo giả {best['fa_per_hour']:.1f}/giờ, phát hiện "
             f"{best['detected']}/{best['events']}, video Normal {best['alarm_pct_normal_videos']:.1%}"
             + (f", --thr {best['on_fq']}" if best["on_fq"] is not None else ""))
    log.info(f"Báo cáo → {rep / 'alarm_tuning.md'}")


if __name__ == "__main__":
    main()
