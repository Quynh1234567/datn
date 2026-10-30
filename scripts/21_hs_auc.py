#!/usr/bin/env python3
"""A4 – CÓ CẦN TRAIN LẠI KHÔNG? Đặc trưng từ HS số nguyên (đường phần cứng) → mô hình OC-SVM hiện có → AUC.

Trên cùng tập Test (và val để đo báo động giả), so ba đường:
  (1) GỐC     : đặc trưng số thực trong cache (pipeline train)     → mô hình số thực
  (2) HS-NGUYÊN: Gauss ESP32 (preproc_q) → HS/lưới/log FPGA (hs_q) → mô hình số thực
  (3) PHẦN CỨNG: như (2) → z-score Q5.10 → OC-SVM số nguyên (FixedPointOCSVM) = f_q trên board
Quy tắc (kế hoạch giai đoạn A): AUC (3) giảm ≤ --tol so với (1) → giữ mô hình; ngược lại → trích lại
đặc trưng cả 358 video bằng HS số nguyên (--splits train val test) rồi train lại cùng ν, γ, 128 SV.

Khung 160×120 lấy từ cache outputs/frames (không giải mã lại video). Đặc trưng HS số nguyên cache theo
video trong outputs/features_hw/<hash>/<split>/ → dừng giữa chừng chạy lại sẽ tiếp tục.

    python scripts/21_hs_auc.py --jobs 8
    python scripts/21_hs_auc.py --jobs 8 --max-videos 10        # thử nhanh
    python scripts/21_hs_auc.py --jobs 8 --splits train val test # chuẩn bị train lại
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, load_detector_checked, setup  # noqa: E402
from ocsvm_fpga.config import resolve_path  # noqa: E402
from ocsvm_fpga.data.dataset import find_annotation, load_frame_labels, load_test_classes  # noqa: E402
from ocsvm_fpga.data.video_reader import video_info  # noqa: E402
from ocsvm_fpga.evaluation.metrics import frame_metrics, per_video_auc, video_level_auc  # noqa: E402
from ocsvm_fpga.features.pipeline import frames_signature, load_sensor_frames, load_split  # noqa: E402
from ocsvm_fpga.hwmodel import hs_q as HQ, preproc_q as PQ  # noqa: E402
from ocsvm_fpga.models.base import smooth_per_video  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402


# --------------------------------------------------------------------------- trích đặc trưng HS số nguyên

def hw_signature(cfg, p: HQ.HSParams) -> str:
    key = json.dumps({"frames": frames_signature(cfg), "hs": asdict(p), "gauss": "gauss5_q/reflect101",
                      "dark": int(cfg.preprocess.get("skip_dark_below", 0) or 0)}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def hw_video(item, cfg, out_root: Path, p: HQ.HSParams) -> dict:
    """Một video → log_q (n, 48) Q5.10 trước chuẩn hóa, đúng như FPGA (chưa phụ thuộc bộ chuẩn hóa)."""
    frames, fidx = load_sensor_frames(item, cfg, out_root)
    dark = int(cfg.preprocess.get("skip_dark_below", 0) or 0)
    lut = HQ.log_lut(p)
    feats, ids, n_sat = [], [], 0
    prev = None
    for idx, raw in zip(fidx, frames):
        g = PQ.gauss5_q(raw)                           # ESP32 gửi ảnh đã Gauss
        if dark > 0 and PQ.is_dark(g, dark):           # khung tối → ESP32 báo FPGA ngắt cặp
            prev = None
            continue
        if prev is not None:
            tr = HQ.features_q(prev, g, p, lut=lut)
            feats.append(tr.log_q)
            ids.append(int(idx))                       # nhãn của cặp = nhãn khung thứ hai (như train)
            n_sat += int(tr.hs.n_sat)
        prev = g
    return {"log_q": np.asarray(feats, np.int16).reshape(-1, HQ.DIM), "frame_idx": np.asarray(ids, np.int32),
            "n_frames": np.int32(video_info(item.path)["frames"]), "n_sat": np.int64(n_sat)}


def hw_path(item, cfg, out_root: Path, sig: str) -> Path:
    return out_root / "features_hw" / sig / item.split / f"{item.uid}.npz"


def _process(item, cfg, out_root: Path, p: HQ.HSParams, sig: str):
    path = hw_path(item, cfg, out_root, sig)
    if path.exists():
        return item.uid, -1, None
    try:
        res = hw_video(item, cfg, out_root, p)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(path)
        return item.uid, len(res["frame_idx"]), None
    except Exception as exc:  # noqa: BLE001 – một video lỗi không làm dừng cả tập
        return item.uid, 0, repr(exc)


def extract(items, cfg, out_root, p, sig, jobs, log):
    todo = [it for it in items if not hw_path(it, cfg, out_root, sig).exists()]
    if not todo:
        return
    log.info(f"[{items[0].split}] HS số nguyên cho {len(todo)}/{len(items)} video với {jobs} tiến trình ...")
    t0 = time.time()
    if jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=jobs, verbose=5)(delayed(_process)(it, cfg, out_root, p, sig) for it in todo)
    else:
        from ocsvm_fpga.utils import progress
        res = [_process(it, cfg, out_root, p, sig) for it in progress(todo, desc=f"[{items[0].split}]")]
    for uid, _, err in res:
        if err:
            log.error(f"Lỗi video {uid}: {err}")
    log.info(f"[{items[0].split}] xong sau {time.time() - t0:.0f}s")


def labels_for(item, cfg, n_frames: int):
    lab = item.labels_path if item.labels_path and Path(item.labels_path).exists() else None
    if lab is None and cfg.data.annotation_dir:
        lab = find_annotation(item.path.stem, resolve_path(cfg.data.annotation_dir), cfg.data.annotation_patterns,
                              item.path)
    return load_frame_labels(lab, n_frames) if lab is not None else None


def load_hw(items, cfg, out_root, sig) -> dict:
    """Ghép cache HS số nguyên; video_id = chỉ số trong splits (trùng với load_split). Test chỉ giữ video có nhãn."""
    xs, ys, vids, fids, n_sat = [], [], [], [], 0
    for vid, it in enumerate(items):
        path = hw_path(it, cfg, out_root, sig)
        if not path.exists():
            continue
        d = np.load(path)
        if len(d["frame_idx"]) == 0:
            continue
        y = None
        if it.split == "test":
            lab = labels_for(it, cfg, int(d["n_frames"]))
            if lab is None:
                continue
            y = lab[np.clip(d["frame_idx"], 0, len(lab) - 1)].astype(np.uint8)
        xs.append(d["log_q"].astype(np.int64))
        ys.append(y if y is not None else np.zeros(len(d["frame_idx"]), np.uint8))
        vids.append(np.full(len(d["frame_idx"]), vid, np.int32))
        fids.append(d["frame_idx"])
        n_sat += int(d["n_sat"])
    if not xs:
        return {"log_q": np.zeros((0, HQ.DIM), np.int64), "y": np.zeros(0, np.uint8),
                "video_id": np.zeros(0, np.int32), "frame_idx": np.zeros(0, np.int32), "n_sat": 0}
    return {"log_q": np.concatenate(xs), "y": np.concatenate(ys), "video_id": np.concatenate(vids),
            "frame_idx": np.concatenate(fids), "n_sat": n_sat}


# --------------------------------------------------------------------------- đánh giá

def path_metrics(name, y, raw, vid, det, classes=None, stems=None):
    sm = smooth_per_video(raw, vid, det.smoothing)
    r = {"name": name, "raw": frame_metrics(y, raw, det.threshold), "smoothed": frame_metrics(y, sm, det.threshold),
         "per_video_auc": per_video_auc(y, sm, vid)}
    vl = video_level_auc(y, sm, vid)
    if vl is not None:
        r["video_level_auc"] = vl[0]
    if classes and stems:
        cls = np.array([classes.get(stems[v], "?") for v in vid])
        r["per_class_auc"] = {c: float(roc_auc_score(y[(cls == c) | (cls == "Normal")],
                                                     sm[(cls == c) | (cls == "Normal")]))
                              for c in sorted(set(cls) - {"Normal", "?"})
                              if len(np.unique(y[(cls == c) | (cls == "Normal")])) == 2}
    return r, sm


def common_pairs(a_vid, a_fid, b_vid, b_fid):
    """Chỉ số các cặp khung có ở cả hai đường (cùng video, cùng frame_idx)."""
    ka = a_vid.astype(np.int64) << 32 | a_fid.astype(np.int64)
    kb = b_vid.astype(np.int64) << 32 | b_fid.astype(np.int64)
    _, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    return ia, ib


def report(path, s):
    f = lambda x: "–" if x is None else f"{x:.4f}"  # noqa: E731
    P = s["paths"]
    L = ["# A4 – Đặc trưng HS số nguyên đưa vào mô hình OC-SVM hiện có", "",
         f"Thời gian {s['time']} · {s['test_videos']} video Test có nhãn · tham số HS: {s['hs_params']} · "
         f"làm mượt k = {s['smoothing']} · ngưỡng score {s['threshold']:.5f}", "",
         "## AUC mức khung trên Test", "",
         "| Đường | Số cặp | AUC (làm mượt) | AUC thô | EER | AUC TB theo video | AUC mức video | P | R | F1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in P:
        sm = r["smoothed"]
        L.append(f"| {r['name']} | {sm['n_frames']} | **{f(sm.get('auc'))}** | {f(r['raw'].get('auc'))} | "
                 f"{f(sm.get('eer'))} | {f(r['per_video_auc'])} | {f(r.get('video_level_auc'))} | "
                 f"{sm['precision']:.3f} | {sm['recall']:.3f} | {sm['f1']:.3f} |")
    L += ["", f"Cặp khung chung giữa đường gốc và đường phần cứng: {s['common']['n']} "
          f"(gốc {s['common']['n_base']}, phần cứng {s['common']['n_hw']} – khác nhau do quyết định khung tối "
          f"tính trên Gauss số nguyên). Trên các cặp chung: AUC gốc {f(s['common']['auc_base'])}, phần cứng "
          f"{f(s['common']['auc_hw'])}, tương quan điểm (làm mượt) {s['common']['score_corr']:.4f}, "
          f"cờ bất thường (làm mượt, cùng ngưỡng) trùng **{s['common']['flag_agree_pct']:.2f}%**.", "",
          f"Bão hòa u/v trong HS số nguyên: {s['n_sat_test']} lần (Test).", ""]
    if s.get("val"):
        v = s["val"]
        L += ["## Báo động giả trên val (video bình thường)", "", "| Đường | Số cặp | Tỉ lệ score > ngưỡng (làm mượt) |",
              "|---|---|---|"] + [f"| {k} | {v['n']} | {x:.3%} |" for k, x in v["far"].items()] + [""]
    if any("per_class_auc" in r for r in P):
        cls = sorted({c for r in P for c in r.get("per_class_auc", {})})
        L += ["## AUC theo loại hành vi (loại + video Normal)", "", "| Loại | " + " | ".join(r["name"] for r in P) + " |",
              "|---|" + "---|" * len(P)]
        L += [f"| {c} | " + " | ".join(f(r.get("per_class_auc", {}).get(c)) for r in P) + " |" for c in cls]
        L.append("")
    d = s["decision"]
    L += ["## Kết luận", "", f"AUC phần cứng − AUC gốc = **{d['delta']:+.4f}** (ngưỡng chấp nhận −{d['tol']}). "
          f"→ **{d['verdict']}**", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--jobs", type=int, default=None, help="số tiến trình (mặc định project.n_jobs)")
    ap.add_argument("--splits", nargs="*", default=["val", "test"], help="tập cần trích HS số nguyên")
    ap.add_argument("--max-videos", type=int, default=0, help="chỉ dùng N video đầu mỗi tập (thử nhanh)")
    ap.add_argument("--tol", type=float, default=0.01, help="AUC được phép giảm tối đa để giữ mô hình cũ")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "21_hs_auc")
    det = load_detector_checked(cfg, out)
    if det.model.support_vectors.shape[1] != HQ.DIM:
        sys.exit(f"Mô hình có {det.model.support_vectors.shape[1]} chiều, khối FPGA tạo {HQ.DIM}")
    splits = get_splits(cfg, out, rng)
    if args.max_videos:
        splits = {k: v[:args.max_videos] for k, v in splits.items()}
    p = HQ.DEFAULT
    sig = hw_signature(cfg, p)
    jobs = args.jobs or int(cfg.project.n_jobs)
    log.info(f"HS số nguyên: u, v Q{p.wu - p.fu - 1}.{p.fu} ({p.wu} bit), {p.n_iter} vòng, |v| {p.mag_mode} · "
             f"cache = features_hw/{sig}")
    for name in args.splits:
        if splits.get(name):
            extract(splits[name], cfg, out, p, sig, jobs, log)

    sc = HQ.quantize_scaler(det.scaler.mean, det.scaler.scale, p)
    fx = FixedPointOCSVM(det, cfg.fpga)
    held = det.meta.get("tuning_val_video_ids", [])
    classes = load_test_classes(resolve_path(cfg.data.get("test_class_file")))
    stems = [it.path.stem for it in splits["test"]]

    def hw_scores(log_q):
        """(score mô hình số thực trên đặc trưng HS số nguyên, score từ f_q của OC-SVM số nguyên)."""
        s_float = det.score(log_q / (1 << p.data_frac), smooth=False)
        f_q = fx.decision_q(HQ.standardize_q(log_q, sc, p))
        return s_float, det.model.score_from_decision(fx.decision_float_equiv(f_q)), f_q

    # (1) gốc
    base = load_split(splits["test"], cfg, out)
    if base["y"] is None or not len(base["X"]):
        sys.exit("Không có đặc trưng Test có nhãn trong cache – chạy scripts/01_extract_features.py trước")
    mb = ~np.isin(base["video_id"], held)
    b_y, b_vid, b_fid = base["y"][mb], base["video_id"][mb], base["frame_idx"][mb]
    r_base, sm_base = path_metrics("(1) Gốc: số thực → mô hình số thực", b_y,
                                   det.score(base["X"][mb], smooth=False), b_vid, det, classes, stems)
    # (2), (3) HS số nguyên
    hw = load_hw(splits["test"], cfg, out, sig)
    mh = ~np.isin(hw["video_id"], held)
    h_y, h_vid, h_fid = hw["y"][mh], hw["video_id"][mh], hw["frame_idx"][mh]
    s_float, s_fx, f_q = hw_scores(hw["log_q"][mh])
    r_hsf, _ = path_metrics("(2) HS số nguyên → mô hình số thực", h_y, s_float, h_vid, det, classes, stems)
    r_hw, sm_hw = path_metrics("(3) Phần cứng: HS số nguyên → OC-SVM số nguyên", h_y, s_fx, h_vid, det,
                               classes, stems)
    flag_raw = fx.anomaly_q(f_q)
    log.info(f"Kiểm tra: cờ f_q < thr_q trùng score thô > ngưỡng ở {np.mean(flag_raw == (s_fx > det.threshold)):.4%} cặp")

    ib, ih = common_pairs(b_vid, b_fid, h_vid, h_fid)
    yc = b_y[ib]
    common = {"n": int(len(ib)), "n_base": int(len(b_y)), "n_hw": int(len(h_y)),
              "auc_base": float(roc_auc_score(yc, sm_base[ib])) if len(np.unique(yc)) == 2 else None,
              "auc_hw": float(roc_auc_score(yc, sm_hw[ih])) if len(np.unique(yc)) == 2 else None,
              "score_corr": float(np.corrcoef(sm_base[ib], sm_hw[ih])[0, 1]),
              "flag_agree_pct": float(np.mean((sm_base[ib] > det.threshold) == (sm_hw[ih] > det.threshold)) * 100),
              "label_agree_pct": float(np.mean(yc == h_y[ih]) * 100)}

    val = None
    if splits.get("val") and "val" in args.splits:
        bv = load_split(splits["val"], cfg, out)
        hv = load_hw(splits["val"], cfg, out, sig)
        if len(bv["X"]) and len(hv["log_q"]):
            vf, vx, _ = hw_scores(hv["log_q"])
            far = lambda s, v: float(np.mean(smooth_per_video(s, v, det.smoothing) > det.threshold))  # noqa: E731
            val = {"n": int(len(hv["log_q"])), "far": {
                "(1) Gốc": far(det.score(bv["X"], smooth=False), bv["video_id"]),
                "(2) HS số nguyên → số thực": far(vf, hv["video_id"]),
                "(3) Phần cứng": far(vx, hv["video_id"])}}

    a_base, a_hw = r_base["smoothed"].get("auc"), r_hw["smoothed"].get("auc")
    delta = a_hw - a_base
    keep = delta >= -args.tol
    s = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "test_videos": int(len(np.unique(h_vid))),
         "hs_params": f"u, v Q{p.wu - p.fu - 1}.{p.fu} {p.wu} bit, {p.n_iter} vòng, |v| {p.mag_mode}",
         "hs_signature": sig, "smoothing": det.smoothing, "threshold": det.threshold,
         "paths": [r_base, r_hsf, r_hw], "common": common, "val": val, "n_sat_test": hw["n_sat"],
         "decision": {"delta": delta, "tol": args.tol, "keep_model": bool(keep),
                      "verdict": "GIỮ mô hình OC-SVM hiện có, không cần train lại" if keep else
                      "TRAIN LẠI: trích đặc trưng HS số nguyên cho cả Train/val/Test "
                      "(--splits train val test) rồi train cùng ν, γ, 128 SV"}}
    rep = out / "hs_auc" / time.strftime("%Y%m%d_%H%M%S")
    rep.mkdir(parents=True, exist_ok=True)
    save_json(s, rep / "summary.json")
    report(rep / "report.md", s)
    for r in s["paths"]:
        log.info(f"{r['name']:<48} AUC = {r['smoothed'].get('auc', float('nan')):.4f} "
                 f"(thô {r['raw'].get('auc', float('nan')):.4f}) · {r['smoothed']['n_frames']} cặp")
    log.info(f"Cặp chung {common['n']}: cờ trùng {common['flag_agree_pct']:.2f}%, tương quan {common['score_corr']:.4f}")
    if val:
        log.info("Báo động giả val: " + " | ".join(f"{k} {x:.3%}" for k, x in val["far"].items()))
    log.info(f"ΔAUC = {delta:+.4f} → {s['decision']['verdict']}")
    log.info(f"Báo cáo → {rep / 'report.md'}")


if __name__ == "__main__":
    main()
