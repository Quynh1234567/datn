#!/usr/bin/env python3
"""QUÉT SỐ VÒNG HORN–SCHUNCK – căn cứ để chọn N_ITER (mặc định 64) cho báo cáo đồ án.

Ba bước, dùng đúng Golden Model số nguyên của phần cứng (hwmodel/hs_q.py, đặc tả 0.4):

(1) HỘI TỤ – nhanh (vài phút), không cần train lại:
    python scripts/22_hs_iter_sweep.py --config configs/hw.yaml
    Lấy ~300 cặp khung IITB (nửa train, nửa test), chạy HS một lần tới --nmax vòng (mặc định 256,
    coi như đã hội tụ) và chụp u, v ở các mốc 8, 16, 24, 32, 48, 64, 96, 128, 192.
    Mỗi mốc đo: sai số dòng quang học (px) so với N = nmax, sai số 48 đặc trưng z (đơn vị σ của bộ
    chuẩn hóa), số cờ bất thường đổi so với N = 64 khi giữ mô hình hiện tại, thời gian ước lượng trên FPGA.

(2) AUC – chậm (mỗi giá trị N là một lần trích đặc trưng + train + đánh giá), in sẵn lệnh:
    python scripts/22_hs_iter_sweep.py --config configs/hw.yaml --plan 32 128
    Mỗi N ghi vào outputs_hw_n<N>/; N = 64 chính là outputs_hw/ đã có (không chạy lại).

(3) TỔNG HỢP – đọc báo cáo của các lần chạy ở bước 2:
    python scripts/22_hs_iter_sweep.py --config configs/hw.yaml --collect

Kết quả: outputs_hw/hs_iter_sweep/<thời gian>/report.md, sweep.json, convergence.png
và outputs_hw/hs_iter_sweep/auc_table.md (bước 3).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, load_detector_checked, setup  # noqa: E402
from ocsvm_fpga.features.pipeline import load_sensor_frames  # noqa: E402
from ocsvm_fpga.hwmodel import hs_q as HQ, preproc_q as PQ  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402

CHECKPOINTS = (8, 16, 24, 32, 48, 64, 96, 128, 192)
# Thời gian trên FPGA (tài liệu kiến trúc RTL, mục 3): mỗi vòng 19 200 điểm × 2 chu kỳ ở 100 MHz
# + xả pipeline, refresh, đổi hàng SDRAM ≈ 0,43 ms; cộng DERIV + FEAT + OC-SVM ≈ 0,5 ms.
MS_PER_ITER, MS_FIXED, BUDGET_MS = 0.43, 0.5, 80.0


def fpga_ms(n: int) -> float:
    return MS_FIXED + MS_PER_ITER * n


# ------------------------------------------------------------------------------- (1) hội tụ

def sample_pairs(splits, cfg, out, n_videos: int, n_pairs: int, rng, log):
    """~n_pairs cặp khung (ảnh đã Gauss số nguyên) từ n_videos video, nửa train (bình thường)
    nửa test (có hành vi bất thường → chuyển động lớn, nơi số vòng ảnh hưởng nhiều nhất)."""
    dark = float(cfg.preprocess.get("skip_dark_below", 0) or 0)
    pools = [("train", splits.get("train") or []), ("test", splits.get("test") or [])]
    pools = [(k, v) for k, v in pools if v] or [("val", splits.get("val") or [])]
    per_pool = max(1, n_videos // len(pools))
    pairs = []
    for name, items in pools:
        pick = rng.choice(len(items), size=min(per_pool, len(items)), replace=False)
        k_per = max(1, n_pairs // (per_pool * len(pools)))
        for i in sorted(int(x) for x in pick):
            item = items[i]
            frames, fidx = load_sensor_frames(item, cfg, out)
            cand, prev = [], None
            for idx, raw in zip(fidx, frames):
                g = PQ.gauss5_q(np.ascontiguousarray(raw, np.uint8))
                if dark > 0 and PQ.frame_sum(g) < dark * g.size:
                    prev = None
                    continue
                if prev is not None:
                    cand.append((prev, g))
                prev = g
            if not cand:
                continue
            for j in np.linspace(0, len(cand) - 1, min(k_per, len(cand))).astype(int):
                pairs.append((name, item.path.stem, cand[int(j)][0], cand[int(j)][1]))
        log.info(f"  {name}: {sum(1 for p in pairs if p[0] == name)} cặp từ {len(pick)} video")
    return pairs


def feats_from_uv(u, v, sc, lut, p):
    m = HQ.magnitude(u, v, p)
    lq = HQ.log_compress_q(HQ.cell_means_q(HQ.cell_sums(u, v, m), p), lut, p)
    return HQ.standardize_q(lq, sc, p)


def converge(pairs, det, cfg, nmax: int, checkpoints, log):
    p = HQ.DEFAULT if nmax == HQ.DEFAULT.n_iter else HQ.HSParams(n_iter=nmax)
    cps = sorted({c for c in checkpoints if 0 < c < nmax}
                 | ({HQ.DEFAULT.n_iter} if HQ.DEFAULT.n_iter < nmax else set()))
    sc = HQ.quantize_scaler(det.scaler.mean, det.scaler.scale, p)
    lut = HQ.log_lut(p)
    fx = FixedPointOCSVM(det, cfg.fpga)
    one = float(1 << p.fu)
    rows = {n: {"flow_px": [], "flow_px_moving": [], "num": 0.0, "den": 0.0, "z_mean": [], "z_max": [],
                "flag": [], "sat": 0} for n in cps}
    flag64, flag_ref, z_ref_all = [], [], []
    t0 = time.time()
    for k, (_, _, g1, g2) in enumerate(pairs):
        tr = HQ.horn_schunck_q(g1, g2, p, snapshots=set(cps))
        ur, vr = tr.u / one, tr.v / one
        mag_ref = np.hypot(ur, vr)
        moving = mag_ref >= 0.5
        z_ref = feats_from_uv(tr.u, tr.v, sc, lut, p)
        z_ref_all.append(z_ref)
        f_ref = fx.decision_q(np.atleast_2d(z_ref))
        flag_ref.append(int(fx.anomaly_q(f_ref)[0]))
        for n in cps:
            u, v = tr.snaps[n]
            du = np.hypot(u / one - ur, v / one - vr)
            r = rows[n]
            r["flow_px"].append(float(du.mean()))
            r["flow_px_moving"].append(float(du[moving].mean()) if moving.any() else 0.0)
            r["num"] += float((du ** 2).sum())
            r["den"] += float((mag_ref ** 2).sum())
            z = feats_from_uv(u, v, sc, lut, p)
            dz = np.abs(z - z_ref) / float(1 << p.data_frac)
            r["z_mean"].append(float(dz.mean()))
            r["z_max"].append(float(dz.max()))
            fl = int(fx.anomaly_q(fx.decision_q(np.atleast_2d(z)))[0])
            r["flag"].append(fl)
            if n == HQ.DEFAULT.n_iter:
                flag64.append(fl)
        if (k + 1) % 50 == 0 or k + 1 == len(pairs):
            log.info(f"  {k + 1}/{len(pairs)} cặp · {time.time() - t0:.0f} s")
    flag64 = np.asarray(flag64) if flag64 else np.asarray(flag_ref)
    out = []
    for n in cps + [nmax]:
        if n == nmax:
            out.append({"n": n, "flow_px": 0.0, "flow_px_moving": 0.0, "flow_rel": 0.0, "z_mean_sigma": 0.0,
                        "z_p95_sigma": 0.0, "z_max_sigma": 0.0,
                        "flag_change_vs_64": float(np.mean(np.asarray(flag_ref) != flag64)),
                        "fpga_ms": fpga_ms(n), "fits_budget": fpga_ms(n) < BUDGET_MS})
            continue
        r = rows[n]
        fl = np.asarray(r["flag"])
        out.append({"n": n, "flow_px": float(np.mean(r["flow_px"])),
                    "flow_px_moving": float(np.mean(r["flow_px_moving"])),
                    "flow_rel": float(np.sqrt(r["num"] / max(r["den"], 1e-12))),
                    "z_mean_sigma": float(np.mean(r["z_mean"])),
                    "z_p95_sigma": float(np.percentile(r["z_max"], 95)),
                    "z_max_sigma": float(np.max(r["z_max"])),
                    "flag_change_vs_64": float(np.mean(fl != flag64)),
                    "fpga_ms": fpga_ms(n), "fits_budget": fpga_ms(n) < BUDGET_MS})
    return out, {"pairs": len(pairs), "nmax": nmax, "anomaly_rate_ref": float(np.mean(flag_ref)),
                 "seconds": time.time() - t0}


def write_converge(res, meta, run_dir: Path):
    save_json({"meta": meta, "rows": res}, run_dir / "sweep.json")
    nmax = meta["nmax"]
    L = ["# Quét số vòng Horn–Schunck – độ hội tụ", "",
         f"{meta['pairs']} cặp khung (nửa train, nửa test), Golden Model số nguyên (đặc tả "
         f"{HQ.SPEC_VERSION}). Tham chiếu \"đã hội tụ\" = **{nmax} vòng**. "
         f"Thời gian FPGA ước lượng = {MS_FIXED} + {MS_PER_ITER}·N ms (100 MHz, tài liệu kiến trúc RTL mục 3).", "",
         "| N | Sai số dòng TB (px) | … ở điểm chuyển động ≥ 0,5 px | Sai số tương đối | "
         "Sai số z TB (σ) | z p95 (σ) | Cờ đổi so với N=64 * | FPGA (ms) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res:
        mark = " **(hiện tại)**" if r["n"] == HQ.DEFAULT.n_iter else (" (tham chiếu)" if r["n"] == nmax else "")
        L.append(f"| {r['n']}{mark} | {r['flow_px']:.3f} | {r['flow_px_moving']:.3f} | {r['flow_rel']:.1%} | "
                 f"{r['z_mean_sigma']:.3f} | {r['z_p95_sigma']:.3f} | {r['flag_change_vs_64']:.2%} | "
                 f"{r['fpga_ms']:.1f}{'' if r['fits_budget'] else ' ⚠ > 80'} |")
    L += ["", "\\* dùng mô hình hiện tại (train với 64 vòng) cho mọi N – chỉ cho biết đổi N mà KHÔNG train "
          "lại thì bao nhiêu cờ thay đổi. Muốn so công bằng phải train lại cho từng N (bước 2, --plan).", "",
          "## Cách đọc", "",
          "- Sai số z tính theo σ của bộ chuẩn hóa – cùng thang với đầu vào OC-SVM; 0,1 σ là nhỏ so với "
          "độ trải của dữ liệu bình thường.",
          "- Số vòng hợp lý là điểm mà đường sai số bắt đầu nằm ngang: thêm vòng tốn thời gian tuyến tính "
          "nhưng đặc trưng gần như không đổi.",
          "- Kết luận cuối cùng dựa vào AUC sau khi train lại (bước 2–3); bảng này chỉ chọn các N đáng thử.", ""]
    good = [r for r in res if r["n"] < nmax and r["z_mean_sigma"] <= 0.05]
    if good:
        n0 = min(good, key=lambda r: r["n"])
        nxt = ("Nên train lại thêm N = 32 và 48 để xem AUC có giữ được với ít vòng hơn không."
               if n0["n"] == HQ.DEFAULT.n_iter else
               f"Nên train lại với N = {n0['n']} và so AUC với N = 64 (bước 2–3).")
        L += [f"Gợi ý: N nhỏ nhất có sai số z TB ≤ 0,05 σ so với {nmax} vòng là **N = {n0['n']}** "
              f"(FPGA ≈ {n0['fpga_ms']:.1f} ms). " + nxt, ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [r["n"] for r in res]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ns, [r["flow_px_moving"] for r in res], "o-", label="điểm chuyển động ≥ 0,5 px")
        ax[0].plot(ns, [r["flow_px"] for r in res], "s--", label="toàn ảnh")
        ax[0].set(xscale="log", xlabel="số vòng N", ylabel=f"|(u,v)_N − (u,v)_{nmax}| TB (px)",
                  title="Sai số dòng quang học")
        ax[1].plot(ns, [r["z_mean_sigma"] for r in res], "o-", label="TB")
        ax[1].plot(ns, [r["z_p95_sigma"] for r in res], "s--", label="p95 của max/cặp")
        ax[1].set(xscale="log", xlabel="số vòng N", ylabel="|z_N − z_ref| (σ)", title="Sai số 48 đặc trưng z")
        for a in ax:
            a.axvline(HQ.DEFAULT.n_iter, color="gray", ls=":", lw=1)
            a.set_xticks(ns)
            a.set_xticklabels([str(n) for n in ns])
            a.grid(alpha=0.3)
            a.legend()
        ax2 = ax[1].twinx()
        ax2.plot(ns, [r["fpga_ms"] for r in res], color="tab:red", alpha=0.6, lw=1)
        ax2.set_ylabel("thời gian FPGA (ms)", color="tab:red")
        fig.tight_layout()
        fig.savefig(run_dir / "convergence.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"(bỏ qua biểu đồ: {e})")


# ------------------------------------------------------------------------------- (2) kế hoạch

def plan(ns, args, log):
    cfg_arg = f"--config {args.config}" if args.config else "--config configs/hw.yaml"
    jobs = f" --jobs {args.jobs}" if args.jobs else ""
    lines = ["# Mỗi N: trích đặc trưng (khung đã cache, không giải mã lại video) → train → đánh giá",
             "# N = 64 = outputs_hw/ đã có, không cần chạy lại."]
    for n in ns:
        if n == HQ.DEFAULT.n_iter:
            continue
        s = f"--set project.output_dir=outputs_hw_n{n} optical_flow.num_iter={n}"
        lines += [f"python scripts/01_extract_features.py {cfg_arg} {s}{jobs}",
                  f"python scripts/02_train.py {cfg_arg} {s}",
                  f"python scripts/03_evaluate.py {cfg_arg} {s}", ""]
    lines.append(f"python scripts/22_hs_iter_sweep.py {cfg_arg} --collect")
    txt = "\n".join(lines)
    print(txt)
    log.info("Thời gian trích đặc trưng tỉ lệ thuận với N (64 vòng ≈ 85 phút với --jobs 8 trên máy chị).")
    return txt


# ------------------------------------------------------------------------------- (3) tổng hợp

def collect(out: Path, log):
    runs = []
    for d in sorted(ROOT.glob("outputs_hw*")):
        m = re.fullmatch(r"outputs_hw(?:_n(\d+))?", d.name)
        ev = d / "reports" / "eval_report.json"
        if not m or not ev.exists():
            continue
        n = int(m.group(1)) if m.group(1) else HQ.DEFAULT.n_iter
        e = json.loads(ev.read_text(encoding="utf-8"))
        tr_p = d / "reports" / "train_report.json"
        t = json.loads(tr_p.read_text(encoding="utf-8")) if tr_p.exists() else {}
        sm, raw = e.get("test_smoothed", {}), e.get("test_raw", {})
        evh = (e.get("events") or {}).get("hysteresis", {}).get("events", {})
        runs.append({"n": n, "dir": d.name, "auc": sm.get("auc"), "auc_raw": raw.get("auc"), "eer": sm.get("eer"),
                     "event_recall": evh.get("event_recall"), "fa_per_h": evh.get("false_alarms_per_hour"),
                     "n_sv": t.get("n_support_vectors"), "fpga_ms": fpga_ms(n)})
    if not runs:
        sys.exit("Chưa có outputs_hw*/reports/eval_report.json nào – chạy bước 2 (--plan) trước")
    runs.sort(key=lambda r: r["n"])
    best = max(r["auc"] or 0 for r in runs)

    def f(x, fmt):
        return "–" if x is None else format(x, fmt)
    L = ["# Quét số vòng Horn–Schunck – AUC sau khi train lại", "",
         "| N | AUC (làm mượt) | AUC thô | EER | Phát hiện sự kiện | Báo giả / giờ | SV | FPGA (ms) | thư mục |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['n']}{' **(hiện tại)**' if r['n'] == HQ.DEFAULT.n_iter else ''} | {f(r['auc'], '.4f')} | "
                 f"{f(r['auc_raw'], '.4f')} | {f(r['eer'], '.4f')} | {f(r['event_recall'], '.1%')} | "
                 f"{f(r['fa_per_h'], '.1f')} | {f(r['n_sv'], 'd')} | {r['fpga_ms']:.1f} | `{r['dir']}` |")
    ok = [r for r in runs if r["auc"] is not None and r["auc"] >= best - 0.005]
    if ok:
        n0 = min(ok, key=lambda r: r["n"])
        L += ["", f"Quy tắc chọn: N nhỏ nhất có AUC ≥ AUC tốt nhất − 0,005 → **N = {n0['n']}** "
              f"(AUC {n0['auc']:.4f}, FPGA ≈ {n0['fpga_ms']:.1f} ms).",
              "Nếu N khác 64: dùng thư mục tương ứng cho 04_export_fpga.py (hs_params.vh tự ghi N_ITER mới)."]
    path = out / "hs_iter_sweep" / "auc_table.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    save_json(runs, path.with_suffix(".json"))
    log.info(f"→ {path}")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--pairs", type=int, default=300, help="số cặp khung cho bước hội tụ")
    ap.add_argument("--videos", type=int, default=30, help="số video lấy mẫu (nửa train, nửa test)")
    ap.add_argument("--nmax", type=int, default=256, help="số vòng coi như đã hội tụ (tham chiếu)")
    ap.add_argument("--checkpoints", type=int, nargs="*", default=list(CHECKPOINTS))
    ap.add_argument("--plan", type=int, nargs="*", default=None, metavar="N", help="in lệnh train lại cho các N")
    ap.add_argument("--collect", action="store_true", help="tổng hợp AUC các outputs_hw*")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "22_hs_iter_sweep")
    if cfg.features.get("descriptor") != "grid_flow_hw":
        sys.exit("Cần --config configs/hw.yaml (descriptor = grid_flow_hw)")
    if args.plan is not None:
        plan(args.plan or [32, 128], args, log)
        return
    if args.collect:
        collect(out, log)
        return
    if int(cfg.optical_flow.num_iter) != HQ.DEFAULT.n_iter:
        log.warning(f"optical_flow.num_iter = {cfg.optical_flow.num_iter}: bước hội tụ luôn so với mô hình "
                    f"trong {out} – nên chạy với cấu hình 64 vòng (outputs_hw)")
    det = load_detector_checked(cfg, out)
    splits = get_splits(cfg, out, rng)
    log.info(f"Lấy mẫu {args.pairs} cặp từ {args.videos} video …")
    pairs = sample_pairs(splits, cfg, out, args.videos, args.pairs, rng, log)
    if not pairs:
        sys.exit("Không lấy được cặp khung nào – kiểm tra cache khung (preprocess.frames_cache_dir)")
    log.info(f"Chạy HS tới {args.nmax} vòng, chụp ở {sorted(set(args.checkpoints))} …")
    res, meta = converge(pairs, det, cfg, args.nmax, args.checkpoints, log)
    run_dir = out / "hs_iter_sweep" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_converge(res, meta, run_dir)
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    log.info(f"→ {run_dir}  (report.md, sweep.json, convergence.png)")


if __name__ == "__main__":
    main()
