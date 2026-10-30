#!/usr/bin/env python3
"""GOLDEN MODEL SỐ NGUYÊN CỦA KHỐI TRÍCH ĐẶC TRƯNG FPGA (A2) – đo sai số (A3) – xuất véc-tơ RTL.

So hai đường xử lý trên cùng cặp khung:
  TRAIN  : ảnh → Gauss số thực → HS số thực → lưới 4×4 → nén log (đúng pipeline đã train)
  PHẦN CỨNG: ảnh → Gauss số nguyên ESP32 (preproc_q) → HS số nguyên FPGA (hs_q) → … → Q5.10
và báo sai số từng đặc trưng, theo đơn vị đặc trưng và theo σ của bộ chuẩn hóa; nếu có mô hình
đã train thì so luôn x_q (LSB), f_q và cờ bất thường của OC-SVM.

    python scripts/20_hs_golden.py --selftest                       # không cần dữ liệu
    python scripts/20_hs_golden.py --video data/IITB-Corridor/Test/<video>.avi --pairs 300
    python scripts/20_hs_golden.py --video ... --sweep              # quét độ rộng u, v và cách tính |v|
    python scripts/20_hs_golden.py --frames outputs/rec/<dir>       # chuỗi ảnh PNG liên tiếp (vd ghi từ OV5640)
    python scripts/20_hs_golden.py --video ... --export 8           # xuất véc-tơ RTL vào outputs/hw_vectors/
Mô hình: tự nạp outputs/models/detector.joblib + configs/default.yaml nếu có (--no-model để tắt).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT / "src" / "ocsvm_fpga" / "hwmodel"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PQ = _load("preproc_q", HW / "preproc_q.py")
HQ = _load("hs_q", HW / "hs_q.py")
HX = _load("hs_export", HW / "hs_export.py")
STAT = ["u", "v", "|v|"]


# --------------------------------------------------------------------------- nguồn khung

def sensor_frame_fallback(frame):
    """Cắt giữa về 4:3, đổi 160×120, xám (dùng khi không nạp được video_reader của repo)."""
    import cv2
    h, w = frame.shape[:2]
    tw = int(round(h * 4 / 3))
    if tw <= w:
        x0 = (w - tw) // 2
        frame = frame[:, x0:x0 + tw]
    else:
        th = int(round(w * 3 / 4))
        y0 = (h - th) // 2
        frame = frame[y0:y0 + th]
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, (HQ.W, HQ.H), interpolation=cv2.INTER_AREA)


def get_repo():
    """(cfg, to_sensor_frame, smooth_frame) của repo nếu nạp được."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from ocsvm_fpga.config import load_config
        from ocsvm_fpga.data.video_reader import smooth_frame, to_sensor_frame
        cfg = load_config(str(ROOT / "configs" / "default.yaml"), [])
        return cfg, to_sensor_frame, smooth_frame
    except Exception as e:  # noqa: BLE001
        print(f"  (không nạp được pipeline repo: {e!r} – dùng cài đặt dự phòng tương đương)")
        return None, None, None


def video_frames(path, start, n, step, repo):
    import cv2
    cfg, to_sensor, _ = repo
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Không mở được video {path}")
    out, i = [], 0
    while len(out) < n + 1:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= start and (i - start) % step == 0:
            out.append(to_sensor(fr, cfg.preprocess) if to_sensor else sensor_frame_fallback(fr))
        i += 1
    cap.release()
    return [np.ascontiguousarray(f, np.uint8) for f in out]


def dir_frames(path, n):
    import cv2
    files = sorted([p for p in Path(path).iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp", ".pgm")])
    out = []
    for f in files[:n + 1]:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im.shape != (HQ.H, HQ.W):
            im = sensor_frame_fallback(im)
        out.append(im)
    return out


def synthetic_sequence(rng, speeds=((0, 0), (1, 0), (2, 1), (3, 0), (5, 2), (7, 0)), per=4):
    """Cảnh tổng hợp: nền có kết cấu + 1–2 "người" di chuyển với nhiều tốc độ, có nhiễu cảm biến."""
    import cv2
    bg = cv2.GaussianBlur(rng.uniform(0, 255, (H2 := HQ.H + 60, W2 := HQ.W + 60)).astype(np.float32), (0, 0), 2.5)
    bg = cv2.normalize(bg, None, 40, 200, cv2.NORM_MINMAX)
    pairs, names = [], []
    for sx, sy in speeds:
        for k in range(per):
            y0, x0 = 30 + 7 * k, 20 + 25 * k
            f1, f2 = bg[30:30 + HQ.H, 30:30 + HQ.W].copy(), None
            f2 = f1.copy()
            person = bg[k * 5:k * 5 + 50, 100:122] * 0.4 + 90
            f1[y0:y0 + 50, x0:x0 + 22] = person
            f2[y0 + sy:y0 + sy + 50, x0 + sx:x0 + sx + 22] = person
            if sx >= 3:   # người thứ hai đi ngược
                f1[20:60, 120:136] = 200
                f2[20:60, 120 - sx // 2:136 - sx // 2] = 200
            n1 = np.clip(f1 + rng.normal(0, 2, f1.shape), 0, 255).astype(np.uint8)
            n2 = np.clip(f2 + rng.normal(0, 2, f2.shape), 0, 255).astype(np.uint8)
            pairs.append((n1, n2))
            names.append(f"tổng hợp {sx},{sy} px/khung #{k}")
    return pairs, names


def pairs_from_frames(frames, names_prefix="", start=0, step=1):
    """Ghép cặp liên tiếp, ngắt cặp ở khung tối (quy tắc train: TB ảnh đã làm mượt < 15)."""
    pairs, names, prev, last_i = [], [], None, 0
    for i, f in enumerate(frames):
        if PQ.is_dark(PQ.gauss5_q(f)):
            prev = None
            continue
        if prev is not None:
            pairs.append((prev, f))
            pidx = last_i
            names.append(f"{names_prefix}khung {start + pidx * step}→{start + i * step}")
        prev, last_i = f, i
    return pairs, names


# --------------------------------------------------------------------------- mô hình

class Model:
    def __init__(self, det, fx):
        self.det, self.fx = det, fx
        self.scaler_q = HQ.quantize_scaler(det.scaler.mean, det.scaler.scale)

    def x_q_from_float(self, feat):         # đường train: z số thực → Q5.10 như exporter cũ
        return self.fx.quantize_input(np.atleast_2d(feat))[0][0]

    def f_flag(self, x_q):
        fq = self.fx.decision_q(np.atleast_2d(x_q))
        return int(fq[0]), int(self.fx.anomaly_q(fq)[0])


def load_model(path):
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from ocsvm_fpga.config import load_config
        from ocsvm_fpga.models.base import AnomalyDetector
        from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM
        det = AnomalyDetector.load(path)
        cfg = load_config(str(ROOT / "configs" / "default.yaml"), [])
        if det.model.support_vectors.shape[1] != HQ.DIM:
            print(f"  ⚠ mô hình có {det.model.support_vectors.shape[1]} chiều, khối FPGA tạo {HQ.DIM} – bỏ qua mô hình")
            return None
        return Model(det, FixedPointOCSVM(det, cfg.fpga))
    except SystemExit:
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  (không nạp được mô hình {path}: {e!r})")
        return None


# --------------------------------------------------------------------------- đo sai số

def float_gauss(raw):
    import cv2
    return cv2.GaussianBlur(raw.astype(np.float32), (5, 5), 1.0)


def evaluate(pairs, p=HQ.DEFAULT, model: Model | None = None, train_smooth=None):
    """Trả về các mảng sai số (n, 48) và thống kê."""
    lut = HQ.log_lut(p)
    sc = model.scaler_q if model else None
    rows = {"hw": [], "train": [], "hsonly": [], "z_q": [], "x_q_train": [], "f": [], "sat": [], "maxN": [],
            "flow_max": []}
    for r1, r2 in pairs:
        g1, g2 = PQ.gauss5_q(r1), PQ.gauss5_q(r2)
        tr = HQ.features_q(g1, g2, p, sc, lut)
        s1 = train_smooth(r1) if train_smooth else float_gauss(r1)
        s2 = train_smooth(r2) if train_smooth else float_gauss(r2)
        ft, _ = HQ.features_float(s1, s2, p.log_c)
        fh, (uf, vf) = HQ.features_float(g1, g2, p.log_c)
        rows["flow_max"].append(float(max(np.abs(uf).max(), np.abs(vf).max())))
        rows["hw"].append(tr.log_q / (1 << p.data_frac))
        rows["train"].append(ft)
        rows["hsonly"].append(fh)
        rows["sat"].append(tr.hs.n_sat)
        rows["maxN"].append(tr.hs.max_abs_n)
        if model:
            rows["z_q"].append(tr.z_q)
            xt = model.x_q_from_float(ft)
            rows["x_q_train"].append(xt)
            rows["f"].append(model.f_flag(tr.z_q) + model.f_flag(xt))
    out = {k: np.asarray(v) for k, v in rows.items() if len(v)}
    out["err_total"] = np.abs(out["hw"] - out["train"])
    out["err_hs"] = np.abs(out["hw"] - out["hsonly"])
    out["err_gauss"] = np.abs(out["hsonly"] - out["train"])
    if model:
        out["err_sigma"] = out["err_total"] / model.det.scaler.scale
        out["lsb"] = np.abs(out["z_q"] - out["x_q_train"])
        f = out["f"]
        out["flag_agree"] = float(np.mean(f[:, 1] == f[:, 3]))
        out["f_rel"] = np.abs(f[:, 0] - f[:, 2]) / (np.abs(f[:, 2]).mean() + 1)
    return out


def summarize(ev, label=""):
    fm = ev["flow_max"]
    s = {"pairs": int(len(ev["hw"])), "sat_total": int(ev["sat"].sum()), "maxN": int(ev["maxN"].max()),
         "flow_max": float(fm.max()), "flow_p99": float(np.percentile(fm, 99)),
         "pairs_over_range": int(np.sum(fm >= 2 ** (HQ.DEFAULT.wu - HQ.DEFAULT.fu - 1))),
         "N_fits_25bit": bool(ev["maxN"].max() < 1 << 24)}
    for k in ("err_total", "err_hs", "err_gauss"):
        e = ev[k]
        s[k] = {"max": float(e.max()), "mean": float(e.mean()), "p99": float(np.percentile(e, 99)),
                "by_stat_max": [float(e[:, j::3].max()) for j in range(3)]}
    if "lsb" in ev:
        s["sigma"] = {"max": float(ev["err_sigma"].max()), "mean": float(ev["err_sigma"].mean())}
        s["x_q_lsb"] = {"max": int(ev["lsb"].max()), "mean": float(ev["lsb"].mean()),
                        "within_2lsb_pct": float(np.mean(ev["lsb"] <= 2) * 100)}
        s["flag_agree_pct"] = ev["flag_agree"] * 100
        s["f_rel_max"] = float(ev["f_rel"].max())
    s["label"] = label
    return s


def sweep(pairs, model):
    cfgs = [("Q5.7 13 bit (bản 0.3)", HQ.HSParams(wu=13, fu=7)), ("Q6.6 13 bit (đặc tả 0.4)", HQ.HSParams()),
            ("Q6.6 13 bit, |v| floor", HQ.HSParams(mag_mode="floor")), ("Q6.7 14 bit", HQ.HSParams(wu=14, fu=7)),
            ("Q6.10 17 bit", HQ.HSParams(wu=17, fu=10))]
    res = []
    for name, p in cfgs:
        ev = evaluate(pairs, p, model)
        s = summarize(ev, name)
        res.append(s)
        extra = f", x_q lệch max {s['x_q_lsb']['max']} LSB, cờ trùng {s['flag_agree_pct']:.1f}%" if "x_q_lsb" in s else ""
        print(f"  {name:24s} sai số HS max {s['err_hs']['max']:.4f} TB {s['err_hs']['mean']:.5f} · "
              f"tổng max {s['err_total']['max']:.4f} · bão hòa {s['sat_total']}{extra}")
    return res


# --------------------------------------------------------------------------- báo cáo

def report(out, s, sw, src, exported, model):
    f = lambda x, n=4: f"{x:.{n}f}"  # noqa: E731
    L = ["# Golden Model số nguyên khối trích đặc trưng FPGA – sai số so với pipeline train", "",
         f"Thời gian {time.strftime('%Y-%m-%d %H:%M:%S')} · nguồn: {src} · {s['pairs']} cặp khung · "
         f"tham số: u, v Q{HQ.DEFAULT.wu - HQ.DEFAULT.fu - 1}.{HQ.DEFAULT.fu} {HQ.DEFAULT.wu} bit, a/b 15 bit 18 bit lẻ, |v| {HQ.DEFAULT.mag_mode}, bảng log 2 vùng", "",
         "## Sai số đặc trưng (sau nén log, trước chuẩn hóa)", "",
         "| Nguồn sai số | max | p99 | TB | max theo u / v / \\|v\\| |", "|---|---|---|---|---|"]
    for k, name in (("err_hs", "Số học FPGA (HS + lưới + log), cùng ảnh Gauss 8 bit"),
                    ("err_gauss", "Gauss số nguyên ESP32 so với Gauss số thực"),
                    ("err_total", "**Tổng: phần cứng so với pipeline train**")):
        e = s[k]
        L.append(f"| {name} | {f(e['max'])} | {f(e['p99'])} | {f(e['mean'], 5)} | "
                 + " / ".join(f(x) for x in e["by_stat_max"]) + " |")
    L += ["", f"Bão hòa u/v: {s['sat_total']} lần · |N| lớn nhất {s['maxN']} (vừa 25 bit: {s['N_fits_25bit']}) · "
          f"max(|u|,|v|) số thực mỗi cặp: p99 {s['flow_p99']:.1f} px, lớn nhất {s['flow_max']:.1f} px, "
          f"{s['pairs_over_range']} cặp vượt dải ±{2 ** (HQ.DEFAULT.wu - HQ.DEFAULT.fu - 1)} px", ""]
    if "x_q_lsb" in s:
        L += ["## So với mô hình OC-SVM đã train", "", "| Chỉ số | Giá trị |", "|---|---|",
              f"| Sai số theo σ của bộ chuẩn hóa | max {f(s['sigma']['max'])}, TB {f(s['sigma']['mean'], 5)} |",
              f"| x_q (Q5.10) lệch so với đường train | max {s['x_q_lsb']['max']} LSB, TB {s['x_q_lsb']['mean']:.2f}, "
              f"{s['x_q_lsb']['within_2lsb_pct']:.1f}% trong ±2 LSB |",
              f"| Cờ bất thường trùng | **{s['flag_agree_pct']:.2f}%** |",
              f"| f_q lệch tương đối lớn nhất | {s['f_rel_max']:.4f} |",
              f"| 1/σ: số bit lẻ (INV_FRAC) | {model.scaler_q.inv_frac} |", ""]
    if sw:
        L += ["## Quét độ rộng u, v và cách tính |v|", "",
              "| Cấu hình | Số học max | Số học TB | Tổng max | Bão hòa | x_q lệch max (LSB) | Cờ trùng |",
              "|---|---|---|---|---|---|---|"]
        for r in sw:
            L.append(f"| {r['label']} | {f(r['err_hs']['max'])} | {f(r['err_hs']['mean'], 5)} | "
                     f"{f(r['err_total']['max'])} | {r['sat_total']} | {r.get('x_q_lsb', {}).get('max', '–')} | "
                     f"{f(r['flag_agree_pct'], 2) + '%' if 'flag_agree_pct' in r else '–'} |")
        L.append("")
    if exported:
        L += [f"Véc-tơ RTL: {len(exported)} ca → `outputs/hw_vectors/` (xem index.md).", ""]
    if (out / "errors.png").exists():
        L += ["![sai số](errors.png)", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def plot(out, ev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for j, k in enumerate(("err_hs", "err_gauss", "err_total")):
        ax[j].hist(ev[k].ravel(), bins=60, log=True)
        ax[j].set(title={"err_hs": "Số học FPGA", "err_gauss": "Gauss số nguyên", "err_total": "Tổng so với train"}[k],
                  xlabel="|sai số| đặc trưng")
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- selftest

def selftest():
    import cv2
    fails = 0

    def check(ok, msg):
        nonlocal fails
        fails += not ok
        print(f"  {'ĐẠT ' if ok else 'LỖI '} {msg}")

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (HQ.H, HQ.W)).astype(np.uint8)
    b = rng.integers(0, 256, (HQ.H, HQ.W)).astype(np.uint8)
    # 1. đạo hàm chính xác tuyệt đối
    ix, iy, it = HQ.derivatives4(a, b)
    A, B = a.astype(np.float32), b.astype(np.float32)
    s = HQ._shift
    fx = 0.25 * ((s(A, 0, 1) - A) + (s(A, 1, 1) - s(A, 1, 0)) + (s(B, 0, 1) - B) + (s(B, 1, 1) - s(B, 1, 0)))
    check(np.array_equal(ix / 4, fx) and max(np.abs(ix).max(), np.abs(iy).max(), np.abs(it).max()) <= 1020,
          "tầng 1: 4·Ix khớp tuyệt đối công thức số thực, |·| ≤ 1020 (11 bit)")
    # 2. hệ số a, b
    d16, qa, qb = HQ.coeffs(ix, iy)
    ref = ix / d16 * (1 << 18)
    err = np.abs(qa - ref)
    check(err.max() <= 1.0 and d16.max() < 1 << 21, f"tầng 2: a sai ≤ 1 LSB (max {err.max():.3f}; chỉ bão hòa khi a = 1/16), D16 < 2²¹")
    # 3. trung bình láng giềng
    u = rng.integers(-4096, 4096, (HQ.H, HQ.W))
    k = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], np.float64) / 12
    uf = cv2.filter2D(u.astype(np.float64), -1, k, borderType=cv2.BORDER_REPLICATE)
    e = np.abs(HQ.avg12(u) - uf)
    check(e.max() <= 0.51 + 4096 * 3e-5, f"tầng 3: ū sai ≤ 0,5 LSB + sai số hằng 1/12 (max {e.max():.3f})")
    # 4. dịch chuyển đều 1 px → u ≈ 1
    import cv2 as _c
    base = _c.GaussianBlur(rng.uniform(0, 255, (140, 180)).astype(np.float32), (0, 0), 3)
    i1 = np.clip(base[10:130, 10:170], 0, 255).astype(np.uint8)
    i2 = np.clip(base[10:130, 9:169], 0, 255).astype(np.uint8)
    tr = HQ.horn_schunck_q(i1, i2)
    um = tr.u[10:-10, 10:-10].mean() / (1 << HQ.DEFAULT.fu)
    uff, _ = HQ.horn_schunck_float(i1, i2)
    check(abs(um - uff[10:-10, 10:-10].mean()) < 0.01 and abs(um - 1) < 0.15,
          f"tầng 3: dịch 1 px → u TB {um:.3f} (số thực {uff[10:-10, 10:-10].mean():.3f})")
    # 5. |v|
    uu = rng.integers(-4096, 4096, 100000)
    vv = rng.integers(-4096, 4096, 100000)
    m = HQ.magnitude(uu, vv)
    check(np.max(np.abs(m - np.hypot(uu, vv))) <= 0.5 + 1e-9 and m.max() < 1 << 13,
          "tầng 4: |v| làm tròn đúng (sai ≤ 0,5 LSB), vừa 13 bit")
    # 6. bảng log
    x = np.arange(-32768, 32768)
    eg = np.abs(HQ.log_compress_q(x, HQ.log_lut()) - HQ.compress_float(x / 1024) * 1024).max() / 1024
    check(eg < 0.0065, f"tầng 5: bảng log sai max {eg:.5f} trên toàn dải Q5.10")
    # 7. chuẩn hóa
    mean = rng.normal(0, 0.5, 48)
    scale = rng.uniform(0.02, 1.5, 48)
    sq = HQ.quantize_scaler(mean, scale)
    g = rng.integers(-2100, 2100, (500, 48))
    zf = (g / 1024 - mean) / scale * 1024
    zq = HQ.standardize_q(g, sq)
    ok = np.abs(zq - np.clip(zf, -32768, 32767)).max()
    check(ok <= 2 + 1024 * np.abs(g / 1024 - mean).max() / scale.min() * (2 ** -sq.inv_frac) * 1.01 + 1,
          f"tầng 6: z lệch max {ok:.2f} LSB (INV_FRAC = {sq.inv_frac})")
    # 8. đặc trưng tổng hợp
    pairs, names = synthetic_sequence(rng)
    ev = evaluate(pairs)
    s = summarize(ev)
    check(s["err_hs"]["max"] < 0.01 and s["sat_total"] == 0 and s["N_fits_25bit"],
          f"toàn khối: sai số số học max {s['err_hs']['max']:.4f} (< 0,01), không bão hòa, N vừa 25 bit "
          f"({len(pairs)} cặp, tới 7 px/khung)")
    # 9. xuất véc-tơ và đọc lại
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sq2 = HQ.quantize_scaler(mean, scale)
        sm = [(PQ.gauss5_q(x), PQ.gauss5_q(y)) for x, y in (pairs[0], pairs[-1])]
        rows = HX.export_all(sm, tmp, scaler=sq2, names=[names[0], names[-1]])
        d = tmp / "case_01"
        tr = HQ.features_q(*sm[1], scaler=sq2, keep_iter1=True)
        cs = HX.read_mem(d / "cell_sums.mem", HX.SUM_W).reshape(16, 3)
        uv = HX.read_mem(d / "uv_final.mem", 26, signed=False)
        u_back = np.where(uv & 0x1FFF >= 4096, (uv & 0x1FFF) - 8192, uv & 0x1FFF)
        der = HX.read_mem(d / "deriv.mem", 64, signed=False)
        a_back = (der >> 33) & 0x7FFF
        a_back = np.where(a_back >= 1 << 14, a_back - (1 << 15), a_back)
        ok = (np.array_equal(cs, tr.sums) and np.array_equal(HQ.cell_means_q(cs), HX.read_mem(d / "feat_mean.mem", 16))
              and np.array_equal(u_back, tr.hs.u.ravel()) and np.array_equal(a_back, tr.hs.a.ravel())
              and np.array_equal(HX.read_mem(d / "feat_z.mem", 16), tr.z_q)
              and (tmp / "hs_params.vh").exists() and len(HX.read_mem(tmp / "mem" / "log_lut.mem", 12)) == 1504
              and all(r["n_sat"] == 0 for r in rows))
        check(ok, "xuất .mem (ảnh đã Gauss, gồm ca 7 px/khung): đọc lại tổng ô, u, a, đặc trưng, z khớp Golden; không bão hòa")
    print("\nSELFTEST:", "TẤT CẢ ĐẠT" if not fails else f"{fails} lỗi")
    return int(fails > 0)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--selftest", action="store_true")
    src.add_argument("--video", help="video (vd IITB-Corridor) → khung 160×120 như lúc train")
    src.add_argument("--frames", help="thư mục ảnh liên tiếp (png/jpg)")
    src.add_argument("--synthetic", action="store_true", help="cảnh tổng hợp")
    ap.add_argument("--pairs", type=int, default=200, help="số cặp khung tối đa")
    ap.add_argument("--start", type=int, default=0, help="khung bắt đầu trong video")
    ap.add_argument("--step", type=int, default=2, help="frame_step (IITB 25 fps → 2 = 80 ms)")
    ap.add_argument("--sweep", action="store_true", help="quét độ rộng u, v và |v| floor/round")
    ap.add_argument("--export", type=int, default=0, help="xuất N ca véc-tơ RTL (chọn các cặp chuyển động mạnh)")
    ap.add_argument("--model", default=str(ROOT / "outputs" / "models" / "detector.joblib"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    repo = get_repo() if (a.video or a.frames) else (None, None, None)
    if a.video:
        frames = video_frames(a.video, a.start, a.pairs, a.step, repo)
        pairs, names = pairs_from_frames(frames, Path(a.video).stem + " ", a.start, a.step)
        src_name = f"{Path(a.video).name} (từ khung {a.start}, bước {a.step})"
    elif a.frames:
        pairs, names = pairs_from_frames(dir_frames(a.frames, a.pairs), "")
        src_name = a.frames
    else:
        pairs, names = synthetic_sequence(np.random.default_rng(0))
        src_name = "tổng hợp"
    pairs, names = pairs[:a.pairs], names[:a.pairs]
    if not pairs:
        sys.exit("Không có cặp khung nào")
    model = None if a.no_model or not Path(a.model).exists() else load_model(a.model)
    print(f"{len(pairs)} cặp khung · mô hình: {'có' if model else 'không'}")
    _, _, smooth = repo
    train_smooth = None
    if smooth is not None:
        train_smooth = lambda r: np.asarray(smooth(r, repo[0].preprocess), np.float32)  # noqa: E731

    out = Path(a.out) if a.out else ROOT / "outputs" / "hs_golden" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    ev = evaluate(pairs, HQ.DEFAULT, model, train_smooth)
    s = summarize(ev, "đặc tả 0.4")
    print(f"Sai số số học FPGA max {s['err_hs']['max']:.4f} · Gauss {s['err_gauss']['max']:.4f} · "
          f"tổng max {s['err_total']['max']:.4f} (TB {s['err_total']['mean']:.5f}) · bão hòa {s['sat_total']} · "
          f"vận tốc lớn nhất {s['flow_max']:.1f} px ({s['pairs_over_range']} cặp vượt dải)")
    if "x_q_lsb" in s:
        print(f"x_q lệch max {s['x_q_lsb']['max']} LSB · cờ bất thường trùng {s['flag_agree_pct']:.2f}%")
    sw = sweep(pairs[:min(len(pairs), 80)], model) if a.sweep else None
    exported = None
    if a.export:
        motion = [float(np.abs(f[2::3]).sum()) for f in ev["train"]]
        idx = sorted(set([int(i) for i in np.argsort(motion)[::-1][:max(1, a.export - 1)]] + [int(np.argmin(motion))]))
        vec_dir = ROOT / "outputs" / "hw_vectors"
        smoothed = [(PQ.gauss5_q(pairs[i][0]), PQ.gauss5_q(pairs[i][1])) for i in idx]   # FPGA nhận ảnh đã Gauss
        exported = HX.export_all(smoothed, vec_dir, scaler=model.scaler_q if model else None,
                                 ocsvm=model.f_flag if model else None, names=[names[i] for i in idx])
        print(f"Đã xuất {len(exported)} ca véc-tơ RTL → {vec_dir}")
    plot(out, ev)
    report(out, s, sw, src_name, exported, model)
    (out / "summary.json").write_text(json.dumps({"spec": s, "sweep": sw}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã lưu báo cáo → {out / 'report.md'}")


if __name__ == "__main__":
    main()
