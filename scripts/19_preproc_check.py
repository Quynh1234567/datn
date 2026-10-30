#!/usr/bin/env python3
"""KIỂM TRA TIỀN XỬ LÝ TRÊN ESP32-S3 (đặc tả A1, mục 2.2–2.3).

Đọc log của firmware/esp32_preproc qua cổng COM (hoặc file log), rồi:
  • tự kiểm tra trên chip: 20 vector (10 ảnh mẫu × 2 kiểu biên) đúng CRC của Golden Model
  • BIT-EXACT TRÊN ẢNH THẬT: tính lại Gauss bằng Golden Model trên ảnh gốc ESP32 gửi về,
    so CRC ảnh làm mượt, tổng, cờ khung tối với kết quả ESP32
  • sai khác so với cv2.GaussianBlur số thực (pipeline lúc train) – lượng hóa mục 2.4
  • thời gian Gauss và cả vòng xử lý so với ngân sách 80 ms, khoảng cách khung, rớt khung
  • khung tối ngắt cặp đúng (chạy lại bộ chọn cặp bằng Python và so với ESP32)
  • lưu log, CSV, ảnh, biểu đồ, report.md vào outputs/preproc/<thời gian>/

    pip install pyserial numpy matplotlib opencv-python
    python scripts/19_preproc_check.py --port COM5          # nhấn RST trên board; che ống kính ~3 s lúc đo
    python scripts/19_preproc_check.py --log outputs/preproc/xxx/log.txt
    python scripts/19_preproc_check.py --selftest           # kiểm tra Golden ↔ C (gcc) ↔ script, không cần board
    python scripts/19_preproc_check.py --gen-vectors        # sinh lại main/selftest_vectors.h
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "esp32_preproc"
GOLDEN_PATH = ROOT / "src" / "ocsvm_fpga" / "hwmodel" / "preproc_q.py"
W, H = 160, 120
SEL_SKIP, SEL_FIRST, SEL_OK, SEL_BAD, SEL_NONE, SEL_DARK = range(6)
SEL_NAME = {0: "skip", 1: "first", 2: "pair_ok", 3: "pair_bad", 4: "none", 5: "dark"}
COLS = ["idx", "ts_us", "t_gauss_us", "t_loop_us", "sum", "dark", "sel", "crc_raw", "crc_smooth", "mean_raw"]
BORDER_NAMES = {0: "reflect101", 1: "replicate"}


def load_golden():
    spec = importlib.util.spec_from_file_location("preproc_q", GOLDEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preproc_q"] = mod
    spec.loader.exec_module(mod)
    return mod


G = load_golden()


# ------------------------------------------------------------------------------ đọc log

def kv(tokens) -> dict:
    return dict(t.split("=", 1) for t in tokens if "=" in t)


@dataclass
class Log:
    info: dict = field(default_factory=dict)
    selftest: list = field(default_factory=list)
    selftest_result: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)
    pp_params: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)       # idx → ảnh gốc
    smooth: dict = field(default_factory=dict)    # idx → ảnh làm mượt do ESP32 tính
    done: bool = False


class Parser:
    def __init__(self):
        self.log = Log()
        self.in_pp = False
        self.img_hdr = None
        self.img_buf: list[str] = []

    def feed(self, line: str):
        line = line.rstrip("\r\n")
        lg = self.log
        if self.img_hdr is not None:
            if line == "END":
                tag, idx, w, h = self.img_hdr
                self.img_hdr = None
                raw = base64.b64decode("".join(self.img_buf))
                if len(raw) != w * h:
                    return f"  ⚠ ảnh {tag} {idx} lỗi: {len(raw)} byte"
                img = np.frombuffer(raw, np.uint8).reshape(h, w)
                (lg.raw if tag == "RAW" else lg.smooth)[idx] = img
                return f"  ✔ nhận ảnh {tag} khung {idx}"
            self.img_buf.append(line.strip())
            return None
        if self.in_pp:
            if line.startswith("END_PP"):
                self.in_pp = False
                return f"END_PP – đã nhận {len(lg.rows)} bản ghi"
            if line.startswith("P,"):
                p = line.split(",")
                if len(p) == 11 and p[1] != "idx":
                    try:
                        lg.rows.append([int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]),
                                        int(p[7]), int(p[8], 16), int(p[9], 16), int(p[10])])
                    except ValueError:
                        pass
            return None
        t = line.split()
        if not t:
            return None
        if t[0] == "#INFO":
            lg.info.update(kv(t[1:]))
            return line
        if t[0] == "SELFTEST":
            lg.selftest.append(kv(t[1:]))
            return None
        if t[0] == "SELFTEST_RESULT":
            lg.selftest_result = kv(t[1:])
            return line
        if t[0] == "BENCH":
            lg.bench = kv(t[1:])
            return line
        if t[0] == "BEGIN_PP":
            lg.pp_params = kv(t[1:])
            self.in_pp = True
            return f"{line}   (đang nhận bản ghi...)"
        if t[0] == "SUMMARY":
            lg.summary = kv(t[1:])
            return None
        if t[0] in ("RAW", "SMOOTH") and len(t) == 4:
            self.img_hdr, self.img_buf = (t[0], int(t[1]), int(t[2]), int(t[3])), []
            return None
        if line.strip() == "DONE":
            lg.done = True
        return line


# ----------------------------------------------------------------- bộ chọn cặp (Python)

def pacer_with_dark(ts_us, dark, target_ms, tol_pct):
    """Giống frame_pacer.c; khung tối → pacer_reset() và ghi SEL_DARK."""
    target, tol = int(target_ms * 1000), int(target_ms * 1000) * int(tol_pct) // 100
    period = last_raw = last_sel = 0
    has_raw = has_sel = False
    out = np.zeros(len(ts_us), np.int8)
    for i, (t, d) in enumerate(zip(ts_us, dark)):
        if d:
            has_sel = False
            out[i] = SEL_DARK
            continue
        t = int(t)
        if has_raw:
            dd = t - last_raw
            if 0 < dd < 4 * target:
                period = (3 * period + dd) // 4 if period else dd
        last_raw, has_raw = t, True
        if not has_sel:
            has_sel, last_sel = True, t
            out[i] = SEL_FIRST
            continue
        dt = t - last_sel
        if dt < target - period // 2:
            out[i] = SEL_SKIP
            continue
        last_sel = t
        out[i] = SEL_OK if abs(dt - target) <= tol else SEL_BAD
    return out


# --------------------------------------------------------------------------- phân tích

def float_gauss(img, border):
    import cv2
    b = cv2.BORDER_REFLECT_101 if border == "reflect101" else cv2.BORDER_REPLICATE
    return cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.0, borderType=b)


def analyze(lg: Log) -> dict:
    r: dict = {}
    border = BORDER_NAMES.get(int(lg.pp_params.get("border", lg.info.get("border", 0))), "reflect101")
    dark_mean = int(lg.pp_params.get("dark_mean", lg.info.get("dark_mean", 15)))
    target_ms = float(lg.pp_params.get("target_ms", 80))
    tol_pct = float(lg.pp_params.get("tol_pct", 20))
    r.update(border=border, dark_mean=dark_mean, target_ms=target_ms, tol_pct=tol_pct)

    # 1. tự kiểm tra trên chip
    st_ok = sum(int(s.get("ok", 0)) for s in lg.selftest)
    r["selftest_pass"], r["selftest_n"] = st_ok, len(lg.selftest)
    r["selftest_us"] = [int(s.get("us", 0)) for s in lg.selftest]

    # 2. bản ghi thời gian thực
    a = np.asarray(lg.rows, np.int64).reshape(-1, len(COLS))
    c = {k: a[:, i] for i, k in enumerate(COLS)}
    r["c"] = c
    got = c["sel"] != SEL_NONE
    r["n"], r["n_fail"] = len(a), int((~got).sum())
    g, lp = c["t_gauss_us"][got], c["t_loop_us"][got]
    budget_us = target_ms * 1000
    if len(g):
        r.update(g_avg=float(g.mean()), g_p99=float(np.percentile(g, 99)), g_max=int(g.max()),
                 loop_avg=float(lp.mean()), loop_max=int(lp.max()), budget_pct=float(lp.max() / budget_us * 100))
    ts = c["ts_us"][got]
    dt = np.diff(ts) / 1000.0 if len(ts) > 1 else np.array([])
    r["dt"] = dt
    if len(dt):
        med = float(np.median(dt))
        r.update(dt_avg=float(dt.mean()), dt_med=med, dt_min=float(dt.min()), dt_max=float(dt.max()),
                 drops=int(np.sum(dt > 1.5 * med)))
    dark = c["dark"].astype(bool)
    r["n_dark"] = int(dark.sum())
    # cờ tối phải đúng với tổng mà chính ESP32 báo
    r["dark_rule_ok"] = bool(np.array_equal(dark[got], c["sum"][got] < dark_mean * W * H))
    sel_py = pacer_with_dark(c["ts_us"], dark, target_ms, tol_pct)
    sel_py[~got] = SEL_NONE                         # khung lỗi: không so
    r["pacer_match"] = bool(np.array_equal(sel_py[got], c["sel"][got])) if not (~got).any() else None
    r["pair_ok"] = int(np.sum(c["sel"] == SEL_OK))
    r["pair_bad"] = int(np.sum(c["sel"] == SEL_BAD))
    r["pair_first"] = int(np.sum(c["sel"] == SEL_FIRST))
    # sau mỗi đoạn tối, khung sáng đầu tiên phải là FIRST (bắt đầu cặp mới)
    after_dark = [i for i in range(1, len(a)) if dark[i - 1] and not dark[i] and got[i]]
    r["after_dark_first"] = all(c["sel"][i] == SEL_FIRST for i in after_dark)
    r["n_dark_runs"] = len(after_dark) + int(dark[-1]) if len(dark) else 0

    # 3. bit-exact trên ảnh thật + so với số thực
    idx_map = {int(k): i for i, k in enumerate(c["idx"])}
    samples = []
    for idx, raw in sorted(lg.raw.items()):
        row = idx_map.get(idx)
        if row is None:
            continue
        q = G.gauss5_q(raw, border)
        s = {"idx": idx, "raw": raw, "q": q,
             "crc_raw_ok": G.crc32(raw) == int(c["crc_raw"][row]),
             "crc_smooth_ok": G.crc32(q) == int(c["crc_smooth"][row]),
             "sum_ok": G.frame_sum(q) == int(c["sum"][row]),
             "dark_ok": G.is_dark(q, dark_mean) == bool(c["dark"][row]),
             "dark": bool(c["dark"][row]), "mean": float(q.mean())}
        if idx in lg.smooth:
            s["pixel_ok"] = bool(np.array_equal(lg.smooth[idx], q))
        try:
            f = float_gauss(raw, border)
            d = np.abs(q.astype(np.float64) - f)
            s.update(f_max=float(d.max()), f_mean=float(d.mean()), f_gt05=float(np.mean(d > 0.5) * 100),
                     other_border_max=float(np.abs(q.astype(np.float64) - float_gauss(
                         raw, "replicate" if border == "reflect101" else "reflect101")).max()))
        except ImportError:
            pass
        samples.append(s)
    r["samples"] = samples
    r["bitexact"] = bool(samples) and all(s["crc_smooth_ok"] and s["sum_ok"] and s["dark_ok"]
                                          and s.get("pixel_ok", True) for s in samples)
    r["transfer_ok"] = bool(samples) and all(s["crc_raw_ok"] for s in samples)
    if samples and "f_max" in samples[0]:
        r["f_max"] = max(s["f_max"] for s in samples)
        r["f_mean"] = float(np.mean([s["f_mean"] for s in samples]))

    # 4. kết luận
    v = {
        "selftest_on_chip": r["selftest_n"] > 0 and st_ok == r["selftest_n"],
        "transfer_ok": r["transfer_ok"],
        "bitexact_real_frames": r["bitexact"],
        "float_error_rounding_only": r.get("f_max", 9) <= 0.6,
        "gauss_fast": r.get("g_max", 1e9) < 0.1 * budget_us,
        "within_budget": r.get("loop_max", 1e9) < budget_us,
        "no_frame_drop": r["n_fail"] == 0 and r.get("drops", 1) == 0,
        "dark_rule": r["dark_rule_ok"] and all(s["dark_ok"] for s in samples),
        "pacer_ok": r["pacer_match"] in (True, None) and r["after_dark_first"],
    }
    r["verdict"] = v
    r["dark_tested"] = r["n_dark"] > 0
    return r


# ------------------------------------------------------------------------------ xuất

def fmt(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def save_outputs(lg: Log, r: dict, out: Path, no_plot: bool):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preproc_records.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for row in lg.rows:
            w.writerow(row[:7] + [f"{row[7]:08X}", f"{row[8]:08X}", row[9]])
    try:
        import cv2
        for s in r["samples"]:
            cv2.imwrite(str(out / f"raw_{s['idx']:04d}.png"), s["raw"])
            cv2.imwrite(str(out / f"smooth_{s['idx']:04d}.png"), s["q"])
    except ImportError:
        pass
    if no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    c = r["c"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(c["t_gauss_us"] / 1000, lw=0.8, label="Gauss")
    ax[0].plot(c["t_loop_us"] / 1000, lw=0.8, label="cả vòng")
    ax[0].set(title="Thời gian xử lý mỗi khung", xlabel="khung", ylabel="ms")
    ax[0].legend()
    mean = c["sum"] / (W * H)
    ax[1].plot(mean, lw=0.8, color="tab:gray")
    ax[1].axhline(r["dark_mean"], color="red", ls="--", lw=0.8, label=f"ngưỡng tối {r['dark_mean']}")
    dk = np.flatnonzero(c["dark"])
    if len(dk):
        ax[1].scatter(dk, mean[dk], s=6, color="red", label="khung tối")
    ax[1].set(title="Độ sáng TB ảnh đã làm mượt", xlabel="khung")
    ax[1].legend()
    if len(r["dt"]):
        ax[2].hist(r["dt"], bins=50)
    ax[2].set(title="Khoảng cách khung (ms)", xlabel="ms")
    fig.tight_layout()
    fig.savefig(out / "preproc_timing.png", dpi=110)
    plt.close(fig)
    if r["samples"]:
        s = next((x for x in r["samples"] if not x["dark"]), r["samples"][0])
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].imshow(s["raw"], cmap="gray", vmin=0, vmax=255)
        ax[0].set_title(f"Ảnh gốc khung {s['idx']}")
        ax[1].imshow(s["q"], cmap="gray", vmin=0, vmax=255)
        ax[1].set_title("Gauss số nguyên (ESP32 = Golden)")
        try:
            d = s["q"].astype(np.float32) - float_gauss(s["raw"], r["border"])
            im = ax[2].imshow(d, cmap="coolwarm", vmin=-0.6, vmax=0.6)
            fig.colorbar(im, ax=ax[2], fraction=0.035)
            ax[2].set_title("Số nguyên − cv2 số thực")
        except ImportError:
            pass
        for x in ax:
            x.axis("off")
        fig.tight_layout()
        fig.savefig(out / "preproc_images.png", dpi=110)
        plt.close(fig)


def report(lg: Log, r: dict, out: Path) -> str:
    i, v = lg.info, r["verdict"]
    yn = lambda b: "✅" if b else "❌"  # noqa: E731
    L = ["# Báo cáo tiền xử lý ESP32-S3 (mục 2.2–2.3)", "",
         f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')} · board {i.get('board', '?')} · IDF {i.get('idf', '?')} · "
         f"FPS đặt {int(i.get('target_fps_x10', 0)) / 10:g} ({i.get('fps_ctrl', '?')}) · biên {r['border']} · "
         f"ngưỡng tối {r['dark_mean']} · hệ số {i.get('kernel', '56,250,412,250,56/1024')}", "",
         "## Tiêu chí", "", "| Tiêu chí | Kết quả | Đạt |", "|---|---|---|",
         f"| Tự kiểm tra trên chip (10 ảnh mẫu × 2 biên) | {r['selftest_pass']}/{r['selftest_n']} vector | {yn(v['selftest_on_chip'])} |",
         f"| Truyền ảnh gốc về PC nguyên vẹn (CRC) | {len(r['samples'])} ảnh | {yn(v['transfer_ok'])} |",
         f"| **Bit-exact với Golden Model trên ảnh camera thật** | CRC, tổng, cờ tối{', từng pixel' if lg.smooth else ''} | {yn(v['bitexact_real_frames'])} |",
         f"| Sai khác so với cv2.GaussianBlur số thực ≤ 0,6 (chỉ làm tròn) | max {fmt(r.get('f_max'), 3)}, TB {fmt(r.get('f_mean'), 3)} | {yn(v['float_error_rounding_only'])} |",
         f"| Gauss < 10% ngân sách | TB {fmt(r.get('g_avg'), 0)} µs, max {fmt(r.get('g_max'))} µs | {yn(v['gauss_fast'])} |",
         f"| Cả vòng xử lý < ngân sách {r['target_ms']:.0f} ms | max {fmt(r.get('loop_max'))} µs ({fmt(r.get('budget_pct'))}%) | {yn(v['within_budget'])} |",
         f"| Không rớt khung khi có xử lý | lỗi {r['n_fail']}, chỗ hở {r.get('drops', '–')}, khoảng cách {fmt(r.get('dt_min'))}–{fmt(r.get('dt_max'))} ms | {yn(v['no_frame_drop'])} |",
         f"| Quy tắc khung tối (Σ < {r['dark_mean']}×19 200) | {r['n_dark']} khung tối | {yn(v['dark_rule'])} |",
         f"| Khung tối ngắt cặp, bộ chọn cặp khớp Python | khớp = {r['pacer_match']}, sau đoạn tối là FIRST = {r['after_dark_first']} | {yn(v['pacer_ok'])} |",
         ""]
    if not r["dark_tested"]:
        L += ["> ⚠ Chưa có khung tối nào trong lần đo – che kín ống kính khoảng 3 s trong lúc đo để kiểm tra mục 2.3.", ""]
    b = lg.bench
    if b:
        L += ["## Benchmark Gauss (một khung camera, lặp nhiều lần)", "",
              f"{b.get('iters')} lần · min {b.get('us_min')} µs · TB {b.get('us_avg')} µs · max {b.get('us_max')} µs · "
              f"{b.get('cyc_avg')} chu kỳ ({b.get('cyc_per_px')} chu kỳ/điểm ảnh) · bộ nhớ {b.get('src')}", ""]
    L += ["## Thời gian thực", "", "| Thông số | Giá trị |", "|---|---|",
          f"| Số khung | {r['n']} (lỗi {r['n_fail']}) |",
          f"| Gauss TB / p99 / max | {fmt(r.get('g_avg'), 0)} / {fmt(r.get('g_p99'), 0)} / {fmt(r.get('g_max'))} µs |",
          f"| Cả vòng TB / max | {fmt(r.get('loop_avg'), 0)} / {fmt(r.get('loop_max'))} µs |",
          f"| Khoảng cách khung TB / min / max | {fmt(r.get('dt_avg'))} / {fmt(r.get('dt_min'))} / {fmt(r.get('dt_max'))} ms |",
          f"| Khung tối / số đoạn tối | {r['n_dark']} / {r['n_dark_runs']} |",
          f"| Cặp hợp lệ / bị loại / chuỗi mới | {r['pair_ok']} / {r['pair_bad']} / {r['pair_first']} |", ""]
    if r["samples"]:
        L += ["## Ảnh mẫu – Golden Model so với ESP32 và với số thực", "",
              "| Khung | Tối | CRC gốc | CRC làm mượt | Tổng | Cờ tối | Độ sáng TB | max \\|q − float\\| | TB | % > 0,5 | max nếu sai kiểu biên |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in r["samples"]:
            L.append(f"| {s['idx']} | {'có' if s['dark'] else ''} | {yn(s['crc_raw_ok'])} | {yn(s['crc_smooth_ok'])} | "
                     f"{yn(s['sum_ok'])} | {yn(s['dark_ok'])} | {s['mean']:.1f} | {fmt(s.get('f_max'), 3)} | "
                     f"{fmt(s.get('f_mean'), 3)} | {fmt(s.get('f_gt05'), 1)} | {fmt(s.get('other_border_max'), 1)} |")
        L.append("")
    if (out / "preproc_images.png").exists():
        L += ["![ảnh](preproc_images.png)", ""]
    if (out / "preproc_timing.png").exists():
        L += ["![thời gian](preproc_timing.png)", ""]
    return "\n".join(L)


def finish(lg: Log, out: Path, no_plot=False) -> dict:
    if not lg.rows:
        sys.exit("Không có bản ghi (thiếu BEGIN_PP … END_PP). Nhấn RST và chạy lại.")
    r = analyze(lg)
    save_outputs(lg, r, out, no_plot)
    (out / "report.md").write_text(report(lg, r, out), encoding="utf-8")
    v = r["verdict"]
    print(f"\nTự kiểm tra trên chip {r['selftest_pass']}/{r['selftest_n']} · bit-exact ảnh thật: {r['bitexact']} "
          f"({len(r['samples'])} ảnh) · |q − float| max {fmt(r.get('f_max'), 3)}")
    print(f"Gauss TB {fmt(r.get('g_avg'), 0)} µs (max {fmt(r.get('g_max'))}) · cả vòng max {fmt(r.get('loop_max'))} µs "
          f"= {fmt(r.get('budget_pct'))}% ngân sách · khung tối {r['n_dark']} · cặp {r['pair_ok']}/{r['pair_bad']}")
    bad = [k for k, ok in v.items() if not ok]
    print("Tiêu chí chưa đạt: " + (", ".join(bad) if bad else "không có"))
    if not r["dark_tested"]:
        print("⚠ Chưa có khung tối – che kín ống kính ~3 s trong lúc đo để kiểm tra mục 2.3.")
    print(f"Đã lưu kết quả vào {out}")
    return r


# ----------------------------------------------------------------------------- nguồn log

def read_serial(port, baud, timeout_s, raw_path: Path, parser: Parser):
    try:
        import serial
    except ImportError:
        sys.exit("Thiếu thư viện: pip install pyserial")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Đang nghe {port} @ {baud}. Nhấn RST trên board. Khi thấy 'measuring', che kín ống kính ~3 s.")
    t_last = time.time()
    with open(raw_path, "w", encoding="utf-8") as fo:
        try:
            while True:
                b = ser.readline()
                if not b:
                    if parser.log.done or (timeout_s and time.time() - t_last > timeout_s):
                        break
                    continue
                t_last = time.time()
                line = b.decode("utf-8", errors="replace")
                fo.write(line if line.endswith("\n") else line + "\n")
                shown = parser.feed(line)
                if shown is not None:
                    print(shown)
                if parser.log.done:
                    break
        except KeyboardInterrupt:
            print("\nDừng theo yêu cầu.")
        finally:
            ser.close()


# ------------------------------------------------------------------------------ selftest

def synthetic_log(rng, n=200, dark_range=(90, 115), corrupt=False) -> str:
    """Log giả lập đúng định dạng firmware, tính bằng Golden Model."""
    L = ["#INFO test=preproc board=sim idf=sim", "#INFO cfg target_fps_x10=125 border=0 dark_mean=15",
         "#INFO fps_ctrl=OK"]
    for (name, b), (crc, s) in G.expected_crcs().items():
        L.append(f"SELFTEST pattern={name} border={G.BORDER_ID[b]} crc={crc:08X} expect={crc:08X} sum={s} "
                 f"expect_sum={s} us=2500 ok=1")
    L += ["SELFTEST_RESULT pass=20 fail=0", "BENCH iters=200 us_min=2400 us_avg=2450.0 us_max=2600 cyc_avg=588000",
          f"BEGIN_PP n={n} border=0 dark_mean=15 target_ms=80 tol_pct=20 extra_load_ms=0",
          "P,idx,ts_us,t_gauss_us,t_loop_us,sum,dark,sel,crc_raw,crc_smooth,mean_raw"]
    imgs, rows = {}, []
    ts = 1_000_000 + np.arange(n) * 80_000 + rng.integers(-20, 21, n)
    darks = np.zeros(n, bool)
    import cv2
    base = cv2.GaussianBlur(rng.uniform(0, 255, (H + 40, W + 40)).astype(np.float32), (0, 0), 2)
    for k in range(n):
        if dark_range[0] <= k < dark_range[1]:
            img = rng.integers(0, 8, (H, W)).astype(np.uint8)
        else:
            img = np.clip(base[k % 40:k % 40 + H, 20:20 + W] + rng.normal(0, 3, (H, W)), 0, 255).astype(np.uint8)
        q = G.gauss5_q(img)
        darks[k] = G.is_dark(q)
        imgs[k] = img
        rows.append([k, int(ts[k]), 2500, 3100, G.frame_sum(q), int(darks[k]), 0, G.crc32(img), G.crc32(q),
                     int(img.mean())])
    sel = pacer_with_dark(ts, darks, 80, 20)
    for k, row in enumerate(rows):
        row[6] = int(sel[k])
        if corrupt and k == 30:
            row[8] ^= 1
        L.append("P," + ",".join(str(x) if j not in (7, 8) else f"{x:08X}" for j, x in enumerate(row)))
    L += ["END_PP", "SUMMARY n=%d" % n]
    for k in (10, 30, 100, 150):
        L.append(f"RAW {k} {W} {H}")
        b64 = base64.b64encode(imgs[k].tobytes()).decode()
        L += [b64[j:j + 76] for j in range(0, len(b64), 76)] + ["END"]
    L.append(f"SMOOTH 10 {W} {H}")
    b64 = base64.b64encode(G.gauss5_q(imgs[10]).tobytes()).decode()
    L += [b64[j:j + 76] for j in range(0, len(b64), 76)] + ["END", "DONE"]
    return "\n".join(L) + "\n"


def selftest(out_root: Path, cc: str) -> int:
    fails = 0
    rng = np.random.default_rng(0)

    # 1. header nhúng trong firmware khớp Golden Model
    hdr = FW / "main" / "selftest_vectors.h"
    same = hdr.exists() and hdr.read_text(encoding="utf-8") == G.vectors_header()
    fails += not same
    print(f"  {'ĐẠT ' if same else 'LỖI '} selftest_vectors.h khớp Golden Model")

    # 2. C (gcc) ↔ Golden trên ảnh ngẫu nhiên, làm mờ, mẫu thử, cả hai kiểu biên
    if shutil.which(cc):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exe = tmp / ("tpp.exe" if sys.platform.startswith("win") else "tpp")
            cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", f"-I{FW / 'main'}", str(FW / "test" / "test_preproc_pc.c"),
                   str(FW / "main" / "preproc.c"), "-o", str(exe)]
            rr = subprocess.run(cmd, capture_output=True, text=True)
            if rr.returncode:
                print("  LỖI  biên dịch C:\n" + rr.stderr)
                return 1
            rv = subprocess.run([str(exe), "vectors"], capture_output=True, text=True)
            ok = rv.returncode == 0
            fails += not ok
            print(f"  {'ĐẠT ' if ok else 'LỖI '} C tự kiểm tra 20 vector: {rv.stdout.strip().splitlines()[-1]}")
            import cv2
            imgs = [rng.integers(0, 256, (H, W)).astype(np.uint8) for _ in range(30)]
            imgs += [np.clip(cv2.GaussianBlur(rng.uniform(0, 255, (H, W)).astype(np.float32), (0, 0), s), 0, 255)
                     .astype(np.uint8) for s in (1, 2, 4) for _ in range(10)]
            imgs += [G.test_pattern(p) for p in range(len(G.PATTERN_NAMES))]
            arr = np.stack(imgs)
            fin = tmp / "in.bin"
            with open(fin, "wb") as f:
                np.array([len(arr)], np.int32).tofile(f)
                arr.tofile(f)
            for bid, bname in BORDER_NAMES.items():
                fout = tmp / f"out{bid}.bin"
                subprocess.run([str(exe), "run", str(fin), str(fout), str(bid)], check=True, capture_output=True)
                data = np.fromfile(fout, np.uint8)
                outs = data[:len(arr) * W * H].reshape(len(arr), H, W)
                sums = data[len(arr) * W * H:].view(np.uint32)
                gold = np.stack([G.gauss5_q(im, bname) for im in arr])
                ok = np.array_equal(outs, gold) and np.array_equal(sums, gold.reshape(len(arr), -1).sum(1))
                fails += not ok
                print(f"  {'ĐẠT ' if ok else 'LỖI '} C ↔ Golden bit-exact, {len(arr)} ảnh, biên {bname}")
    else:
        print(f"  BỎ QUA so C ↔ Golden: không tìm thấy '{cc}' (Windows: cài MinGW-w64 hoặc dùng --cc)")

    # 3. phân tích log giả lập: đúng thì đạt hết, hỏng 1 CRC thì phải phát hiện
    for corrupt in (False, True):
        p = Parser()
        for line in synthetic_log(np.random.default_rng(1), corrupt=corrupt).splitlines():
            p.feed(line)
        r = analyze(p.log)
        want = not corrupt
        ok = (r["bitexact"] == want and r["verdict"]["pacer_ok"] and r["verdict"]["dark_rule"]
              and r["n_dark"] == 25 and r["transfer_ok"])
        fails += not ok
        print(f"  {'ĐẠT ' if ok else 'LỖI '} script phân tích log giả lập ({'hỏng 1 CRC → phải báo lỗi' if corrupt else 'đúng'}): "
              f"bit-exact={r['bitexact']}, khung tối={r['n_dark']}, |q−float| max={fmt(r.get('f_max'), 3)}")
        if not corrupt:
            finish(p.log, out_root / "selftest", no_plot=False)
    print("\nSELFTEST:", "TẤT CẢ ĐẠT" if not fails else f"{fails} lỗi")
    return int(fails > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="cổng COM của ESP32 (vd COM5, /dev/ttyACM0)")
    src.add_argument("--log", help="phân tích lại file log đã lưu")
    src.add_argument("--selftest", action="store_true", help="kiểm tra Golden ↔ C ↔ script, không cần board")
    src.add_argument("--gen-vectors", action="store_true", help="sinh lại firmware/esp32_preproc/main/selftest_vectors.h")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0)
    ap.add_argument("--cc", default="gcc", help="trình biên dịch C cho --selftest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    out_root = ROOT / "outputs" / "preproc"
    if a.gen_vectors:
        path = FW / "main" / "selftest_vectors.h"
        path.write_text(G.vectors_header(), encoding="utf-8")
        print(f"Đã ghi {path}")
        return
    if a.selftest:
        sys.exit(selftest(out_root, a.cc))
    out = Path(a.out) if a.out else out_root / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    p = Parser()
    if a.port:
        read_serial(a.port, a.baud, a.timeout, out / "log.txt", p)
    else:
        for line in Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines():
            p.feed(line)
    finish(p.log, out, a.no_plot)


if __name__ == "__main__":
    main()
