#!/usr/bin/env python3
"""ĐO THỜI GIAN KHUNG CAMERA ESP32-S3 + OV5640 (đặc tả A1, mục 2.1).

Đọc log của firmware/esp32_cam_timing qua cổng COM (hoặc từ file log đã lưu), rồi:
  • kiểm tra camera: mã cảm biến OV5640, đúng 160×120 xám, không lỗi lấy khung
  • thống kê FPS thực tế, khoảng cách khung, jitter, rớt khung, khung lặp, độ sáng
  • đánh giá bộ chọn cặp khung 80 ms ±20% và kết luận:
      dùng được mô hình IITB (80 ms) hay phải train lại ở khoảng cách đo được
  • lưu log gốc, CSV, ảnh mẫu, biểu đồ và báo cáo Markdown vào outputs/camtiming/<thời gian>/

    pip install pyserial numpy matplotlib opencv-python
    python scripts/18_cam_timing.py --port COM5                 # nhấn RST trên board sau khi chạy
    python scripts/18_cam_timing.py --log outputs/camtiming/xxx/log.txt   # phân tích lại log cũ
    python scripts/18_cam_timing.py --selftest                  # tự kiểm tra script bằng log giả lập
"""
from __future__ import annotations

import argparse
import base64
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 160, 120
PIXFORMAT_GRAYSCALE = 3            # giá trị enum trong esp32-camera
SEL_SKIP, SEL_FIRST, SEL_OK, SEL_BAD, SEL_NONE = range(5)
COLS = ["idx", "ts_us", "get_us", "len", "w", "h", "fmt", "mean", "crc", "sel", "ok"]


# --------------------------------------------------------------------------- đọc log

@dataclass
class Run:
    label: str
    params: dict
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


@dataclass
class Log:
    info: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    sweep: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    done: bool = False


def kv(tokens) -> dict:
    out = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out


class Parser:
    """Phân tích log theo từng dòng – dùng chung cho cổng COM và file."""

    def __init__(self):
        self.log = Log()
        self.cur: Run | None = None
        self.frame_hdr = None
        self.frame_buf: list[str] = []

    def feed(self, line: str) -> str | None:
        """Trả về dòng nên in ra màn hình (None = không in)."""
        line = line.rstrip("\r\n")
        lg = self.log
        if self.frame_hdr is not None:                       # đang nhận ảnh base64
            if line == "END":
                w, h = self.frame_hdr
                raw = base64.b64decode("".join(self.frame_buf))
                self.frame_hdr = None
                if len(raw) == w * h:
                    lg.frames.append(np.frombuffer(raw, np.uint8).reshape(h, w))
                    return f"  ✔ nhận ảnh mẫu {len(lg.frames)} ({w}×{h})"
                return f"  ⚠ ảnh lỗi: {len(raw)} byte, cần {w * h}"
            self.frame_buf.append(line.strip())
            return None
        if line.startswith("T,") and self.cur is not None:
            p = line.split(",")
            if len(p) == 12 and p[1] != "idx":
                try:
                    r = [int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7]),
                         int(p[8]), int(p[9], 16), int(p[10]), int(p[11])]
                    self.cur.rows.append(r)
                except ValueError:
                    pass
            return None
        if line.startswith("#INFO"):
            lg.info.update(kv(line.split()[1:]))
            return line
        if line.startswith("BEGIN_TIMING"):
            t = line.split()
            self.cur = Run(t[1] if len(t) > 1 else f"run{len(lg.runs)}", kv(t[2:]))
            return f"{line}   (đang nhận bản ghi...)"
        if line.startswith("END_TIMING"):
            if self.cur is not None:
                lg.runs.append(self.cur)
                n = len(self.cur.rows)
                self.cur = None
                return f"END_TIMING – đã nhận {n} bản ghi"
            return line
        if line.startswith("SUMMARY"):
            t = line.split()
            for r in lg.runs:
                if len(t) > 1 and r.label == t[1]:
                    r.summary = kv(t[2:])
            return None
        if line.startswith("SWEEP"):
            lg.sweep.append(kv(line.split()[1:]))
            return line
        if line.startswith("FRAME "):
            _, w, h = line.split()[:3]
            self.frame_hdr, self.frame_buf = (int(w), int(h)), []
            return None
        if line.strip() == "DONE":
            lg.done = True
            return line
        return line                                           # log khác (ESP_LOG, bảng kết quả...)


# ------------------------------------------------------------------- bộ chọn cặp khung

def pacer(ts_us: np.ndarray, target_ms: float, tol_pct: float, reset_mask=None) -> np.ndarray:
    """Bản Python của firmware/esp32_cam_timing/main/frame_pacer.c (giống từng bước)."""
    target, tol = int(target_ms * 1000), int(target_ms * 1000) * int(tol_pct) // 100
    period, last_raw, last_sel, has_raw, has_sel = 0, 0, 0, False, False
    out = np.zeros(len(ts_us), np.int8)
    for i, t in enumerate(ts_us):
        if reset_mask is not None and reset_mask[i]:
            has_sel = False
            out[i] = SEL_NONE
            continue
        t = int(t)
        if has_raw:
            d = t - last_raw
            if 0 < d < 4 * target:
                period = (3 * period + d) // 4 if period else d
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


def pair_intervals(ts_us: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Khoảng cách (ms) của các cặp hợp lệ và của mọi cặp được tạo."""
    ok, allp, prev = [], [], None
    for t, s in zip(ts_us, sel):
        if s == SEL_NONE:
            prev = None
        elif s == SEL_FIRST:
            prev = t
        elif s in (SEL_OK, SEL_BAD):
            if prev is not None:
                d = (t - prev) / 1000.0
                allp.append(d)
                if s == SEL_OK:
                    ok.append(d)
            prev = t
    return np.asarray(ok), np.asarray(allp)


# ---------------------------------------------------------------------- phân tích

def analyze(run: Run, info: dict, target_ms: float, tol_pct: float) -> dict:
    a = np.asarray(run.rows, np.int64).reshape(-1, len(COLS))
    c = {k: a[:, i] for i, k in enumerate(COLS)}
    got = c["sel"] != SEL_NONE
    ok_fmt = got & (c["ok"] == 1)
    ts = c["ts_us"][got]
    res = {"label": run.label, "params": run.params, "n": len(a), "n_got": int(got.sum()),
           "n_fail": int((~got).sum()), "n_badfmt": int((got & (c["ok"] == 0)).sum())}

    # khoảng cách liên tiếp, bỏ qua chỗ có lỗi lấy khung
    idx = np.flatnonzero(got)
    consecutive = np.diff(idx) == 1
    dt = (np.diff(ts) / 1000.0)[consecutive]
    res["dt"] = dt
    if len(dt):
        med = float(np.median(dt))
        gaps = dt > 1.5 * med
        res.update(
            fps=len(np.diff(ts)) / ((ts[-1] - ts[0]) / 1e6) if ts[-1] > ts[0] else float("nan"),
            dt_mean=float(dt.mean()), dt_med=med, dt_std=float(dt.std()), dt_min=float(dt.min()),
            dt_max=float(dt.max()), dt_p1=float(np.percentile(dt, 1)), dt_p99=float(np.percentile(dt, 99)),
            jitter_pct=float(dt[~gaps].std() / med * 100) if (~gaps).any() else float("nan"),
            drop_gaps=int(gaps.sum()), lost_est=int(np.sum(np.round(dt[gaps] / med) - 1)),
        )
        res["drop_rate_pct"] = res["lost_est"] / (res["n_got"] + res["lost_est"]) * 100
    crc = c["crc"][got]
    res["dup"] = int(np.sum((crc[1:] == crc[:-1])[consecutive]))
    br = c["mean"][got]
    res.update(bright_min=int(br.min()) if len(br) else 0, bright_avg=float(br.mean()) if len(br) else 0,
               bright_max=int(br.max()) if len(br) else 0, dark=int(np.sum(br < 15)))
    # lag: thời gian từ lúc driver gắn timestamp đến lúc ứng dụng nhận khung
    lag = (c["get_us"][got] - c["ts_us"][got]) / 1000.0
    res.update(lag_med=float(np.median(lag)) if len(lag) else float("nan"),
               lag_max=float(lag.max()) if len(lag) else float("nan"))

    # bộ chọn cặp: kết quả do ESP32 tính và bản Python tính lại (phải trùng)
    sel_dev = c["sel"]
    sel_py = pacer(c["ts_us"], target_ms, tol_pct, reset_mask=~got)
    res["pacer_match"] = bool(np.array_equal(sel_dev, sel_py)) if (
        float(run.params.get("target_ms", target_ms)) == target_ms
        and float(run.params.get("tol_pct", tol_pct)) == tol_pct) else None
    pok, pall = pair_intervals(c["ts_us"], sel_py)
    res.update(pair_ok=int(np.sum(sel_py == SEL_OK)), pair_bad=int(np.sum(sel_py == SEL_BAD)),
               pair_first=int(np.sum(sel_py == SEL_FIRST)), pair_dt=pok, pair_all=pall)
    if len(pok):
        res.update(pair_mean=float(pok.mean()), pair_std=float(pok.std()), pair_min=float(pok.min()),
                   pair_max=float(pok.max()))
    n_pairs = res["pair_ok"] + res["pair_bad"]
    res["pair_bad_pct"] = res["pair_bad"] / n_pairs * 100 if n_pairs else float("nan")

    # các lựa chọn lấy 1 trong k khung
    if len(dt):
        res["options"] = [{"k": k, "interval": k * res["dt_med"],
                           "pairs_per_s": 1000.0 / (k * res["dt_med"]),
                           "dev_pct": (k * res["dt_med"] - target_ms) / target_ms * 100} for k in range(1, 6)]

    # ---- kết luận
    pid = info.get("sensor_pid", "?").lower()
    v = {}
    v["camera_init"] = info.get("cam_init") == "OK"
    v["ov5640"] = pid in ("0x5640",)
    v["format"] = res["n_badfmt"] == 0 and res["n_got"] > 0
    v["no_fail"] = res["n_fail"] == 0
    v["enough_frames"] = res["n_got"] >= 200
    v["stable"] = len(dt) > 0 and res["jitter_pct"] < 5 and res["drop_rate_pct"] < 2
    v["pacer_ok"] = res["pacer_match"] in (True, None)
    v["pairs_usable"] = n_pairs > 0 and res["pair_bad_pct"] < 5
    v["match_80"] = "pair_mean" in res and abs(res["pair_mean"] - target_ms) <= 0.05 * target_ms
    res["verdict"] = v
    return res


def recommendation(res: dict, target_ms: float, tol_pct: float = 20.0) -> str:
    v = res["verdict"]
    if not (v["camera_init"] and v["format"]):
        return ("Camera chưa đạt: kiểm tra sơ đồ chân trong board_pins.h, cáp camera, nguồn 3,3 V, "
                "thử XCLK 10 MHz.")
    if v["match_80"] and v["stable"] and v["pairs_usable"]:
        return (f"Dùng được mô hình IITB: cặp khung trung bình {res['pair_mean']:.1f} ms "
                f"(lệch {res['pair_mean'] - target_ms:+.1f} ms), rớt khung {res['drop_rate_pct']:.2f}%.")
    best = min(res.get("options", []), key=lambda o: abs(o["dev_pct"]), default=None)
    msg = []
    if "pair_mean" in res and not v["match_80"]:
        msg.append(f"Cặp khung thực tế {res['pair_mean']:.1f} ms khác {target_ms:.0f} ms lúc train "
                   f"({(res['pair_mean'] - target_ms) / target_ms * 100:+.1f}%) → vận tốc u, v sẽ lệch tương ứng.")
    if best and not v["match_80"]:
        msg.append(f"Khoảng cách tự nhiên gần nhất: lấy 1 trong {best['k']} khung = {best['interval']:.1f} ms. "
                   f"Train lại mô hình trên dữ liệu OV5640 ở đúng khoảng cách này "
                   f"(đặt target_ms = {round(best['interval'])} cho firmware), hoặc chỉnh XCLK để camera đạt 25/12,5 fps.")
    if v["match_80"] and not msg:
        msg.append(f"Khoảng cách cặp khớp {target_ms:.0f} ms nhưng:")
    if not v["pairs_usable"]:
        nearest = min(res.get("options", []), key=lambda o: abs(o["dev_pct"]), default=None)
        if res.get("drop_rate_pct", 100) < 2 and nearest and abs(nearest["dev_pct"]) > tol_pct:
            msg.append(f"{res['pair_bad_pct']:.1f}% cặp bị loại vì chu kỳ camera {res['dt_med']:.2f} ms không "
                       f"cho khoảng cách nào trong {target_ms:.0f} ms ±{tol_pct:.0f}% (không phải rớt khung). "
                       f"Cách sửa triệt để: đặt FPS camera đúng {1000 / target_ms:.1f} hoặc {2000 / target_ms:.1f} "
                       f"bằng thanh ghi VTS của OV5640.")
        else:
            msg.append(f"{res['pair_bad_pct']:.1f}% cặp bị loại vì sai khoảng cách (rớt khung).")
    if not v["stable"]:
        msg.append(f"Thời gian khung chưa ổn định (jitter {res.get('jitter_pct', float('nan')):.1f}%, "
                   f"rớt {res.get('drop_rate_pct', float('nan')):.2f}%): thử fb_count = 2, GRAB_LATEST, "
                   "giảm tải xử lý hoặc tăng ánh sáng.")
    return " ".join(msg)


# ------------------------------------------------------------------------- xuất kết quả

def save_csv(run: Run, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in run.rows:
            w.writerow(r[:8] + [f"{r[8]:08X}"] + r[9:])


def plot(res: dict, path: Path, target_ms: float, tol_pct: float) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    dt, pdt = res["dt"], res["pair_all"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(dt, lw=0.8)
    ax[0].axhline(res.get("dt_med", 0), color="gray", ls="--", lw=0.8, label="trung vị")
    ax[0].set(title="Khoảng cách giữa hai khung liên tiếp", xlabel="khung", ylabel="ms")
    ax[0].legend()
    ax[1].hist(dt, bins=60)
    ax[1].set(title=f"Phân bố khoảng cách khung (FPS {res.get('fps', 0):.2f})", xlabel="ms", ylabel="số khung")
    if len(pdt):
        ax[2].hist(pdt, bins=60, color="tab:orange")
    lo, hi = target_ms * (1 - tol_pct / 100), target_ms * (1 + tol_pct / 100)
    ax[2].axvline(target_ms, color="green", lw=1.2, label=f"{target_ms:.0f} ms (train)")
    ax[2].axvspan(lo, hi, color="green", alpha=0.08, label=f"±{tol_pct:.0f}%")
    ax[2].set(title="Khoảng cách các cặp khung được chọn", xlabel="ms", ylabel="số cặp")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(log: Log, results: list, out: Path, target_ms: float, tol_pct: float) -> str:
    i = log.info
    L = ["# Báo cáo đo camera ESP32-S3 + OV5640 (mục 2.1)", "",
         f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')} · board {i.get('board', '?')} · "
         f"IDF {i.get('idf', '?')} · PSRAM {i.get('psram_mb', '?')} MB", "",
         "## Cấu hình", "",
         f"XCLK {i.get('xclk_mhz', '?')} MHz · fb_count {i.get('fb_count', '?')} · grab {i.get('grab', '?')} · "
         f"buffer {i.get('fb_in', '?')} · warm-up {i.get('warmup', '?')} khung · tải giả lập {i.get('load_ms', '?')} ms · "
         f"khóa phơi sáng {i.get('lock_exposure', '?')} · cặp danh định {target_ms:.0f} ms ±{tol_pct:.0f}%", ""]
    if log.sweep:
        L += ["## Quét XCLK × fb_count", "", "| XCLK (MHz) | fb_count | Trạng thái | FPS | Trung vị (ms) | Độ lệch (ms) | Chỗ rớt |",
              "|---|---|---|---|---|---|---|"]
        for s in log.sweep:
            L.append(f"| {s.get('xclk_mhz')} | {s.get('fb_count')} | {s.get('status')} | {s.get('fps', '–')} | "
                     f"{s.get('dt_med', '–')} | {s.get('dt_std', '–')} | {s.get('drop_gaps', '–')} |")
        L.append("")
    for r in results:
        v = r["verdict"]
        yn = lambda b: "✅" if b else "❌"  # noqa: E731
        L += [f"## Lần đo `{r['label']}` ({r['n']} khung)", "",
              "| Tiêu chí | Kết quả | Đạt |", "|---|---|---|",
              f"| Khởi tạo camera | {i.get('cam_init', '?')} | {yn(v['camera_init'])} |",
              f"| Cảm biến | PID {i.get('sensor_pid', '?')} ({i.get('sensor_name', '?')}) | {yn(v['ov5640'])} |",
              f"| Định dạng 160×120 xám, 19 200 byte | sai định dạng {r['n_badfmt']} khung | {yn(v['format'])} |",
              f"| Lỗi lấy khung | {r['n_fail']} | {yn(v['no_fail'])} |",
              f"| Số khung ghi timestamp ≥ 200 | {r['n_got']} | {yn(v['enough_frames'])} |",
              f"| Ổn định (jitter < 5%, rớt < 2%) | jitter {fmt(r.get('jitter_pct'))}%, rớt {fmt(r.get('drop_rate_pct'))}% | {yn(v['stable'])} |",
              f"| Bộ chọn cặp ESP32 trùng bản Python | {r['pacer_match']} | {yn(v['pacer_ok'])} |",
              f"| Cặp bị loại vì sai khoảng cách < 5% | {fmt(r.get('pair_bad_pct'))}% | {yn(v['pairs_usable'])} |",
              f"| Cặp khung khớp {target_ms:.0f} ms (±5%) | TB {fmt(r.get('pair_mean'))} ms | {yn(v['match_80'])} |",
              "", "| Thông số | Giá trị |", "|---|---|",
              f"| FPS thực tế | {fmt(r.get('fps'))} |",
              f"| Khoảng cách khung TB / trung vị / độ lệch | {fmt(r.get('dt_mean'))} / {fmt(r.get('dt_med'))} / {fmt(r.get('dt_std'))} ms |",
              f"| Min / max · phân vị 1% / 99% | {fmt(r.get('dt_min'))} / {fmt(r.get('dt_max'))} · {fmt(r.get('dt_p1'))} / {fmt(r.get('dt_p99'))} ms |",
              f"| Rớt khung | {r.get('drop_gaps', 0)} chỗ hở, ước lượng mất {r.get('lost_est', 0)} khung |",
              f"| Khung lặp (CRC trùng) | {r['dup']} |",
              f"| Trễ từ lúc chụp đến lúc ứng dụng nhận | trung vị {fmt(r['lag_med'])} ms, lớn nhất {fmt(r['lag_max'])} ms |",
              f"| Độ sáng TB min / TB / max · khung tối (< 15) | {r['bright_min']} / {fmt(r['bright_avg'], 1)} / {r['bright_max']} · {r['dark']} |",
              f"| Cặp hợp lệ / bị loại / chuỗi mới | {r['pair_ok']} / {r['pair_bad']} / {r['pair_first']} |",
              f"| Khoảng cách cặp TB / độ lệch / min / max | {fmt(r.get('pair_mean'))} / {fmt(r.get('pair_std'))} / {fmt(r.get('pair_min'))} / {fmt(r.get('pair_max'))} ms |",
              ""]
        if r.get("options"):
            L += ["| Lấy 1 trong k khung | Khoảng cách | Cặp/giây | Lệch so với 80 ms |", "|---|---|---|---|"]
            L += [f"| {o['k']} | {o['interval']:.1f} ms | {o['pairs_per_s']:.2f} | {o['dev_pct']:+.1f}% |"
                  for o in r["options"]]
            L.append("")
        L += [f"**Kết luận:** {recommendation(r, target_ms, tol_pct)}", ""]
        if (out / f"{r['label']}_timing.png").exists():
            L += [f"![biểu đồ]({r['label']}_timing.png)", ""]
    if log.frames:
        L += ["## Ảnh mẫu", ""] + [f"![ảnh {k + 1}](frame_{k + 1:02d}.png)" for k in range(len(log.frames))] + [""]
    return "\n".join(L)


def finish(log: Log, out: Path, target_ms: float, tol_pct: float, no_plot: bool) -> list:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for run in log.runs:
        if not run.rows:
            continue
        save_csv(run, out / f"{run.label}_timing.csv")
        r = analyze(run, log.info, target_ms, tol_pct)
        if not no_plot:
            plot(r, out / f"{run.label}_timing.png", target_ms, tol_pct)
        results.append(r)
    for k, img in enumerate(log.frames, 1):
        try:
            import cv2
            cv2.imwrite(str(out / f"frame_{k:02d}.png"), img)
            cv2.imwrite(str(out / f"frame_{k:02d}_x4.png"), cv2.resize(img, (W * 4, H * 4),
                                                                        interpolation=cv2.INTER_NEAREST))
        except ImportError:
            np.save(out / f"frame_{k:02d}.npy", img)
    (out / "report.md").write_text(report(log, results, out, target_ms, tol_pct), encoding="utf-8")
    for r in results:
        print(f"\n[{r['label']}] FPS {fmt(r.get('fps'))} · khoảng cách khung trung vị {fmt(r.get('dt_med'))} ms "
              f"(jitter {fmt(r.get('jitter_pct'))}%) · rớt {fmt(r.get('drop_rate_pct'))}% · "
              f"cặp TB {fmt(r.get('pair_mean'))} ms")
        failed = [k for k, ok in r["verdict"].items() if not ok]
        print("  Tiêu chí chưa đạt: " + (", ".join(failed) if failed else "không có"))
        print("  " + recommendation(r, target_ms, tol_pct))
    print(f"\nĐã lưu kết quả vào {out}")
    return results


# ------------------------------------------------------------------------------ nguồn log

def read_serial(port: str, baud: int, timeout_s: float, raw_path: Path, parser: Parser):
    try:
        import serial
    except ImportError:
        sys.exit("Thiếu thư viện: pip install pyserial")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Đang nghe {port} @ {baud}. Nhấn nút RST trên board để bắt đầu (Ctrl+C để dừng).")
    t_last = time.time()
    with open(raw_path, "w", encoding="utf-8") as raw:
        try:
            while True:
                b = ser.readline()
                if not b:
                    if parser.log.done and time.time() - t_last > 2:
                        break
                    if timeout_s and time.time() - t_last > timeout_s:
                        print(f"Hết {timeout_s:.0f} s không có dữ liệu – dừng.")
                        break
                    continue
                t_last = time.time()
                line = b.decode("utf-8", errors="replace")
                raw.write(line if line.endswith("\n") else line + "\n")
                shown = parser.feed(line)
                if shown is not None:
                    print(shown)
                if parser.log.done:
                    ser.timeout = 0.5
        except KeyboardInterrupt:
            print("\nDừng theo yêu cầu.")
        finally:
            ser.close()


def synthetic_log(fps=25.0, n=500, drop_every=97, jitter_us=200, seed=0) -> str:
    """Log giả lập đúng định dạng firmware – dùng cho --selftest."""
    rng = np.random.default_rng(seed)
    period = 1e6 / fps
    lines = ["#INFO test=cam_timing board=sim idf=sim", "#INFO cam_init=OK err=0x0",
             "#INFO sensor_pid=0x5640 sensor_name=OV5640 is_ov5640=1",
             "#INFO cfg xclk_mhz=20 fb_count=2 grab=latest fb_in=dram warmup=30 n_frames=%d target_ms=80 "
             "tol_pct=20 load_ms=0 lock_exposure=0" % n,
             f"BEGIN_TIMING run0 n={n} xclk_mhz=20 fb_count=2 grab=latest target_ms=80 tol_pct=20 load_ms=0",
             "T,idx,ts_us,get_us,len,w,h,fmt,mean,crc,sel,ok"]
    ts, k = [], 0
    while len(ts) < n:
        if drop_every and k % drop_every == drop_every // 2:
            k += 1
        ts.append(int(1_000_000 + k * period + rng.integers(-jitter_us, jitter_us + 1)))
        k += 1
    ts = np.asarray(ts)
    sel = pacer(ts, 80, 20)
    for i, t in enumerate(ts):
        lines.append(f"T,{i},{t},{t + 1500},19200,160,120,3,{100 + i % 7},{rng.integers(0, 2**32):08X},{sel[i]},1")
    lines += ["END_TIMING", "SUMMARY run0 n=%d" % n]
    img = (np.add.outer(np.arange(H), np.arange(W)) % 256).astype(np.uint8)
    b64 = base64.b64encode(img.tobytes()).decode()
    lines += ["FRAME 160 120"] + [b64[i:i + 76] for i in range(0, len(b64), 76)] + ["END", "DONE"]
    return "\n".join(lines) + "\n"


def selftest(out_root: Path) -> int:
    fails = 0
    cases = [("25fps", 25.0, True), ("12.5fps", 12.5, True), ("15fps", 15.0, False), ("30fps", 30.0, False)]
    for name, fps, expect_match in cases:
        p = Parser()
        for line in synthetic_log(fps=fps).splitlines():
            p.feed(line)
        r = analyze(p.log.runs[0], p.log.info, 80, 20)
        ok = (r["verdict"]["match_80"] == expect_match and r["pacer_match"] and r["verdict"]["format"]
              and len(p.log.frames) == 1 and abs(r["fps"] - fps) / fps < 0.03)
        fails += not ok
        print(f"  {'ĐẠT ' if ok else 'LỖI '} {name:8s} FPS {r['fps']:.2f}, cặp TB {fmt(r.get('pair_mean'))} ms, "
              f"khớp 80 ms = {r['verdict']['match_80']} (mong đợi {expect_match}), rớt {r['drop_rate_pct']:.2f}%")
    p = Parser()
    for line in synthetic_log(fps=15.0).splitlines():
        p.feed(line)
    finish(p.log, out_root / "selftest", 80, 20, no_plot=False)
    print("\nSELFTEST:", "TẤT CẢ ĐẠT" if not fails else f"{fails} lỗi")
    return int(fails > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="cổng COM của ESP32 (vd COM5, /dev/ttyACM0)")
    src.add_argument("--log", help="phân tích lại file log đã lưu")
    src.add_argument("--selftest", action="store_true", help="kiểm tra script bằng log giả lập")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--target-ms", type=float, default=80.0, help="khoảng cách cặp khung lúc train (mặc định 80)")
    ap.add_argument("--tol-pct", type=float, default=20.0, help="sai lệch cho phép một cặp (mặc định 20%%)")
    ap.add_argument("--timeout", type=float, default=0, help="dừng sau N giây không có dữ liệu (0 = chờ mãi)")
    ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định outputs/camtiming/<thời gian>)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    out_root = ROOT / "outputs" / "camtiming"
    if a.selftest:
        sys.exit(selftest(out_root))
    out = Path(a.out) if a.out else out_root / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    p = Parser()
    if a.port:
        read_serial(a.port, a.baud, a.timeout, out / "log.txt", p)
    else:
        for line in Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines():
            p.feed(line)
    if not p.log.runs:
        sys.exit("Không có lần đo nào trong log (thiếu BEGIN_TIMING … END_TIMING). Nhấn RST và chạy lại.")
    finish(p.log, out, a.target_ms, a.tol_pct, a.no_plot)


if __name__ == "__main__":
    main()
