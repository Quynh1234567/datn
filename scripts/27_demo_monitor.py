#!/usr/bin/env python3
"""MÀN HÌNH DEMO 1 – đọc kết quả FPGA (và ảnh xem trước từ ESP32), hiển thị trực tiếp, ghi số liệu.

    python scripts/27_demo_monitor.py --fpga COM5                       # chỉ kết quả FPGA
    python scripts/27_demo_monitor.py --fpga COM5 --esp COM7            # + ảnh camera 80×60 từ ESP32
    python scripts/27_demo_monitor.py --fpga COM5 --esp COM7 --minutes 5 --no-gui

FPGA (115 200 baud) gửi:  SD E=0000 W=0800 K=36      (tự kiểm tra SDRAM, K = MHz hex)
                          HS N=… Q=seq C=chu kỳ HS T=chu kỳ tới kết quả F=f_q A=cờ S=bão hòa Z=Σz D=bỏ R=nhận
ESP32 (921 600 baud) gửi: F seq=… g=µs Gauss s=µs SPI …  ·  PV seq 80 60 / base64 / END
Phím: q thoát · s chụp màn hình.  Kết quả: outputs_hw/demo_runs/<thời gian>/ (CSV + summary.json)
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RE_HS = re.compile(r"HS N=([0-9A-F]{4}) Q=([0-9A-F]{2}) C=([0-9A-F]{8}) T=([0-9A-F]{8}) F=([0-9A-F]{8}) "
                   r"A=([0-9A-F]) S=([0-9A-F]{8}) Z=([0-9A-F]{4}) D=([0-9A-F]{2}) R=([0-9A-F]{4})")
RE_SD = re.compile(r"SD E=([0-9A-F]{4}) W=([0-9A-F]{4}) K=([0-9A-F]{2})")
RE_F = re.compile(r"F seq=(\d+) ts=(-?\d+) dt=(-?\d+) g=(\d+) s=(\d+) fl=(\d+) d=(\S+)")


def s32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def reader(port: str, baud: int, tag: str, q: queue.Queue, stop: threading.Event):
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:  # noqa: BLE001
        q.put(("err", tag, str(e)))
        return
    pv = None
    while not stop.is_set():
        try:
            raw = ser.readline().decode("ascii", "replace").strip()
        except Exception as e:  # noqa: BLE001
            q.put(("err", tag, str(e)))
            break
        if not raw:
            continue
        t = time.time()
        if pv is not None:                                   # đang nhận ảnh xem trước
            if raw == "END":
                try:
                    data = base64.b64decode("".join(pv["b64"]))
                    if len(data) >= pv["w"] * pv["h"]:
                        img = np.frombuffer(data[:pv["w"] * pv["h"]], np.uint8).reshape(pv["h"], pv["w"])
                        q.put(("pv", tag, (pv["seq"], img, t)))
                except Exception:  # noqa: BLE001
                    pass
                pv = None
            else:
                pv["b64"].append(raw)
            continue
        if raw.startswith("PV "):
            p = raw.split()
            if len(p) == 4:
                pv = {"seq": int(p[1]), "w": int(p[2]), "h": int(p[3]), "b64": []}
            continue
        m = RE_HS.search(raw)
        if m:
            n, qs, c, tt, f, a, s, z, d, r = m.groups()
            q.put(("hs", tag, {"t": t, "N": int(n, 16), "seq": int(qs, 16), "C": int(c, 16), "T": int(tt, 16),
                               "F": s32(int(f, 16)), "A": int(a, 16), "S": int(s, 16), "Z": int(z, 16),
                               "D": int(d, 16), "R": int(r, 16)}))
            continue
        m = RE_SD.search(raw)
        if m:
            q.put(("sd", tag, {"E": int(m.group(1), 16), "W": int(m.group(2), 16), "K": int(m.group(3), 16)}))
            continue
        m = RE_F.search(raw)
        if m:
            q.put(("f", tag, {"t": t, "seq": int(m.group(1)), "g": int(m.group(4)), "s": int(m.group(5)),
                              "fl": int(m.group(6)), "d": m.group(7)}))
            continue
        q.put(("txt", tag, raw))
    ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpga", required=True, help="cổng COM của Tang Nano 20K (BL616)")
    ap.add_argument("--esp", default=None, help="cổng COM của ESP32-S3 (ảnh xem trước)")
    ap.add_argument("--fpga-baud", type=int, default=115200)
    ap.add_argument("--esp-baud", type=int, default=921600)
    ap.add_argument("--n-iter", type=int, default=64, help="số vòng HS (để tính chu kỳ / điểm / vòng)")
    ap.add_argument("--cpu-ms", type=float, default=899.9,
                    help="thời gian HS trên ESP32-S3 để so (mặc định 899,9 ms: C số nguyên, SRAM, đo 31/10)")
    ap.add_argument("--smooth", type=int, default=3, help="làm mượt cờ (đa số trong k kết quả)")
    ap.add_argument("--calib-sec", type=float, default=0,
                    help="hiệu chỉnh ngưỡng cho cảnh demo: N giây đầu là cảnh BÌNH THƯỜNG, ngưỡng = phân vị --calib-pct "
                         "của f_q; sau đó cờ hiển thị = f_q < ngưỡng (cờ gốc của FPGA vẫn được ghi trong CSV)")
    ap.add_argument("--calib-pct", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=None, help="ngưỡng f_q cố định (đã hiệu chỉnh từ lần trước)")
    ap.add_argument("--minutes", type=float, default=0, help="tự dừng sau N phút (0 = tới khi bấm q)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs_hw" / "demo_runs"))
    a = ap.parse_args()
    try:
        import serial  # noqa: F401
    except ImportError:
        sys.exit("Cần pyserial: pip install pyserial")
    gui = not a.no_gui
    if gui:
        try:
            import cv2
        except ImportError:
            gui = False
    run = Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    f_hs = open(run / "fpga_results.csv", "w", encoding="utf-8")
    f_hs.write("pc_time,N,seq,C,T,f_q,A,S,Z,D,R,hs_ms,lat_ms\n")
    f_es = open(run / "esp_frames.csv", "w", encoding="utf-8")
    f_es.write("pc_time,seq,gauss_us,spi_us,flags,pacer\n")

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    ths = [threading.Thread(target=reader, args=(a.fpga, a.fpga_baud, "fpga", q, stop), daemon=True)]
    if a.esp:
        ths.append(threading.Thread(target=reader, args=(a.esp, a.esp_baud, "esp", q, stop), daemon=True))
    for t in ths:
        t.start()

    mhz = 54.0
    sd = None
    res, flags_hist, fq_hist = [], deque(maxlen=a.smooth), deque(maxlen=150)
    esp_frames, preview, pv_seq = 0, None, -1
    last = None
    thr, calib = a.thr, []
    t0 = time.time()
    if a.calib_sec:
        print(f"HIỆU CHỈNH: {a.calib_sec:.0f} s đầu chỉ để cảnh BÌNH THƯỜNG trước camera …")
    print(f"Ghi số liệu vào {run}. Bấm q (cửa sổ) hoặc Ctrl+C để dừng.")
    try:
        while True:
            if a.minutes and time.time() - t0 > a.minutes * 60:
                break
            try:
                ev = q.get(timeout=0.05)
            except queue.Empty:
                ev = None
            if ev:
                kind, tag, d = ev
                if kind == "err":
                    print(f"[{tag}] LỖI cổng: {d}")
                elif kind == "sd":
                    sd, mhz = d, float(d["K"])
                    print(f"[fpga] SDRAM tự kiểm tra: {d['E']} từ sai / {d['W']} · xung nhịp {mhz:.0f} MHz"
                          + ("  → OK" if d["E"] == 0 else "  → LỖI SDRAM (xem hướng dẫn mục sự cố)"))
                elif kind == "hs":
                    d["hs_ms"], d["lat_ms"] = d["C"] / (mhz * 1e3), d["T"] / (mhz * 1e3)
                    if a.calib_sec and thr is None:
                        calib.append(d["F"])
                        if time.time() - t0 >= a.calib_sec and len(calib) >= 10:
                            thr = float(np.percentile(calib, a.calib_pct))
                            print(f"HIỆU CHỈNH XONG: {len(calib)} kết quả, ngưỡng f_q = {thr:.0f} "
                                  f"(lần sau dùng --thr {thr:.0f})")
                    d["A_disp"] = int(d["F"] < thr) if thr is not None else (0 if a.calib_sec else d["A"])
                    res.append(d)
                    last = d
                    flags_hist.append(d["A_disp"])
                    fq_hist.append(d["F"])
                    f_hs.write(f"{d['t']:.3f},{d['N']},{d['seq']},{d['C']},{d['T']},{d['F']},{d['A']},{d['S']},"
                               f"{d['Z']},{d['D']},{d['R']},{d['hs_ms']:.3f},{d['lat_ms']:.3f}\n")
                    f_hs.flush()
                    if not gui:
                        sm = int(sum(flags_hist) * 2 > len(flags_hist))
                        print(f"[fpga] #{d['N']:4d} seq {d['seq']:3d} · HS {d['hs_ms']:6.2f} ms · trễ {d['lat_ms']:6.2f} ms · "
                              f"f_q {d['F']:>12d} · cờ {d['A']} (mượt {sm}) · bỏ {d['D']} · nhận {d['R']}")
                elif kind == "f":
                    esp_frames += 1
                    f_es.write(f"{d['t']:.3f},{d['seq']},{d['g']},{d['s']},{d['fl']},{d['d']}\n")
                elif kind == "pv":
                    pv_seq, preview, _ = d
                elif kind == "txt" and not gui:
                    print(f"[{tag}] {d}")
            if gui:
                import cv2
                W, H = 960, 560
                cv = np.full((H, W, 3), 245, np.uint8)
                anom = int(sum(flags_hist) * 2 > len(flags_hist)) if flags_hist else 0
                col = (0, 0, 220) if anom else (40, 160, 40)
                if preview is not None:
                    big = cv2.resize(preview, (480, 360), interpolation=cv2.INTER_NEAREST)
                    cv[20:380, 20:500] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
                    cv2.putText(cv, f"camera seq {pv_seq}", (28, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(cv, "(khong co anh xem truoc: them --esp COMx)", (40, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
                cv2.rectangle(cv, (16, 16), (504, 384), col, 6)
                cv2.putText(cv, "BAT THUONG" if anom else "BINH THUONG", (530, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
                el = max(time.time() - t0, 1e-6)
                lines = []
                if last:
                    ms = [r["hs_ms"] for r in res[-50:]]
                    fps = len([r for r in res if r["t"] > time.time() - 5]) / 5.0
                    lines = [f"Ket qua #{last['N']}  (seq {last['seq']})",
                             f"HS tren FPGA : {last['hs_ms']:.2f} ms  (TB {np.mean(ms):.2f})",
                             f"Tu luc du anh: {last['lat_ms']:.2f} ms",
                             f"Chu ky/diem/vong: {last['C'] / (19200 * a.n_iter):.2f}",
                             f"ESP32-S3 CPU : {a.cpu_ms:.0f} ms  ->  nhanh hon {a.cpu_ms / last['hs_ms']:.0f}x",
                             f"Toc do ket qua: {fps:.1f} cap/s",
                             f"f_q = {last['F']}   co FPGA = {last['A']}" + (f"  nguong {thr:.0f}" if thr is not None else ""),
                             f"Bao hoa u,v: {last['S']}   bo cap: {last['D']}   nhan: {last['R']}",
                             f"SDRAM: {'OK' if sd and sd['E'] == 0 else ('LOI' if sd else '?')}  {mhz:.0f} MHz",
                             f"ESP32 da gui: {esp_frames} khung   thoi gian: {el:.0f} s"]
                else:
                    lines = ["Cho ket qua tu FPGA ...", f"SDRAM: {'OK' if sd and sd['E'] == 0 else '?'}"]
                if a.calib_sec and thr is None:
                    lines.insert(0, f"HIEU CHINH: canh binh thuong ({time.time() - t0:.0f}/{a.calib_sec:.0f} s)")
                for i, s in enumerate(lines):
                    cv2.putText(cv, s, (530, 110 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
                if len(fq_hist) > 1:                             # đồ thị f_q
                    v = np.array(fq_hist, np.float64)
                    lo, hi = v.min(), v.max()
                    hi = hi if hi > lo else lo + 1
                    pts = [(20 + int(i * 920 / (fq_hist.maxlen - 1)), 540 - int((x - lo) / (hi - lo) * 130))
                           for i, x in enumerate(v)]
                    cv2.polylines(cv, [np.array(pts, np.int32)], False, (180, 90, 20), 2)
                    cv2.putText(cv, "f_q (150 ket qua gan nhat)", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
                cv2.imshow("DATN - HS + OC-SVM tren Tang Nano 20K", cv)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    cv2.imwrite(str(run / f"screen_{int(time.time())}.png"), cv)
    except KeyboardInterrupt:
        pass
    stop.set()
    f_hs.close()
    f_es.close()
    summ = {"threshold_display": thr, "results": len(res), "esp_frames_sent": esp_frames, "clock_mhz": mhz, "sdram_errors": sd["E"] if sd else None,
            "n_iter": a.n_iter, "duration_s": time.time() - t0}
    if res:
        hs = np.array([r["hs_ms"] for r in res])
        lat = np.array([r["lat_ms"] for r in res])
        summ.update({"hs_ms_median": float(np.median(hs)), "hs_ms_p95": float(np.percentile(hs, 95)),
                     "hs_ms_max": float(hs.max()), "latency_ms_median": float(np.median(lat)),
                     "cycles_median": int(np.median([r["C"] for r in res])),
                     "cycles_per_pixel_iter": float(np.median([r["C"] for r in res]) / (19200 * a.n_iter)),
                     "results_per_s": len(res) / max(res[-1]["t"] - res[0]["t"], 1e-6) if len(res) > 1 else None,
                     "pairs_dropped": res[-1]["D"], "frames_received": res[-1]["R"],
                     "anomaly_rate": float(np.mean([r["A"] for r in res])),
                     "speedup_vs_esp32": a.cpu_ms / float(np.median(hs))})
    (run / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f"→ {run}")
    if gui:
        import cv2
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
