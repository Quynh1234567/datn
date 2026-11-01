#!/usr/bin/env python3
"""ĐO THỜI GIAN KHỐI HORN–SCHUNCK TRÊN CPU – mốc so sánh cho bộ tăng tốc FPGA.

Đầu vào là ĐÚNG các ca kiểm thử của RTL (outputs_hw/fpga/hs/case_XX: img1/img2.mem đã Gauss), nên
kết quả CPU được kiểm khớp từng bit với uv_final.mem và feat_z.mem rồi mới tính thời gian.
Đo riêng 3 tầng giống 3 khối RTL: DERIV (đạo hàm + a, b) · HS_ITER (N vòng Jacobi) · FEAT (48 z).

Các cài đặt được đo (CPU máy tính, 1 luồng):
  (1) Python số thực  – đúng code pipeline train cũ (features/optical_flow.py, numpy + OpenCV)
  (2) Python số nguyên – Golden Model hs_q.py (numpy, đã tối ưu)
  (3) C số thực float32 – firmware/esp32_hs_bench/main/hs_core.c, biên dịch -O2
  (4) C số nguyên      – cùng file, KHỚP TỪNG BIT phần cứng (mốc so sánh chính)
(3), (4) cần trình biên dịch C: gcc / clang (Windows: MinGW-w64 hoặc MSYS2, hoặc chạy trong WSL).

    python scripts/23_hs_cpu_bench.py --config configs/hw.yaml
    python scripts/23_hs_cpu_bench.py --config configs/hw.yaml --repeat 20 --cc gcc
    # tạo véc-tơ cho bản đo trên ESP32-S3 (firmware/esp32_hs_bench):
    python scripts/23_hs_cpu_bench.py --config configs/hw.yaml --esp32-header firmware/esp32_hs_bench/main/test_vectors.h

Kết quả: outputs_hw/hs_cpu_bench/<thời gian>/report.md, bench.json
"""
from __future__ import annotations

import ctypes
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, setup  # noqa: E402
from ocsvm_fpga.hwmodel import hs_q as HQ  # noqa: E402
from ocsvm_fpga.hwmodel.hs_export import read_mem  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402

C_SRC = ROOT / "firmware" / "esp32_hs_bench" / "main" / "hs_core.c"
W, H, N = HQ.W, HQ.H, HQ.W * HQ.H
# Thời gian FPGA ở 100 MHz – ƯỚC LƯỢNG THIẾT KẾ (tài liệu kiến trúc RTL mục 3), thay bằng số đo khi có board
FPGA_MS = {"deriv": 0.5, "iter_per": 0.43, "feat": 0.01}


# ------------------------------------------------------------------------------- dữ liệu

def vh_param(text: str, name: str) -> int | None:
    m = re.search(rf"localparam\s+integer\s+{name}\s*=\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def load_cases(d: Path):
    vh = (d / "hs_params.vh").read_text(encoding="ascii")
    n_iter, inv_frac = vh_param(vh, "N_ITER"), vh_param(vh, "INV_FRAC")
    if vh_param(vh, "UV_FORMAT_CHECK") != 1306:
        sys.exit(f"{d}/hs_params.vh không phải bản Q6.6 (UV_FORMAT_CHECK ≠ 1306) – xuất lại bằng 04_export_fpga.py")
    shared = {"lut": read_mem(d / "mem" / "log_lut.mem", 12, signed=False).astype(np.uint16),
              "mean": read_mem(d / "mem" / "scaler_mean_q.mem", 16).astype(np.int16),
              "inv": read_mem(d / "mem" / "scaler_inv_q.mem", 16, signed=False).astype(np.uint16),
              "n_iter": n_iter, "inv_frac": inv_frac}
    cases = []
    for c in sorted(d.glob("case_*")):
        uv = read_mem(c / "uv_final.mem", 26, signed=False)
        u, v = uv & 0x1FFF, (uv >> 13) & 0x1FFF
        u, v = np.where(u >= 4096, u - 8192, u), np.where(v >= 4096, v - 8192, v)
        cases.append({"name": c.name,
                      "i1": read_mem(c / "img1.mem", 8, signed=False).astype(np.uint8).reshape(H, W),
                      "i2": read_mem(c / "img2.mem", 8, signed=False).astype(np.uint8).reshape(H, W),
                      "u": u.astype(np.int16), "v": v.astype(np.int16),
                      "z": read_mem(c / "feat_z.mem", 16).astype(np.int16)})
    if not cases:
        sys.exit(f"Không có case_XX trong {d} – chạy scripts/04_export_fpga.py --config configs/hw.yaml")
    return shared, cases


# ------------------------------------------------------------------------------- đo

def timeit(fn, repeat: int) -> list[float]:
    fn()                                                   # chạy nóng (cache, cấp phát)
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def ptr(a: np.ndarray, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def build_c(cc: str | None, cflags: str, work: Path, log):
    cc = cc or next((c for c in ("gcc", "clang", "cc") if shutil.which(c)), None)
    if not cc or not shutil.which(cc):
        log.warning("Không tìm thấy gcc/clang → bỏ qua bản C. Windows: cài MinGW-w64 (MSYS2) rồi chạy lại "
                    "với --cc gcc, hoặc chạy script trong WSL.")
        return None, None
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    lib = work / f"hs_core{ext}"
    cmd = [cc, *cflags.split(), "-shared", "-o", str(lib), str(C_SRC)]
    if sys.platform != "win32":
        cmd.insert(1, "-fPIC")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning(f"Biên dịch C lỗi → bỏ qua bản C:\n{' '.join(cmd)}\n{r.stderr}")
        return None, None
    ver = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()
    L = ctypes.CDLL(str(lib))
    i16, u16, i32, f32, u8 = (ctypes.POINTER(t) for t in
                              (ctypes.c_int16, ctypes.c_uint16, ctypes.c_int32, ctypes.c_float, ctypes.c_uint8))
    L.hs_deriv.argtypes = [u8, u8, i16, i16, i16, i16, i16]
    L.hs_iter.argtypes = [i16] * 7 + [ctypes.c_int32]
    L.hs_iter.restype = ctypes.c_int32
    L.hs_feat.argtypes = [i16, i16, u16, ctypes.c_int32, i16, u16, ctypes.c_int32, i32, i16, i16, i16]
    L.hsf_deriv.argtypes = [u8, u8, f32, ctypes.c_float]
    L.hsf_iter.argtypes = [f32, f32, f32, ctypes.c_int32]
    return L, f"{cc} {cflags} ({ver[0] if ver else '?'})"


def bench_case(cs, sh, cfg, L, repeat: int):
    n_iter = sh["n_iter"]
    p = HQ.DEFAULT if n_iter == HQ.DEFAULT.n_iter else HQ.HSParams(n_iter=n_iter)
    sc = HQ.ScalerQ(sh["mean"].astype(np.int64), sh["inv"].astype(np.int64), sh["inv_frac"])
    lut = sh["lut"].astype(np.int64)
    i1, i2 = cs["i1"], cs["i2"]
    res, chk = {}, {}

    # (1) Python số thực – code pipeline train (features/optical_flow.py + grid_flow.py)
    from ocsvm_fpga.features import optical_flow as OF
    from ocsvm_fpga.features.grid_flow import compress, grid_flow_descriptor, grid_layout
    f1, f2 = i1.astype(np.float32), i2.astype(np.float32)
    ye, xe = grid_layout(cfg.features, H, W)
    t_d = timeit(lambda: OF.image_derivatives(f1, f2), repeat)
    t_all = timeit(lambda: OF.horn_schunck(f1, f2, 2.0, n_iter), repeat)
    uf, vf = OF.horn_schunck(f1, f2, 2.0, n_iter)
    t_f = timeit(lambda: compress(grid_flow_descriptor(uf, vf, stats="basic", ye=ye, xe=xe), 0.5), repeat)
    res["py_float"] = {"deriv": np.median(t_d), "iter": np.median(t_all) - np.median(t_d), "feat": np.median(t_f)}

    # (2) Python số nguyên – Golden Model
    t_d = timeit(lambda: HQ.coeffs(*HQ.derivatives4(i1, i2)[:2], p), repeat)
    t_all = timeit(lambda: HQ.horn_schunck_q(i1, i2, p), repeat)
    tr = HQ.horn_schunck_q(i1, i2, p)

    def feat_py():
        m = HQ.magnitude(tr.u, tr.v, p)
        lq = HQ.log_compress_q(HQ.cell_means_q(HQ.cell_sums(tr.u, tr.v, m), p), lut, p)
        return HQ.standardize_q(lq, sc, p)
    t_f = timeit(feat_py, repeat)
    res["py_int"] = {"deriv": np.median(t_d), "iter": np.median(t_all) - np.median(t_d), "feat": np.median(t_f)}
    chk["py_int"] = bool(np.array_equal(tr.u.ravel(), cs["u"]) and np.array_equal(tr.v.ravel(), cs["v"])
                         and np.array_equal(feat_py(), cs["z"]))

    if L is not None:
        a8, b8 = np.ascontiguousarray(i1).ravel(), np.ascontiguousarray(i2).ravel()
        u8 = ctypes.c_uint8
        # (3) C float32
        ws = np.zeros(4 * N, np.float32)
        uc, vc = np.zeros(N, np.float32), np.zeros(N, np.float32)
        t_d = timeit(lambda: L.hsf_deriv(ptr(a8, u8), ptr(b8, u8), ptr(ws, ctypes.c_float), 2.0), repeat)
        t_i = timeit(lambda: L.hsf_iter(ptr(ws, ctypes.c_float), ptr(uc, ctypes.c_float),
                                        ptr(vc, ctypes.c_float), n_iter), repeat)
        res["c_float"] = {"deriv": np.median(t_d), "iter": np.median(t_i), "feat": float("nan")}
        chk["c_float_vs_py_float_px"] = float(max(np.abs(uc.reshape(H, W) - uf).max(),
                                                  np.abs(vc.reshape(H, W) - vf).max()))
        # (4) C số nguyên
        d = {k: np.zeros(N, np.int16) for k in ("ix", "iy", "it", "a", "b", "u", "v")}
        sums, mq, lq, z = np.zeros(48, np.int32), np.zeros(48, np.int16), np.zeros(48, np.int16), np.zeros(48, np.int16)
        i16 = ctypes.c_int16

        def c_deriv():
            L.hs_deriv(ptr(a8, u8), ptr(b8, u8), *(ptr(d[k], i16) for k in ("ix", "iy", "it", "a", "b")))

        def c_iter():
            return L.hs_iter(*(ptr(d[k], i16) for k in ("ix", "iy", "it", "a", "b", "u", "v")), n_iter)

        def c_feat():
            L.hs_feat(ptr(d["u"], i16), ptr(d["v"], i16), ptr(sh["lut"], ctypes.c_uint16), len(sh["lut"]),
                      ptr(sh["mean"], i16), ptr(sh["inv"], ctypes.c_uint16), sh["inv_frac"],
                      ptr(sums, ctypes.c_int32), ptr(mq, i16), ptr(lq, i16), ptr(z, i16))
        t_d, t_i = timeit(c_deriv, repeat), timeit(c_iter, repeat)
        t_f = timeit(c_feat, repeat)
        res["c_int"] = {"deriv": np.median(t_d), "iter": np.median(t_i), "feat": np.median(t_f)}
        chk["c_int"] = bool(np.array_equal(d["u"], cs["u"]) and np.array_equal(d["v"], cs["v"])
                            and np.array_equal(z, cs["z"]))
    for r in res.values():
        r["total"] = r["deriv"] + r["iter"] + (0.0 if np.isnan(r["feat"]) else r["feat"])
    return res, chk


# ------------------------------------------------------------------------------- báo cáo

NAMES = {"py_float": "Python số thực (pipeline train cũ)", "py_int": "Python số nguyên (Golden Model)",
         "c_float": "C float32, 1 luồng", "c_int": "C số nguyên, 1 luồng (khớp bit RTL)"}


def system_info(compiler):
    info = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "compiler": compiler}
    try:
        import cv2
        info["opencv"] = cv2.__version__
        info["opencv_threads"] = cv2.getNumThreads()
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        f = psutil.cpu_freq()
        info["cpu_mhz_max"] = f.max if f else None
        info["cores"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        pass
    return info


def report(all_res, checks, sh, info, repeat, run_dir: Path):
    n_iter = sh["n_iter"]
    fpga = {"deriv": FPGA_MS["deriv"], "iter": FPGA_MS["iter_per"] * n_iter, "feat": FPGA_MS["feat"]}
    fpga["total"] = sum(fpga.values())
    agg = {}
    for k in NAMES:
        rows = [r[k] for r in all_res if k in r]
        if rows:
            agg[k] = {s: float(np.median([r[s] for r in rows])) for s in ("deriv", "iter", "feat", "total")}
    L = ["# Thời gian khối Horn–Schunck trên CPU (mốc so sánh cho FPGA)", "",
         f"- Máy: {info['processor']} · {info['platform']}",
         f"- Python {info['python']}, numpy {info['numpy']}" + (f", OpenCV {info['opencv']}" if "opencv" in info else ""),
         f"- Trình biên dịch C: {info['compiler'] or 'không có (bỏ qua bản C)'}",
         f"- {len(all_res)} ca kiểm thử RTL (ảnh 160×120 đã Gauss), N = {n_iter} vòng; trung vị của {repeat} lần "
         "chạy mỗi ca, rồi trung vị qua các ca. Mọi bản chạy 1 luồng.", "",
         "| Cài đặt | DERIV (ms) | HS_ITER (ms) | 1 vòng (ms) | FEAT (ms) | Tổng (ms) | FPGA nhanh hơn |",
         "|---|---|---|---|---|---|---|"]
    for k, r in agg.items():
        feat = "–" if np.isnan(r["feat"]) else f"{r['feat']:.3f}"
        L.append(f"| {NAMES[k]} | {r['deriv']:.3f} | {r['iter']:.2f} | {r['iter'] / n_iter:.3f} | {feat} | "
                 f"**{r['total']:.2f}** | {r['total'] / fpga['total']:.1f}× |")
    L.append(f"| **FPGA 100 MHz (ước lượng thiết kế)** | {fpga['deriv']:.2f} | {fpga['iter']:.1f} | "
             f"{FPGA_MS['iter_per']:.2f} | {fpga['feat']:.2f} | **{fpga['total']:.1f}** | 1× |")
    ok_py = all(c.get("py_int") for c in checks)
    ok_c = [c.get("c_int") for c in checks if "c_int" in c]
    L += ["", "## Kiểm chứng", "",
          f"- Python số nguyên khớp uv_final.mem và feat_z.mem: **{'đạt' if ok_py else 'KHÔNG ĐẠT'}** "
          f"({sum(bool(c.get('py_int')) for c in checks)}/{len(checks)} ca)."]
    if ok_c:
        L.append(f"- C số nguyên khớp từng bit: **{'đạt' if all(ok_c) else 'KHÔNG ĐẠT'}** ({sum(ok_c)}/{len(ok_c)} ca).")
        dmax = max(c["c_float_vs_py_float_px"] for c in checks if "c_float_vs_py_float_px" in c)
        L.append(f"- C float32 so với Python số thực: lệch u, v tối đa {dmax:.2e} px (chỉ sai số làm tròn float).")
    L += ["", "## Ghi chú", "",
          "- Cột FPGA là ước lượng từ thiết kế (2 chu kỳ/điểm do băng thông SDRAM, + refresh/đổi hàng); thay bằng "
          "số chu kỳ đo trong mô phỏng RTL / trên board khi có.",
          "- Bản C là mốc công bằng nhất cho CPU: cùng phép tính số nguyên với RTL, biên dịch tối ưu, không tính "
          "thời gian đọc ảnh.",
          "- Bản ESP32-S3 (firmware/esp32_hs_bench) đo cùng hs_core.c trên vi điều khiển – mốc cho hệ nhúng.", ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    save_json({"system": info, "n_iter": n_iter, "repeat": repeat, "fpga_ms_estimate": fpga, "median": agg,
               "per_case": [{k: {s: float(x) for s, x in v.items()} for k, v in r.items()} for r in all_res],
               "checks": checks}, run_dir / "bench.json")
    return "\n".join(L)


# ------------------------------------------------------------------------------- ESP32

def c_array(name: str, ctype: str, a, per_line: int = 24) -> str:
    vals = [str(int(x)) for x in np.ravel(a)]
    body = ",\n".join("    " + ", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))
    return f"static const {ctype} {name}[{len(vals)}] = {{\n{body}\n}};\n"


def write_esp32_header(path: Path, sh, cs, log):
    h = ["/* test_vectors.h - sinh tu dong boi scripts/23_hs_cpu_bench.py, KHONG sua tay.",
         f" * Ca {cs['name']} cua outputs_hw/fpga/hs (giong testbench RTL). */",
         "#pragma once", "#include <stdint.h>", "",
         f'#define TV_CASE "{cs["name"]}"', f"#define TV_N_ITER {sh['n_iter']}",
         f"#define TV_INV_FRAC {sh['inv_frac']}", f"#define TV_LUT_N {len(sh['lut'])}", "",
         c_array("TV_IMG1", "uint8_t", cs["i1"]), c_array("TV_IMG2", "uint8_t", cs["i2"]),
         c_array("TV_U", "int16_t", cs["u"]), c_array("TV_V", "int16_t", cs["v"]),
         c_array("TV_Z", "int16_t", cs["z"]), c_array("TV_LUT", "uint16_t", sh["lut"]),
         c_array("TV_MEAN", "int16_t", sh["mean"]), c_array("TV_INV", "uint16_t", sh["inv"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(h), encoding="ascii")
    log.info(f"→ {path} (ca {cs['name']}, N = {sh['n_iter']}, INV_FRAC = {sh['inv_frac']})")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--cases", default=None, help="thư mục véc-tơ (mặc định <output_dir>/fpga/hs)")
    ap.add_argument("--repeat", type=int, default=10, help="số lần đo mỗi tầng mỗi ca")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--cc", default=None, help="trình biên dịch C (mặc định gcc → clang → cc)")
    ap.add_argument("--cflags", default="-O2", help="cờ biên dịch (mặc định -O2, giống firmware)")
    ap.add_argument("--esp32-header", default=None, help="chỉ sinh test_vectors.h cho firmware ESP32 rồi thoát")
    ap.add_argument("--case-index", type=int, default=1, help="ca dùng cho ESP32 (mặc định 1: chuyển động lớn)")
    args = ap.parse_args()
    cfg, out, log, _ = setup(args, "23_hs_cpu_bench")
    d = Path(args.cases) if args.cases else out / "fpga" / "hs"
    sh, cases = load_cases(d)
    if args.esp32_header:
        write_esp32_header(Path(args.esp32_header), sh, cases[min(args.case_index, len(cases) - 1)], log)
        return
    if args.max_cases:
        cases = cases[:args.max_cases]
    run_dir = out / "hs_cpu_bench" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    L, compiler = build_c(args.cc, args.cflags, run_dir, log)
    log.info(f"{len(cases)} ca từ {d}, N = {sh['n_iter']} vòng, {args.repeat} lần/tầng …")
    all_res, checks = [], []
    for cs in cases:
        r, c = bench_case(cs, sh, cfg, L, args.repeat)
        all_res.append(r)
        checks.append(c)
        log.info(f"  {cs['name']}: " + " · ".join(f"{k} {v['total']:.1f} ms" for k, v in r.items())
                 + f" · khớp bit: py {c.get('py_int')}, C {c.get('c_int', '–')}")
    txt = report(all_res, checks, sh, system_info(compiler), args.repeat, run_dir)
    print(txt)
    log.info(f"→ {run_dir}")


if __name__ == "__main__":
    main()
