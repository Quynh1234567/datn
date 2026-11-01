#!/usr/bin/env python3
"""MÔ PHỎNG RTL KHỐI HORN–SCHUNCK (Icarus Verilog) – so từng tầng với véc-tơ Golden Model.

Testbench (rtl/sim/hs/):
  deriv  : hs_deriv      img1/img2  → deriv.mem
  iter1  : hs_iter N=1   deriv.mem  → uv_iter1.mem            (nhanh, gỡ lỗi)
  feat   : hs_feat       uv_final   → mag, cell_sums, feat_mean/log/z
  iter64 : hs_iter N=64  deriv.mem  → uv_final.mem            (~40 s / ca)
  engine : hs_engine + ocsvm_core   img1/img2 → z, f_q, cờ    (~5 phút / ca, in số chu kỳ thật)

    python scripts/24_hs_rtl_sim.py --config configs/hw.yaml                 # deriv, iter1, feat × 8 ca
    python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --full          # + iter64, engine
    python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --stress        # + ca khó (bão hòa, a = ±1/16, …)
    python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --tb engine --max-cases 1
    python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --sim xsim       # dùng Vivado xsim thay Icarus

Trước khi chạy engine: python scripts/15_export_rtl.py --config configs/hw.yaml --lanes 8
(rtl/build/ phải chứa ocsvm_params.vh, sv_lanes.mem, alpha.mem, exp_lut.mem của CÙNG mô hình).
Kết quả: outputs_hw/rtl_sim/<thời gian>/report.md
"""
from __future__ import annotations

import os
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

SRC, SIM = ROOT / "rtl" / "src", ROOT / "rtl" / "sim" / "hs"
HS_SRC = [SRC / "hs" / f for f in ("hs_div_pipe.v", "hs_deriv.v", "hs_uv_ram.v", "hs_iter.v",
                                   "hs_isqrt_pipe.v", "hs_feat.v", "hs_engine.v")]
TBS = {
    "deriv":  {"src": ["hs_div_pipe.v", "hs_deriv.v"], "tb": "tb_hs_deriv.v", "P": {}},
    "iter1":  {"src": ["hs_uv_ram.v", "hs_iter.v"], "tb": "tb_hs_iter.v", "P": {"tb_hs_iter.NIT": 1}},
    "feat":   {"src": ["hs_isqrt_pipe.v", "hs_feat.v"], "tb": "tb_hs_feat.v", "P": {}},
    "iter64": {"src": ["hs_uv_ram.v", "hs_iter.v"], "tb": "tb_hs_iter.v", "P": {"tb_hs_iter.NIT": None}},
    "engine": {"src": None, "tb": "tb_hs_engine.v", "P": {"tb_hs_engine.NIT": None}, "svm": True},
    # hệ thống demo 1: SPI → SDRAM (mô hình) → HS → OC-SVM → UART (~30 phút / ca với Icarus)
    "system": {"src": "SYS", "tb": "../sys/tb_hs_system.v", "P": {}, "svm": True},
}
SYS_SRC = [SRC / "hs" / f for f in ("hs_div_pipe.v", "hs_deriv.v", "hs_uv_ram.v", "hs_iter.v", "hs_isqrt_pipe.v",
                                    "hs_feat.v")] + sorted((SRC / "sys").glob("*.v")) + \
    [SRC / "sys" / "third_party" / "sdram.v", SRC / "uart_tx.v",
     ROOT / "rtl" / "sim" / "sys" / "gowin_rpll_stub.v", ROOT / "rtl" / "sim" / "sys" / "sdram_model.v"]
QUICK, FULL = ["deriv", "iter1", "feat"], ["deriv", "iter1", "feat", "iter64", "engine"]


def vh_int(path: Path, name: str):
    m = re.search(rf"localparam\s+integer\s+{name}\s*=\s*(-?\d+)", path.read_text(encoding="ascii"))
    return int(m.group(1)) if m else None


def _cmd(exe: str) -> list[str]:
    """Windows: các lệnh Vivado là file .bat → chạy qua cmd /c."""
    return ["cmd", "/c", exe] if sys.platform == "win32" and exe.lower().endswith((".bat", ".cmd")) else [exe]


def _win(cmd):
    """Windows + file .bat: cmd.exe coi dấu "=" (và , ;) là dấu phân cách tham số, nên "NIT=1" tới xelab thành
    "NIT 1", "CASE=D:/..." tới xsim thành "CASE D:/...". Tự ghép dòng lệnh và đặt các tham số đó trong ngoặc kép."""
    if not (sys.platform == "win32" and isinstance(cmd, list) and cmd[:2] == ["cmd", "/c"]):
        return cmd
    special = set(' =,;&()^"')

    def q(a):
        a = str(a)
        return a if (a.startswith('"') or not (set(a) & special)) else f'"{a}"'
    return " ".join(q(a) for a in cmd)


def find_vivado(name: str, given: str | None) -> list[str]:
    if given:
        # Windows: thư mục bin của Vivado có CẢ "xvlog" (script Linux) lẫn "xvlog.bat" → phải ưu tiên .bat,
        # chạy nhầm file không đuôi sẽ báo WinError 193
        win = sys.platform == "win32"
        names = [f"{name}.bat", name] if win else [name, f"{name}.bat"]
        for d in (Path(given), Path(given) / "bin"):
            for nm in names:
                cand = d / nm
                if cand.exists():
                    return _cmd(str(cand))
    exe = shutil.which(f"{name}.bat") if sys.platform == "win32" else None
    exe = exe or shutil.which(name)
    if exe:
        return _cmd(exe)
    for root in (Path(r"C:\Xilinx\Vivado"), Path(r"C:\AMDDesignTools"), Path(r"D:\AMDDesignTools"),
                 Path("/tools/Xilinx/Vivado")):
        hits = sorted(root.glob(f"*/bin/{name}.bat")) + sorted(root.glob(f"*/Vivado/bin/{name}.bat")) \
            + sorted(root.glob(f"*/bin/{name}"))
        if hits:
            return _cmd(str(hits[-1]))
    sys.exit(f"Không tìm thấy {name} của Vivado. Mở 'Vivado Tcl Shell' / chạy settings64.bat, "
             "hoặc chỉ thư mục bằng --vivado C:\\Xilinx\\Vivado\\2024.2\\bin")


def find_tool(name: str, given: str | None) -> str:
    if given:
        p = Path(given)
        cand = p / name if p.is_dir() else p.with_name(name)
        if cand.exists() or cand.with_suffix(".exe").exists():
            return str(cand)
    exe = shutil.which(name)
    if exe:
        return exe
    for d in (r"C:\iverilog\bin", r"C:\Program Files\iverilog\bin"):
        if (Path(d) / f"{name}.exe").exists():
            return str(Path(d) / f"{name}.exe")
    sys.exit(f"Không tìm thấy {name}. Cài Icarus Verilog (Windows: bleyer.org/icarus, tick 'Add to PATH') "
             "hoặc chỉ đường dẫn bằng --iverilog C:\\iverilog\\bin")


# ------------------------------------------------------------------------------- ca khó

def make_stress(hs_dir: Path, out_dir: Path, log):
    """Ca khó cho RTL (không cần mô hình): độ dốc cực đại, a = ±1/16, chuyển động lớn, bão hòa u/v."""
    from ocsvm_fpga.hwmodel import hs_export as HX, hs_q as HQ
    vh = hs_dir / "hs_params.vh"
    n_iter, inv_frac = vh_int(vh, "N_ITER"), vh_int(vh, "INV_FRAC")
    sc = HQ.ScalerQ(HX.read_mem(hs_dir / "mem" / "scaler_mean_q.mem", 16),
                    HX.read_mem(hs_dir / "mem" / "scaler_inv_q.mem", 16, signed=False), inv_frac)
    p = HQ.DEFAULT if n_iter == HQ.DEFAULT.n_iter else HQ.HSParams(n_iter=n_iter)
    H, W = HQ.H, HQ.W
    rng = np.random.default_rng(7)
    x = np.arange(W)
    ramp = np.where(x < W // 2, 2 * x, 2 * (W - 1 - x)).astype(np.uint8)       # Ix4 = ±8, Iy4 = 0 → a = ±16 384
    ramp = np.tile(ramp, (H, 1))
    stripes = np.tile(np.where(x % 2 == 0, 0, 255).astype(np.uint8), (H, 1))   # |Ix4| = 1020
    import cv2
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (H + 80, W + 80)).astype(np.float32), (0, 0), 2)
    tex = np.clip((tex - tex.mean()) * 4 + 128, 0, 255).astype(np.uint8)
    pairs = [(ramp, ramp), (stripes, np.roll(stripes, 1, axis=1)),
             (tex[20:20 + H, 20:20 + W], tex[20:20 + H, 40:40 + W]),               # dịch 20 px
             (rng.integers(0, 256, (H, W), dtype=np.uint8), rng.integers(0, 256, (H, W), dtype=np.uint8)),
             (np.zeros((H, W), np.uint8), np.full((H, W), 255, np.uint8))]
    names = ["ramp: a = ±1/16 (bão hòa hệ số)", "sọc 0/255 dịch 1 px: |Ix4| = 1020",
             "kết cấu dịch 20 px", "hai ảnh nhiễu độc lập: |N| lớn, bão hòa u/v", "đen → trắng: It4 = 1020"]
    rows = HX.export_all(pairs, out_dir, p, scaler=sc, ocsvm=None, names=names)
    log.info(f"Ca khó → {out_dir}: " + ", ".join(f"{r['case']:02d} bão hòa {r['n_sat']}" for r in rows))
    return out_dir


# ------------------------------------------------------------------------------- chạy

def compile_tb(sim: dict, name: str, hs_dir: Path, build: Path, work: Path, n_iter: int, log):
    t = TBS[name]
    srcs = HS_SRC if t["src"] is None else (SYS_SRC if t["src"] == "SYS" else [SRC / "hs" / f for f in t["src"]])
    if t.get("svm"):
        srcs = list(srcs) + [SRC / "ocsvm_core.v"]
    if sim["kind"] == "xsim":                                     # Vivado: xvlog → xelab → snapshot
        inc = ["-i", str(hs_dir)] + (["-i", str(build)] if t.get("svm") else [])
        c1 = sim["xvlog"] + ["-sv"] + inc + [str(s) for s in srcs] + [str(SIM / t["tb"])]
        top = Path(t["tb"]).stem
        c2 = sim["xelab"] + ["-debug", "off", top, "-s", f"snap_{name}", "-timescale", "1ns/1ps"]
        for k, v in t["P"].items():
            c2 += ["-generic_top", f"{k.split('.')[-1]}={n_iter if v is None else v}"]
        for c in (c1, c2):
            r = subprocess.run(_win(c), capture_output=True, text=True, cwd=work)
            if r.returncode != 0:
                log.error(f"Biên dịch {name} (xsim) lỗi:\n{' '.join(c)}\n{r.stdout[-3000:]}{r.stderr[-2000:]}")
                return None
        return f"snap_{name}"
    iv = sim["iverilog"]
    out = work / f"{name}.vvp"
    cmd = [iv, "-g2012", "-I", str(hs_dir)] + (["-I", str(build)] if t.get("svm") else [])
    for k, v in t["P"].items():
        cmd += ["-P", f"{k}={n_iter if v is None else v}"]
    cmd += ["-o", str(out)] + [str(s) for s in srcs] + [str(SIM / t["tb"])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"Biên dịch {name} lỗi:\n{' '.join(cmd)}\n{r.stdout}{r.stderr}")
        return None
    return out


def run_case(sim: dict, vvp_file, case: Path, work: Path, stall: int, rate: int | None, timeout: int):
    args = [f"CASE={case.resolve().as_posix()}", f"STALL={stall}"] + ([f"RATE={rate}"] if rate else [])
    if sim["kind"] == "xsim":
        cmd = sim["xsim"] + [vvp_file, "-R"]
        for a in args:
            cmd += ["-testplusarg", a]
    else:
        cmd = [sim["vvp"], "-n", str(vvp_file)] + [f"+{a}" for a in args]
    t0 = time.time()
    try:
        r = subprocess.run(_win(cmd), capture_output=True, text=True, cwd=work, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "FAIL", f"quá {timeout} s", time.time() - t0
    line = next((ln for ln in out.splitlines() if ln.startswith(("PASS", "FAIL"))), None)
    if line is None:
        return "FAIL", out.strip().splitlines()[-1] if out.strip() else "không có kết quả", time.time() - t0
    return line.split()[0], line, time.time() - t0


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--cases", default=None, help="thư mục véc-tơ (mặc định <output_dir>/fpga/hs)")
    ap.add_argument("--tb", nargs="*", default=None, help=f"chọn testbench trong {list(TBS)}")
    ap.add_argument("--full", action="store_true", help="chạy cả iter64 và engine (lâu)")
    ap.add_argument("--stress", action="store_true", help="sinh và chạy thêm ca khó")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--stall", type=int, default=30, help="%% chu kỳ chèn chờ ngẫu nhiên (0 = tắt)")
    ap.add_argument("--rate", type=int, default=2, help="chu kỳ / gói DERIV khi đọc lại (2 = như SDRAM)")
    ap.add_argument("--build", default="rtl/build", help="thư mục ocsvm_params.vh + sv_lanes.mem (cho engine)")
    ap.add_argument("--sim", choices=["icarus", "xsim"], default="icarus", help="trình mô phỏng (xsim = Vivado)")
    ap.add_argument("--iverilog", default=None, help="đường dẫn iverilog (hoặc thư mục bin)")
    ap.add_argument("--vivado", default=None, help="thư mục bin của Vivado (khi --sim xsim)")
    ap.add_argument("--timeout", type=int, default=1800, help="giây tối đa mỗi lần chạy")
    args = ap.parse_args()
    cfg, out, log, _ = setup(args, "24_hs_rtl_sim")
    if args.sim == "xsim":
        sim = {"kind": "xsim", "xvlog": find_vivado("xvlog", args.vivado), "xelab": find_vivado("xelab", args.vivado),
               "xsim": find_vivado("xsim", args.vivado)}
        iv = " ".join(sim["xsim"])
    else:
        sim = {"kind": "icarus", "iverilog": find_tool("iverilog", args.iverilog), "vvp": find_tool("vvp", args.iverilog)}
        iv = sim["iverilog"]
    hs_dir = (Path(args.cases) if args.cases else out / "fpga" / "hs").resolve()
    if not (hs_dir / "hs_params.vh").exists():
        sys.exit(f"Thiếu {hs_dir / 'hs_params.vh'} – chạy scripts/04_export_fpga.py --config configs/hw.yaml")
    if vh_int(hs_dir / "hs_params.vh", "UV_FORMAT_CHECK") != 1306:
        sys.exit("hs_params.vh không phải bản Q6.6 (UV_FORMAT_CHECK ≠ 1306) – xuất lại bằng 04_export_fpga.py")
    n_iter = vh_int(hs_dir / "hs_params.vh", "N_ITER")
    names = args.tb or (FULL if args.full else QUICK)
    bad = [n for n in names if n not in TBS]
    if bad:
        sys.exit(f"Testbench không có: {bad}; chọn trong {list(TBS)}")
    build = ((ROOT / args.build) if not Path(args.build).is_absolute() else Path(args.build)).resolve()
    if ({"engine", "system"} & set(names)) and not (build / "ocsvm_params.vh").exists():
        sys.exit(f"Thiếu {build}/ocsvm_params.vh – chạy python scripts/15_export_rtl.py --config configs/hw.yaml --lanes 8")

    run_dir = out / "rtl_sim" / time.strftime("%Y%m%d_%H%M%S")
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    for f in (hs_dir / "mem").glob("*.mem"):                       # $readmemh đọc theo thư mục làm việc
        shutil.copy(f, work / f.name)
    if {"engine", "system"} & set(names):
        for f in ("sv_lanes.mem", "alpha.mem", "exp_lut.mem"):
            shutil.copy(build / f, work / f)

    groups = [("IITB", hs_dir)]
    if args.stress:
        groups.append(("ca khó", make_stress(hs_dir, run_dir / "stress", log)))

    rows, fails = [], 0
    for name in names:
        vvp_file = compile_tb(sim, name, hs_dir, build, work, n_iter, log)
        if vvp_file is None:
            rows.append((name, "-", "FAIL", "lỗi biên dịch", 0.0))
            fails += 1
            continue
        for gname, gdir in groups:
            cases = sorted(gdir.glob("case_*"))
            if args.max_cases:
                cases = cases[:args.max_cases]
            for c in cases:
                if name in ("engine", "system") and gname != "IITB":
                    continue                                       # ca khó không có expected.txt / mô hình
                rate = args.rate if name in ("iter1", "iter64", "engine") else None
                res, line, sec = run_case(sim, vvp_file, c, work, args.stall, rate,
                                          args.timeout * (4 if name == "system" else 1))
                fails += res != "PASS"
                rows.append((name, f"{gname}/{c.name}", res, line, sec))
                log.info(f"  {res}  {name:<6} {gname}/{c.name}  ({sec:.0f} s)" + ("" if res == "PASS" else f"\n      {line}"))

    L = ["# Mô phỏng RTL khối Horn–Schunck", "",
         f"- Véc-tơ: `{hs_dir}` (N_ITER = {n_iter}) · trình mô phỏng {args.sim}: `{iv}`",
         f"- Chèn chờ ngẫu nhiên {args.stall} % · đọc gói DERIV 1 gói / {args.rate} chu kỳ", "",
         "| Testbench | Ca | Kết quả | Thời gian | Chi tiết |", "|---|---|---|---|---|"]
    for name, case, res, line, sec in rows:
        det = re.sub(r"^(PASS|FAIL) \S+ \S+:\s*", "", line).replace("|", "·")
        L.append(f"| {name} | {case} | **{res}** | {sec:.0f} s | {det} |")
    eng = [ln for n, _, r, ln, _ in rows if n == "engine" and r == "PASS"]
    cyc = [int(m.group(1)) for ln in eng if (m := re.search(r"HS (\d+) chu ky", ln))]
    if cyc:
        L += ["", f"**Thời gian khối HS đo trong mô phỏng:** {np.median(cyc):.0f} chu kỳ = "
              f"{np.median(cyc) / 1e5:.2f} ms ở 100 MHz (trung vị {len(cyc)} ca, chưa tính refresh SDRAM)."]
    L += ["", f"**Tổng: {len(rows) - fails}/{len(rows)} PASS**", ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    log.info(f"→ {run_dir / 'report.md'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()