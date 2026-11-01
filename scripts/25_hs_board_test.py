#!/usr/bin/env python3
"""CHẠY THỬ KHỐI HS + OC-SVM TRÊN BOARD TANG NANO 20K (top rtl/src/hs/hs_board_test.v).

Board tự sinh ảnh thử bằng công thức (không cần camera, SDRAM), chạy DERIV → 64 vòng HS → FEAT
→ OC-SVM liên tục và gửi mỗi lần một dòng qua UART 115 200 baud:
    HS N=0001 C=0028A3F1 F=B1E9C1AF A=1 S=00000000 Z=1A2B D1
Script này: (1) tính dòng MONG ĐỢI bằng Golden Model; (2) dựng thư mục project Gowin;
(3) đọc cổng COM và so từng dòng.

    python scripts/25_hs_board_test.py --config configs/hw.yaml                       # in dòng mong đợi
    python scripts/25_hs_board_test.py --config configs/hw.yaml --gowin fpga_gowin/hs_board_test
    python scripts/25_hs_board_test.py --config configs/hw.yaml --port COM5           # đọc board và so

Trước đó: python scripts/15_export_rtl.py --config configs/hw.yaml --lanes 8   (rtl/build: tham số OC-SVM)
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, load_detector_checked, setup  # noqa: E402

DX, DY = 1, 0                                        # khớp tham số hs_pixgen
CLK_HZ = 27_000_000


def pattern(dx: int = 0, dy: int = 0) -> np.ndarray:
    """A(x, y) = ((5x XOR 3y) + (x·y >> 3)) mod 256 – giống hệt hs_pixgen.v."""
    y, x = np.mgrid[0:120, 0:160].astype(np.int64)
    x, y = x + dx, y + dy
    return ((((5 * x) ^ (3 * y)) + ((x * y) >> 3)) & 0xFF).astype(np.uint8)


def vh_int(path: Path, name: str):
    m = re.search(rf"localparam\s+integer\s+{name}\s*=\s*(-?\d+)", path.read_text(encoding="ascii"))
    return int(m.group(1)) if m else None


def expected(cfg, out: Path, hs_dir: Path, log) -> dict:
    from ocsvm_fpga.hwmodel import hs_export as HX, hs_q as HQ
    from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM
    vh = hs_dir / "hs_params.vh"
    n_iter, inv_frac = vh_int(vh, "N_ITER"), vh_int(vh, "INV_FRAC")
    p = HQ.DEFAULT if n_iter == HQ.DEFAULT.n_iter else HQ.HSParams(n_iter=n_iter)
    sc = HQ.ScalerQ(HX.read_mem(hs_dir / "mem" / "scaler_mean_q.mem", 16),
                    HX.read_mem(hs_dir / "mem" / "scaler_inv_q.mem", 16, signed=False), inv_frac)
    lut = HX.read_mem(hs_dir / "mem" / "log_lut.mem", 12, signed=False)
    i1, i2 = pattern(), pattern(DX, DY)
    tr = HQ.features_q(i1, i2, p, sc, lut)
    det = load_detector_checked(cfg, out)
    fx = FixedPointOCSVM(det, cfg.fpga)
    fqa = fx.decision_q(np.atleast_2d(tr.z_q))
    fq, an = int(fqa[0]), int(fx.anomaly_q(fqa)[0])
    hs = tr.hs
    pk = HX.pack([(hs.ix4, 11), (hs.iy4, 11), (hs.it4, 11), (hs.a, p.coef_w), (hs.b, p.coef_w)])
    xr = 0
    for w in pk:
        xr ^= int(w)
    e = {"F": fq & 0xFFFFFFFF, "A": an, "S": int(hs.n_sat), "Z": int(np.sum(tr.z_q)) & 0xFFFF,
         "D": bin(xr).count("1") & 1, "f_q": fq, "n_iter": n_iter,
         "zq": [int(v) for v in np.asarray(tr.z_q).reshape(-1)]}
    e["line"] = f"HS N=xxxx C=(đo trên board) F={e['F']:08X} A={e['A']} S={e['S']:08X} Z={e['Z']:04X} D{e['D']}"
    return e


def parse(line: str) -> dict | None:
    m = re.match(r"HS N=([0-9A-F]{4}) C=([0-9A-F]{8}) F=([0-9A-F]{8}) A=([01]) S=([0-9A-F]{8}) Z=([0-9A-F]{4}) D([01])",
                 line.strip())
    if not m:
        return None
    n, c, f, a, s, z, d = m.groups()
    return {"N": int(n, 16), "C": int(c, 16), "F": int(f, 16), "A": int(a), "S": int(s, 16), "Z": int(z, 16), "D": int(d)}


# ------------------------------------------------------------------------------- Gowin

CST = """// Tang Nano 20K (GW2AR-LV18QN88C8/I7) – top hs_board_top
IO_LOC  "clk" 4;                 // thạch anh 27 MHz
IO_PORT "clk" IO_TYPE=LVCMOS33 PULL_MODE=UP;
IO_LOC  "uart_tx" 69;            // FPGA → BL616 → cổng COM trên PC (kiểm tra theo schematic)
IO_PORT "uart_tx" IO_TYPE=LVCMOS33 PULL_MODE=UP;
IO_LOC  "led_busy_n" 15;         // LED0: đảo mỗi lần chạy xong
IO_PORT "led_busy_n" IO_TYPE=LVCMOS33 PULL_MODE=UP DRIVE=8;
IO_LOC  "led_anomaly_n" 16;      // LED1: sáng khi cờ bất thường = 1
IO_PORT "led_anomaly_n" IO_TYPE=LVCMOS33 PULL_MODE=UP DRIVE=8;
"""
SDC = """// 27 MHz (thạch anh). Muốn xem có đạt 100 MHz không: đổi period thành 10 rồi Run P&R, đọc Timing Report.
create_clock -name clk -period 37.037 [get_ports {clk}]
"""


def gowin_project(dst: Path, hs_dir: Path, build: Path, log):
    src = dst / "src"
    src.mkdir(parents=True, exist_ok=True)
    rtl = ROOT / "rtl" / "src"
    files = sorted((rtl / "hs").glob("*.v")) + [rtl / "ocsvm_core.v", rtl / "uart_tx.v"]
    for f in files:
        shutil.copy(f, src / f.name)
    shutil.copy(hs_dir / "hs_params.vh", src / "hs_params.vh")
    shutil.copy(build / "ocsvm_params.vh", src / "ocsvm_params.vh")
    mem = {}
    for f in ("log_lut.mem", "scaler_mean_q.mem", "scaler_inv_q.mem"):
        shutil.copy(hs_dir / "mem" / f, src / f)
        mem[f] = (src / f).resolve().as_posix()
    for f in ("sv_lanes.mem", "alpha.mem", "exp_lut.mem"):
        shutil.copy(build / f, src / f)
        mem[f] = (src / f).resolve().as_posix()
    top = f"""// hs_board_top.v – sinh bởi scripts/25_hs_board_test.py: top Gowin, đường dẫn .mem tuyệt đối
module hs_board_top (input wire clk, output wire uart_tx, output wire led_busy_n, output wire led_anomaly_n);
    hs_board_test #(
        .LUT_FILE("{mem['log_lut.mem']}"), .MEAN_FILE("{mem['scaler_mean_q.mem']}"),
        .INV_FILE("{mem['scaler_inv_q.mem']}"), .SV_FILE("{mem['sv_lanes.mem']}"),
        .ALPHA_FILE("{mem['alpha.mem']}"), .EXP_FILE("{mem['exp_lut.mem']}")
    ) u (.clk(clk), .uart_tx(uart_tx), .led_busy_n(led_busy_n), .led_anomaly_n(led_anomaly_n));
endmodule
"""
    (src / "hs_board_top.v").write_text(top, encoding="utf-8")
    (src / "hs_board_test.cst").write_text(CST, encoding="utf-8")
    (src / "hs_board_test.sdc").write_text(SDC, encoding="utf-8")
    vfiles = ["hs_board_top.v"] + [f.name for f in files]
    lines = "\n".join(f'        <File path="src/{n}" type="file.verilog" enable="1"/>' for n in vfiles)
    gprj = f"""<?xml version="1" encoding="UTF-8"?>
<!DOCTYPE gowin-fpga-project>
<Project>
    <Template>FPGA</Template>
    <Version>5</Version>
    <Device name="GW2AR-18C" pn="GW2AR-LV18QN88C8/I7">gw2ar18c-000</Device>
    <FileList>
{lines}
        <File path="src/hs_board_test.cst" type="file.cst" enable="1"/>
        <File path="src/hs_board_test.sdc" type="file.sdc" enable="1"/>
    </FileList>
</Project>
"""
    (dst / "hs_board_test.gprj").write_text(gprj, encoding="utf-8")
    log.info(f"Project Gowin → {dst} (mở hs_board_test.gprj; top = hs_board_top; include path = src)")


STAT = ["u", "v", "|v|"]


def check_zv(raw: str, exp: dict, log):
    """Dòng gỡ lỗi 'ZV W=30 z0 … z47': so từng giá trị z mà OC-SVM nhận với Golden Model."""
    m = re.match(r"ZV W=([0-9A-F]{2}) ([0-9A-F]{192})", raw.strip())
    if not m:
        log.info(f"  (ZV lỗi định dạng) {raw[:80]}")
        return
    w = int(m.group(1), 16)
    hx = m.group(2)
    z = [int(hx[4 * i:4 * i + 4], 16) for i in range(48)]
    z = [v - 65536 if v >= 32768 else v for v in z]
    bad = [i for i in range(48) if z[i] != exp["zq"][i]]
    log.info(f"  ZV: số lần ghi x_we = {w} (phải 48) · {48 - len(bad)}/48 giá trị z khớp Golden")
    for i in bad[:16]:
        log.info(f"     z[{i:2d}] ô {i // 3:2d} ({STAT[i % 3]:>3}): board {z[i]:6d} | Golden {exp['zq'][i]:6d} | "
                 f"lệch {z[i] - exp['zq'][i]:+d}")
    if bad:
        cols = sorted({i % 3 for i in bad})
        log.info("     → các giá trị sai thuộc thống kê: " + ", ".join(STAT[c] for c in cols)
                 + f" · {len(bad)} giá trị · tổng lệch {sum(z[i] - exp['zq'][i] for i in bad):+d}")


def listen(port: str, exp: dict, n_lines: int, log):
    try:
        import serial
    except ImportError:
        sys.exit("Cần pyserial: pip install pyserial")
    ok_all = True
    with serial.Serial(port, 115200, timeout=5) as s:
        s.reset_input_buffer()
        got = 0
        t_end = time.time() + 30 + n_lines * 2
        while got < n_lines and time.time() < t_end:
            raw = s.readline().decode("ascii", "replace").strip()
            if not raw:
                continue
            if raw.startswith("ZV "):
                check_zv(raw, exp, log)
                continue
            r = parse(raw)
            if r is None:
                log.info(f"  (bỏ qua) {raw}")
                continue
            got += 1
            bad = [k for k in ("F", "A", "S", "Z", "D") if r[k] != exp[k]]
            ok_all &= not bad
            ms = r["C"] / CLK_HZ * 1e3
            log.info(f"  {raw}  → {'ĐÚNG' if not bad else 'SAI ' + ','.join(bad)} · HS {r['C']} chu kỳ = "
                     f"{ms:.1f} ms ở 27 MHz = {r['C'] / 1e5:.2f} ms nếu chạy 100 MHz")
    if got == 0:
        sys.exit("Không nhận được dòng nào: kiểm tra cổng COM, baud 115200, chân uart_tx trong .cst")
    print("KẾT QUẢ:", "board KHỚP Golden Model" if ok_all else "CÓ DÒNG SAI – xem log")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--cases", default=None, help="thư mục hs (mặc định <output_dir>/fpga/hs)")
    ap.add_argument("--build", default="rtl/build", help="thư mục tham số OC-SVM (15_export_rtl.py)")
    ap.add_argument("--gowin", default=None, help="dựng thư mục project Gowin tại đây")
    ap.add_argument("--port", default=None, help="cổng COM của board (vd COM5, /dev/ttyUSB1)")
    ap.add_argument("--lines", type=int, default=5)
    args = ap.parse_args()
    cfg, out, log, _ = setup(args, "25_hs_board_test")
    hs_dir = (Path(args.cases) if args.cases else out / "fpga" / "hs").resolve()
    build = (ROOT / args.build).resolve() if not Path(args.build).is_absolute() else Path(args.build)
    if not (build / "ocsvm_params.vh").exists():
        sys.exit(f"Thiếu {build}/ocsvm_params.vh – chạy python scripts/15_export_rtl.py --config configs/hw.yaml --lanes 8")
    exp = expected(cfg, out, hs_dir, log)
    print(f"Dòng mong đợi (ảnh thử hs_pixgen, N = {exp['n_iter']} vòng, f_q = {exp['f_q']}):\n  {exp['line']}")
    if args.gowin:
        gowin_project(Path(args.gowin), hs_dir, build, log)
    if args.port:
        listen(args.port, exp, args.lines, log)


if __name__ == "__main__":
    main()
