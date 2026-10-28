#!/usr/bin/env python3
"""Chuẩn bị thư mục rtl/build để MÔ PHỎNG và TỔNG HỢP RTL từ đúng mô hình đã xuất.

  • chép ocsvm_params.vh, alpha.mem, exp_lut.mem, test_*.mem từ outputs/fpga
  • sinh sv_lanes.mem cho P làn PE: từ thứ i·G + g chứa phần tử g·P … g·P+P−1 của SV i
    (G = DIM/P, phần tử nhỏ nhất ở bit thấp) – đúng thứ tự ocsvm_core.v đọc
  • sinh run_sim.bat / run_sim.sh (Icarus Verilog)

    python scripts/15_export_rtl.py --lanes 8
    cd rtl/build && run_sim.bat
"""
import argparse
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocsvm_fpga.config import load_config, resolve_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_mem(path: Path, width: int) -> list[int]:
    vals = []
    for ln in path.read_text().splitlines():
        ln = ln.split("//")[0].strip()
        if ln:
            v = int(ln, 16)
            vals.append(v - (1 << width) if v >> (width - 1) else v)
    return vals


def vh_params(path: Path) -> dict:
    out = {}
    for m in re.finditer(r"localparam\s+integer\s+(\w+)\s*=\s*(-?\d+)", path.read_text()):
        out[m.group(1)] = int(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lanes", type=int, default=8, help="số làn PE song song P (DIM phải chia hết cho P)")
    ap.add_argument("--out", default="rtl/build")
    ap.add_argument("--clk-hz", type=int, default=27_000_000, help="xung nhịp board (Tang Nano: 27 MHz)")
    ap.add_argument("--baud", type=int, default=115_200)
    a = ap.parse_args()

    cfg = load_config(a.config)
    fpga = resolve_path(cfg.project.output_dir) / "fpga"
    if not (fpga / "ocsvm_params.vh").exists():
        sys.exit(f"Chưa có {fpga / 'ocsvm_params.vh'} – chạy scripts/04_export_fpga.py trước")
    prm = vh_params(fpga / "ocsvm_params.vh")
    dim, nsv, dw, P = prm["DIM"], prm["N_SV"], prm["DATA_W"], a.lanes
    if dim % P:
        divs = [d for d in range(1, dim + 1) if dim % d == 0]
        sys.exit(f"DIM = {dim} không chia hết cho P = {P}. Chọn P trong {divs}")
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(fpga / "ocsvm_params.vh", out / "ocsvm_params.vh")
    for f in ("alpha.mem", "exp_lut.mem", "test_input.mem", "test_expected_score.mem",
              "test_expected_label.mem", "test_kernel_vec0.mem"):
        if (fpga / "mem" / f).exists():
            shutil.copy(fpga / "mem" / f, out / f)
    sv = read_mem(fpga / "mem" / "sv.mem", dw)
    if len(sv) != nsv * dim:
        sys.exit(f"sv.mem có {len(sv)} giá trị, cần N_SV × DIM = {nsv * dim}")
    G, mask, nd = dim // P, (1 << dw) - 1, math.ceil(P * dw / 4)
    lines = [f"// sv_lanes.mem: N_SV = {nsv}, DIM = {dim}, P = {P} lanes, G = {G} nhom/SV, {P * dw} bit/tu"]
    for i in range(nsv):
        for g in range(G):
            word = 0
            for l in range(P):
                word |= (sv[i * dim + g * P + l] & mask) << (l * dw)
            lines.append(format(word, f"0{nd}X"))
    (out / "sv_lanes.mem").write_text("\n".join(lines) + "\n")

    src = "../src"
    bat = (f"@echo off\r\nrem Mo phong RTL bang Icarus Verilog (https://bleyer.org/icarus/)\r\n"
           f"iverilog -g2012 -I . -P tb_ocsvm_core.P={P} -o core.vvp {src}/ocsvm_core.v ../sim/tb_ocsvm_core.v || exit /b 1\r\n"
           f"vvp core.vvp\r\n"
           f"iverilog -g2012 -I . -o top.vvp {src}/uart_rx.v {src}/uart_tx.v {src}/ocsvm_core.v {src}/ocsvm_top.v "
           f"../sim/tb_ocsvm_top.v || exit /b 1\r\nvvp top.vvp\r\n")
    (out / "run_sim.bat").write_text(bat)
    (out / "run_sim.sh").write_text(bat.replace("\r\n", "\n").replace("@echo off\n", "#!/bin/sh\nset -e\n")
                                    .replace("rem ", "# ").replace(" || exit /b 1", ""))
    # ---- Vivado xsim (dòng lệnh, chạy trong rtl/build để $readmemh tìm thấy file .mem)
    xs = ("@echo off\r\nrem Mo phong bang Vivado xsim – mo 'Vivado 2026.1 Tcl Shell' hoac them Vivado\\bin vao PATH\r\n"
          "call xvlog -sv -i . ../src/ocsvm_core.v ../sim/tb_ocsvm_core.v || exit /b 1\r\n"
          f"call xelab -generic_top \"P={P}\" -debug typical tb_ocsvm_core -s tb_core || exit /b 1\r\n"
          "call xsim tb_core -R\r\n"
          "call xvlog -sv -i . ../src/uart_rx.v ../src/uart_tx.v ../src/ocsvm_core.v ../src/ocsvm_top.v "
          "../sim/tb_ocsvm_top.v || exit /b 1\r\n"
          "call xelab -debug typical tb_ocsvm_top -s tb_top || exit /b 1\r\n"
          "call xsim tb_top -R\r\n")
    (out / "run_xsim.bat").write_text(xs)
    # ---- top cho board: đường dẫn TUYỆT ĐỐI tới file .mem (Gowin không phụ thuộc thư mục chạy)
    absd = out.resolve().as_posix()
    (out / "ocsvm_board_top.v").write_text(f"""// Sinh bởi scripts/15_export_rtl.py – top module cho Gowin EDA (P = {P}, {a.clk_hz} Hz, {a.baud} baud)
module ocsvm_board_top (
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire led_busy_n,
    output wire led_anomaly_n
);
    ocsvm_top #(
        .CLK_HZ({a.clk_hz}), .BAUD({a.baud}), .P({P}),
        .SV_FILE("{absd}/sv_lanes.mem"),
        .ALPHA_FILE("{absd}/alpha.mem"),
        .LUT_FILE("{absd}/exp_lut.mem")
    ) u_top (.clk(clk), .rst_n(rst_n), .uart_rx(uart_rx), .uart_tx(uart_tx),
             .led_busy_n(led_busy_n), .led_anomaly_n(led_anomaly_n));
endmodule
""", encoding="utf-8")
    (out / "ocsvm.sdc").write_text(
    f"create_clock -name clk -period {1e9 / a.clk_hz:.3f} [get_ports {{clk}}]\n",
    encoding="utf-8"
    )
    shutil.copy(out / "ocsvm_params.vh", ROOT / "rtl" / "src" / "ocsvm_params.vh")   # cho Gowin tìm `include
    cyc = nsv * G + 11
    print(f"Đã ghi {out}: sv_lanes.mem ({nsv * G} từ × {P * dw} bit), ocsvm_params.vh, alpha.mem, exp_lut.mem, test_*.mem")
    print(f"P = {P}: G = {G} nhóm/SV → độ trễ lõi ≈ {cyc} chu kỳ "
          f"({cyc / 27:.1f} µs @ 27 MHz, {cyc / 100:.2f} µs @ 100 MHz)")
    print(f"Mô phỏng Vivado: cd {out} && run_xsim.bat   |   Icarus: run_sim.bat")
    print(f"Gowin: top = ocsvm_board_top ({out / 'ocsvm_board_top.v'}), ràng buộc thời gian {out / 'ocsvm.sdc'}")


if __name__ == "__main__":
    main()