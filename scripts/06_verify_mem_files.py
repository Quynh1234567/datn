#!/usr/bin/env python3
"""Kiểm chứng ĐỘC LẬP các file đã xuất – mô phỏng đúng cách testbench Verilog làm việc.

Script KHÔNG dùng mô hình Python: chỉ đọc ocsvm_params.vh + mem/*.mem, tính lại
f_q bằng số nguyên thuần (vòng lặp như phần cứng tuần tự) rồi so với
test_expected_score.mem / test_expected_label.mem. Nếu PASS thì định dạng file,
thứ tự phần tử, bù hai và tham số trong .vh đều nhất quán.

    python scripts/06_verify_mem_files.py                       # outputs/fpga
    python scripts/06_verify_mem_files.py --dir outputs_synthetic/fpga
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_vh(path: Path) -> dict:
    params = {}
    for m in re.finditer(r"localparam\s+(?:signed\s+\[[^\]]+\]\s+|integer\s+)?(\w+)\s*=\s*([^;]+);",
                         path.read_text()):
        name, expr = m.group(1), m.group(2).strip()
        neg = expr.startswith("-")
        expr = expr.lstrip("-").strip()
        val = int(expr.split("'sd")[1]) if "'sd" in expr else int(expr)
        params[name] = -val if neg else val
    return params


def read_mem(path: Path, width: int, signed: bool = True) -> list[int]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue
        v = int(line, 16)
        if signed and v >= 1 << (width - 1):
            v -= 1 << width
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs" / "fpga"))
    d = Path(ap.parse_args().dir)
    P = read_vh(d / "ocsvm_params.vh")
    n_sv, dim, n_test = P["N_SV"], P["DIM"], P["N_TEST"]
    sv = read_mem(d / "mem" / "sv.mem", P["DATA_W"])
    alpha = read_mem(d / "mem" / "alpha.mem", P["COEF_W"])
    lut = read_mem(d / "mem" / "exp_lut.mem", P["LUT_W"], signed=False)
    assert len(sv) == n_sv * dim and len(alpha) == n_sv and len(lut) == P["LUT_SIZE"], "Sai kích thước file"
    if n_test == 0:
        sys.exit("Không có vector kiểm thử (N_TEST = 0)")
    x = read_mem(d / "mem" / "test_input.mem", P["DATA_W"])
    exp_f = read_mem(d / "mem" / "test_expected_score.mem", P["ACC_W"])
    exp_l = read_mem(d / "mem" / "test_expected_label.mem", 1, signed=False)

    n_err = 0
    for t in range(n_test):
        acc = 0
        for i in range(n_sv):                               # mô phỏng PE tuần tự
            dist = 0
            for j in range(dim):                            # khối MAC bình phương khoảng cách
                diff = x[t * dim + j] - sv[i * dim + j]
                dist += diff * diff
            idx = (dist * P["GAMMA_Q"]) >> P["LUT_SHIFT"]
            k = lut[idx] if idx < P["LUT_SIZE"] else 0      # khối RBF (LUT exp)
            acc += alpha[i] * k                             # khối MAC hệ số
        f = acc - P["RHO_Q"]
        flag = int(abs(f) > P["THR_Q"]) if P["DECISION_MODE"] == 1 else int(f < P["THR_Q"])
        if f != exp_f[t] or flag != exp_l[t]:
            n_err += 1
            if n_err <= 5:
                print(f"  ✗ vector {t}: f = {f} (kỳ vọng {exp_f[t]}), flag = {flag} (kỳ vọng {exp_l[t]})")
    print(f"Kiểm tra {n_test} vector, {n_sv} SV × {dim} chiều: "
          + ("PASS ✔ – khớp bit-exact" if n_err == 0 else f"FAIL ✗ – {n_err} vector sai"))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
