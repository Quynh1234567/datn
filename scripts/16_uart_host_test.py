#!/usr/bin/env python3
"""Kiểm thử bộ tăng tốc TRÊN BOARD qua UART từ PC (trước khi nối ESP32).

Gửi véc-tơ trong test_input.mem đúng giao thức ESP32 dùng, nhận f_q và cờ từ FPGA,
so với test_expected_score.mem / test_expected_label.mem (Golden Model) và đo thời gian
khứ hồi mỗi gói.

    pip install pyserial
    python scripts/16_uart_host_test.py --port COM5 --baud 115200
    python scripts/16_uart_host_test.py --mock          # không cần board: giả lập FPGA bằng Golden Model

Gói vào : 0xAA 0x55 | DIM × int16 LE | XOR byte dữ liệu
Gói ra  : 0x5A | f_q 6 byte LE (bù 2) | cờ | XOR(6 byte f_q + cờ)
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.config import load_config, resolve_path  # noqa: E402


def read_mem(path: Path, width: int) -> np.ndarray:
    vals = []
    for ln in path.read_text().splitlines():
        ln = ln.split("//")[0].strip()
        if ln:
            v = int(ln, 16)
            vals.append(v - (1 << width) if v >> (width - 1) else v)
    return np.array(vals, dtype=np.int64)


def vh(path: Path) -> dict:
    t = path.read_text()
    out = {m.group(1): int(m.group(2)) for m in re.finditer(r"localparam\s+integer\s+(\w+)\s*=\s*(-?\d+)", t)}
    for m in re.finditer(r"localparam\s+signed\s+\[[^\]]+\]\s+(\w+)\s*=\s*(-?)\s*\d+'sd(\d+)", t):
        out[m.group(1)] = -int(m.group(3)) if m.group(2) else int(m.group(3))
    return out


def request(xq: np.ndarray) -> bytes:
    data = b"".join(int(v & 0xFFFF).to_bytes(2, "little") for v in xq)
    chk = 0
    for b in data:
        chk ^= b
    return b"\xAA\x55" + data + bytes([chk])


def parse_response(buf: bytes) -> tuple[int, int]:
    if len(buf) != 9 or buf[0] != 0x5A:
        raise ValueError(f"gói trả lời sai: {buf.hex()}")
    chk = 0
    for b in buf[1:8]:
        chk ^= b
    if chk != buf[8]:
        raise ValueError(f"sai checksum: {buf.hex()}")
    raw = int.from_bytes(buf[1:7], "little")
    if raw >> 47:
        raw -= 1 << 48
    return raw, buf[7] & 1


class MockFPGA:
    """Giả lập FPGA bằng phép tính số nguyên của Golden Model (đọc thẳng các file .mem)."""

    def __init__(self, fpga: Path, prm: dict):
        self.p = prm
        self.sv = read_mem(fpga / "mem" / "sv.mem", prm["DATA_W"]).reshape(prm["N_SV"], prm["DIM"])
        self.alpha = read_mem(fpga / "mem" / "alpha.mem", prm["COEF_W"])
        self.lut = read_mem(fpga / "mem" / "exp_lut.mem", prm["LUT_W"] + 1)
        self.buf = b""

    def write(self, data: bytes):
        self.buf = data

    def read(self, n: int) -> bytes:
        d = self.buf
        xq = np.frombuffer(d[2:-1], dtype="<i2").astype(np.int64)
        p = self.p
        diff = xq[None, :] - self.sv
        t = (diff * diff).sum(1) * p["GAMMA_Q"]
        idx = t >> p["LUT_SHIFT"]
        k = np.where(idx < p["LUT_SIZE"], self.lut[np.minimum(idx, p["LUT_SIZE"] - 1)], 0)
        f = int(k @ self.alpha) - p["RHO_Q"]
        flag = int(f < p["THR_Q"]) if p.get("DECISION_MODE", 0) == 0 else int(abs(f) > p["THR_Q"])
        body = (f & ((1 << 48) - 1)).to_bytes(6, "little") + bytes([flag])
        chk = 0
        for b in body:
            chk ^= b
        return (b"\x5A" + body + bytes([chk]))[:n]

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", default=None, help="cổng COM của board (vd COM5 hoặc /dev/ttyUSB1)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--count", type=int, default=None, help="số véc-tơ gửi (mặc định: tất cả)")
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--mock", action="store_true", help="không cần board: giả lập FPGA")
    a = ap.parse_args()

    cfg = load_config(a.config)
    fpga = resolve_path(cfg.project.output_dir) / "fpga"
    prm = vh(fpga / "ocsvm_params.vh")
    dim, n_test = prm["DIM"], prm["N_TEST"]
    xin = read_mem(fpga / "mem" / "test_input.mem", prm["DATA_W"]).reshape(n_test, dim)
    exp_f = read_mem(fpga / "mem" / "test_expected_score.mem", prm["ACC_W"])
    exp_l = read_mem(fpga / "mem" / "test_expected_label.mem", 2)
    n = min(a.count or n_test, n_test)

    if a.mock:
        ser = MockFPGA(fpga, prm)
        print("Chế độ giả lập (--mock): không dùng board")
    else:
        if not a.port:
            sys.exit("Cần --port (vd --port COM5) hoặc --mock")
        try:
            import serial
        except ImportError:
            sys.exit("Thiếu thư viện: pip install pyserial")
        ser = serial.Serial(a.port, a.baud, timeout=a.timeout)
        time.sleep(0.2)
        ser.reset_input_buffer()

    err_f = err_l = err_io = 0
    rtt = []
    for v in range(n):
        pkt = request(xin[v])
        t0 = time.perf_counter()
        ser.write(pkt)
        resp = ser.read(9)
        rtt.append(time.perf_counter() - t0)
        try:
            f, flag = parse_response(resp)
        except ValueError as e:
            err_io += 1
            print(f"véc-tơ {v}: {e}")
            continue
        if f != exp_f[v]:
            err_f += 1
            if err_f <= 5:
                print(f"véc-tơ {v}: FPGA f = {f}, Golden f = {exp_f[v]}")
        if flag != (exp_l[v] & 1):
            err_l += 1
    ser.close()
    rtt_ms = np.array(rtt) * 1000
    wire_ms = (len(request(xin[0])) + 9) * 10 / a.baud * 1000
    print("-" * 70)
    print(f"{n} véc-tơ | lỗi truyền: {err_io} | sai f_q: {err_f} | sai cờ: {err_l}")
    print(f"Thời gian khứ hồi: trung bình {rtt_ms.mean():.2f} ms, lớn nhất {rtt_ms.max():.2f} ms "
          f"(lý thuyết truyền {wire_ms:.2f} ms @ {a.baud} baud)")
    ok = err_io == 0 and err_f == 0 and err_l == 0
    print("PASS: FPGA khớp từng bit với Golden Model" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
