#!/usr/bin/env python3
"""Chẩn đoán đường UART PC ↔ FPGA, in NGUYÊN VĂN các byte nhận được.

  --mode echo   : dùng với bitstream rtl/debug/uart_echo_top.v (FPGA trả lại mọi byte)
                  → kiểm tra dây, chân, mức điện áp, cổng COM, baud
  --mode packet : dùng với bitstream OC-SVM (ocsvm_board_top) – gửi 1 gói véc-tơ 0,
                  chờ 2 s, in mọi byte trả về (hex) và so với Golden Model

    python scripts/18_uart_probe.py --port COM9 --mode echo
    python scripts/18_uart_probe.py --port COM9 --mode packet
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_host():
    spec = importlib.util.spec_from_file_location("host", ROOT / "scripts" / "16_uart_host_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_all(ser, wait_s):
    t0, data, stamps = time.perf_counter(), bytearray(), []
    while time.perf_counter() - t0 < wait_s:
        n = ser.in_waiting
        if n:
            chunk = ser.read(n)
            stamps.append((time.perf_counter() - t0, len(chunk)))
            data += chunk
        else:
            time.sleep(0.002)
    return bytes(data), stamps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", choices=["echo", "packet"], default="packet")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0)
    time.sleep(0.3)
    junk = ser.read(ser.in_waiting or 0)
    if junk:
        print(f"⚠ Cổng đã có sẵn {len(junk)} byte trước khi gửi (thiết bị khác đang phát?): {junk[:40].hex(' ')}")
    if a.mode == "echo":
        msg = bytes(range(0x30, 0x30 + 40)) + b"\xAA\x55\x00\xFF"
        ser.write(msg)
        got, _ = read_all(ser, 1.0)
        print(f"Gửi  {len(msg)} byte: {msg.hex(' ')}")
        print(f"Nhận {len(got)} byte: {got.hex(' ') if got else '(không có)'}")
        if got == msg:
            print("PASS: dây, chân, cổng COM và baud đều đúng – FPGA nhận và trả được dữ liệu")
        elif not got:
            print("FAIL: không nhận được gì → kiểm tra TX/RX ngược, GND, đúng chân, bitstream echo đã nạp, đúng cổng COM")
        else:
            print("FAIL: nhận sai dữ liệu → sai baud, điện áp 5 V, dây chập chờn hoặc thiếu GND")
    else:
        from ocsvm_fpga.config import load_config, resolve_path
        h = load_host()
        cfg = load_config(a.config)
        fpga = resolve_path(cfg.project.output_dir) / "fpga"
        prm = h.vh(fpga / "ocsvm_params.vh")
        x = h.read_mem(fpga / "mem" / "test_input.mem", prm["DATA_W"]).reshape(prm["N_TEST"], prm["DIM"])[0]
        exp = h.read_mem(fpga / "mem" / "test_expected_score.mem", prm["ACC_W"])[0]
        pkt = h.request(x)
        ser.write(pkt)
        got, stamps = read_all(ser, 2.0)
        print(f"Gửi gói {len(pkt)} byte (đầu: {pkt[:8].hex(' ')} … cuối: {pkt[-3:].hex(' ')})")
        print(f"Nhận {len(got)} byte: {got.hex(' ') if got else '(không có)'}")
        if stamps:
            print("Thời điểm nhận (ms, số byte): " + ", ".join(f"{t * 1000:.1f}:{n}" for t, n in stamps[:10]))
        if len(got) >= 9 and got[0] == 0x5A:
            try:
                f, flag = h.parse_response(got[:9])
                print(f"Gói trả lời hợp lệ: f_q = {f} (Golden {exp}), cờ = {flag} → "
                      + ("PASS" if f == exp else "SAI GIÁ TRỊ"))
            except ValueError as e:
                print(f"Gói trả lời lỗi: {e}")
        elif got[:len(pkt)] == pkt[:len(got)] and got:
            print("Nhận lại đúng gói đã gửi → đang chạy bitstream ECHO hoặc TX/RX bị nối tắt, không phải OC-SVM")
        elif not got:
            print("Không có trả lời → FPGA không nhận được gói (dây/chân/baud) hoặc bitstream chưa đúng")
    ser.close()


if __name__ == "__main__":
    main()
