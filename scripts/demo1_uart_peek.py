#!/usr/bin/env python3
"""Đọc và in các dòng UART trong N giây: python scripts/demo1_uart_peek.py COM5 115200 10"""
import sys
import time

import serial


def main():
    port, baud, sec = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    n = 0
    with serial.Serial(port, baud, timeout=0.2) as s:
        t0 = time.time()
        while time.time() - t0 < sec:
            line = s.readline()
            if line:
                n += 1
                print(line.decode("utf-8", "replace").rstrip(), flush=True)
    print(f"--- {n} dòng trong {sec:.0f} s" + ("" if n else " (KHÔNG có dữ liệu: kiểm tra cổng, nạp FPGA, LED nhịp 1 Hz)"))


if __name__ == "__main__":
    main()
