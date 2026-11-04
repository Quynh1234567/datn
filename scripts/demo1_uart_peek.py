#!/usr/bin/env python3
"""Đọc và in các dòng UART trong N giây: python scripts/demo1_uart_peek.py COM9 115200 10   (thêm \"esp\" cho cổng ESP32)"""
import sys
import time

import serial


def main():
    port, baud, sec = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    esp = len(sys.argv) > 4 and sys.argv[4] == "esp"
    n = 0
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 0.2
    if esp:                # ESP32 (CH343): không kéo DTR/RTS → không reset / kẹt chế độ nạp
        s.dtr = False
        s.rts = False
    # FPGA (BL616 trên Tang Nano): GIỮ DTR mặc định – chip này chỉ chuyển dữ liệu khi DTR được bật
    s.open()
    with s:
        t0 = time.time()
        while time.time() - t0 < sec:
            line = s.readline()
            if line:
                n += 1
                print(line.decode("utf-8", "replace").rstrip(), flush=True)
    print(f"--- {n} dòng trong {sec:.0f} s" + ("" if n else " (KHÔNG có dữ liệu: kiểm tra cổng, nạp FPGA, LED nhịp 1 Hz)"))


if __name__ == "__main__":
    main()
