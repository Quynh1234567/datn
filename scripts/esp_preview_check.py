#!/usr/bin/env python3
"""Đo luồng ảnh xem trước từ ESP32 trong 10 giây (không reset ESP32):
    python scripts/esp_preview_check.py COM5 921600
    python scripts/esp_preview_check.py COM5 921600 reset     (khởi động lại ESP32, đọc cả log khởi động, 20 s)
In: loại ảnh (PJ = JPEG mới, PC = màu không nén cũ, PV = xám), số ảnh/giây, kích thước, số dòng F/giây,
khoảng cách giữa các ảnh – để biết ảnh trên PC chậm do firmware hay do máy tính."""
import sys
import time

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 921600
sec = 10.0
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, baud, 0.2
s.dtr = False
s.rts = False
s.open()
if "reset" in sys.argv:            # khởi động lại ESP32 để đọc cả log khởi động (#INFO fps…)
    s.rts = True
    time.sleep(0.15)
    s.rts = False
    sec = 30.0
    print("Đang đọc 30 s – nếu ESP32 không tự khởi động lại, hãy NHẤN NÚT RST trên board ngay bây giờ …", flush=True)
t0 = time.time()
blocks, cur, nbytes, n_f, infos, gaps, last_t = {}, None, 0, 0, [], [], None
while time.time() - t0 < sec:
    raw = s.readline()
    nbytes += len(raw)
    line = raw.decode("ascii", "replace").strip()
    if not line:
        continue
    if line.startswith(("PJ ", "PC ", "PV ")):
        cur = [line[:2], 0]
        continue
    if cur is not None:
        if line == "END":
            tag, n = cur
            b = blocks.setdefault(tag, [])
            b.append(n * 3 // 4)
            now = time.time()
            if last_t is not None:
                gaps.append(now - last_t)
            last_t = now
            cur = None
        else:
            cur[1] += len(line)
        continue
    if line.startswith("F seq="):
        n_f += 1
    elif not line.startswith(("D ts=",)) and len(infos) < 400 and not (
            line.startswith("I (") and any(k in line for k in ("boot", "esp_image", "octal_psram", "heap_init",
                                                                  "app_init", "efuse", "MSPI", "cpu_start"))):
        infos.append(line)                 # mọi dòng log khác (#INFO, lỗi driver, khởi động ROM …)
s.close()
dt = time.time() - t0
print(f"Cổng {port} @ {baud}: {nbytes / dt / 1024:.1f} KB/s nhận được (tối đa ≈ {baud / 10 / 1024:.0f} KB/s)")
print(f"Dòng F (khung gửi FPGA): {n_f / dt:.1f} /s   (đúng: ≈ 12,5)")
if not blocks:
    print("KHÔNG có ảnh xem trước nào – kiểm tra baud, firmware (menuconfig ảnh màu), hoặc ESP32 đang kẹt.")
for tag, sz in blocks.items():
    name = {"PJ": "JPEG màu (bản mới)", "PC": "màu KHÔNG nén (bản cũ, chậm)", "PV": "xám"}[tag]
    print(f"Ảnh {tag} – {name}: {len(sz) / dt:.1f} ảnh/s · TB {sum(sz) / len(sz) / 1024:.1f} KB/ảnh")
if gaps:
    gaps.sort()
    print(f"Khoảng cách giữa 2 ảnh: trung vị {gaps[len(gaps) // 2] * 1000:.0f} ms · lớn nhất {gaps[-1] * 1000:.0f} ms")
if "reset" in sys.argv and not any(i.startswith("#INFO") for i in infos):
    print("CHƯA thấy log khởi động: board không tự reset qua cổng COM → chạy lại lệnh và NHẤN NÚT RST trên ESP32"
          " ngay sau khi Enter (trong 2 giây đầu).")
for i in infos:
    print(" ", i)
