#!/usr/bin/env python3
"""Nhận ảnh từ chương trình kiểm tra camera (firmware/esp32_camtest) qua cổng COM.

In nguyên văn log của ESP32 (thông tin chip, PSRAM, mã cảm biến, fps) và mỗi khi ESP32 gửi
một khung (FRAME w h / base64 / END) thì giải mã, lưu PNG vào outputs/camtest/ và hiện ảnh.

    pip install pyserial
    python scripts/17_cam_grab.py --port COM5
    python scripts/17_cam_grab.py --port COM5 --no-show --count 3
"""
import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="cổng COM của ESP32 (vd COM7)")
    ap.add_argument("--baud", type=int, default=115200, help="tốc độ console ESP32 (mặc định 115200)")
    ap.add_argument("--count", type=int, default=0, help="dừng sau N ảnh (0 = chạy mãi, Ctrl+C để dừng)")
    ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ ảnh")
    a = ap.parse_args()
    try:
        import serial
    except ImportError:
        sys.exit("Thiếu thư viện: pip install pyserial")
    out = ROOT / "outputs" / "camtest"
    out.mkdir(parents=True, exist_ok=True)
    ser = serial.Serial(a.port, a.baud, timeout=1)
    print(f"Đang nghe {a.port} @ {a.baud} – nhấn nút RST trên board để xem thông tin khởi động")
    got, frame_hdr, buf = 0, None, []
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if frame_hdr is None and line.startswith("FRAME "):
                _, w, h = line.split()
                frame_hdr, buf = (int(w), int(h)), []
                continue
            if frame_hdr is not None:
                if line == "END":
                    w, h = frame_hdr
                    raw = base64.b64decode("".join(buf))
                    frame_hdr = None
                    if len(raw) != w * h:
                        print(f"  ⚠ khung lỗi: {len(raw)} byte, cần {w * h}")
                        continue
                    img = np.frombuffer(raw, np.uint8).reshape(h, w)
                    got += 1
                    path = out / f"frame_{time.strftime('%H%M%S')}_{got:02d}.png"
                    cv2.imwrite(str(path), img)
                    print(f"  ✔ ảnh {got}: {w}×{h}, độ sáng TB {img.mean():.0f} → {path}")
                    if not a.no_show:
                        cv2.imshow("ESP32-S3 OV5640 (160x120, phong to x4)",
                                   cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST))
                        cv2.waitKey(1)
                    if a.count and got >= a.count:
                        break
                else:
                    buf.append(line.strip())
                continue
            print(line)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()
    print(f"Đã nhận {got} ảnh → {out}")


if __name__ == "__main__":
    main()
