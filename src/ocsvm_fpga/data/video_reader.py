"""BƯỚC 1 – Tiếp nhận video và tách khung hình xám {I_1, I_2, ..., I_n}.

Khung hình được đưa về đúng định dạng mà ESP32-S3 + OV5640 cung cấp khi demo
(ảnh xám, 160x120) để đặc trưng lúc huấn luyện và lúc chạy thực tế khớp nhau.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def center_crop_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Cắt giữa ảnh về tỉ lệ rộng/cao = aspect (vd 4:3 như OV5640 QQVGA)."""
    h, w = img.shape[:2]
    if w / h > aspect:                       # ảnh quá rộng (16:9) → cắt hai bên
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / aspect))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def to_sensor_frame(frame: np.ndarray, pp) -> np.ndarray:
    """BGR/xám (bất kỳ kích thước) → ảnh xám uint8 (height, width).

    Đây chính là dữ liệu OV5640 trả về ở chế độ GRAYSCALE/QQVGA, tức đầu vào
    của firmware ESP32 (firmware/esp32_features).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if pp.aspect_mode == "crop":
        gray = center_crop_to_aspect(gray, pp.width / pp.height)
    if gray.shape[:2] != (pp.height, pp.width):
        gray = cv2.resize(gray, (pp.width, pp.height), interpolation=cv2.INTER_AREA)
    return gray.astype(np.uint8)


def smooth_frame(gray_u8: np.ndarray, pp) -> np.ndarray:
    """uint8 → float32 + làm mượt Gaussian (biên REFLECT_101 mặc định của OpenCV)."""
    gray = gray_u8.astype(np.float32)
    if pp.gaussian_ksize and pp.gaussian_ksize > 1:
        gray = cv2.GaussianBlur(gray, (pp.gaussian_ksize, pp.gaussian_ksize), pp.gaussian_sigma)
    return gray


def preprocess_frame(frame: np.ndarray, pp) -> np.ndarray:
    """BGR/xám (bất kỳ kích thước) → ảnh xám float32 kích thước (height, width)."""
    return smooth_frame(to_sensor_frame(frame, pp), pp)


def apply_masks(frame: np.ndarray, path, pp) -> np.ndarray:
    """Che các vùng cố định (vd dòng chữ in sẵn trên video) bằng màu xám hằng số.

    preprocess.mask_regions: [{pattern: "chuoi_trong_ten_file", box: [y0, y1, x0, x1]}, ...]
    Tọa độ tính trên khung GỐC. Vùng che không đổi theo thời gian → không sinh dòng quang học.
    """
    regions = pp.get("mask_regions") or []
    if not regions:
        return frame
    name = Path(str(path)).name.lower()
    for r in regions:
        if str(r.get("pattern", "")).lower() in name:
            y0, y1, x0, x1 = (int(v) for v in r["box"])
            frame = frame.copy()
            frame[y0:y1, x0:x1] = 128
    return frame


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "opened": cap.isOpened(),
    }
    cap.release()
    return info


def iter_frames(path: str | Path, pp) -> Iterator[tuple[int, np.ndarray]]:
    """Sinh (chỉ số khung gốc, ảnh xám đã tiền xử lý), lấy 1 khung mỗi `frame_step` khung."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {path}")
    step = max(1, int(pp.frame_step))
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, preprocess_frame(apply_masks(frame, path, pp), pp)
            else:
                if not cap.grab():               # bỏ qua khung, không cần chuyển màu
                    break
            idx += 1
    finally:
        cap.release()


def iter_sensor_frames(path: str | Path, pp) -> Iterator[tuple[int, np.ndarray]]:
    """Như iter_frames nhưng trả ảnh uint8 CHƯA làm mượt (để lưu cache khung)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {path}")
    step = max(1, int(pp.frame_step))
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, to_sensor_frame(apply_masks(frame, path, pp), pp)
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
