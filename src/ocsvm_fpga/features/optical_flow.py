"""BƯỚC 2 – Dòng quang học Horn–Schunck (Horn & Schunck, 1981).

Tối thiểu hóa hàm năng lượng toàn cục (phương trình (42) trong bài báo):

    E = ∬ [ (I_x·u + I_y·v + I_t)² + γ²(‖∇u‖² + ‖∇v‖²) ] dx dy

Lời giải lặp Jacobi (Euler–Lagrange rời rạc hóa):

    u^{k+1} = ū^k − I_x · (I_x·ū^k + I_y·v̄^k + I_t) / (γ² + I_x² + I_y²)
    v^{k+1} = v̄^k − I_y · (I_x·ū^k + I_y·v̄^k + I_t) / (γ² + I_x² + I_y²)

trong đó ū, v̄ là trung bình lân cận có trọng số (kernel Laplace của HS).
Cài đặt chỉ dùng phép cộng/nhân trên lưới với biên "replicate" và số vòng lặp
cố định → có thể chuyển nguyên văn sang C cho ESP32-S3 và cho kết quả giống hệt.
"""
from __future__ import annotations

import cv2
import numpy as np

# Kernel trung bình lân cận của Horn–Schunck (trọng số 1/6 cạnh, 1/12 góc)
HS_AVG_KERNEL = np.array([[1 / 12, 1 / 6, 1 / 12],
                          [1 / 6, 0.0, 1 / 6],
                          [1 / 12, 1 / 6, 1 / 12]], dtype=np.float32)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a[y+dy, x+dx] với biên replicate (dy, dx ∈ {0, 1})."""
    p = np.pad(a, ((0, dy), (0, dx)), mode="edge")
    return p[dy:dy + a.shape[0], dx:dx + a.shape[1]]


def image_derivatives(i1: np.ndarray, i2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ước lượng I_x, I_y, I_t trên khối lập phương 2×2×2 (đúng công thức gốc của HS)."""
    a00, a01, a10, a11 = i1, _shift(i1, 0, 1), _shift(i1, 1, 0), _shift(i1, 1, 1)
    b00, b01, b10, b11 = i2, _shift(i2, 0, 1), _shift(i2, 1, 0), _shift(i2, 1, 1)
    ix = 0.25 * ((a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10))
    iy = 0.25 * ((a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01))
    it = 0.25 * ((b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11))
    return ix.astype(np.float32), iy.astype(np.float32), it.astype(np.float32)


def neighbor_average(f: np.ndarray) -> np.ndarray:
    return cv2.filter2D(f, -1, HS_AVG_KERNEL, borderType=cv2.BORDER_REPLICATE)


def horn_schunck(i1: np.ndarray, i2: np.ndarray, alpha: float = 10.0, num_iter: int = 64,
                 intensity_scale: float = 1.0,
                 u0: np.ndarray | None = None, v0: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Tính trường dòng quang học OP_i = (u, v) giữa hai khung xám liên tiếp.

    Args:
        i1, i2: ảnh xám float32 cùng kích thước (H, W).
        alpha: γ – trọng số số hạng làm mượt.
        num_iter: số vòng lặp Jacobi.
        intensity_scale: chia cường độ trước khi tính (1 = giữ thang 0..255).
        u0, v0: khởi tạo (warm-start) – tùy chọn.
    Returns:
        u, v: vận tốc ngang / dọc (pixel / bước khung), float32 (H, W).
    """
    i1 = i1.astype(np.float32) / intensity_scale
    i2 = i2.astype(np.float32) / intensity_scale
    ix, iy, it = image_derivatives(i1, i2)
    u = np.zeros_like(i1) if u0 is None else u0.astype(np.float32).copy()
    v = np.zeros_like(i1) if v0 is None else v0.astype(np.float32).copy()
    denom = (alpha * alpha + ix * ix + iy * iy).astype(np.float32)
    for _ in range(int(num_iter)):
        u_avg = neighbor_average(u)
        v_avg = neighbor_average(v)
        der = (ix * u_avg + iy * v_avg + it) / denom
        u = u_avg - ix * der
        v = v_avg - iy * der
    return u, v


def farneback(i1: np.ndarray, i2: np.ndarray, of) -> tuple[np.ndarray, np.ndarray]:
    """Dòng quang học Farneback của OpenCV (đa tỉ lệ) – dùng cho thí nghiệm so sánh.

    Đầu vào float (đã làm mượt) được đưa về uint8 vì OpenCV yêu cầu ảnh 8 bit.
    """
    a = np.clip(i1, 0, 255).astype(np.uint8)
    b = np.clip(i2, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, float(of.get("fb_pyr_scale", 0.5)), int(of.get("fb_levels", 3)),
        int(of.get("fb_winsize", 15)), int(of.get("fb_iterations", 3)),
        int(of.get("fb_poly_n", 5)), float(of.get("fb_poly_sigma", 1.2)), 0)
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def compute_flow(i1: np.ndarray, i2: np.ndarray, of) -> tuple[np.ndarray, np.ndarray]:
    """Chọn thuật toán theo optical_flow.method: "hs" (mặc định, chạy được trên ESP32) | "farneback"."""
    method = of.get("method", "hs")
    if method == "hs":
        return horn_schunck(i1, i2, of.alpha, of.num_iter, of.intensity_scale)
    if method == "farneback":
        return farneback(i1, i2, of)
    raise ValueError(f"optical_flow.method không hỗ trợ: {method}")
