"""Đặc trưng ĐƯỜNG CƠ SỞ: thống kê dòng quang học gộp theo lưới ô.

Dùng để so sánh với đặc trưng ma trận hiệp biến trong chương đánh giá:
cùng dữ liệu, cùng dòng quang học, cùng OC-SVM – chỉ khác cách biến trường
(u, v) thành véc-tơ. Ý tưởng theo repo sanjeebtiwary/Crowd-Anomaly-Detection
(Farneback + lưới 6×8, mỗi ô [dx, dy, |v|], bản "rich" thêm max và std của |v|).

Khác biệt có chủ đích so với repo gốc: repo đó mô hình hóa TỪNG Ô bằng GMM;
ở đây các ô được GHÉP thành MỘT véc-tơ cho cả khung để dùng đúng một OC-SVM
như đặc trưng hiệp biến – nhờ vậy phép so sánh chỉ khác ở bước trích đặc trưng.

    stats = "basic": mỗi ô [mean u, mean v, mean |v|]                → R × C × 3 chiều
    stats = "rich" : mỗi ô [mean u, mean v, mean |v|, max |v|, std |v|] → R × C × 5 chiều
Thứ tự: ô theo hàng (row-major), trong mỗi ô theo thứ tự thống kê ở trên.
"""
from __future__ import annotations

import numpy as np

STAT_NAMES = {
    "basic": ["mean_u", "mean_v", "mean_mag"],
    "rich": ["mean_u", "mean_v", "mean_mag", "max_mag", "std_mag"],
}


def grid_descriptor_dim(rows: int, cols: int, stats: str) -> int:
    return rows * cols * len(STAT_NAMES[stats])


def cell_edges(n: int, parts: int) -> np.ndarray:
    """Biên chia đều [0, n) thành `parts` đoạn (đúng tuyệt đối khi n chia hết cho parts)."""
    return np.round(np.linspace(0, n, parts + 1)).astype(int)


def grid_flow_descriptor(u: np.ndarray, v: np.ndarray, rows: int = 6, cols: int = 8,
                         stats: str = "basic") -> np.ndarray:
    """(u, v) kích thước H×W → véc-tơ float32 độ dài grid_descriptor_dim(rows, cols, stats)."""
    if stats not in STAT_NAMES:
        raise ValueError(f"grid_stats phải là {list(STAT_NAMES)}, nhận '{stats}'")
    h, w = u.shape
    ye, xe = cell_edges(h, rows), cell_edges(w, cols)
    mag = np.hypot(u, v)
    out = np.empty((rows, cols, len(STAT_NAMES[stats])), np.float64)
    for r in range(rows):
        for c in range(cols):
            sl = (slice(ye[r], ye[r + 1]), slice(xe[c], xe[c + 1]))
            m = mag[sl]
            out[r, c, 0] = u[sl].mean()
            out[r, c, 1] = v[sl].mean()
            out[r, c, 2] = m.mean()
            if stats == "rich":
                out[r, c, 3] = m.max()
                out[r, c, 4] = m.std()
    return out.reshape(-1).astype(np.float32)
