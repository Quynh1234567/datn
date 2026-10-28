"""Đặc trưng thống kê dòng quang học theo lưới ô.

Mỗi ô (theo hàng, row-major) cho các thống kê:
    stats = "basic": [mean u, mean v, mean |v|]
    stats = "max"  : + max |v|
    stats = "rich" : + max |v|, std |v|
    grid_diff = true: + mean |I₂ − I₁| (độ lớn hiệu hai khung) – nhạy với vật nhỏ chuyển động
                      nhanh mà Horn–Schunck làm loãng (người chạy ở xa camera)
Biên ô mặc định chia đều; có thể đặt biên tùy ý (lưới phối cảnh: ô nhỏ ở vùng xa) bằng
grid_row_edges / grid_col_edges.

CACHE THEO KHỐI: khi trích xuất, mỗi cặp khung lưu thống kê của các khối 10×10 điểm ảnh
[Σu, Σv, Σ|v|, max|v|, Σ|v|², Σ|ΔI|]. Mọi lưới có biên là bội số của 10 (vd 4×4, 6×8, lưới
phối cảnh) đều được ghép lại từ các khối này lúc ĐỌC cache → đổi cách chia lưới, thống kê, bật
tắt grid_diff KHÔNG phải trích xuất lại. Kết quả trùng với tính trực tiếp (kiểm tra trong tests).
"""
from __future__ import annotations

import numpy as np

STAT_NAMES = {
    "basic": ["mean_u", "mean_v", "mean_mag"],
    "max": ["mean_u", "mean_v", "mean_mag", "max_mag"],
    "rich": ["mean_u", "mean_v", "mean_mag", "max_mag", "std_mag"],
}
BLOCK = 10
N_BLOCK_STATS = 6                       # Σu, Σv, Σ|v|, max|v|, Σ|v|², Σ|ΔI|


def cell_edges(n: int, parts: int) -> np.ndarray:
    """Biên chia đều [0, n) thành `parts` đoạn."""
    return np.round(np.linspace(0, n, parts + 1)).astype(int)


def grid_layout(f, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(biên hàng, biên cột) theo cấu hình: tùy ý nếu có grid_row_edges/grid_col_edges, nếu không chia đều."""
    re_, ce_ = f.get("grid_row_edges"), f.get("grid_col_edges")
    ye = np.asarray(re_, int) if re_ else cell_edges(h, int(f.grid_rows))
    xe = np.asarray(ce_, int) if ce_ else cell_edges(w, int(f.grid_cols))
    for name, e, n in (("grid_row_edges", ye, h), ("grid_col_edges", xe, w)):
        if e[0] != 0 or e[-1] != n or np.any(np.diff(e) <= 0):
            raise ValueError(f"{name} = {e.tolist()} phải tăng dần, bắt đầu 0 và kết thúc {n}")
    return ye, xe


def n_cell_stats(stats: str, diff: bool) -> int:
    return len(STAT_NAMES[stats]) + int(bool(diff))


def grid_dim(f, h: int = 120, w: int = 160) -> int:
    ye, xe = grid_layout(f, h, w)
    return (len(ye) - 1) * (len(xe) - 1) * n_cell_stats(f.get("grid_stats", "basic"), f.get("grid_diff", False))


def grid_descriptor_dim(rows: int, cols: int, stats: str, diff: bool = False) -> int:
    return rows * cols * n_cell_stats(stats, diff)


def grid_flow_descriptor(u: np.ndarray, v: np.ndarray, rows: int = 6, cols: int = 8, stats: str = "basic",
                         ye=None, xe=None, diff_img: np.ndarray | None = None) -> np.ndarray:
    """Tính TRỰC TIẾP từ (u, v) [+ |I₂ − I₁| nếu diff_img khác None] → véc-tơ float32."""
    if stats not in STAT_NAMES:
        raise ValueError(f"grid_stats phải là {list(STAT_NAMES)}, nhận '{stats}'")
    h, w = u.shape
    ye = cell_edges(h, rows) if ye is None else np.asarray(ye, int)
    xe = cell_edges(w, cols) if xe is None else np.asarray(xe, int)
    mag = np.hypot(u, v)
    ns = n_cell_stats(stats, diff_img is not None)
    out = np.empty((len(ye) - 1, len(xe) - 1, ns), np.float64)
    for r in range(len(ye) - 1):
        for c in range(len(xe) - 1):
            sl = (slice(ye[r], ye[r + 1]), slice(xe[c], xe[c + 1]))
            m = mag[sl]
            k = 0
            out[r, c, 0] = u[sl].mean()
            out[r, c, 1] = v[sl].mean()
            out[r, c, 2] = m.mean()
            k = 3
            if stats in ("max", "rich"):
                out[r, c, k] = m.max()
                k += 1
            if stats == "rich":
                out[r, c, k] = m.std()
                k += 1
            if diff_img is not None:
                out[r, c, k] = diff_img[sl].mean()
    return out.reshape(-1).astype(np.float32)


# ------------------------------------------------------------------ cache theo khối 10×10
def block_stats(u: np.ndarray, v: np.ndarray, diff_img: np.ndarray, bs: int = BLOCK) -> np.ndarray:
    """(H/bs, W/bs, 6) thống kê từng khối – lưu vào cache."""
    h, w = u.shape
    if h % bs or w % bs:
        raise ValueError(f"Khung {w}×{h} không chia hết cho khối {bs}")
    mag = np.hypot(u, v).astype(np.float64)

    def blk(a):
        return a.reshape(h // bs, bs, w // bs, bs)

    out = np.empty((h // bs, w // bs, N_BLOCK_STATS), np.float32)
    out[..., 0] = blk(u.astype(np.float64)).sum((1, 3))
    out[..., 1] = blk(v.astype(np.float64)).sum((1, 3))
    out[..., 2] = blk(mag).sum((1, 3))
    out[..., 3] = blk(mag).max((1, 3))
    out[..., 4] = blk(mag * mag).sum((1, 3))
    out[..., 5] = blk(diff_img.astype(np.float64)).sum((1, 3))
    return out


def from_blocks(B: np.ndarray, ye, xe, stats: str, diff: bool, bs: int = BLOCK) -> np.ndarray:
    """Ghép thống kê khối B (n, H/bs, W/bs, 6) thành véc-tơ lưới (n, D)."""
    ye, xe = np.asarray(ye, int), np.asarray(xe, int)
    if np.any(ye % bs) or np.any(xe % bs):
        raise ValueError(f"Biên ô phải là bội số của {bs} điểm ảnh để dùng cache khối: hàng {ye.tolist()}, "
                         f"cột {xe.tolist()}")
    B = B.astype(np.float64)
    n = B.shape[0]
    yb, xb = ye // bs, xe // bs
    cells = []
    for r in range(len(yb) - 1):
        for c in range(len(xb) - 1):
            sub = B[:, yb[r]:yb[r + 1], xb[c]:xb[c + 1], :]
            npx = (yb[r + 1] - yb[r]) * (xb[c + 1] - xb[c]) * bs * bs
            s = sub.sum((1, 2))
            mu = s[:, 2] / npx
            feats = [s[:, 0] / npx, s[:, 1] / npx, mu]
            if stats in ("max", "rich"):
                feats.append(sub[..., 3].max((1, 2)))
            if stats == "rich":
                feats.append(np.sqrt(np.maximum(s[:, 4] / npx - mu * mu, 0.0)))
            if diff:
                feats.append(s[:, 5] / npx)
            cells.append(np.stack(feats, 1))
    return np.concatenate(cells, 1).reshape(n, -1).astype(np.float32)


def compress(vec: np.ndarray, scale: float) -> np.ndarray:
    """Nén biên độ: c·sign(x)·log(1 + |x|/c)."""
    if not scale or scale <= 0:
        return vec
    v = np.asarray(vec, np.float32)
    return (scale * np.sign(v) * np.log1p(np.abs(v) / scale)).astype(np.float32)
