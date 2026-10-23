"""BƯỚC 3 & 4 – Véc-tơ đặc trưng điểm ảnh F(x, y) và ma trận hiệp biến C.

Bước 3: từ trường (u, v) tính đạo hàm bậc 1 (u_x, u_y, v_x, v_y) và bậc 2
        (u_xx, u_yy, v_xx, v_yy); mỗi điểm ảnh k → z_k = F(x, y) ∈ R^d (Bảng 1).
Bước 4: μ = (1/n) Σ z_k ;  C = 1/(n−1) Σ (z_k − μ)(z_k − μ)^T   (phương trình (44)).

Để SVM/FPGA làm việc với véc-tơ thay vì ma trận, C (thuộc nhóm Lie các ma trận
SPD) được ánh xạ sang không gian tiếp tuyến bằng logarit ma trận (Log-Euclidean):

    vec(C) = [ L_11, √2·L_12, ..., √2·L_1d, L_22, ..., L_dd ],  L = logm(C)

Khi đó ‖vec(C_i) − vec(C_j)‖² = ‖log C_i − log C_j‖_F², nên kernel RBF thông
thường trên vec(C) chính là kernel Gauss Log-Euclidean – xấp xỉ chuẩn của kernel
nhóm Lie exp(−‖log(C_i^{-1} C_j)‖ / 2σ²) ở phương trình (45); hai kernel trùng
nhau khi C_i, C_j giao hoán. Lợi ích: phần cứng FPGA chỉ cần tính ‖x − s‖².
"""
from __future__ import annotations

import numpy as np

# Bảng 1 trong bài báo
FEATURE_SETS: dict[str, list[str]] = {
    "F1": ["y", "x", "u", "v", "ux", "uy"],
    "F2": ["y", "x", "u", "v", "vx", "vy"],
    "F3": ["y", "x", "u", "v", "ux", "uy", "vx", "vy"],
    "F4": ["y", "x", "u", "v", "ux", "uy", "vx", "vy", "uxx", "uyy", "vxx", "vyy"],
}


def feature_dim(feature_set: str) -> int:
    return len(FEATURE_SETS[feature_set])


def descriptor_dim(feature_set: str) -> int:
    d = feature_dim(feature_set)
    return d * (d + 1) // 2


def flow_derivatives(u: np.ndarray, v: np.ndarray, needed: set[str]) -> dict[str, np.ndarray]:
    """Đạo hàm riêng sai phân trung tâm của u, v theo x (cột) và y (hàng)."""
    out: dict[str, np.ndarray] = {}
    for name, f in (("u", u), ("v", v)):
        if not needed & {f"{name}x", f"{name}y", f"{name}xx", f"{name}yy"}:
            continue
        fy, fx = np.gradient(f)                      # axis0 = y, axis1 = x
        out[f"{name}x"], out[f"{name}y"] = fx, fy
        if f"{name}xx" in needed:
            out[f"{name}xx"] = np.gradient(fx, axis=1)
        if f"{name}yy" in needed:
            out[f"{name}yy"] = np.gradient(fy, axis=0)
    return out


def pixel_feature_matrix(u: np.ndarray, v: np.ndarray, feature_set: str = "F3", stride: int = 1,
                         border: int = 2, normalize_coords: bool = True,
                         motion_threshold: float = 0.0, min_motion_pixels: int = 50) -> np.ndarray:
    """Xây ma trận Z (n × d): mỗi hàng là z_k = F(x, y) của một điểm ảnh được lấy mẫu."""
    names = FEATURE_SETS[feature_set]
    h, w = u.shape
    maps: dict[str, np.ndarray] = {"u": u, "v": v}
    maps.update(flow_derivatives(u, v, set(names)))

    ys = np.arange(border, h - border, stride)
    xs = np.arange(border, w - border, stride)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.ravel(), gx.ravel()

    if motion_threshold > 0:
        mag = np.hypot(u[gy, gx], v[gy, gx])
        keep = mag > motion_threshold
        if keep.sum() >= min_motion_pixels:
            gy, gx = gy[keep], gx[keep]

    cols = []
    for n in names:
        if n == "y":
            cols.append(gy / h if normalize_coords else gy.astype(np.float64))
        elif n == "x":
            cols.append(gx / w if normalize_coords else gx.astype(np.float64))
        else:
            cols.append(maps[n][gy, gx])
    return np.stack(cols, axis=1).astype(np.float64)


def covariance_matrix(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """C = 1/(n−1) Σ (z_k − μ)(z_k − μ)^T  (+ εI để đảm bảo SPD)."""
    n = z.shape[0]
    mu = z.mean(axis=0)
    zc = z - mu
    c = (zc.T @ zc) / max(n - 1, 1)
    c = 0.5 * (c + c.T)
    return c + eps * np.eye(c.shape[0])


def spd_logm(c: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """logm của ma trận SPD qua phân rã trị riêng: V·diag(log λ)·V^T."""
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, floor)
    return (v * np.log(w)) @ v.T


def vech_scaled(m: np.ndarray) -> np.ndarray:
    """Tam giác trên (gồm đường chéo); phần tử ngoài đường chéo nhân √2."""
    iu = np.triu_indices(m.shape[0])
    vec = m[iu].copy()
    vec[iu[0] != iu[1]] *= np.sqrt(2.0)
    return vec


def unvech_scaled(vec: np.ndarray, d: int) -> np.ndarray:
    m = np.zeros((d, d))
    iu = np.triu_indices(d)
    vals = vec.copy()
    vals[iu[0] != iu[1]] /= np.sqrt(2.0)
    m[iu] = vals
    return m + np.triu(m, 1).T


def covariance_descriptor(u: np.ndarray, v: np.ndarray, fcfg) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) → (véc-tơ mô tả cho SVM, ma trận hiệp biến C)."""
    z = pixel_feature_matrix(u, v, fcfg.feature_set, fcfg.pixel_stride, fcfg.border,
                             fcfg.normalize_coords, fcfg.motion_threshold, fcfg.min_motion_pixels)
    c = covariance_matrix(z, fcfg.cov_epsilon)
    vec = vech_scaled(spd_logm(c)) if fcfg.log_euclidean else vech_scaled(c)
    return vec.astype(np.float32), c.astype(np.float32)


def multi_threshold_descriptors(u: np.ndarray, v: np.ndarray, fcfg, thresholds) -> tuple[np.ndarray, np.ndarray]:
    """Tính véc-tơ mô tả cho NHIỀU ngưỡng chuyển động trong một lần.

    Đạo hàm và ma trận đặc trưng Z chỉ tính một lần trên toàn lưới; với mỗi ngưỡng
    chỉ chọn lại các hàng có |flow| > ngưỡng rồi tính C và logm (rất rẻ, ma trận d×d).
    Kết quả với từng ngưỡng giống hệt covariance_descriptor(motion_threshold=ngưỡng).

    Returns: (vecs (T, D), C của ngưỡng đầu tiên (d, d))
    """
    z = pixel_feature_matrix(u, v, fcfg.feature_set, fcfg.pixel_stride, fcfg.border,
                             fcfg.normalize_coords, 0.0, fcfg.min_motion_pixels)
    h, w = u.shape
    ys = np.arange(fcfg.border, h - fcfg.border, fcfg.pixel_stride)
    xs = np.arange(fcfg.border, w - fcfg.border, fcfg.pixel_stride)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    mag = np.hypot(u[gy.ravel(), gx.ravel()], v[gy.ravel(), gx.ravel()])
    vecs, c0 = [], None
    for thr in thresholds:
        zz = z
        if thr > 0:
            keep = mag > thr
            if keep.sum() >= fcfg.min_motion_pixels:
                zz = z[keep]
        c = covariance_matrix(zz, fcfg.cov_epsilon)
        if c0 is None:
            c0 = c
        vecs.append(vech_scaled(spd_logm(c)) if fcfg.log_euclidean else vech_scaled(c))
    return np.asarray(vecs, np.float32), c0.astype(np.float32)
