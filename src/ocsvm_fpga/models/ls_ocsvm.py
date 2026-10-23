"""LS-OC-SVM thưa dùng từ điển coherence (Wang et al., Sensors 2013, mục 3.2–3.4).

Mô hình so sánh với OC-SVM chuẩn. Hàm mất mát bình phương (4) cho lời giải
bằng hệ phương trình tuyến tính thay vì quy hoạch toàn phương:

  Từ điển x_D chọn theo tiêu chí coherence (32):
      ε_t = max_j |κ(x_t, x_wj)| ;  thêm x_t vào từ điển nếu ε_t ≤ μ0
  Hệ số (30):
      [ K_D(x) K_D⁻¹ K_D(x)ᵀ + I/C   1 ] [  α ]   [ 0 ]
      [            1ᵀ                0 ] [ −ρ ] = [ 1 ]
  Trọng số từ điển (24):  β = K_D⁻¹ K_D(x)ᵀ α
  Siêu phẳng (21):        f(x) = Σ_j β_j κ(x, x_wj) − ρ
  Khoảng cách (22):       dis(x) = |f(x)| / ‖β‖ ;  bất thường nếu dis ≥ T_dis (55)

Ghi chú: phương trình (11)/(30) in "CI" nhưng từ (7) Cξ_i = α_i suy ra số hạng
đúng là I/C – cài đặt dùng I/C. Lời giải batch này trùng với kết quả online
(bài báo khẳng định online cho kết quả như batch), phù hợp cho huấn luyện trên PC.
"""
from __future__ import annotations

import numpy as np

from .base import KernelExpansion, rbf_kernel
from .ocsvm import make_prototypes


def build_coherence_dictionary(z: np.ndarray, gamma: float, mu0: float, max_size: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Chọn chỉ số từ điển theo tiêu chí coherence (32), giới hạn max_size phần tử."""
    order = rng.permutation(len(z))
    dict_idx = [order[0]]
    for t in order[1:]:
        if len(dict_idx) >= max_size:
            break
        eps_t = rbf_kernel(z[t:t + 1], z[dict_idx], gamma).max()
        if eps_t <= mu0:
            dict_idx.append(t)
    return np.asarray(dict_idx)


def train_ls_ocsvm(z: np.ndarray, gamma: float, C: float, mu0: float, max_sv: int,
                   n_prototypes: int, seed: int = 0, logger=None) -> tuple[KernelExpansion, dict]:
    rng = np.random.default_rng(seed)
    x, _ = make_prototypes(z, n_prototypes, seed)
    d_idx = build_coherence_dictionary(x, gamma, mu0, max_sv, rng)
    xd = x[d_idx]
    n, m = len(x), len(xd)

    kd = rbf_kernel(xd, xd, gamma) + 1e-8 * np.eye(m)           # K_D
    kdx = rbf_kernel(x, xd, gamma)                              # K_D(x): n × D
    kd_inv_kdxT = np.linalg.solve(kd, kdx.T)                    # K_D⁻¹ K_D(x)ᵀ : D × n
    a = kdx @ kd_inv_kdxT + np.eye(n) / C
    lhs = np.zeros((n + 1, n + 1))
    lhs[:n, :n] = a
    lhs[:n, n] = 1.0
    lhs[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    sol = np.linalg.solve(lhs, rhs)
    alpha, rho = sol[:n], -sol[n]
    beta = kd_inv_kdxT @ alpha                                  # (24)

    # Chuẩn hóa để f(x) chính là khoảng cách dis(x) ở (22)
    nb = float(np.linalg.norm(beta))
    km = KernelExpansion(support_vectors=xd, coefs=beta / nb, rho=rho / nb, gamma=gamma, mode="abs")
    if logger:
        logger.info(f"  LS-OC-SVM: {n} nguyên mẫu, từ điển μ0={mu0} → {m} phần tử")
    train_dis = np.abs(km.decision(x))
    return km, {"n_prototypes": n, "dict_size": m, "T_dis_max_train": float(train_dis.max())}
