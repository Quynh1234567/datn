"""One-Class SVM (Schölkopf, 2001) – thuật toán chính của đồ án.

    min_{w,ξ,ρ} ½‖w‖² − ρ + 1/(νn) Σ ξ_i   s.t.  ⟨w, Φ(x_i)⟩ ≥ ρ − ξ_i, ξ_i ≥ 0   (1)
    f(x) = Σ α_i κ(x_i, x) − ρ ;  f(x) < 0 ⇒ bất thường                           (3)

Ràng buộc phần cứng: số Support Vector (SV) quyết định dung lượng BRAM và độ trễ
của bộ tăng tốc (mỗi SV = một lần tính kernel). Tập huấn luyện từ 208 video có
hàng trăm nghìn khung → được nén bằng MiniBatchKMeans thành các "nguyên mẫu"
(prototype) có trọng số = số khung thuộc cụm, rồi huấn luyện OC-SVM có trọng số.
Nếu số SV vượt `max_support_vectors`, số nguyên mẫu được giảm dần và huấn luyện lại.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import OneClassSVM

from .base import KernelExpansion


def resolve_gamma(gamma, z: np.ndarray) -> float:
    if isinstance(gamma, str):
        if gamma != "scale":
            raise ValueError(f"gamma không hợp lệ: {gamma}")
        var = float(z.var())
        return 1.0 / (z.shape[1] * var) if var > 0 else 1.0
    return float(gamma)


def make_prototypes(z: np.ndarray, n_prototypes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Nén z thành n_prototypes tâm cụm kèm trọng số (số phần tử mỗi cụm)."""
    if len(z) <= n_prototypes:
        return z, np.ones(len(z))
    km = MiniBatchKMeans(n_clusters=n_prototypes, random_state=seed, batch_size=4096,
                         n_init=3, max_iter=100)
    labels = km.fit_predict(z)
    counts = np.bincount(labels, minlength=n_prototypes).astype(np.float64)
    keep = counts > 0
    return km.cluster_centers_[keep], counts[keep]


def train_ocsvm(z: np.ndarray, nu: float, gamma: float, max_sv: int, n_prototypes: int,
                seed: int = 0, logger=None) -> tuple[KernelExpansion, dict]:
    """Huấn luyện OC-SVM thỏa ngân sách SV. z đã được chuẩn hóa."""
    n_proto = min(n_prototypes, len(z))
    history = []
    while True:
        protos, weights = make_prototypes(z, n_proto, seed)
        # chuẩn hóa trọng số về trung bình 1 để ν giữ đúng ý nghĩa
        w = weights * (len(weights) / weights.sum())
        clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma, cache_size=1024, tol=1e-4)
        clf.fit(protos, sample_weight=w)
        n_sv = len(clf.support_)
        history.append({"n_prototypes": int(len(protos)), "n_sv": int(n_sv)})
        if logger:
            logger.info(f"  OC-SVM: {len(protos)} nguyên mẫu → {n_sv} SV (ngân sách {max_sv})")
        if n_sv <= max_sv or n_proto <= max(2 * max_sv // 3, 16):
            break
        n_proto = max(int(len(protos) * min(0.85, max_sv / n_sv)), 16)

    km = KernelExpansion(
        support_vectors=clf.support_vectors_.astype(np.float64),
        coefs=clf.dual_coef_.ravel().astype(np.float64),
        rho=float(-clf.intercept_[0]),
        gamma=float(gamma),
        mode="lower",
    )
    if km.n_sv > max_sv:   # vẫn vượt → giữ các SV có |α| lớn nhất (cảnh báo)
        order = np.argsort(-np.abs(km.coefs))[:max_sv]
        km.support_vectors, km.coefs = km.support_vectors[order], km.coefs[order]
        if logger:
            logger.warning(f"  Cắt bớt còn {max_sv} SV có |α| lớn nhất – nên giảm ν.")
    # kiểm tra: biểu diễn KernelExpansion trùng với sklearn
    ref = clf.decision_function(protos[:200])
    err = float(np.max(np.abs(km.decision(protos[:200]) - ref))) if km.n_sv == len(clf.support_) else None
    return km, {"history": history, "n_train_points": int(len(z)), "sklearn_consistency_err": err}
