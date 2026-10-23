"""Biểu diễn chung cho mọi mô hình một lớp dùng kernel RBF.

Cả OC-SVM (Schölkopf) lẫn LS-OC-SVM (bài báo) đều có hàm quyết định dạng

    f(x) = Σ_i a_i · exp(−γ‖x − s_i‖²) − ρ                    (phương trình (3), (21))

Đây chính là phép tính mà bộ tăng tốc FPGA thực hiện, nên mọi mô hình đều được
quy về `KernelExpansion` (s_i, a_i, ρ, γ) trước khi lượng tử hóa và xuất file.

Quy ước điểm bất thường (anomaly score, càng lớn càng bất thường):
    mode = "lower": score = −f(x)   (OC-SVM: f < 0 là bất thường)
    mode = "abs"  : score = |f(x)|  (LS-OC-SVM: khoảng cách tới siêu phẳng, (54)–(55))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np


@dataclass
class Standardizer:
    """z = (x − mean) / scale ; đặc trưng hằng số giữ scale = 1."""
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, x: np.ndarray, min_std: float = 1e-6) -> "Standardizer":
        self.mean = x.mean(axis=0).astype(np.float64)
        std = x.std(axis=0).astype(np.float64)
        self.scale = np.where(std < min_std, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) / self.scale


def rbf_kernel(x: np.ndarray, s: np.ndarray, gamma: float, chunk: int = 4096) -> np.ndarray:
    """K[i, j] = exp(−γ‖x_i − s_j‖²), tính theo khối để tiết kiệm bộ nhớ."""
    x = np.atleast_2d(x).astype(np.float64)
    s = np.atleast_2d(s).astype(np.float64)
    s2 = (s * s).sum(1)
    out = np.empty((len(x), len(s)))
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        d2 = (xb * xb).sum(1)[:, None] + s2[None, :] - 2.0 * xb @ s.T
        out[i:i + chunk] = np.exp(-gamma * np.maximum(d2, 0.0))
    return out


@dataclass
class KernelExpansion:
    support_vectors: np.ndarray            # (n_sv, D) – đã chuẩn hóa
    coefs: np.ndarray                      # (n_sv,)
    rho: float
    gamma: float
    mode: str = "lower"                    # lower | abs

    @property
    def n_sv(self) -> int:
        return len(self.coefs)

    def decision(self, z: np.ndarray) -> np.ndarray:
        return rbf_kernel(z, self.support_vectors, self.gamma) @ self.coefs - self.rho

    def score_from_decision(self, f: np.ndarray) -> np.ndarray:
        return np.abs(f) if self.mode == "abs" else -f

    def decision_threshold_from_score(self, thr_score: float) -> float:
        """Đổi ngưỡng trên miền score sang miền f(x) (dùng cho FPGA)."""
        return thr_score if self.mode == "abs" else -thr_score


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Trung bình trượt nhân quả (chỉ dùng quá khứ) – thực hiện được theo thời gian thực."""
    if k <= 1 or len(x) == 0:
        return x
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


@dataclass
class AnomalyDetector:
    """Gói hoàn chỉnh: chuẩn hóa + mô hình kernel + ngưỡng + làm mượt thời gian."""
    scaler: Standardizer
    model: KernelExpansion
    threshold: float = 0.0                 # trên miền anomaly score
    smoothing: int = 1
    meta: dict = field(default_factory=dict)

    def decision(self, x_raw: np.ndarray) -> np.ndarray:
        return self.model.decision(self.scaler.transform(x_raw))

    def score(self, x_raw: np.ndarray, video_id: np.ndarray | None = None, smooth: bool = True) -> np.ndarray:
        s = self.model.score_from_decision(self.decision(x_raw))
        if smooth and self.smoothing > 1:
            s = smooth_per_video(s, video_id, self.smoothing)
        return s

    def predict(self, x_raw: np.ndarray, video_id: np.ndarray | None = None) -> np.ndarray:
        return (self.score(x_raw, video_id) > self.threshold).astype(np.uint8)

    def save(self, path: str | Path) -> None:
        """Ghi ra file tạm rồi đổi tên → không bao giờ để lại file hỏng nếu bị ngắt giữa chừng."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(self, tmp)
        tmp.replace(path)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"Chưa có mô hình {path} – chạy scripts/02_train.py trước.")
        try:
            return joblib.load(path)
        except Exception as exc:
            raise SystemExit(
                f"Không đọc được {path} ({path.stat().st_size} byte): {exc!r}\n"
                "File mô hình bị hỏng/ghi dở. Xóa file này rồi chạy lại: python scripts/02_train.py") from exc


def smooth_per_video(s: np.ndarray, video_id: np.ndarray | None, k: int) -> np.ndarray:
    if k <= 1:
        return s
    if video_id is None:
        return moving_average(s, k)
    out = np.empty_like(s, dtype=np.float64)
    for v in np.unique(video_id):
        m = video_id == v
        out[m] = moving_average(s[m], k)
    return out
