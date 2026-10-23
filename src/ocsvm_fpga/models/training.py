"""Điều phối huấn luyện: chuẩn hóa → (tìm siêu tham số) → huấn luyện → hiệu chỉnh ngưỡng."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .base import AnomalyDetector, Standardizer
from .ls_ocsvm import train_ls_ocsvm
from .ocsvm import resolve_gamma, train_ocsvm


def fit_detector(x_train: np.ndarray, cfg, rng: np.random.Generator, nu: float | None = None,
                 gamma_mult: float = 1.0, logger=None) -> tuple[AnomalyDetector, dict]:
    m = cfg.model
    if len(x_train) > m.max_train_samples:
        x_train = x_train[rng.choice(len(x_train), m.max_train_samples, replace=False)]
    scaler = Standardizer().fit(x_train)
    z = scaler.transform(x_train)
    gamma = resolve_gamma(m.gamma, z) * gamma_mult
    nu = m.nu if nu is None else nu
    seed = int(cfg.project.seed)
    if m.type == "ocsvm":
        km, info = train_ocsvm(z, nu, gamma, m.max_support_vectors, m.n_prototypes, seed, logger)
    elif m.type == "ls_ocsvm":
        km, info = train_ls_ocsvm(z, gamma, m.ls_C, m.coherence_mu0, m.max_support_vectors,
                                  m.n_prototypes, seed, logger)
    else:
        raise ValueError(f"model.type không hỗ trợ: {m.type}")
    det = AnomalyDetector(scaler=scaler, model=km, smoothing=int(cfg.threshold.temporal_smoothing),
                          meta={"model_type": m.type, "nu": nu, "gamma": gamma,
                                "feature_set": cfg.features.feature_set, **info})
    det.meta["_train_scores"] = det.score(x_train[: min(len(x_train), 20000)], smooth=False)
    return det, info


def calibrate_threshold(det: AnomalyDetector, cfg, val: dict | None, logger=None) -> float:
    """Chọn ngưỡng trên miền anomaly score (đã làm mượt)."""
    t = cfg.threshold
    train_scores = det.meta.pop("_train_scores")
    method = t.method
    if method == "zero" and det.model.mode == "abs":
        method = "max_train"
    if method == "zero":
        thr = 0.0
    elif method == "max_train":
        thr = float(np.max(train_scores))
    elif method == "percentile":
        if val is not None and len(val["X"]):
            s = det.score(val["X"], val["video_id"])
            src = "validation"
        else:
            s, src = train_scores, "train"
        thr = float(np.percentile(s, 100.0 - float(t.percentile)))
        if logger:
            logger.info(f"  Ngưỡng = phân vị {100 - t.percentile:.1f}% điểm {src} bình thường")
    else:
        raise ValueError(f"threshold.method không hỗ trợ: {method}")
    det.threshold = thr
    det.meta["threshold_method"] = method
    return thr


def split_by_video(data: dict, ratio: float, rng: np.random.Generator) -> tuple[dict, dict]:
    vids = np.unique(data["video_id"])
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * ratio)))
    val_mask = np.isin(data["video_id"], vids[:n_val])

    def sub(m):
        return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in data.items()}
    return sub(val_mask), sub(~val_mask)


def tune_hyperparams(x_train, test: dict, cfg, rng, logger=None) -> tuple[dict, list[dict], dict]:
    """Lưới (ν, hệ số γ) – chọn theo AUC trên phần validation của tập Test.

    Trả về (tham số tốt nhất, bảng kết quả, phần Test còn lại dùng để báo cáo).
    """
    val, report = split_by_video(test, cfg.tuning.val_test_ratio, rng)
    nus = cfg.tuning.nu_grid if cfg.model.type == "ocsvm" else [cfg.model.nu]
    rows = []
    for nu in nus:
        for gm in cfg.tuning.gamma_scale_grid:
            det, _ = fit_detector(x_train, cfg, np.random.default_rng(cfg.project.seed), nu, gm)
            det.meta.pop("_train_scores", None)
            s = det.score(val["X"], val["video_id"])
            auc = float(roc_auc_score(val["y"], s))
            rows.append({"nu": nu, "gamma_mult": gm, "gamma": det.model.gamma,
                         "n_sv": det.model.n_sv, "val_auc": auc})
            if logger:
                logger.info(f"  ν={nu:<5} γ×{gm:<5} → SV={det.model.n_sv:<4} AUC(val)={auc:.4f}")
    best = max(rows, key=lambda r: r["val_auc"])
    return best, rows, report
