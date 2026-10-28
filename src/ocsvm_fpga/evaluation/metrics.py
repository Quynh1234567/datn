"""Đánh giá mức khung hình: AUC, EER, Precision/Recall/F1 và các biểu đồ báo cáo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def frame_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """y: 1 = bất thường; score: càng lớn càng bất thường."""
    pred = (score > threshold).astype(int)
    out = {
        "n_frames": int(len(y)),
        "abnormal_ratio": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix[tn,fp,fn,tp]": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }
    if len(np.unique(y)) == 2:
        fpr, tpr, thr = roc_curve(y, score)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["auc"] = float(roc_auc_score(y, score))
        out["eer"] = float((fpr[i] + fnr[i]) / 2)
        out["eer_threshold"] = float(thr[i])
        j = int(np.argmax(tpr - fpr))
        out["youden_threshold"] = float(thr[j])
    return out


def per_video_auc(y: np.ndarray, score: np.ndarray, video_id: np.ndarray) -> float | None:
    """AUC trung bình theo từng video (chỉ các video có cả hai lớp)."""
    aucs = [roc_auc_score(y[video_id == v], score[video_id == v])
            for v in np.unique(video_id) if len(np.unique(y[video_id == v])) == 2]
    return float(np.mean(aucs)) if aucs else None


def video_level_auc(y: np.ndarray, score: np.ndarray, video_id: np.ndarray, agg: str = "mean"):
    """AUC mức VIDEO: điểm video = trung bình (hoặc max) điểm khung; nhãn video = 1 nếu có khung bất thường.
    Dùng cho các bộ chỉ có nhãn cả video (vd thư mục Normal/Abnormal)."""
    vids = np.unique(video_id)
    f = np.mean if agg == "mean" else np.max
    vs = np.array([f(score[video_id == v]) for v in vids])
    vy = np.array([int(y[video_id == v].max()) for v in vids])
    if len(np.unique(vy)) < 2:
        return None
    return float(roc_auc_score(vy, vs)), int((vy == 0).sum()), int((vy == 1).sum())


def per_class_auc(y: np.ndarray, score: np.ndarray, cls: np.ndarray) -> dict[str, float]:
    """AUC từng loại hành vi: khung của video loại đó + khung của các video "Normal"."""
    out = {}
    for c in sorted(set(cls.tolist()) - {"Normal", "?"}):
        m = (cls == c) | (cls == "Normal")
        if len(np.unique(y[m])) == 2:
            out[c] = float(roc_auc_score(y[m], score[m]))
    return out


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, video_id: np.ndarray, n_boot: int = 200,
                     seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Khoảng tin cậy AUC bằng bootstrap THEO VIDEO (lấy mẫu lại cả video, không lấy lẻ khung,
    vì các khung trong một video tương quan mạnh với nhau)."""
    rng = np.random.default_rng(seed)
    vids = np.unique(video_id)
    idx_by_vid = {v: np.flatnonzero(video_id == v) for v in vids}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(vids, len(vids), replace=True)
        idx = np.concatenate([idx_by_vid[v] for v in pick])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], score[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_roc(y, score, path: Path, title="ROC – mức khung hình"):
    plt = _plt()
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_hist(scores: dict[str, np.ndarray], threshold: float, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s in scores.items():
        if len(s):
            ax.hist(s, bins=80, alpha=.5, density=True, label=name)
    ax.axvline(threshold, c="r", ls="--", label=f"ngưỡng = {threshold:.4f}")
    ax.set(xlabel="Điểm bất thường (anomaly score)", ylabel="Mật độ", title="Phân bố điểm số")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timelines(score, y, video_id, threshold, path: Path, max_videos: int = 6):
    plt = _plt()
    vids = np.unique(video_id)[:max_videos]
    fig, axes = plt.subplots(len(vids), 1, figsize=(10, 2.2 * len(vids)), squeeze=False)
    for ax, v in zip(axes[:, 0], vids):
        m = video_id == v
        s = score[m]
        ax.plot(s, lw=1, label="score")
        if y is not None:
            yy = y[m]
            ax.fill_between(np.arange(len(s)), s.min(), s.max(), where=yy > 0,
                            color="red", alpha=.15, label="nhãn bất thường")
        ax.axhline(threshold, c="k", ls="--", lw=1)
        ax.set_ylabel(f"video {v}")
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Chỉ số cặp khung")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
