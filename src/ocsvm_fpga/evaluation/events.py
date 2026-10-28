"""Cảnh báo có trễ (hysteresis) và đánh giá theo SỰ KIỆN.

Hysteresis: cảnh báo BẬT khi điểm (đã làm mượt) vượt ngưỡng bật `on`; chỉ TẮT khi điểm
xuống dưới ngưỡng tắt `off` (< on) LIÊN TỤC `hold` cặp khung. Nhờ vậy điểm dao động quanh
ngưỡng (người khuất sau cột, đổi hướng…) không làm cảnh báo bật/tắt liên tục.

    off = phân vị `off_percentile` (%) điểm của dữ liệu bình thường kiểm định (mặc định 85%),
          không vượt quá on. (Cách cũ: off = on − off_margin × (on − trung vị) khi off_percentile = null.)
    hold = round(hold_seconds × fps / frame_step)

Đánh giá theo sự kiện: mỗi đoạn liên tục có nhãn bất thường trong một video là MỘT sự kiện.
  • phát hiện  : có ít nhất một khung cảnh báo trong đoạn [bắt đầu, kết thúc]
  • độ trễ     : thời gian từ khung bắt đầu sự kiện đến khung cảnh báo đầu tiên trong đoạn
  • báo động giả: đoạn cảnh báo liên tục KHÔNG giao với sự kiện nào (tính theo số lần / giờ
                 trên tổng thời lượng phần bình thường)
"""
from __future__ import annotations

import numpy as np


def alarm_settings(cfg) -> dict:
    a = cfg.get("alarm") or {}
    op = a.get("off_percentile", 85.0)
    return {"hysteresis": bool(a.get("hysteresis", True)), "off_margin": float(a.get("off_margin", 0.5)),
            "off_percentile": None if op is None else float(op),
            "hold_seconds": float(a.get("hold_seconds", 1.0)), "video_fps": float(a.get("video_fps", 25.0))}


def alarm_params(det, cfg, normal_scores=None) -> dict:
    """{on, off, hold} – lấy trong det.meta['alarm'] nếu có, nếu không tính từ điểm bình thường."""
    s = alarm_settings(cfg)
    on = float(det.threshold)
    hold = max(0, int(round(s["hold_seconds"] * s["video_fps"] / max(1, int(cfg.preprocess.frame_step)))))
    if not s["hysteresis"]:
        return {"on": on, "off": on, "hold": 0, "median_normal": None, "settings": s}
    meta = det.meta.get("alarm") if hasattr(det, "meta") else None
    if (normal_scores is None and meta and meta.get("off") is not None and meta.get("settings") == s
            and abs(float(meta.get("on", on)) - on) < 1e-12):
        return meta                                      # đã tính lúc huấn luyện với đúng cấu hình này
    if normal_scores is None or len(normal_scores) == 0:
        return {"on": on, "off": on, "hold": hold, "median_normal": None, "settings": s}
    med = float(np.median(normal_scores))
    if s["off_percentile"] is not None:
        off = min(on, float(np.percentile(normal_scores, s["off_percentile"])))
    else:
        off = on - s["off_margin"] * max(on - med, 0.0)
    return {"on": on, "off": off, "hold": hold, "median_normal": med, "settings": s}


def alarm_params_need_normal(det, cfg) -> bool:
    """True nếu tham số lưu trong mô hình không khớp cấu hình hiện tại → cần điểm dữ liệu bình thường."""
    meta = det.meta.get("alarm")
    return not (meta and meta.get("off") is not None and meta.get("settings") == alarm_settings(cfg)
                and abs(float(meta.get("on", det.threshold)) - float(det.threshold)) < 1e-12)


def hysteresis(scores: np.ndarray, on: float, off: float, hold: int) -> np.ndarray:
    """Cảnh báo 0/1 cho MỘT chuỗi điểm liên tục (một video)."""
    out = np.zeros(len(scores), np.uint8)
    active, below = False, 0
    for i, s in enumerate(scores):
        if not active:
            if s > on:
                active, below = True, 0
        else:
            below = below + 1 if s < off else 0
            if below > hold:
                active = False
        out[i] = active
    return out


def hysteresis_per_video(scores, video_id, on, off, hold) -> np.ndarray:
    out = np.zeros(len(scores), np.uint8)
    for v in np.unique(video_id):
        m = np.flatnonzero(video_id == v)
        out[m] = hysteresis(scores[m], on, off, hold)
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn liên tục mask = 1, dạng (đầu, cuối) theo chỉ số (bao gồm cả cuối)."""
    m = np.asarray(mask).astype(np.int8)
    if not len(m):
        return []
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def event_metrics(y, alarm, video_id, frame_idx, fps: float, classes=None) -> dict:
    """Chỉ số theo sự kiện. frame_idx = chỉ số khung gốc của từng cặp (để tính độ trễ theo giây)."""
    y, alarm = np.asarray(y).astype(np.uint8), np.asarray(alarm).astype(np.uint8)
    events, false_alarms, normal_seconds = [], 0, 0.0
    for v in np.unique(video_id):
        m = np.flatnonzero(video_id == v)
        yv, av, fv = y[m], alarm[m], frame_idx[m]
        if len(fv) > 1:
            dt = np.median(np.diff(fv)) / fps
            normal_seconds += float((yv == 0).sum() * dt)
        for s, e in runs(yv):
            hit = np.flatnonzero(av[s:e + 1])
            ev = {"video": int(v), "start_frame": int(fv[s]), "end_frame": int(fv[e]),
                  "detected": bool(len(hit)), "cls": classes.get(int(v), "?") if classes else "?"}
            if len(hit):
                ev["delay_s"] = float((fv[s + hit[0]] - fv[s]) / fps)
                ev["coverage"] = float(av[s:e + 1].mean())
            events.append(ev)
        for s, e in runs(av):
            if not yv[s:e + 1].any():
                false_alarms += 1
    det = [e for e in events if e["detected"]]
    rep = {
        "events": len(events), "detected": len(det),
        "event_recall": len(det) / len(events) if events else float("nan"),
        "delay_median_s": float(np.median([e["delay_s"] for e in det])) if det else float("nan"),
        "delay_mean_s": float(np.mean([e["delay_s"] for e in det])) if det else float("nan"),
        "coverage_mean": float(np.mean([e["coverage"] for e in det])) if det else float("nan"),
        "false_alarm_events": false_alarms,
        "normal_hours": normal_seconds / 3600,
        "false_alarms_per_hour": false_alarms / (normal_seconds / 3600) if normal_seconds else float("nan"),
        "list": events,
    }
    if classes:
        per = {}
        for c in sorted({e["cls"] for e in events} - {"?"}):
            ec = [e for e in events if e["cls"] == c]
            dc = [e for e in ec if e["detected"]]
            per[c] = {"events": len(ec), "recall": len(dc) / len(ec),
                      "delay_median_s": float(np.median([e["delay_s"] for e in dc])) if dc else float("nan")}
        rep["per_class"] = per
    return rep


def frame_metrics_binary(y, alarm) -> dict:
    y, a = np.asarray(y).astype(bool), np.asarray(alarm).astype(bool)
    tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "false_alarm_rate": float(a[~y].mean()) if (~y).any() else float("nan")}
