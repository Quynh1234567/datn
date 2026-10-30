"""Chuỗi xử lý đầy đủ: video → {I_i} → {OP_i} → {C_i} → véc-tơ mô tả.

Kết quả của mỗi video được lưu cache dạng .npz trong outputs/features/<split>/,
nên có thể dừng/chạy lại mà không phải tính lại các video đã xong.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import resolve_path
from ..hwmodel import hs_q as HQ
from ..hwmodel import preproc_q as PQ
from ..data.dataset import VideoItem, find_annotation, load_frame_labels
from ..data.video_reader import iter_sensor_frames, smooth_frame, video_info
from .covariance import covariance_descriptor, multi_threshold_descriptors
from .grid_flow import block_stats, compress, from_blocks, grid_flow_descriptor, grid_layout
from .optical_flow import compute_flow

# Khóa cấu hình chỉ ảnh hưởng tới đặc trưng lưới / Farneback. Khi dùng mặc định
# (hiệp biến + Horn–Schunck) chúng bị loại khỏi hash → cache cũ vẫn dùng được.
GRID_KEYS = ("descriptor", "grid_rows", "grid_cols", "grid_stats", "grid_log_scale", "grid_diff",
             "grid_row_edges", "grid_col_edges")
COV_ONLY_KEYS = ("feature_set", "pixel_stride", "border", "normalize_coords",
                 "motion_threshold", "motion_threshold_grid", "min_motion_pixels",
                 "cov_epsilon", "log_euclidean")
FLOW_EXTRA_KEYS = ("method", "fb_pyr_scale", "fb_levels", "fb_winsize",
                   "fb_iterations", "fb_poly_n", "fb_poly_sigma")
HS_KEYS = ("alpha", "num_iter", "intensity_scale")


def descriptor_kind(cfg) -> str:
    kind = cfg.features.get("descriptor", "covariance") or "covariance"
    if kind not in ("covariance", "grid_flow", "grid_flow_hw"):
        raise SystemExit(f"features.descriptor phải là covariance | grid_flow | grid_flow_hw, nhận '{kind}'")
    if kind == "grid_flow_hw":
        check_hw_config(cfg)
    return kind


def check_hw_config(cfg) -> None:
    """grid_flow_hw = đặc trưng tính bằng ĐÚNG Golden Model số nguyên của phần cứng
    (hwmodel/preproc_q: Gauss ESP32 · hwmodel/hs_q: Horn–Schunck + lưới + nén log FPGA).
    Các tham số của phần cứng là hằng số → cấu hình phải khớp, nếu không thì báo rõ."""
    pp, of, f = cfg.preprocess, cfg.optical_flow, cfg.features
    want = {
        "preprocess.width": (pp.width, HQ.W), "preprocess.height": (pp.height, HQ.H),
        "preprocess.gaussian_ksize": (pp.gaussian_ksize, 5), "preprocess.gaussian_sigma": (float(pp.gaussian_sigma), 1.0),
        "optical_flow.method": (of.get("method", "hs"), "hs"), "optical_flow.alpha": (float(of.alpha), 2.0),
        "optical_flow.num_iter": (int(of.num_iter), HQ.DEFAULT.n_iter),
        "optical_flow.intensity_scale": (float(of.intensity_scale), 1.0),
        "features.grid_rows": (int(f.grid_rows), HQ.GRID_R), "features.grid_cols": (int(f.grid_cols), HQ.GRID_C),
        "features.grid_stats": (f.get("grid_stats", "basic"), "basic"),
        "features.grid_log_scale": (float(f.get("grid_log_scale", 0) or 0), HQ.DEFAULT.log_c),
        "features.grid_diff": (bool(f.get("grid_diff", False)), False),
        "features.grid_row_edges": (f.get("grid_row_edges") or None, None),
        "features.grid_col_edges": (f.get("grid_col_edges") or None, None),
    }
    bad = [f"{k} = {got!r} (phần cứng cần {need!r})" for k, (got, need) in want.items() if got != need]
    if bad:
        raise SystemExit("descriptor = grid_flow_hw nhưng cấu hình khác phần cứng:\n  - " + "\n  - ".join(bad))


def threshold_grid(cfg) -> list[float]:
    """Các ngưỡng chuyển động được tính sẵn trong MỘT lần trích xuất (chỉ đặc trưng hiệp biến)."""
    if descriptor_kind(cfg) != "covariance":
        return []
    grid = sorted({float(t) for t in (cfg.features.get("motion_threshold_grid") or [0.0])})
    thr = float(cfg.features.motion_threshold)
    if not any(abs(g - thr) < 1e-9 for g in grid):
        raise SystemExit(f"features.motion_threshold={thr} không nằm trong motion_threshold_grid={grid}. "
                         "Chọn một giá trị trong lưới (không phải trích xuất lại), hoặc thêm giá trị "
                         "vào lưới (sẽ trích xuất lại – nhanh nếu bật preprocess.cache_frames).")
    return grid


def feature_signature(cfg) -> str:
    """Hash các tham số ảnh hưởng tới đặc trưng → tự vô hiệu cache khi đổi cấu hình.

    motion_threshold KHÔNG nằm trong hash: mọi ngưỡng trong motion_threshold_grid
    được lưu chung một cache, đổi ngưỡng trong lưới không phải trích xuất lại.
    """
    kind = descriptor_kind(cfg)
    if kind == "covariance":
        ft = {k: v for k, v in dict(cfg.features).items()
              if k not in ("motion_threshold", "motion_threshold_grid") + GRID_KEYS}
    elif kind == "grid_flow_hw":
        # đặc trưng do Golden Model tính → phụ thuộc phiên bản đặc tả và mọi tham số số học
        from dataclasses import asdict
        ft = {"descriptor": "grid_flow_hw", "spec": getattr(HQ, "SPEC_VERSION", "?"),
              "hs": asdict(HQ.DEFAULT), "gauss": [int(k) for k in PQ.K]}
    else:
        # cache theo khối 10×10: cách chia lưới, thống kê, grid_diff, nén biên độ đều áp dụng lúc ĐỌC cache
        ft = {"descriptor": "grid_flow", "cache": "blocks10"}
    of = dict(cfg.optical_flow)
    if of.get("method", "hs") == "hs":
        of = {k: v for k, v in of.items() if k not in FLOW_EXTRA_KEYS}
    else:
        of = {k: v for k, v in of.items() if k not in HS_KEYS}
    pp = {k: v for k, v in dict(cfg.preprocess).items()
          if k not in ("cache_frames", "frames_cache_dir") and not (k == "mask_regions" and not v)}
    key = json.dumps({"pp": pp, "of": of, "ft": ft,
                      "grid": threshold_grid(cfg)}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def frames_signature(cfg) -> str:
    """Hash riêng cho cache khung hình: chỉ phụ thuộc kích thước, cách crop, frame_step."""
    pp = cfg.preprocess
    k = {"w": pp.width, "h": pp.height, "a": pp.aspect_mode, "s": pp.frame_step}
    if pp.get("mask_regions"):                       # chỉ thêm khi có → cache cũ không đổi
        k["m"] = pp.mask_regions
    key = json.dumps(k)                               # giữ nguyên thứ tự → hash cũ không đổi
    return hashlib.md5(key.encode()).hexdigest()[:8]


def frame_pair_descriptor(prev: np.ndarray, cur: np.ndarray, cfg):
    """Một cặp khung (I_i, I_{i+1}) → (vec, C hoặc None, |flow| trung bình).

    grid_flow_hw: prev, cur là ảnh uint8 CHƯA làm mượt (to_sensor_frame) – Gauss số nguyên làm bên trong."""
    if descriptor_kind(cfg) == "grid_flow_hw":
        if prev.dtype != np.uint8 or cur.dtype != np.uint8:
            raise ValueError("grid_flow_hw cần ảnh uint8 chưa làm mượt (to_sensor_frame), không phải smooth_frame")
        vq, motion = hw_pair_features(PQ.gauss5_q(prev), PQ.gauss5_q(cur))
        return vq.astype(np.float64) / (1 << HQ.DEFAULT.data_frac), None, motion
    u, v = compute_flow(prev, cur, cfg.optical_flow)
    motion = float(np.mean(np.hypot(u, v)))
    if descriptor_kind(cfg) == "grid_flow":
        f = cfg.features
        ye, xe = grid_layout(f, *u.shape)
        diff = np.abs(cur.astype(np.float32) - prev.astype(np.float32)) if f.get("grid_diff", False) else None
        vec = grid_flow_descriptor(u, v, stats=f.get("grid_stats", "basic"), ye=ye, xe=xe, diff_img=diff)
        return compress(vec, float(f.get("grid_log_scale", 0) or 0)), None, motion
    vec, c = covariance_descriptor(u, v, cfg.features)
    return vec, c, motion


def hw_pair_features(g1: np.ndarray, g2: np.ndarray, lut=None) -> tuple[np.ndarray, float]:
    """Hai ảnh đã Gauss số nguyên → (48 đặc trưng Q5.10 int16 sau nén log, |v| trung bình px).
    Giống từng bit khối trích đặc trưng trên FPGA."""
    tr = HQ.features_q(g1, g2, lut=lut)
    return tr.log_q.astype(np.int16), float(tr.mag.mean()) / (1 << HQ.DEFAULT.fu)


def frames_cache_root(cfg, out_root: Path | None) -> Path | None:
    """Thư mục chứa cache khung: preprocess.frames_cache_dir (dùng chung giữa các outputs_*) hoặc out_root."""
    d = cfg.preprocess.get("frames_cache_dir")
    return resolve_path(d) if d else out_root


def load_sensor_frames(item: VideoItem, cfg, out_root: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Khung uint8 (N, H, W) + chỉ số khung gốc. Dùng cache nếu preprocess.cache_frames.

    Giải mã video 1080p là phần tốn thời gian nhất; cache này chỉ phụ thuộc kích
    thước/crop/frame_step nên đổi tham số HS hay đặc trưng không phải giải mã lại.
    """
    root = frames_cache_root(cfg, out_root)
    use_cache = bool(cfg.preprocess.get("cache_frames", False)) and root is not None
    path = root / "frames" / frames_signature(cfg) / item.split / f"{item.uid}.npz" if use_cache else None
    if path is not None and path.exists():
        d = np.load(path)
        return d["frames"], d["frame_idx"]
    idx, frames = [], []
    for i, f in iter_sensor_frames(item.path, cfg.preprocess):
        idx.append(i)
        frames.append(f)
    frames_arr = np.stack(frames) if frames else np.zeros((0, cfg.preprocess.height, cfg.preprocess.width), np.uint8)
    idx_arr = np.asarray(idx, np.int32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, frames=frames_arr, frame_idx=idx_arr)
        tmp.replace(path)
    return frames_arr, idx_arr


def extract_video(item: VideoItem, cfg, out_root: Path | None = None) -> dict:
    """Trích xuất véc-tơ mô tả của một video cho MỌI ngưỡng trong motion_threshold_grid."""
    grid = threshold_grid(cfg)
    kind = descriptor_kind(cfg)
    frames, fidx = load_sensor_frames(item, cfg, out_root)
    f = cfg.features
    dark = float(cfg.preprocess.get("skip_dark_below", 0) or 0)
    vecs, covs, frame_ids, motion = [], [], [], []
    prev = None
    if kind == "grid_flow_hw":
        lut = HQ.log_lut()
        for idx, raw in zip(fidx, frames):
            g = PQ.gauss5_q(np.ascontiguousarray(raw, np.uint8))
            if dark > 0 and PQ.frame_sum(g) < dark * g.size:   # = TB ảnh Gauss số nguyên < ngưỡng (giống ESP32)
                prev = None
                continue
            if prev is not None:
                vq, m = hw_pair_features(prev, g, lut)
                vecs.append(vq)
                frame_ids.append(int(idx))
                motion.append(m)
            prev = g
    float_frames = () if kind == "grid_flow_hw" else zip(fidx, frames)   # đường số thực (covariance / grid_flow)
    for idx, raw in float_frames:
        frame = smooth_frame(raw, cfg.preprocess)
        if dark > 0 and float(frame.mean()) < dark:     # khung đen/chuyển cảnh → ngắt cặp
            prev = None
            continue
        if prev is not None:
            u, v = compute_flow(prev, frame, cfg.optical_flow)
            if kind == "covariance":
                vg, c = multi_threshold_descriptors(u, v, f, grid)
                covs.append(c)
            else:
                vg = block_stats(u, v, np.abs(frame.astype(np.float32) - prev.astype(np.float32)))
            vecs.append(vg)
            frame_ids.append(int(idx))                  # nhãn của cặp = nhãn khung thứ hai
            motion.append(float(np.mean(np.hypot(u, v))))
        prev = frame
    info = video_info(item.path)
    out = {}
    if kind == "grid_flow_hw":
        out["Xq"] = np.stack(vecs).astype(np.int16) if vecs else np.zeros((0, HQ.DIM), np.int16)
    elif kind == "covariance":
        d = len(vecs[0][0]) if vecs else 0
        out["Xg"] = np.asarray(vecs, np.float32).reshape(len(vecs), len(grid), d)
        out["thr_grid"] = np.asarray(grid, np.float64)
        out["C"] = np.asarray(covs, np.float32)
    else:                                              # đặc trưng lưới: một véc-tơ / cặp khung
        out["B"] = (np.stack(vecs).astype(np.float32) if vecs
                    else np.zeros((0, frames.shape[1] // 10, frames.shape[2] // 10, 6), np.float32))
    out.update({
        "frame_idx": np.asarray(frame_ids, np.int32),
        "motion": np.asarray(motion, np.float32),
        "n_frames": np.int32(info["frames"]),
    })
    if item.labels_path is not None:
        labels = load_frame_labels(item.labels_path, info["frames"])
        fi = np.clip(out["frame_idx"], 0, len(labels) - 1)
        out["y"] = labels[fi].astype(np.uint8)
    return out


def select_threshold(d: dict, cfg) -> np.ndarray:
    """Lấy véc-tơ ứng với cfg.features.motion_threshold từ cache nhiều ngưỡng."""
    if "Xg" not in d:                                  # cache lưới (khối) hoặc định dạng cũ
        return d.get("X")
    grid = np.asarray(d["thr_grid"])
    thr = float(cfg.features.motion_threshold)
    k = int(np.argmin(np.abs(grid - thr)))
    if abs(grid[k] - thr) > 1e-9:
        raise ValueError(f"motion_threshold={thr} không có trong cache {grid.tolist()} – "
                         "thêm vào features.motion_threshold_grid rồi trích xuất lại")
    return d["Xg"][:, k, :]


def cache_path(item: VideoItem, cfg, out_root: Path) -> Path:
    return out_root / "features" / feature_signature(cfg) / item.split / f"{item.uid}.npz"


def _process(item: VideoItem, cfg, out_root: Path) -> tuple[str, int, str | None]:
    path = cache_path(item, cfg, out_root)
    if path.exists():
        return item.uid, -1, None
    try:
        res = extract_video(item, cfg, out_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **res, video=str(item.path))
        tmp.replace(path)
        return item.uid, len(res["frame_idx"]), None
    except Exception as exc:                        # một video lỗi không làm dừng cả tập
        return item.uid, 0, repr(exc)


def extract_split(items: list[VideoItem], cfg, out_root: Path, n_jobs: int = 1, logger=None):
    """Trích xuất song song cho cả một tập video (bỏ qua video đã có cache)."""
    from ..utils import progress
    if not items:
        return
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_process)(it, cfg, out_root) for it in items)
    else:
        results = [_process(it, cfg, out_root) for it in progress(items, desc=f"[{items[0].split}]")]
    for uid, n, err in results:
        if err and logger:
            logger.error(f"Lỗi video {uid}: {err}")


def load_split(items: list[VideoItem], cfg, out_root: Path) -> dict:
    """Ghép cache của cả tập → X (N, D), y (N,) nếu có, video_id (N,), frame_idx (N,).

    Nếu chỉ MỘT PHẦN video có nhãn (đang gán nhãn dở), chỉ giữ các video có nhãn
    để số liệu đánh giá không bị sai; số video bị bỏ ghi ở khóa "n_unlabeled_dropped".
    """
    per_video = []
    for vid, item in enumerate(items):
        path = cache_path(item, cfg, out_root)
        if not path.exists():
            continue
        d = dict(np.load(path))
        d["X"] = select_threshold(d, cfg)
        if descriptor_kind(cfg) == "grid_flow_hw":       # đã nén log trong phần cứng → chỉ đổi Q5.10 → số thực
            d["X"] = d["Xq"].astype(np.float64) / (1 << HQ.DEFAULT.data_frac)
        elif descriptor_kind(cfg) == "grid_flow":
            f = cfg.features
            if "B" in d:                                 # ghép lưới từ thống kê khối
                ye, xe = grid_layout(f, int(cfg.preprocess.height), int(cfg.preprocess.width))
                d["X"] = from_blocks(d["B"], ye, xe, f.get("grid_stats", "basic"), bool(f.get("grid_diff", False)))
            if float(f.get("grid_log_scale", 0) or 0) > 0:
                d["X"] = compress(d["X"], float(f.grid_log_scale))
        if len(d["X"]) == 0:
            continue
        # Nhãn được đọc lại MỖI LẦN nạp (không phụ thuộc lúc trích đặc trưng),
        # nên có thể gán nhãn sau khi đã trích xuất mà không phải tính lại.
        if item.split == "test":
            lab = item.labels_path if item.labels_path and Path(item.labels_path).exists() else None
            if lab is None:
                ann = resolve_path(cfg.data.annotation_dir) if cfg.data.annotation_dir else None
                lab = find_annotation(item.path.stem, ann, cfg.data.annotation_patterns, item.path)
            if lab is not None:
                labels = load_frame_labels(lab, int(d["n_frames"]))
                d["y"] = labels[np.clip(d["frame_idx"], 0, len(labels) - 1)].astype(np.uint8)
            else:
                d.pop("y", None)
        per_video.append((vid, d))
    labeled = [p for p in per_video if "y" in p[1]]
    dropped = 0
    if labeled and len(labeled) < len(per_video):
        dropped = len(per_video) - len(labeled)
        per_video = labeled
    if not per_video:
        return {"X": np.zeros((0, 0), np.float32), "y": None, "video_id": np.zeros(0, np.int32),
                "frame_idx": np.zeros(0, np.int32), "motion": np.zeros(0, np.float32),
                "n_unlabeled_dropped": 0}
    has_labels = all("y" in d for _, d in per_video)
    return {
        "X": np.concatenate([d["X"] for _, d in per_video]),
        "y": np.concatenate([d["y"] for _, d in per_video]) if has_labels else None,
        "video_id": np.concatenate([np.full(len(d["X"]), v, np.int32) for v, d in per_video]),
        "frame_idx": np.concatenate([d["frame_idx"] for _, d in per_video]),
        "motion": np.concatenate([d["motion"] for _, d in per_video]),
        "n_unlabeled_dropped": dropped,
    }
