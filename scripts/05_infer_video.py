#!/usr/bin/env python3
"""Suy luận trên một video bất kỳ (vd video ghi từ ESP32-S3/OV5640) hoặc webcam.

Mô phỏng đúng luồng demo: khung → HS → hiệp biến → chuẩn hóa → OC-SVM (float và
fixed-point bit-accurate như FPGA) → làm mượt → cảnh báo.

    python scripts/05_infer_video.py --video path/to/clip.avi --save-video
    python scripts/05_infer_video.py --camera 0
"""
import csv
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, setup  # noqa: E402
from ocsvm_fpga.data.video_reader import iter_frames, preprocess_frame  # noqa: E402
from ocsvm_fpga.features.pipeline import frame_pair_descriptor  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402


def camera_frames(index, pp):
    cap = cv2.VideoCapture(index)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, pp.frame_step) == 0:
            yield i, preprocess_frame(frame, pp), frame
        i += 1


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--save-video", action="store_true", help="ghi video có chú thích kết quả")
    args = ap.parse_args()
    if not args.video and args.camera is None:
        ap.error("cần --video <đường dẫn video> hoặc --camera <số>")
    cfg, out, log, _ = setup(args, "05_infer")
    det = AnomalyDetector.load(out / "models" / "detector.joblib")
    fx = FixedPointOCSVM(det, cfg.fpga)
    k = max(1, det.smoothing)
    win_float, win_fix = deque(maxlen=k), deque(maxlen=k)

    if args.video:
        src = ((i, g, None) for i, g in iter_frames(args.video, cfg.preprocess))
        name = Path(args.video).stem
    elif args.camera is not None:
        src, name = camera_frames(args.camera, cfg.preprocess), f"camera{args.camera}"
    else:
        ap.error("cần --video hoặc --camera")

    res_dir = out / "inference"
    res_dir.mkdir(parents=True, exist_ok=True)
    writer, rows, prev = None, [], None
    for idx, gray, bgr in src:
        if prev is not None:
            vec, _, motion = frame_pair_descriptor(prev, gray, cfg)
            s_float = float(det.model.score_from_decision(det.decision(vec[None]))[0])
            xq, _ = fx.quantize_input(vec[None])
            f_q = fx.decision_q(xq)
            s_fix = float(det.model.score_from_decision(fx.decision_float_equiv(f_q))[0])
            win_float.append(s_float)
            win_fix.append(s_fix)
            sm_float, sm_fix = np.mean(win_float), np.mean(win_fix)
            alarm = int(sm_fix > det.threshold)
            rows.append([idx, s_float, s_fix, sm_float, sm_fix, int(fx.anomaly_q(f_q)[0]), alarm, motion])
            vis = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis = cv2.resize(vis, (vis.shape[1] * 4, vis.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            color = (0, 0, 255) if alarm else (0, 200, 0)
            cv2.putText(vis, f"{'ABNORMAL' if alarm else 'NORMAL'}  s={sm_fix:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 4)
            if args.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(res_dir / f"{name}_result.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.camera is not None:
                cv2.imshow("OC-SVM anomaly", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        prev = gray
    if writer:
        writer.release()
    with open(res_dir / f"{name}_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "score_float", "score_fixed", "smooth_float", "smooth_fixed",
                    "fpga_flag_raw", "alarm", "mean_motion"])
        w.writerows(rows)
    if rows:
        arr = np.array(rows)
        log.info(f"{len(rows)} cặp khung | tỉ lệ cảnh báo = {arr[:, 6].mean():.2%} | "
                 f"sai số float/fixed TB = {np.abs(arr[:, 1] - arr[:, 2]).mean():.2e}")
    log.info(f"Kết quả → {res_dir}")


if __name__ == "__main__":
    main()
