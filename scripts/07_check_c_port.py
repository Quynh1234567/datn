#!/usr/bin/env python3
"""So sánh thư viện C cho ESP32 (firmware/esp32_features) với pipeline Python.

Biên dịch thư viện C bằng gcc trên PC, đưa cùng chuỗi khung hình vào cả hai
phía rồi so: véc-tơ đặc trưng, x_q gửi FPGA, f_q và cờ bất thường. Đạt yêu cầu
nghĩa là firmware ESP32 tạo ra đúng đầu vào mà mô hình đã được huấn luyện.

    python scripts/07_check_c_port.py                      # video đầu tiên của tập test/val
    python scripts/07_check_c_port.py --video clip.avi --frames 120
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, get_splits, setup  # noqa: E402
from ocsvm_fpga.data.video_reader import smooth_frame, to_sensor_frame  # noqa: E402
from ocsvm_fpga.features.pipeline import frame_pair_descriptor  # noqa: E402
from ocsvm_fpga.models.base import AnomalyDetector  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402
from ocsvm_fpga.utils import save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "esp32_features"


def sensor_frames(path, pp, n_max):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while len(out) < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, pp.frame_step) == 0:
            out.append(to_sensor_frame(frame, pp))
        i += 1
    cap.release()
    return out


def expected_dim(cfg) -> int:
    f = cfg.features
    if f.get("descriptor", "covariance") == "grid_flow":
        from ocsvm_fpga.features.grid_flow import grid_dim
        return grid_dim(f, int(cfg.preprocess.height), int(cfg.preprocess.width))
    from ocsvm_fpga.features.covariance import descriptor_dim
    return descriptor_dim(f.feature_set)


def preflight(cfg, out, fpga_dir):
    """Kiểm tra cấu hình ↔ mô hình ↔ file .h ↔ firmware khớp nhau trước khi biên dịch,
    để báo đúng nguyên nhân thay vì lỗi biên dịch / lỗi kích thước khó hiểu."""
    import re
    grid = cfg.features.get("descriptor", "covariance") == "grid_flow"
    want = expected_dim(cfg)
    kind = (f"lưới {cfg.features.grid_rows}×{cfg.features.grid_cols} ({cfg.features.grid_stats})" if grid
            else f"hiệp biến {cfg.features.feature_set}")
    problems = []
    det = AnomalyDetector.load(out / "models" / "detector.joblib")
    got = int(det.model.support_vectors.shape[1])
    if got != want:
        problems.append(f"Mô hình outputs/models/detector.joblib có {got} chiều, cấu hình hiện tại ({kind}) cần "
                        f"{want} chiều → huấn luyện lại:  python scripts/02_train.py   (rồi chạy lại 04)")
    hdr = (fpga_dir / "ocsvm_params.h").read_text(encoding="utf-8", errors="ignore")
    m_dim = re.search(r"#define\s+OCSVM_DIM\s+(\d+)", hdr)
    m_desc = re.search(r"#define\s+OCSVM_DESCRIPTOR\s+(\d+)", hdr)
    if m_dim and int(m_dim.group(1)) != want:
        problems.append(f"outputs/fpga/ocsvm_params.h có OCSVM_DIM = {m_dim.group(1)}, cần {want} "
                        "→ xuất lại:  python scripts/04_export_fpga.py")
    if grid and (m_desc is None or m_desc.group(1) != "1" or "GRID_ROWS" not in hdr):
        problems.append("outputs/fpga/ocsvm_params.h chưa có hằng số lưới (OCSVM_DESCRIPTOR 1, GRID_ROWS…) "
                        "→ chép src/ocsvm_fpga/export/fpga_exporter.py bản mới rồi chạy lại "
                        "python scripts/04_export_fpga.py")
    fw_c = (FW / "ocsvm_features.c").read_text(encoding="utf-8", errors="ignore")
    fw_h = (FW / "ocsvm_features.h").read_text(encoding="utf-8", errors="ignore")
    if grid and ("ofe_grid_descriptor" not in fw_c or "OCSVM_DESCRIPTOR != 0\n#error" in fw_h
                 or "Firmware chi ho tro dac trung hiep bien" in fw_h):
        problems.append("Firmware chưa hỗ trợ đặc trưng lưới → chép bản mới của "
                        "firmware/esp32_features/ocsvm_features.c và ocsvm_features.h")
    if problems:
        sys.exit("Chưa chạy được bước 07:\n  - " + "\n  - ".join(problems))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cc", default="gcc")
    args = ap.parse_args()
    cfg, out, log, rng = setup(args, "07_check_c")
    if cfg.optical_flow.get("method", "hs") != "hs":
        sys.exit("Firmware ESP32 chỉ cài đặt Horn–Schunck. Farneback là đường cơ sở để so sánh trên PC, "
                 "không có bản C.")
    fpga_dir = out / "fpga"
    if not (fpga_dir / "ocsvm_params.h").exists() or not (fpga_dir / "ocsvm_model_data.h").exists():
        sys.exit("Chưa có outputs/fpga/*.h – chạy scripts/04_export_fpga.py trước.")
    preflight(cfg, out, fpga_dir)
    if shutil.which(args.cc) is None:
        sys.exit(f"Không tìm thấy trình biên dịch '{args.cc}'.\n"
                 "Windows: cài gcc (MinGW-w64), ví dụ:  winget install BrechtSanders.WinLibs.POSIX.UCRT\n"
                 "rồi mở lại PowerShell, hoặc chỉ đường dẫn trực tiếp:  --cc \"C:\\...\\bin\\gcc.exe\"")

    video = args.video
    if video is None:
        splits = get_splits(cfg, out, rng)
        items = splits["test"] or splits["val"] or splits["train"]
        video = str(items[0].path)
    frames = sensor_frames(video, cfg.preprocess, args.frames)
    if len(frames) < 2:
        sys.exit("Video quá ngắn")
    h, w = frames[0].shape
    log.info(f"Video: {video} – {len(frames)} khung {w}×{h}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / ("test_pc.exe" if os.name == "nt" else "test_pc")
        cmd = [args.cc, "-O2", "-std=c11", "-Wall", "-Wextra",
               f"-I{FW}", f"-I{fpga_dir}", str(FW / "test_pc.c"), str(FW / "ocsvm_features.c"),
               str(FW / "ocsvm_soft.c"), "-lm", "-o", str(exe)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit("Biên dịch C lỗi:\n" + r.stderr)
        if r.stderr.strip():
            log.warning("Cảnh báo biên dịch:\n" + r.stderr.strip())
        fin, fout = tmp / "frames.bin", tmp / "result.bin"
        with open(fin, "wb") as f:
            np.array([len(frames), w, h], np.int32).tofile(f)
            np.stack(frames).astype(np.uint8).tofile(f)
        run = subprocess.run([str(exe), str(fin), str(fout)], capture_output=True, text=True, check=True)
        log.info("C: " + run.stdout.strip())
        det = AnomalyDetector.load(out / "models" / "detector.joblib")
        fx = FixedPointOCSVM(det, cfg.fpga)
        dim = fx.sv_q.shape[1]
        rec = np.dtype([("vec", "<f4", dim), ("xq", "<i2", dim), ("fq", "<i8"), ("flag", "u1")])
        c_res = np.fromfile(fout, dtype=rec)

    # ---------------------------------------------- phía Python
    # Cùng quy tắc với firmware và với 01_extract_features: khung đen (vd khung 0 của IITB)
    # làm NGẮT cặp – khung kế tiếp trở thành khung "trước" mới.
    dark = float(cfg.preprocess.get("skip_dark_below", 0) or 0)
    py_list, prev = [], None
    for g in frames:
        if dark > 0 and float(g.mean()) < dark:
            prev = None
            continue
        cur = smooth_frame(g, cfg.preprocess)
        if prev is not None:
            py_list.append(frame_pair_descriptor(prev, cur, cfg)[0])
        prev = cur
    py_vec = np.array(py_list)
    if len(py_vec) != len(c_res):
        sys.exit(f"Số cặp khung khác nhau: C = {len(c_res)}, Python = {len(py_vec)} – "
                 "kiểm tra SKIP_DARK_BELOW trong ocsvm_params.h có khớp preprocess.skip_dark_below")
    py_xq, _ = fx.quantize_input(py_vec)
    py_fq = fx.decision_q(py_xq)
    py_flag = fx.anomaly_q(py_fq)

    vec_err = np.abs(c_res["vec"] - py_vec)
    xq_diff = np.abs(c_res["xq"].astype(int) - py_xq)
    c_fq_from_py_xq = fx.decision_q(c_res["xq"])          # datapath C dùng chính x_q của C
    f_rel = np.abs(c_res["fq"] - py_fq) / (np.abs(py_fq).mean() + 1)
    rep = {
        "pairs": int(len(c_res)),
        "vec_abs_err_max": float(vec_err.max()),
        "vec_rel_err_max": float((vec_err / (np.abs(py_vec) + 1e-3)).max()),
        "xq_max_lsb_diff": int(xq_diff.max()),
        "xq_ratio_within_1lsb": float(np.mean(xq_diff <= 1)),
        "soft_svm_bit_exact": bool(np.array_equal(c_fq_from_py_xq, c_res["fq"])),
        "fq_rel_err_max": float(f_rel.max()),
        "flag_agreement": float(np.mean(c_res["flag"] == py_flag)),
        "c_stdout": run.stdout.strip(),
    }
    save_json(rep, out / "reports" / "c_port_report.json")
    log.info(f"Véc-tơ đặc trưng: sai số tuyệt đối max = {rep['vec_abs_err_max']:.2e} | "
             f"x_q lệch tối đa {rep['xq_max_lsb_diff']} LSB ({rep['xq_ratio_within_1lsb']:.2%} trong ±1 LSB)")
    log.info(f"OC-SVM số nguyên C bit-exact với Golden Model: {rep['soft_svm_bit_exact']} | "
             f"cờ bất thường trùng: {rep['flag_agreement']:.2%}")
    if cfg.features.motion_threshold > 0 and rep["xq_max_lsb_diff"] > 1:
        log.info("Ghi chú: motion_threshold > 0 – vài điểm ảnh có |flow| sát ngưỡng bị chọn khác nhau "
                 "do sai khác làm tròn float giữa C và numpy, gây lệch nhỏ ở một số véc-tơ (không phải lỗi).")
    ok = (rep["soft_svm_bit_exact"] and rep["xq_ratio_within_1lsb"] >= 0.99
          and rep["flag_agreement"] >= 0.98)
    log.info("KẾT LUẬN: " + ("PASS ✔ firmware C khớp pipeline Python" if ok else "FAIL ✗ – xem c_port_report.json"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
