#!/usr/bin/env python3
"""SO GOLDEN MODEL VỚI BOARD TRÊN CẢNH DEMO THẬT – trả lời câu hỏi: "chạm đáy −ρ là do mô hình/cảnh quay
hay do phần cứng?"

Bước 1 (thu): chạy demo với ảnh xem trước được lưu
    python scripts/27_demo_monitor.py --fpga COM9 --esp COM5 --esp-baud 2000000 --save-preview
    (hoặc bấm phím p trong cửa sổ demo để bật/tắt lưu) – cử động BÌNH THƯỜNG 20–30 giây.
Bước 2 (so):
    python scripts/30_golden_preview.py --config configs/hw.yaml                 # lần chạy demo mới nhất
    python scripts/30_golden_preview.py --config configs/hw.yaml --run outputs_hw/demo_runs/20261104_101500

Mỗi ảnh xem trước (JPEG màu 160×120) → xám → Gauss số nguyên (giống ESP32) → ghép cặp khung liền nhau
(seq k−1, k: cách 80 ms) → HS 64 vòng + đặc trưng + chuẩn hóa (hs_q) → f_q của OC-SVM số nguyên.
So với f_q board báo cho cùng cặp (fpga_results.csv, khớp theo seq).

Lưu ý: ảnh xem trước đã nén JPEG nên Golden ở đây là GẦN ĐÚNG (không khớp từng bit với board); điều cần xem
là TỈ LỆ CHẠM ĐÁY −ρ của Golden so với board.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocsvm_fpga.cli import base_parser, load_detector_checked, setup  # noqa: E402


def vh_int(path: Path, name: str):
    m = re.search(rf"localparam\s+integer\s+{name}\s*=\s*(-?\d+)", path.read_text(encoding="ascii"))
    return int(m.group(1)) if m else None


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--run", default=None, help="thư mục một lần chạy demo (mặc định: mới nhất)")
    ap.add_argument("--cases", default=None, help="thư mục hs (mặc định <output_dir>/fpga/hs)")
    args = ap.parse_args()
    cfg, out, log, _ = setup(args, "30_golden_preview")
    from ocsvm_fpga.hwmodel import hs_export as HX, hs_q as HQ, preproc_q as PQ
    from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM

    runs = ROOT / "outputs_hw" / "demo_runs"
    run = Path(args.run) if args.run else max((d for d in runs.iterdir() if (d / "preview").is_dir()),
                                              key=lambda d: d.name, default=None)
    if run is None or not (run / "preview").is_dir():
        sys.exit("Không thấy thư mục preview – chạy demo với --save-preview (hoặc phím p) trước.")
    hs_dir = (Path(args.cases) if args.cases else out / "fpga" / "hs").resolve()
    vh = hs_dir / "hs_params.vh"
    n_iter, inv_frac = vh_int(vh, "N_ITER"), vh_int(vh, "INV_FRAC")
    p = HQ.DEFAULT if n_iter == HQ.DEFAULT.n_iter else HQ.HSParams(n_iter=n_iter)
    sc = HQ.ScalerQ(HX.read_mem(hs_dir / "mem" / "scaler_mean_q.mem", 16),
                    HX.read_mem(hs_dir / "mem" / "scaler_inv_q.mem", 16, signed=False), inv_frac)
    lut = HX.read_mem(hs_dir / "mem" / "log_lut.mem", 12, signed=False)
    fx = FixedPointOCSVM(load_detector_checked(cfg, out), cfg.fpga)

    frames = {}
    for f in sorted((run / "preview").glob("pv_*.png")):
        seq = int(f.stem.split("_")[1])
        frames[seq] = PQ.gauss5_q(np.ascontiguousarray(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)))
    board = {}
    if (run / "fpga_results.csv").exists():
        with open(run / "fpga_results.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                board.setdefault(int(r["seq"]) & 0xFF, []).append(int(r["f_q"]))
    rows, zs = [], []
    for k in sorted(frames):
        if k - 1 not in frames:
            continue
        tr = HQ.features_q(frames[k - 1], frames[k], p, sc, lut)
        fq = int(fx.decision_q(np.atleast_2d(tr.z_q))[0])
        zs.append(np.asarray(tr.z_q).reshape(-1))
        b = board.get(k & 0xFF)
        rows.append((k, fq, int(np.median(b)) if b else None, float(tr.mag.mean()) / (1 << HQ.DEFAULT.fu)))
    if not rows:
        sys.exit("Không có cặp ảnh liền nhau (seq k−1, k) – để demo chạy lâu hơn khi đang lưu ảnh.")
    rho_floor = int(fx.decision_q(np.full((1, 48), 32767, np.int64))[0])   # mọi kernel = 0 → f_q = −ρ
    g = np.array([r[1] for r in rows], np.int64)
    floor = int(g.min())
    at_floor_g = float(np.mean(g <= rho_floor + 1_000_000))
    bb = np.array([r[2] for r in rows if r[2] is not None], np.int64)
    out_csv = run / "golden_vs_board.csv"
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("seq,f_q_golden,f_q_board,mean_speed_px\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{'' if r[2] is None else r[2]},{r[3]:.3f}\n")
    print(f"Lần chạy: {run}  ·  {len(rows)} cặp ảnh  ·  đáy −ρ = {rho_floor}")
    print(f"Golden (từ ảnh xem trước): trung vị f_q {int(np.median(g))} · thấp nhất {floor} · "
          f"{at_floor_g * 100:.0f}% cặp CHẠM ĐÁY")
    if len(bb):
        print(f"Board (cùng cặp)        : trung vị f_q {int(np.median(bb))} · "
              f"{np.mean(bb <= rho_floor + 1_000_000) * 100:.0f}% cặp CHẠM ĐÁY")
    sp = np.array([r[3] for r in rows])
    print(f"Tốc độ trung bình |v| (px/80 ms): trung vị {np.median(sp):.2f} · lớn nhất {sp.max():.2f}")
    z = np.vstack(zs)
    big = np.abs(z) > 3 * 1024
    print(f"Giá trị z vượt ±3σ: {big.mean() * 100:.1f}% · chiều hay vượt nhất: "
          + ", ".join(f"z[{i}]" for i in np.argsort(-big.mean(axis=0))[:5]))
    print("Kết luận gợi ý: Golden cũng chạm đáy nhiều như board → do MÔ HÌNH/CẢNH QUAY (đặt camera xa hơn hoặc"
          " train lại); Golden ít chạm đáy mà board chạm đáy → xem lại PHẦN CỨNG.")
    print(f"Chi tiết từng cặp: {out_csv}")


if __name__ == "__main__":
    main()
