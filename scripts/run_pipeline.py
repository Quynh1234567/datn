#!/usr/bin/env python3
"""Chạy toàn bộ pipeline: kiểm tra dữ liệu → trích xuất → huấn luyện → đánh giá → xuất FPGA.

    python scripts/run_pipeline.py                      # cấu hình mặc định
    python scripts/run_pipeline.py --config configs/my.yaml --jobs 8
    python scripts/run_pipeline.py --set features.feature_set=F4 model.nu=0.02
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_inspect_dataset.py", "01_extract_features.py", "02_train.py",
         "03_evaluate.py", "04_export_fpga.py"]


def main():
    argv = sys.argv[1:]
    jobs = []
    if "--jobs" in argv:
        i = argv.index("--jobs")
        jobs = argv[i:i + 2]
        argv = argv[:i] + argv[i + 2:]
    for step in STEPS:
        extra = jobs if step.startswith("01") else []
        print(f"\n{'=' * 70}\n▶ {step}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step), *argv, *extra])
        if r.returncode != 0:
            sys.exit(f"Dừng tại {step} (mã lỗi {r.returncode})")
    print("\n✔ Hoàn tất. Xem outputs/reports và outputs/fpga")


if __name__ == "__main__":
    main()
