# esp32_hs_bench – thời gian khối Horn–Schunck trên ESP32-S3

Mốc so sánh cho bộ tăng tốc FPGA: chạy **đúng phép tính số nguyên của RTL** (DERIV → HS_ITER 64 vòng → FEAT 48 z) trên ESP32-S3 240 MHz, 1 lõi, với **cùng véc-tơ của testbench RTL**, kiểm khớp từng bit rồi mới in thời gian.

```
firmware/esp32_hs_bench/
├── main/
│   ├── hs_core.c/.h       # HS số nguyên (khớp bit Golden Model) + bản float32 – dùng chung với PC
│   ├── app_main.c         # đo DERIV / HS_ITER / FEAT, 5 lần mỗi tầng, SRAM trong và PSRAM
│   └── test_vectors.h     # SINH TỰ ĐỘNG (bước 1), không có sẵn trong repo
├── CMakeLists.txt, sdkconfig.defaults
scripts/23_hs_cpu_bench.py  # cùng hs_core.c trên CPU máy tính + Python; sinh test_vectors.h
```

## Chạy

```bash
# 1. sinh véc-tơ từ ca kiểm thử RTL (mặc định case_01 – chuyển động lớn)
python scripts/23_hs_cpu_bench.py --config configs/hw.yaml --esp32-header firmware/esp32_hs_bench/main/test_vectors.h

# 2. build + nạp
cd firmware/esp32_hs_bench
idf.py set-target esp32s3
idf.py build flash monitor
```

Log cuối:

```
=== TOM TAT (trung vi 5 lan, ESP32-S3 240 MHz, 1 loi, N = 64 vong, ca case_01) ===
impl     mem      DERIV ms   HS_ITER ms  1 vong ms    FEAT ms    Tong ms
c_int    sram        …
c_int    psram       …
c_float  psram       …
FPGA 100 MHz (uoc luong thiet ke): DERIV 0.50 + HS_ITER 27.5 + FEAT 0.01 = 28.0 ms
KET LUAN: so nguyen khop tung bit voi Golden Model: sram=DAT psram=DAT
```

- `c_int sram`: dữ liệu (269 KB) trong SRAM trong – trường hợp tốt nhất của ESP32. Nếu không đủ chỗ, dòng này bị bỏ qua và log ghi rõ.
- `c_int psram`: dữ liệu trong PSRAM – giống firmware thật khi còn camera, Wi-Fi chiếm SRAM.
- `c_float psram`: cùng thuật toán bằng float32 (pipeline train cũ), 461 KB nên chỉ chạy ở PSRAM.
- `KET LUAN … DAT`: u, v sau 64 vòng và 48 giá trị z khớp từng bit với `uv_final.mem`, `feat_z.mem`.

Watchdog tác vụ bị tắt (`CONFIG_ESP_TASK_WDT_INIT=n`) vì một lần chạy 64 vòng có thể mất hơn 1 giây.
