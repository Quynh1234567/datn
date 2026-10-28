# Firmware ESP32-S3 – trích xuất đặc trưng & giao tiếp FPGA

```
firmware/
├── esp32_features/            # thư viện C thuần (chạy được cả trên PC lẫn ESP32)
│   ├── ocsvm_features.h/.c    # bước 1–4: Gaussian → Horn–Schunck → C → logm → vech → Q5.10, UART, làm mượt
│   ├── ocsvm_soft.h/.c        # OC-SVM số nguyên trên CPU, bit-exact với FPGA (baseline đo tăng tốc)
│   ├── test_pc.c              # chương trình kiểm thử trên PC (dùng bởi scripts/07_check_c_port.py)
│   └── CMakeLists.txt         # thành phần ESP-IDF
├── esp32_camview/             # xem camera MÀU 640×480 trực tiếp qua Wi-Fi (MJPEG, trình duyệt)
├── esp32_camtest/             # kiểm tra board mới: chip/flash/PSRAM, mã cảm biến OV5640, fps, gửi ảnh về PC
└── esp32_app/                 # project ESP-IDF mẫu: OV5640 → đặc trưng → UART → FPGA → LED cảnh báo
    ├── CMakeLists.txt  sdkconfig.defaults
    └── main/app_main.c  main/CMakeLists.txt  main/idf_component.yml
```

## 1. Kiểm chứng trên PC trước khi nạp

```bash
python scripts/04_export_fpga.py          # sinh outputs/fpga/ocsvm_params.h + ocsvm_model_data.h
python scripts/07_check_c_port.py         # biên dịch C bằng gcc, so với Python
```

Kết quả mong đợi: véc-tơ đặc trưng lệch ~1e-6, `x_q` lệch tối đa 1 LSB (do làm tròn float), OC-SVM số nguyên của C khớp bit với Golden Model, cờ bất thường trùng 100%. Báo cáo ở `outputs/reports/c_port_report.json`. Nếu bật `features.motion_threshold > 0`, vài điểm ảnh có |flow| sát ngưỡng có thể được chọn khác nhau giữa C và numpy, gây lệch nhỏ ở một số véc-tơ – đây không phải lỗi.

## 2. Build cho ESP32-S3 (ESP-IDF ≥ 5.1)

```bash
cp outputs/fpga/ocsvm_params.h outputs/fpga/ocsvm_model_data.h firmware/esp32_features/
cd firmware/esp32_app
idf.py set-target esp32s3
idf.py build flash monitor
```

Trước khi build cần sửa trong `app_main.c`: sơ đồ chân camera theo board của bạn (mặc định theo ESP32-S3-EYE / Freenove ESP32-S3-WROOM CAM), chân UART nối FPGA, tốc độ baud (phải trùng với khối UART trên FPGA), chân LED cảnh báo. `USE_FPGA 0` cho phép chạy toàn bộ trên CPU khi chưa có FPGA – cũng là baseline để đo hệ số tăng tốc.

Bộ đệm làm việc cần khoảng 9 × 75 KB nên được cấp phát trong PSRAM (board phải có PSRAM, `CONFIG_SPIRAM=y`).

## 3. Giao thức UART

```
ESP32 → FPGA : 0xAA 0x55 | DIM × int16 (little-endian, Q5.10) | XOR các byte dữ liệu
FPGA → ESP32 : 0x5A | f_q (ACC_W/8 byte, little-endian, bù hai) | flag | XOR(f_q..flag)
```

Với F3 (DIM = 36) một yêu cầu dài 75 byte; ở 921600 baud mất ≈0.8 ms.

## 4. Lưu ý quan trọng về tốc độ khung

Mô hình học chuyển động giữa hai khung cách nhau `frame_step / fps_gốc` giây (mặc định 2/30 ≈ 67 ms). Trên ESP32-S3, Horn–Schunck 160×120 với 32–64 vòng lặp mất cỡ hàng trăm ms mỗi khung (hãy đo bằng log `feat=...us`), nên hai khung liên tiếp được xử lý sẽ cách nhau xa hơn và vận tốc (u, v) đo được lớn hơn lúc huấn luyện. Hai cách khắc phục, có thể dùng cùng lúc:

1. Đặt `preprocess.frame_step` sao cho khoảng cách thời gian khớp với ESP32 (ví dụ ESP32 đạt 5 fps → `frame_step: 6` với video 30 fps) rồi huấn luyện lại.
2. Tốt nhất: ghi video bình thường bằng chính ESP32 tại khu vực demo, ở đúng tốc độ xử lý thực tế, và huấn luyện lại trên dữ liệu đó.

Giảm `optical_flow.num_iter` (ví dụ 16–32) hoặc tăng `features.pixel_stride` để tăng tốc; nhớ huấn luyện lại sau khi đổi.
