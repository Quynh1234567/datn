# esp32_cam_timing – Kiểm tra camera ESP32-S3 + OV5640 và đo thời gian khung

Hoàn thành **mục 2.1** của đặc tả A1 (`claude/hs_fixed_point_spec.md`): xác nhận camera chạy đúng 160 × 120 xám, đo FPS thực tế, ghi timestamp 200–2 000 khung để đánh giá độ ổn định, và kiểm tra bộ chọn cặp khung 80 ms trên dữ liệu thật. Kết quả quyết định **dùng được mô hình IITB (80 ms)** hay **phải train lại** ở khoảng cách khung của OV5640.

```
firmware/esp32_cam_timing/
├── CMakeLists.txt, sdkconfig.defaults
├── main/
│   ├── app_main.c          # đo đạc, thống kê, in log
│   ├── frame_pacer.c/.h    # bộ chọn cặp khung (C thuần, dùng lại trong firmware chính)
│   ├── board_pins.h        # sơ đồ chân camera theo board
│   ├── Kconfig.projbuild   # mọi tham số chỉnh trong menuconfig
│   └── idf_component.yml   # kéo thư viện espressif/esp32-camera
└── test/test_pacer.c       # kiểm thử bộ chọn cặp trên PC
scripts/18_cam_timing.py    # thu log qua COM, phân tích, vẽ biểu đồ, viết báo cáo
```

## 1. Chuẩn bị

- ESP-IDF ≥ 5.1 (đã thử cú pháp với API của 5.x và esp32-camera 2.x).
- Python: `pip install pyserial numpy matplotlib opencv-python`
- **Kiểm tra sơ đồ chân** trong `main/board_pins.h` khớp với board của bạn. Có sẵn 2 bộ:
  - ESP32-S3-EYE / Freenove ESP32-S3-WROOM CAM / đa số board "ESP32-S3-CAM N16R8" (mặc định)
  - Seeed XIAO ESP32S3 Sense
  Board khác: chọn *Custom* trong menuconfig rồi sửa khối `#else` trong `board_pins.h`.

## 2. Build và nạp

```bash
cd firmware/esp32_cam_timing
idf.py set-target esp32s3
idf.py menuconfig          # Camera timing test (ĐATN – mục 2.1): chọn board, XCLK, số khung...
idf.py build flash
```

Lần build đầu, ESP-IDF tự tải thư viện `espressif/esp32-camera`.

- Board có PSRAM Quad (vd N4R2): đổi `CONFIG_SPIRAM_MODE_OCT` thành `CONFIG_SPIRAM_MODE_QUAD` trong `sdkconfig.defaults` (xóa file `sdkconfig` rồi build lại). Không có PSRAM vẫn chạy được.
- Board chỉ có cổng USB native (không có chip USB-UART): menuconfig → Component config → ESP System Settings → Channel for console output → USB Serial/JTAG Controller.

## 3. Chạy và thu kết quả

```bash
python scripts/18_cam_timing.py --port COM5       # Linux: /dev/ttyACM0 hoặc /dev/ttyUSB0
```

Nhấn nút **RST** trên board. Firmware in thông tin chip, khởi tạo camera, bỏ 30 khung đầu (chờ tự động phơi sáng ổn định), **ghi 500 khung vào RAM mà không in gì** (để việc in không làm sai thời gian), rồi in toàn bộ bản ghi, bảng tóm tắt và 3 ảnh mẫu. Script tự dừng khi nhận `DONE`.

Kết quả nằm trong `outputs/camtiming/<thời gian>/`:

| File | Nội dung |
|---|---|
| `log.txt` | log gốc (phân tích lại được bằng `--log`) |
| `run0_timing.csv` | timestamp từng khung |
| `run0_timing.png` | khoảng cách khung theo thời gian, phân bố, khoảng cách cặp |
| `frame_XX.png`, `frame_XX_x4.png` | ảnh mẫu 160 × 120 (và phóng to 4 lần) |
| `report.md` | bảng tiêu chí đạt/không đạt và kết luận – đưa thẳng vào báo cáo |

Không có PC: mở `idf.py monitor`, bảng **KẾT QUẢ ĐO** in trực tiếp trên console.

## 4. Quy trình đo đề xuất

Đổi tham số trong menuconfig, build, nạp và chạy script cho từng kịch bản. Mỗi kịch bản lưu một thư mục kết quả riêng.

| # | Kịch bản | Cài đặt | Mục đích |
|---|---|---|---|
| 1 | Quét cấu hình | `CAMT_SWEEP = y` | tìm XCLK và fb_count cho FPS cao, ổn định |
| 2 | Chuẩn | cấu hình tốt nhất từ #1, 500 khung, ánh sáng phòng bình thường | số liệu chính |
| 3 | Có tải xử lý | `CAMT_SIM_LOAD_MS = 40` | ESP32 bận (Gauss + gửi SPI) có làm rớt khung không |
| 4 | Ánh sáng yếu | như #2 trong phòng tối | OV5640 có tự giảm FPS khi tối không |
| 5 | Khóa phơi sáng | như #4, `CAMT_LOCK_EXPOSURE = y` | giữ FPS cố định trong điều kiện tối |
| 6 | Dài | 2 000 khung (khoảng 1–3 phút) | độ ổn định lâu dài |

## 5. Tiêu chí đạt và cách đọc kết quả

| Tiêu chí | Ngưỡng |
|---|---|
| Khởi tạo camera, mã cảm biến | `cam_init=OK`, PID `0x5640` |
| Định dạng | mọi khung 160 × 120, 19 200 byte, xám |
| Lỗi lấy khung | 0 |
| Số khung có timestamp | ≥ 200 |
| Ổn định | jitter < 5% chu kỳ, rớt khung < 2% |
| Cặp bị loại vì sai khoảng cách | < 5% |
| Khớp mô hình IITB | khoảng cách cặp trung bình trong 80 ms ± 5% |

**Kết luận theo FPS đo được:**

| FPS camera | Bộ chọn giữ | Khoảng cách cặp | Hành động |
|---|---|---|---|
| ≈ 25 | 1 trong 2 khung | 80 ms | dùng được mô hình IITB |
| ≈ 12,5 | mọi khung | 80 ms | dùng được mô hình IITB |
| ≈ 15 | mọi khung | 66,7 ms | train lại ở 67 ms, đặt `target_ms = 67` |
| ≈ 30 | 1 trong 2 khung | 66,7 ms | train lại ở 67 ms, hoặc chỉnh XCLK về 25 fps |

Bảng "Lấy 1 trong k khung" trong `report.md` liệt kê mọi lựa chọn và độ lệch so với 80 ms.

**Các con số khác:**
- *Khung lặp (CRC trùng)*: hai khung liên tiếp giống hệt từng byte – driver trả lại khung cũ. Phải bằng 0.
- *Trễ từ lúc chụp đến lúc ứng dụng nhận*: thời gian khung nằm trong buffer. Lớn và tăng dần nghĩa là ứng dụng xử lý chậm hơn camera.
- *Bộ chọn cặp ESP32 trùng bản Python*: script chạy lại thuật toán trên cùng timestamp; `True` nghĩa là firmware cài đúng.

## 6. Đặt FPS chính xác (`CAMT_TARGET_FPS_X10`)

Lần đo đầu trên board thật cho **19,73 fps (50,68 ms)** ở cả hai mức sáng: FPS do cấu hình thời gian của cảm biến quyết định, không phụ thuộc ánh sáng. Không có cách lấy khung nào cho đúng 80 ms (mọi khung 50,7 ms, 1 trong 2 khung 101,4 ms).

OV5640 có FPS = PCLK / (HTS × VTS), nên chu kỳ khung tỉ lệ thuận với VTS (thanh ghi 0x380E/0x380F). Khi `CAMT_TARGET_FPS_X10 > 0`, firmware:
1. đo chu kỳ hiện tại, đọc HTS/VTS (log `#INFO fps_ctrl_before ...`);
2. ghi VTS mới = VTS · T_mục_tiêu / T_đo, tắt night mode (0x3A00 bit 2) để AEC không tự chèn hàng làm đổi FPS;
3. (`CAMT_AEC_MAX_FOLLOW_VTS`) nới giới hạn phơi sáng tối đa (0x3A02/03, 0x3A14/15) = VTS − 4 để ảnh sáng hơn;
4. đo lại, lặp tối đa 3 vòng tới khi lệch < 0,2% (log `#INFO fps_ctrl_iter=...`, `#INFO fps_ctrl=OK ...`).

| Giá trị | FPS | Bộ chọn giữ | Ghi chú |
|---|---|---|---|
| **125 (mặc định)** | 12,5 | mọi khung, 80 ms | khuyến nghị: thời gian phơi sáng tối đa dài nhất (ảnh sáng hơn), ngân sách xử lý 80 ms/khung |
| 250 | 25 | 1 trong 2 khung, 80 ms | thời gian phơi sáng tối đa ngắn hơn → tối hơn trong phòng thiếu sáng |
| 0 | mặc định driver | | đo FPS gốc |

Nếu `fps_ctrl=FAIL` hoặc `APPROX`, gửi log `fps_ctrl_*` để kiểm tra lại cấu hình thời gian của driver. Ảnh tối (độ sáng TB < 40): tăng ánh sáng, đặt `CAMT_AE_LEVEL = 1` hoặc `2`.

**Lưu ý cho firmware chính và việc train lại:** dùng đúng đoạn `set_target_fps()` này, để camera lúc quay dữ liệu train và lúc demo chạy cùng một FPS.

## 7. Bộ chọn cặp khung (`frame_pacer`)

Đây là phần dùng lại nguyên văn trong firmware chính (B2 trong kế hoạch):

```c
frame_pacer_t pacer;
pacer_init(&pacer, 80, 20);                  // 80 ms ± 20%
...
pacer_decision_t d = pacer_feed(&pacer, ts_us);   // ts_us = timestamp khung từ driver
if (d == PACER_PAIR_OK)      { /* gửi khung + cờ "ghép với khung trước" */ }
else if (d == PACER_FIRST || d == PACER_PAIR_BAD) { /* gửi khung + cờ RESET_PAIR */ }
/* PACER_SKIP: bỏ khung */
if (khung_tối) pacer_reset(&pacer);         // ngắt cặp
```

Thuật toán: giữ khung đầu tiên có khoảng cách tới khung giữ trước ≥ target − T/2 (T là chu kỳ camera ước lượng), tức khung gần target nhất. Cặp lệch quá ±tol bị loại và khung đó bắt đầu một cặp mới.

Kiểm thử trên PC:

```bash
cd firmware/esp32_cam_timing/test
gcc -O2 -I ../main test_pacer.c ../main/frame_pacer.c -lm -o test_pacer && ./test_pacer
```

## 8. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `cam_init=FAIL err=0x105` hoặc `0x20001` | không dò thấy cảm biến: sai chân SCCB (SIOD/SIOC) hoặc XCLK, cáp lỏng, camera chưa cấp nguồn – kiểm tra `board_pins.h` |
| `cam_init=FAIL err=0x101` | thiếu bộ nhớ: tắt `CAMT_FB_IN_PSRAM` hoặc giảm `fb_count` |
| PID khác `0x5640` | camera không phải OV5640 (vd OV2640 = `0x2642`) |
| Ảnh sọc, lệch, nhiễu | thử XCLK 10 MHz; kiểm tra thứ tự chân D0–D7 |
| FPS giảm mạnh khi tối | tự động phơi sáng kéo dài thời gian phơi; tăng ánh sáng hoặc bật `CAMT_LOCK_EXPOSURE` |
| Ảnh quá tối (độ sáng TB < 40) | tăng ánh sáng; dùng 12,5 fps với `CAMT_AEC_MAX_FOLLOW_VTS`; `CAMT_AE_LEVEL = 1…2` |
| FPS không đổi sau khi đặt | driver ghi đè VTS hoặc AEC đang chèn hàng: gửi log `fps_ctrl_*` |
| `sai định dạng` > 0 | driver không hỗ trợ xám ở cấu hình này – cập nhật esp32-camera |
| Script không nhận dữ liệu | sai cổng COM; console đang ở USB Serial/JTAG nhưng mở cổng UART (hoặc ngược lại) |

## 9. Ghi chú về timestamp

`ts_us` là thời điểm driver esp32-camera gắn cho khung (theo `esp_timer`, cùng đồng hồ với `esp_timer_get_time()`), nên phản ánh thời điểm cảm biến xuất khung, không phụ thuộc lúc ứng dụng gọi `esp_camera_fb_get()`. Nếu phiên bản driver không gắn timestamp (bằng 0), firmware dùng thời điểm `fb_get` trả về thay thế; khi đó jitter đo được sẽ lớn hơn thực tế một chút.
