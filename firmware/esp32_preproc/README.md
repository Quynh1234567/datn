# esp32_preproc – Gauss 5×5 số nguyên và khung tối trên ESP32-S3

Hoàn thành **mục 2.2–2.3** của đặc tả A1 (`claude/hs_fixed_point_spec.md`): làm mượt ảnh 160 × 120 bằng Gauss 5 × 5 số nguyên, phát hiện khung tối, đo thời gian chạy trên ESP32-S3, và chứng minh firmware **khớp từng bit** với Golden Model Python trên ảnh camera thật. Đây là ảnh sẽ gửi sang FPGA để tính Horn–Schunck.

```
firmware/esp32_preproc/
├── main/
│   ├── preproc.c/.h          # Gauss 5×5 số nguyên, khung tối, CRC-32, ảnh mẫu (C thuần – dùng lại trong firmware chính)
│   ├── selftest_vectors.h    # CRC mong đợi do Golden Model sinh (KHÔNG sửa tay)
│   ├── ov5640_fps.c/.h       # khởi động camera + đặt 12,5 fps bằng VTS (từ esp32_cam_timing)
│   ├── frame_pacer.c/.h      # bộ chọn cặp khung 80 ms (từ esp32_cam_timing)
│   ├── app_main.c            # tự kiểm tra, benchmark, vòng thời gian thực, gửi ảnh mẫu
│   ├── board_pins.h, Kconfig.projbuild, idf_component.yml
├── test/test_preproc_pc.c    # chạy preproc.c trên PC để so với Golden
src/ocsvm_fpga/hwmodel/preproc_q.py   # Golden Model (Python, số nguyên)
scripts/19_preproc_check.py           # thu log, kiểm tra bit-exact, báo cáo
```

## 1. Thuật toán (khớp từng bit giữa C và Python)

```
k = [56, 250, 412, 250, 56]                       # = 1024·kernel Gauss σ=1, tổng đúng 1024
h[y][x] = Σ k[i]·I[y][x+i−2]                      # lọc ngang, ≤ 261 120 (18 bit)
s[y][x] = Σ k[j]·h[y+j−2][x]                      # lọc dọc, ≤ 267 386 880 < 2³²
G[y][x] = (s + 2¹⁹) >> 20                         # làm tròn MỘT lần, luôn 0…255
biên: REFLECT_101 (mặc định cv2.GaussianBlur)
khung tối: Σ G < 15 × 19 200 = 288 000            # trên ảnh đã làm mượt, như features/pipeline.py
```

- Bộ hệ số chia 1 024 lệch tối đa **0,59** so với `cv2.GaussianBlur` số thực (chỉ còn sai số làm tròn). Bộ chia 256 (`[14, 62, 104, 62, 14]`) trong bản đặc tả đầu lệch tới 1,33 nên đã được thay.
- Lọc dọc dùng **vòng đệm 5 hàng** (3,2 KB), không cần ảnh trung gian.
- **Kiểu biên phải giống code train.** Sai kiểu biên (REPLICATE thay vì REFLECT_101) làm lệch tới khoảng 40 mức xám ở viền ảnh. Kiểm tra `src/ocsvm_fpga/data/video_reader.py → smooth_frame()`: nếu gọi `cv2.GaussianBlur` không có `borderType` thì là REFLECT_101 (mặc định ở đây).

## 2. Kiểm tra trên PC trước (không cần board)

```bash
python scripts/19_preproc_check.py --selftest
```

Kiểm tra: `selftest_vectors.h` khớp Golden, `preproc.c` biên dịch bằng gcc khớp Golden trên 70 ảnh × 2 kiểu biên, và script phân tích log đúng (phát hiện được khi cố ý làm hỏng 1 CRC). Windows cần gcc (MinGW-w64, giống `07_check_c_port.py`) hoặc chỉ đường dẫn bằng `--cc`.

## 3. Build, nạp, đo

```bash
cd firmware/esp32_preproc
idf.py set-target esp32s3
idf.py menuconfig        # Preproc test (ĐATN – mục 2.2–2.3): board, kiểu biên, ngưỡng tối...
idf.py build flash
cd ../..
python scripts/19_preproc_check.py --port COM5
```

Nhấn **RST**. Firmware chạy lần lượt:
1. **Tự kiểm tra trên chip** (chưa cần camera): 10 ảnh mẫu × 2 kiểu biên, so CRC với Golden.
2. Khởi động camera, đặt **12,5 fps** (giống mục 2.1).
3. **Benchmark** Gauss 200 lần trên một khung camera (µs và chu kỳ CPU).
4. **Vòng thời gian thực 300 khung (24 s)**: Gauss → khung tối → bộ chọn cặp 80 ms. **Khi thấy dòng `measuring`, che kín ống kính khoảng 3 giây rồi bỏ ra** để thử khung tối.
5. Gửi 6 ảnh gốc rải đều (và 1 ảnh đã làm mượt) về PC.

Kết quả ở `outputs/preproc/<thời gian>/`: `report.md` (bảng tiêu chí), `preproc_records.csv`, `preproc_timing.png`, `preproc_images.png` (ảnh gốc, ảnh làm mượt, sai khác so với số thực), `raw_*.png`, `smooth_*.png`, `log.txt`.

## 4. Tiêu chí đạt

| Tiêu chí | Ngưỡng |
|---|---|
| Tự kiểm tra trên chip | 20/20 vector |
| **Bit-exact trên ảnh camera thật** | CRC ảnh làm mượt, tổng, cờ tối của ESP32 = Golden Model (và từng pixel với ảnh SMOOTH) |
| Sai khác so với cv2 số thực | ≤ 0,6 (chỉ do làm tròn) |
| Thời gian Gauss | < 10% ngân sách (8 ms) |
| Cả vòng xử lý | < 80 ms |
| Không rớt khung | khoảng cách khung vẫn 80 ms khi có xử lý |
| Khung tối | cờ tối đúng quy tắc Σ < 288 000; khung sáng đầu tiên sau đoạn tối bắt đầu cặp mới |

Nếu không thấy khung tối nào: OV5640 tự tăng độ lợi khi bị che, ảnh có thể chưa đủ tối. Che kín hơn (ngón tay áp sát, hoặc úp vào lòng bàn tay), hoặc tạm đặt `PP_DARK_MEAN` cao hơn chỉ để thử luồng xử lý.

## 5. Dùng trong firmware chính

```c
#include "preproc.h"
static pp_work_t work;
uint32_t sum = pp_gauss5(fb->buf, smooth, PP_BORDER_REFLECT101, &work);
if (pp_is_dark(sum, 15)) { pacer_reset(&pacer); /* bỏ khung, gửi RESET_PAIR */ }
else if (pacer_feed(&pacer, ts) == PACER_PAIR_OK) { /* gửi 'smooth' sang FPGA */ }
```

Khi đổi hệ số hoặc kiểu biên: sửa `preproc_q.py` và `preproc.c` cùng lúc, rồi chạy `python scripts/19_preproc_check.py --gen-vectors` và `--selftest`.
