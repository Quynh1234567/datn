# Ảnh màu gốc trên màn hình demo + sửa f_q đứng yên

## 1. Ảnh màu gốc (ESP32 + PC)
- Camera chuyển GRAYSCALE → YUV422 (160×120). Kênh Y chính là ảnh xám Y8 mà cảm biến xuất ở chế độ
  GRAYSCALE → ESP32 tách Y, làm mượt Gauss, gửi FPGA như cũ (HS/OC-SVM không đổi).
- Ảnh gửi PC: dòng `PC seq 160 120` = ảnh GỐC chưa làm mượt, Y 160×120 + U, V 80×60 (28 800 byte).
- `27_demo_monitor.py` hiển thị màu (phóng 480×360). Phím `c` đổi U/V nếu màu da ra xanh/tím
  (hoặc chạy với `--swap-uv`). Firmware cũ (dòng PV xám) vẫn hiển thị được.
- Menuconfig mới: "Ảnh xem trước MÀU gốc" (mặc định bật), "Vị trí byte Y" (mặc định tự dò),
  "Gửi ảnh mỗi N khung" (mặc định 8 ≈ 1,6 ảnh/s).

### Nạp lại ESP32
    cd firmware/esp32_demo
    idf.py menuconfig      → Demo 1 → bật "Ảnh xem trước MÀU gốc", đặt "mỗi N khung" = 8
    idf.py build flash monitor
(sdkconfig cũ giữ N = 6 → ảnh màu ~80 % đường truyền 921 600; nên đổi thành 8.)
Log khởi động phải có `color=1` và `#INFO yuv_detect … y_off=0 (YUYV)`.

### Chạy màn hình
    python scripts/27_demo_monitor.py --fpga COM9 --esp COM7

## 2. f_q luôn = -1073408562, luôn BAT THUONG
-1073408562 = -ρ: OC-SVM thấy véc-tơ z ở rất xa mọi SV. Nguyên nhân là 2 lỗi mất dấu khi Gowin
tổng hợp (xem FIX_DAU_MU.md): μ âm trong hs_feat và z âm trong ocsvm_core (xmem). Bitstream demo
hiện tại build trước bản sửa → phải build lại cả project hệ thống:
    python scripts/26_gowin_system.py ...   (như hướng dẫn demo), xóa impl/, Run All, nạp.
Sau đó f_q phải dao động theo cảnh (đồ thị không còn đường thẳng).

## 3. "SDRAM: ?"
FPGA chỉ in dòng `SD E=… ` một lần lúc khởi động. Mở monitor TRƯỚC rồi bấm reset FPGA (hoặc cắm lại USB).
