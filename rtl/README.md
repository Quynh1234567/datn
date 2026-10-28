# RTL bộ tăng tốc OC-SVM

| File | Nội dung |
|---|---|
| `src/ocsvm_core.v` | lõi OC-SVM: P làn PE → cây cộng → cộng dồn D_i → nhân γ → bảng tra exp → MAC α·k → so ngưỡng |
| `src/ocsvm_top.v` | FSM nhận gói UART, nạp véc-tơ, chạy lõi, gửi gói trả lời; LED trạng thái |
| `src/uart_rx.v`, `src/uart_tx.v` | UART 8N1 |
| `sim/tb_ocsvm_core.v` | so lõi với Golden Model trên toàn bộ `test_input.mem` (khớp từng bit) |
| `sim/tb_ocsvm_top.v` | kiểm thử toàn hệ thống qua UART (gói như ESP32) |
| `constraints/*.cst` | gán chân Tang Nano 9K / 20K (kiểm tra lại theo schematic) |

## Quy trình

1. Xuất mô hình: `python scripts/04_export_fpga.py`
2. Chuẩn bị thư mục build: `python scripts/15_export_rtl.py --lanes 8`
3. Mô phỏng (Icarus Verilog): `cd rtl/build && run_sim.bat` → phải in `PASS` cho cả hai testbench
4. Tổng hợp trên Gowin EDA: thêm `src/*.v`, đặt include path tới `rtl/build`, chép các file `.mem`
   (`sv_lanes.mem`, `alpha.mem`, `exp_lut.mem`) vào thư mục project (hoặc truyền đường dẫn tuyệt đối qua
   tham số `SV_FILE/ALPHA_FILE/LUT_FILE` của `ocsvm_top`), thêm file `.cst`, top module = `ocsvm_top`
5. Nạp board, kiểm thử từ PC: `python scripts/16_uart_host_test.py --port COMx --baud 115200`

Tham số `ocsvm_top`: `CLK_HZ` (27 MHz mặc định), `BAUD` (115 200 mặc định; 921 600 cần PLL tần số cao
hơn để sai lệch baud nhỏ), `P` (số làn PE, DIM phải chia hết cho P).
