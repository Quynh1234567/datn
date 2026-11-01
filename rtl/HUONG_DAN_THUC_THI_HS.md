# Thực thi khối Horn–Schunck: mô phỏng bằng Vivado, nạp board bằng Gowin

Hai việc độc lập:

| | Công cụ | Mục đích | Kết quả mong đợi |
|---|---|---|---|
| **A** | Vivado (xsim) | Kiểm RTL khớp từng bit với Golden Model, xem dạng sóng khi gỡ lỗi | mọi testbench in `PASS` |
| **B** | Gowin EDA + board Tang Nano 20K | Chạy thật trên FPGA (bản thử chưa cần SDRAM), lấy số tài nguyên / Fmax | board gửi qua UART dòng trùng dòng mong đợi |

Chuẩn bị chung (một lần, sau mỗi lần train lại):

```powershell
python scripts/04_export_fpga.py --config configs/hw.yaml            # outputs_hw/fpga/hs: véc-tơ + hs_params.vh
python scripts/15_export_rtl.py  --config configs/hw.yaml --lanes 8  # rtl/build: tham số + ROM OC-SVM
```

---

## A. Mô phỏng bằng Vivado

### A1. Tự động cả ma trận testbench (khuyên dùng)

Mở **Vivado Tcl Shell** (hoặc cmd đã chạy `C:\Xilinx\Vivado\<bản>\settings64.bat`), `cd` vào thư mục repo:

```powershell
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --sim xsim                      # deriv, iter1, feat × 8 ca
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --sim xsim --tb iter64 engine --max-cases 2
```

Nếu không mở Tcl Shell: thêm `--vivado C:\Xilinx\Vivado\2024.2\bin` (đúng phiên bản của chị).
Script gọi `xvlog -sv` → `xelab` → `xsim -R -testplusarg CASE=…` và in bảng PASS/FAIL như bản Icarus.

### A2. Bằng giao diện Vivado (khi cần xem dạng sóng)

Ví dụ testbench `tb_hs_iter` với 1 vòng:

1. **Create Project** → RTL Project → bỏ qua thêm nguồn → chọn part bất kỳ (chỉ mô phỏng).
2. **Add Sources → Add or create simulation sources**, thêm:
   - `rtl/src/hs/hs_uv_ram.v`, `rtl/src/hs/hs_iter.v`, `rtl/sim/hs/tb_hs_iter.v`
   - (với `tb_hs_feat`: thêm `hs_isqrt_pipe.v`, `hs_feat.v` và **3 file** `outputs_hw/fpga/hs/mem/*.mem` – Vivado tự chép .mem vào thư mục chạy)
   - (với `tb_hs_engine`: mọi file `rtl/src/hs/*.v`, `rtl/src/ocsvm_core.v`, 3 file `mem/*.mem`, và `rtl/build/sv_lanes.mem`, `alpha.mem`, `exp_lut.mem`)
3. Chuột phải `tb_hs_iter` → **Set as Top**.
4. **Settings → General**:
   - *Verilog options → Include Files Search Paths*: thêm `outputs_hw/fpga/hs` (và `rtl/build` cho `tb_hs_engine`).
   - *Generics/Parameters*: `NIT = 1` (đủ 64 vòng: `NIT = 64`).
5. **Settings → Simulation**:
   - *Simulation → xsim.simulate.runtime* = `all`
   - *xsim.simulate.xsim.more_options* = `-testplusarg CASE=D:/New folder/ocsvm_anomaly_fpga/outputs_hw/fpga/hs/case_00 -testplusarg STALL=30 -testplusarg RATE=2`
     (dùng dấu `/`; đường dẫn có dấu cách vẫn được)
6. **Run Simulation → Run Behavioral Simulation**. Tcl Console phải có dòng `PASS tb_hs_iter …`. Kéo các tín hiệu `dut/*` vào cửa sổ sóng để xem.

Testbench sai sẽ in điểm sai đầu tiên `(y, x)`, trường và hai giá trị; đường dẫn sai in `khong doc duoc file .mem`.

---

## B. Nạp board Tang Nano 20K bằng Gowin

### B0. Bản thử này làm gì

Chưa có bộ điều khiển SDRAM nên top `hs_board_test` chạy **toàn bộ khối thật** (DERIV → 64 vòng HS → FEAT → OC-SVM) trên ảnh **tự sinh bằng công thức** (`hs_pixgen`), còn "đọc lại gói DERIV từ SDRAM" được thay bằng một khối DERIV thứ hai tính lại gói mỗi vòng (cùng kết quả, cùng tốc độ ~2 chu kỳ/gói). Board chạy liên tục và gửi qua UART mỗi lần một dòng:

```
HS N=0001 C=00263B74 F=B1EC7A6F A=1 S=00000000 Z=6C53 D1
```

| Trường | Ý nghĩa | So với |
|---|---|---|
| `N` | số lần chạy | tăng dần |
| `C` | số chu kỳ khối HS | **00263B74 = 2 505 588** (mô phỏng); 92,8 ms ở 27 MHz = 25,1 ms nếu chạy 100 MHz |
| `F`, `A` | f_q (32 bit thấp), cờ bất thường | dòng mong đợi của script (phụ thuộc mô hình) |
| `S`, `Z`, `D` | số lần bão hòa, tổng 48 z, bit kiểm tra gói DERIV | dòng mong đợi |

Ví dụ trên là của mô hình thử của mình – **F, A, Z, D của chị sẽ khác**, lấy từ bước B1. Riêng `C` không phụ thuộc mô hình.

### B1. Dựng project và tính dòng mong đợi

```powershell
python scripts/25_hs_board_test.py --config configs/hw.yaml --gowin fpga_gowin/hs_board_test
```

In ra dòng mong đợi (ghi lại) và tạo:

```
fpga_gowin/hs_board_test/
├── hs_board_test.gprj      project Gowin
└── src/  hs_board_top.v (top, đường dẫn .mem tuyệt đối) · rtl HS + ocsvm_core + uart_tx
          hs_params.vh · ocsvm_params.vh · 6 file .mem · hs_board_test.cst · hs_board_test.sdc
```

### B2. Mở project trong Gowin EDA

1. **File → Open Project** → `fpga_gowin/hs_board_test/hs_board_test.gprj`.
   Nếu Gowin không mở được file này: **File → New → FPGA Design Project**, chọn thiết bị **GW2AR-LV18QN88C8/I7** (series GW2AR, device GW2AR-18C, package QFN88, speed C8/I7), rồi **Add Files** toàn bộ `src/*.v`, `src/*.cst`, `src/*.sdc`.
2. **Project → Configuration**:
   - *Synthesize → General*: **Top Module/Entity** = `hs_board_top`; **Include Path** = đường dẫn tới thư mục `src`; Verilog Language = *Verilog 2001* (hoặc *System Verilog 2017*).
   - *Place & Route → Dual-Purpose Pin*: nếu P&R báo chân 69 bận, tick *Use SSPI as regular IO* / *Use MSPI as regular IO*.

### B3. Tổng hợp, đọc tài nguyên và Fmax

1. **Run Synthesize** – xem *Synthesis Report → Resource*: ghi lại LUT, REG, BSRAM, SSRAM, DSP/MULT.
2. **Run Place & Route** – xem *Timing Report*. File `.sdc` đặt 27 MHz. Muốn biết có đạt **100 MHz** không: sửa `period` thành `10` trong `hs_board_test.sdc`, chạy lại P&R, đọc *Max Frequency* / slack (bản chạy board vẫn dùng 27 MHz).

Lưu ý: top thử có **thêm một khối DERIV và một bộ tạo ảnh** so với thiết kế thật (thay cho SDRAM) – số LUT cao hơn bản cuối khoảng 1 000–1 500. BSRAM dự kiến khoảng 30 (u, v) + 2 (log) + 10 (OC-SVM).

### B4. Nạp và kiểm tra

1. Cắm board qua USB-C. **Tools → Programmer** → chọn thiết bị GW2AR-18C → *Operation* = **SRAM Program** (nạp tạm, mất khi rút điện; dùng *External Flash Mode* khi muốn giữ) → file `impl/pnr/hs_board_test.fs` → **Program/Configure**.
2. Quan sát: **LED0 nhấp nháy nhanh** (mỗi lần chạy ~0,1 s đảo một lần) = khối đang chạy liên tục; LED1 sáng nếu cờ bất thường = 1.
3. Đọc UART (BL616 trên board tạo một cổng COM; xem trong Device Manager):

   ```powershell
   python scripts/25_hs_board_test.py --config configs/hw.yaml --port COM5
   ```

   Script đọc 5 dòng, so từng trường và in `board KHỚP Golden Model` cùng thời gian đo trên board. Có thể xem thô bằng PuTTY / Arduino Serial Monitor, 115 200 baud.

### B5. Khi không như mong đợi

| Hiện tượng | Kiểm tra |
|---|---|
| Không có dòng UART, LED0 không nháy | chân trong `hs_board_test.cst` (clk = 4, uart_tx = 69, LED = 15, 16) theo schematic Tang Nano 20K; đã nạp đúng `.fs` chưa |
| LED0 nháy nhưng không có UART | sai cổng COM hoặc chân uart_tx; thử chân 70 nếu bảng chân của chị ghi ngược |
| Có dòng nhưng `F/Z/D` sai | `.mem` trong `src/` có cùng mô hình với dòng mong đợi không (chạy lại B1 sau khi train); xem Synthesis Report có cảnh báo "$readmemh" không tìm thấy file |
| `C` khác 00263B74 | báo mình – nghĩa là phần điều khiển chạy khác mô phỏng |
| Timing không đạt 100 MHz | bình thường ở lần đầu: tách tầng nhân a·N (P6 trong `hs_iter.v`) thành 2 tầng; kết quả không đổi |

---

## Sau bước này

Khi B4 khớp: khối HS + OC-SVM đã chạy đúng trên phần cứng thật. Bước tiếp theo là bộ điều khiển SDRAM (thay khối DERIV thứ hai bằng đọc/ghi SDRAM thật), rồi SPI nhận ảnh thật từ ESP32 và rPLL 100 MHz.
