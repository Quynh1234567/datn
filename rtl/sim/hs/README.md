# RTL khối Horn–Schunck – bước 1: viết RTL và mô phỏng từng tầng

Khối trích đặc trưng trên Tang Nano 20K: **DERIV → HS_ITER (64 vòng Jacobi) → FEAT → 48 giá trị z → `ocsvm_core`**.
Mọi phép tính khớp từng bit Golden Model `src/ocsvm_fpga/hwmodel/hs_q.py` (đặc tả `claude/hs_fixed_point_spec.md` bản 0.4, u, v Q6.6).

```
rtl/src/hs/
├── hs_div_pipe.v     bộ chia pipeline 15 tầng (a = Ix4/D16, b = Iy4/D16)
├── hs_deriv.v        DERIV: 2 ảnh → gói 64 bit / điểm (= một dòng deriv.mem)
├── hs_uv_ram.v       u, v 19 200 × 26 bit (30 BSRAM)
├── hs_iter.v         HS_ITER: N vòng Jacobi tại chỗ, pipeline 8 tầng
├── hs_isqrt_pipe.v   căn bậc hai nguyên 14 tầng
├── hs_feat.v         FEAT: |v|, 48 tổng ô, trung bình, bảng log, z → ghi ocsvm_core
└── hs_engine.v       ghép 3 khối; cổng bộ nhớ là luồng valid/ready (chưa gắn SDRAM)
rtl/sim/hs/
├── tb_hs_deriv.v     img1/img2 → so deriv.mem
├── tb_hs_iter.v      deriv.mem × N → so uv_iter1.mem (N = 1) / uv_final.mem (N = 64)
├── tb_hs_feat.v      uv_final → so mag, cell_sums, feat_mean, feat_log, feat_z
└── tb_hs_engine.v    img1/img2 → hs_engine → ocsvm_core → so z, f_q, cờ (expected.txt)
scripts/24_hs_rtl_sim.py   chạy cả ma trận testbench × 8 ca, in bảng PASS/FAIL
```

## Bước 0 – Cài Icarus Verilog (một lần)

- Windows: tải bản **12.0** tại <https://bleyer.org/icarus/>, khi cài tick **"Add executable folder(s) to the user PATH"**.
- Mở PowerShell mới, kiểm tra: `iverilog -V` → `Icarus Verilog version 12.0`.
- Nếu không thêm vào PATH: thêm `--iverilog C:\iverilog\bin` vào mọi lệnh bên dưới.

## Bước 1 – Chuẩn bị véc-tơ kiểm thử

Véc-tơ đã có trong `outputs_hw/fpga/hs/` (sinh lúc chạy `04_export_fpga.py`). Kiểm tra nhanh:

```powershell
dir outputs_hw\fpga\hs            # phải có hs_params.vh, mem\, case_00 … case_07
```

Nếu thiếu hoặc vừa train lại: `python scripts/04_export_fpga.py --config configs/hw.yaml`.

Testbench toàn khối (`engine`) cần thêm tham số OC-SVM cho **cùng mô hình**:

```powershell
python scripts/15_export_rtl.py --config configs/hw.yaml --lanes 8      # → rtl/build/ocsvm_params.vh, sv_lanes.mem, …
```

## Bước 2 – Chạy kiểm thử nhanh (~1 phút)

```powershell
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml
```

Chạy `deriv`, `iter1`, `feat` trên 8 ca, có chèn chờ ngẫu nhiên 30 % (kiểm logic valid/ready) và đọc gói DERIV 1 gói / 2 chu kỳ như SDRAM. Kết quả mong đợi:

```
| deriv | IITB/case_00 | **PASS** | 3 s | 19200 goi khop deriv.mem, … |
| iter1 | IITB/case_00 | **PASS** | 1 s | N=1 vong, u v khop uv_iter1.mem, … |
| feat  | IITB/case_00 | **PASS** | 1 s | |v|, 48 tong o, mean, log, z va 48 lan ghi OC-SVM khop, … |
…
**Tổng: 24/24 PASS**
```

Báo cáo lưu ở `outputs_hw/rtl_sim/<thời gian>/report.md`.

## Bước 3 – Chạy đủ 64 vòng và toàn khối

```powershell
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --tb iter64                    # ~50 s / ca
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --tb engine --max-cases 2      # ~5 phút / ca
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --full                         # tất cả (~50 phút)
```

`engine` in **số chu kỳ thật** của khối HS, ví dụ `HS 2496503 chu ky = 24.97 ms o 100 MHz (RATE=2 …)` – đây là số thay cho "ước lượng thiết kế" trong các bảng so sánh (chưa tính refresh SDRAM). Chạy `--stall 0` để có số chu kỳ không bị chèn chờ.

## Bước 4 – Ca khó (khuyên chạy một lần)

```powershell
python scripts/24_hs_rtl_sim.py --config configs/hw.yaml --stress --tb deriv iter1 feat iter64
```

Sinh thêm 5 ca mà video thật ít gặp: a = ±1/16 (bão hòa hệ số, kể cả −16 384), sọc 0/255 (|Ix4| = 1020), dịch 20 px, hai ảnh nhiễu độc lập (|N| lớn), đen → trắng (It4 = 1020).

## Chạy tay một testbench (khi gỡ lỗi)

```powershell
cd outputs_hw\fpga\hs\mem                  # $readmemh đọc log_lut / scaler theo thư mục hiện tại
iverilog -g2012 -I .. -P tb_hs_iter.NIT=1 -o iter1.vvp ^
    ..\..\..\..\rtl\src\hs\hs_uv_ram.v ..\..\..\..\rtl\src\hs\hs_iter.v ..\..\..\..\rtl\sim\hs\tb_hs_iter.v
vvp -n iter1.vvp +CASE=../case_00 +STALL=30 +RATE=2
```

Tham số: `+CASE=<thư mục ca>` (bắt buộc), `+STALL=<0…90>` (% chu kỳ chờ ngẫu nhiên), `+RATE=<chu kỳ / gói>`, `+SEED=<số>`.
Khi sai, testbench in điểm sai đầu tiên: `(y, x)`, trường (Ix4, a, u, v, cell_sums[d] …), giá trị RTL và mong đợi.
Nếu đường dẫn sai, testbench báo `khong doc duoc file .mem` (không bao giờ PASS khi thiếu file).

## Giao diện các khối

| Khối | Vào | Ra | Thời gian (100 MHz) |
|---|---|---|---|
| `hs_deriv` | luồng `{i2, i1}` 16 bit, 19 200 nhịp | luồng gói 64 bit, có `out_ready` | ~38 500 chu kỳ = 0,39 ms |
| `hs_iter` | luồng gói 64 bit, 19 200 × N nhịp | luồng `(u', v')` vòng cuối, `n_sat` | 1,03 chu kỳ / điểm / vòng nếu cấp đủ; 2 nếu đọc SDRAM 2 từ / gói |
| `hs_feat` | luồng `(u, v)` 13 bit | 48 lần ghi `x_we/x_addr/x_data`, `done` | chạy cùng vòng cuối + ~340 chu kỳ |
| `hs_engine` | 3 luồng bộ nhớ (`pix_*`, `dw_*`, `dr_*`) + `start` | `x_*`, `svm_start`, `cycles` | ≈ 2,5 triệu chu kỳ = 25 ms |

Tất cả nằm trong **một miền xung nhịp**; luồng dùng quy tắc valid/ready (dữ liệu chuyển khi cả hai = 1).

## Khác với tài liệu kiến trúc (đã cập nhật khi viết RTL)

- **Bộ chia 15 tầng** thay vì 28: D16 ≥ 64 + n² nên thương ≤ 16 384 < 2¹⁵. Hai bộ chia chạy song song (a và b) thay vì một bộ dùng chung, để đơn giản luồng.
- **HS_ITER dùng 4 bộ đệm hàng** (4 × 160 × 26 = 16,6 Kbit SSRAM) thay vì 2: bộ đệm thứ tư nạp trước hàng y+2 trong lúc tính hàng y nên không phải chờ. Vẫn nằm trong 41 Kbit SSRAM.
- **DERIV nạp một hàng rồi quét một hàng** (~2 chu kỳ / điểm, 0,39 ms) – đơn giản hơn đọc song song; không ảnh hưởng tổng thời gian.
- Pipeline HS_ITER 8 tầng (P1…P8): S6 (nhân 15 × 25 bit) là đường dài nhất – nếu Gowin báo không đạt 100 MHz, tách P6 thành 2 tầng (chỉ cần thêm thanh ghi, kết quả không đổi).

## Bước tiếp theo (bước 2–3 của kế hoạch)

1. Tổng hợp thử `hs_engine` trên Gowin EDA (include path: `outputs_hw/fpga/hs`, chép 3 file `mem/*.mem` vào thư mục project) để lấy số LUT, FF, BSRAM, DSP, Fmax thật.
2. Viết bộ điều khiển SDRAM + phân xử nối vào 3 luồng `pix_*`, `dw_*`, `dr_*` (địa chỉ theo mục 7.3 tài liệu kiến trúc).
3. SPI slave nhận ảnh từ ESP32 ghi vào vòng đệm 3 khung.
