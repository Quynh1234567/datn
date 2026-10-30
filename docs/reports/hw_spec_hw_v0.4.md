# Bảng đặc tả bộ tăng tốc OC-SVM trên FPGA

Sinh tự động từ `fpga/ocsvm_params.vh` và `model_float.json` – khớp đúng mô hình đã xuất.

## Tóm tắt

| Thông số | Giá trị |
|---|---|
| DIM × N_SV | 48 × 109 |
| Bộ nhớ trên chip (SV + α + LUT + x) | 102 608 bit (12.53 KB) |
| Phép nhân-cộng mỗi khung | 5 450 |
| Độ trễ FPGA, P = 1 / P = 48 @ 100 MHz | 52.44 µs / 1.21 µs |
| Truyền UART mỗi khung @ 921 600 baud | 1.17 ms |
| Ngân sách thời gian mỗi khung | 80.0 ms |
| AUC (Test, làm mượt) | 0.7348 |

## 1. Thông số mô hình

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Thuật toán | One-Class SVM, kernel Gauss RBF | f(x) = Σ αᵢ·exp(−γ‖x−sᵢ‖²) − ρ |
| Loại đặc trưng | lưới 4×4 [basic] | grid_flow_hw |
| DIM – số chiều véc-tơ đầu vào | 48 | số phần tử gửi qua UART mỗi khung |
| N_SV – số Support Vector | 109 | số lần tính kernel mỗi khung |
| γ (gamma) | 0.042 | GAMMA_Q = 44040, 20 bit lẻ |
| ρ (rho) | 0.106608 | RHO_Q = 1073408562 |
| Ngưỡng quyết định | 0.100774 (miền điểm) | THR_Q = -1014665947 |
| Quy tắc quyết định | bất thường nếu f(x) < THR | DECISION_MODE = 0 |
| Làm mượt sau FPGA | 3 khung | trung bình trượt trên ESP32 |

## 2. Tiền xử lý và trích đặc trưng (ESP32-S3)

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Độ phân giải xử lý | 160×120 xám | OV5640 QQVGA |
| Khoảng cách 2 khung | frame_step = 2 → 80.0 ms | video gốc 25 fps |
| Làm mượt | Gaussian 5×5, σ = 1.0 |  |
| Dòng quang học | hs: α = 2.0, 64 vòng lặp | chạy trên ESP32 |
| Bỏ khung tối | độ sáng TB < 15 |  |
| Chuẩn hóa đầu vào | z = (x − mean)·inv_std | trên ESP32 (ocsvm_params.h) hoặc FPGA (scaler_*.mem) |

## 3. Định dạng điểm cố định và độ rộng bit

| Tín hiệu | Định dạng | Ghi chú |
|---|---|---|
| x, SV (đầu vào, support vector) | Q5.10 có dấu, 16 bit | phạm vi ±32, bước 9.77e-04 |
| diff = x − sv | 17 bit có dấu | 1 bộ trừ / làn |
| dist = Σ diff² | 40 bit không dấu, Q·20 | tích lũy 48 số hạng |
| γ | 16 bit không dấu, 20 bit lẻ | GAMMA_Q = 44040 (γ = GAMMA_Q / 2^20) |
| t = dist · γ | 56 bit, Q·40 | 1 bộ nhân / SV |
| Địa chỉ LUT = t ≫ LUT_SHIFT | 10 bit (LUT_SHIFT = 34) | t ≥ 16 → k = 0 |
| k = exp(−t) (LUT) | Q0.16 không dấu, 16 bit | 1024 phần tử, bước Δt = 0.01562 |
| α (hệ số) | Q1.14 có dấu, 16 bit | đã chuẩn hóa: giá trị tuyệt đối lớn nhất của α = 1 |
| α · k | 33 bit có dấu | 1 bộ nhân / SV |
| Bộ cộng dồn f | 48 bit có dấu, Q·30 | cần tối thiểu 40 bit (trường hợp xấu nhất) |

## 4. Bộ nhớ

| Bộ nhớ | Độ sâu × độ rộng | Dung lượng | BRAM18K (ước lượng) | Ghi chú |
|---|---|---|---|---|
| SV (sv.mem, 1 phần tử/từ) | 5232 × 16 | 83 712 bit (10.22 KB) | 6 | đọc tuần tự P = 1 |
| SV (sv_packed.mem, cả véc-tơ/từ) | 109 × 768 | 83 712 bit (10.22 KB) | 22 | đọc song song P = DIM |
| α (alpha.mem) | 109 × 16 | 1 744 bit (0.21 KB) | 1 |  |
| LUT exp (exp_lut.mem) | 1024 × 16 | 16 384 bit (2.00 KB) | 1 | ROM |
| Bộ đệm đầu vào x | 48 × 16 | 768 bit (0.09 KB) | 1 | thanh ghi hoặc LUTRAM |
| Scaler mean + inv_std (tùy chọn) | 96 × 32 | 3 072 bit (0.38 KB) | 1 | chỉ khi chuẩn hóa trên FPGA |
| **Tổng (SV tuần tự + α + LUT + x)** |  | 102 608 bit (12.53 KB) | 8 |  |

## 5. Khối lượng tính toán mỗi lần suy luận

| Phép toán | Số lần | Độ rộng |
|---|---|---|
| Phép trừ (x − sv) | 5 232 |  |
| Phép nhân bình phương diff² | 5 232 | 17×17 bit |
| Phép cộng tích lũy dist | 5 232 |  |
| Phép nhân dist·γ | 109 | 40×16 bit |
| Tra LUT exp | 109 |  |
| Phép nhân α·k | 109 | 16×16 bit |
| Phép cộng tích lũy f | 109 | 48 bit |
| **Tổng phép nhân-cộng (MAC) mỗi khung** | 5 450 |  |

## 6. Độ trễ và tài nguyên FPGA theo mức song song P (ước lượng)

| f_clk | P | Chu kỳ | Độ trễ | Suy luận/giây | DSP (ước lượng) | BRAM18K cho SV |
|---|---|---|---|---|---|---|
| 100 MHz | 1 | 5 244 | 52.44 µs | 19 069 | 4 | 6 |
| 100 MHz | 4 | 1 320 | 13.20 µs | 75 757 | 7 | 6 |
| 100 MHz | 8 | 666 | 6.66 µs | 150 150 | 11 | 8 |
| 100 MHz | 16 | 339 | 3.39 µs | 294 985 | 19 | 8 |
| 100 MHz | 48 | 121 | 1.21 µs | 826 446 | 51 | 22 |

Chu kỳ ≈ N_SV × ⌈DIM/P⌉ + 12 (khởi động pipeline). DSP ≈ P (bình phương) + 2 (dist·γ) + 1 (α·k), tính theo DSP48 25×18 của Xilinx; LUT exp và α không phụ thuộc P.

## 7. Truyền UART và ngân sách thời gian (1 khung = 80.0 ms)

| Baud | ESP32 → FPGA | FPGA → ESP32 | Tổng truyền | % ngân sách khung |
|---|---|---|---|---|
| 115 200 | 99 B → 8.59 ms | 9 B → 0.781 ms | 9.38 ms | 11.7% |
| 921 600 | 99 B → 1.07 ms | 9 B → 0.098 ms | 1.17 ms | 1.5% |

Gói yêu cầu: 0xAA 0x55 | DIM × int16 | XOR. Gói trả lời: 0x5A | f_q (6 B) | cờ | XOR. Thời gian truyền UART lớn hơn thời gian tính trên FPGA hàng trăm lần → trong hệ thống thật, UART và trích đặc trưng trên ESP32 mới là phần quyết định độ trễ.

## 8. So sánh thời gian suy luận OC-SVM

| Nền tảng | Thời gian / khung | Ghi chú |
|---|---|---|
| FPGA, P = 1, 100 MHz | 52.44 µs | ước lượng |
| FPGA, P = 48, 100 MHz | 1.21 µs | ước lượng |
| OC-SVM phần mềm | – | thêm --pc-svm-us / --esp32-svm-us để tính hệ số tăng tốc |

## 9. Độ chính xác và kiểm chứng

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Sai số tuyệt đối f_fixed − f_float lớn nhất | 4.807e-04 |  |
| Sai số trung bình | 6.141e-05 |  |
| Tỉ lệ trùng nhãn float ↔ fixed | 99.99% | mỗi khung |
| AUC float / fixed (thô) | 0.7185 / 0.7185 | trên tập kiểm chứng của bước 04 |
| AUC mô hình (làm mượt) | 0.7348 | 90594 cặp khung Test |
| EER | 0.3215 |  |
| Vector kiểm thử cho testbench | 200 vector | test_input.mem / test_expected_score.mem |

## 10. File bàn giao

| File | Nội dung |
|---|---|
| ocsvm_params.vh | localparam cho RTL (mọi thông số ở trên) |
| mem/sv.mem, sv_packed.mem | support vector (tuần tự / song song) |
| mem/alpha.mem | hệ số α |
| mem/exp_lut.mem | bảng tra exp(−t) |
| mem/test_input.mem, test_expected_score.mem, test_expected_label.mem | vector kiểm thử + kết quả mong đợi |
| mem/test_kernel_vec0.mem | giá trị kernel từng SV của vector 0 (gỡ lỗi PE) |
| coe/*.coe | khởi tạo Block Memory Generator (Vivado) |
| ocsvm_params.h, ocsvm_model_data.h | firmware ESP32 |
