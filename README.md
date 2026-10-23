# OC-SVM Anomaly Detection → FPGA Accelerator

**Đồ án tốt nghiệp:** *Thiết kế bộ tăng tốc phần cứng phát hiện hành vi bất thường trong giám sát an ninh sử dụng thuật toán One-Class SVM trên FPGA*
Học viện Công nghệ Bưu chính Viễn thông – Khoa Kỹ thuật Điện tử 1

Project này hiện thực toàn bộ phần **phần mềm trên PC**: đọc 208 video `.avi` của IITB-Corridor → trích xuất đặc trưng chuyển động theo bài báo *Wang et al., "Online Least Squares One-Class SVM-Based Abnormal Visual Event Detection", Sensors 2013* → huấn luyện One-Class SVM → lượng tử hóa điểm cố định → xuất các file `.mem` / `.coe` / `.vh` / `.h` để nạp vào FPGA và ESP32-S3.

---

## 1. Kiến trúc hệ thống (HW/SW Co-design)

```
 ┌──────────────────────── HUẤN LUYỆN (PC – project này) ────────────────────────┐
 │ IITB-Corridor .avi → khung xám 160×120 → Horn–Schunck (u,v) → F(x,y) → C (d×d) │
 │ → logm(C) → vech → chuẩn hóa → One-Class SVM → lượng tử hóa → .mem/.vh/.h      │
 └───────────────────────────────────────────────────────────────┬────────────────┘
                                                                 │ nạp tham số
 ┌──────────── DEMO (thời gian thực) ────────────┐   UART  ┌─────▼──────────────────────────┐
 │ OV5640 → ESP32-S3:                             │ ──────► │ FPGA – bộ tăng tốc OC-SVM      │
 │  ảnh xám QQVGA → Horn–Schunck → hiệp biến C    │ x_q[DIM]│  BRAM: sv.mem, alpha.mem       │
 │  → logm → chuẩn hóa (ocsvm_params.h) → Q5.10   │ ◄────── │  PE: Σ(x−sv)² → ·γ → LUT exp   │
 │  ← nhận f(x), cờ bất thường → làm mượt → cảnh báo│ f_q,flag│  MAC: Σ α·k − ρ → so ngưỡng    │
 └────────────────────────────────────────────────┘         └────────────────────────────────┘
```

Hàm quyết định mà FPGA tính (giống hệt phương trình (3) của bài báo):

$$f(\mathbf{x}) = \sum_{i=1}^{N_{SV}} \alpha_i \exp\left(-\gamma \lVert \mathbf{x}-\mathbf{s}_i \rVert^2\right) - \rho ,\qquad f(\mathbf{x}) < T \Rightarrow \text{bất thường}$$

---

## 2. Cấu trúc project

```
ocsvm_anomaly_fpga/
├── configs/
│   ├── default.yaml              # MỌI tham số (dữ liệu, HS, đặc trưng, SVM, ngưỡng, fixed-point)
│   └── synthetic.yaml            # cấu hình kiểm tra nhanh trên dữ liệu giả lập
├── scripts/                      # các bước chạy theo thứ tự
│   ├── 00_inspect_dataset.py     # kiểm tra video, nhãn, chia train/val/test
│   ├── 01_extract_features.py    # Bước 1→4: video → véc-tơ hiệp biến (song song, có cache)
│   ├── 02_train.py               # Bước 5: huấn luyện OC-SVM (+ tìm siêu tham số tùy chọn)
│   ├── 03_evaluate.py            # Bước 6: AUC / EER / F1 + biểu đồ
│   ├── 04_export_fpga.py         # Bước 7: lượng tử hóa, Golden Model, xuất .mem/.coe/.vh/.h
│   ├── 05_infer_video.py         # chạy thử 1 video / webcam (float + fixed-point)
│   ├── 06_verify_mem_files.py    # kiểm chứng độc lập các file .mem như một testbench
│   ├── 07_check_c_port.py        # so thư viện C của ESP32 với Python (biên dịch bằng gcc)
│   ├── 08_annotate_videos.py     # công cụ gán nhãn khung bất thường (giao diện / CSV)
│   ├── 09_compare_features.py    # thí nghiệm so sánh đặc trưng: hiệp biến vs lưới dòng quang học
│   ├── run_pipeline.py           # chạy 00 → 04
│   └── make_synthetic_dataset.py # sinh dữ liệu giả để test pipeline
├── src/ocsvm_fpga/
│   ├── config.py · cli.py · utils.py
│   ├── data/
│   │   ├── dataset.py            # tìm video, đọc nhãn (.npy/.txt/.csv/.mat), chia tập
│   │   └── video_reader.py       # BƯỚC 1: tách khung, crop 4:3, resize, làm mượt
│   ├── features/
│   │   ├── optical_flow.py       # BƯỚC 2: Horn–Schunck (phương trình 42)
│   │   ├── covariance.py         # BƯỚC 3–4: F1–F4 (Bảng 1), ma trận C (44), Log-Euclidean
│   │   ├── grid_flow.py          # ĐƯỜNG CƠ SỞ: thống kê dòng quang học theo lưới ô
│   │   └── pipeline.py           # ghép các bước, cache .npz, xử lý song song
│   ├── models/
│   │   ├── base.py               # KernelExpansion, Standardizer, AnomalyDetector
│   │   ├── ocsvm.py              # One-Class SVM + ràng buộc số SV cho phần cứng
│   │   ├── ls_ocsvm.py           # LS-OC-SVM thưa của bài báo (mô hình so sánh)
│   │   └── training.py           # huấn luyện, hiệu chỉnh ngưỡng, tìm (ν, γ)
│   ├── evaluation/metrics.py     # AUC, EER, P/R/F1, ROC, histogram, timeline
│   ├── quantization/
│   │   ├── fixed_point.py        # Q(m.n), bù hai, hex
│   │   └── golden_model.py       # mô hình bit-accurate của datapath FPGA
│   └── export/fpga_exporter.py   # ghi .mem, .coe, ocsvm_params.vh, ocsvm_params.h
├── firmware/                     # C cho ESP32-S3 (xem firmware/README.md)
│   ├── esp32_features/           # bước 1–4 bằng C, OC-SVM số nguyên, UART, làm mượt
│   └── esp32_app/                # project ESP-IDF mẫu: OV5640 → UART → FPGA → LED
├── tests/                        # kiểm thử đơn vị (python -m unittest discover tests)
├── data/                         # đặt dữ liệu IITB-Corridor ở đây
└── outputs/                      # sinh ra khi chạy: features, models, reports, fpga
```

---

## 3. Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.9. Không cần GPU.

## 4. Chuẩn bị dữ liệu IITB-Corridor

Giải nén bộ Train và Test (link + mật khẩu trong ghi chú dữ liệu của bạn) vào:

```
data/IITB-Corridor/Train/...   ← 208 video .avi bình thường (thư mục con bất kỳ)
data/IITB-Corridor/Test/...    ← video kiểm thử + file nhãn khung hình
```

Video và nhãn được tìm **đệ quy**. File nhãn được ghép với video theo tên (`000001.avi` ↔ `000001.npy`); nếu bản bạn tải về đặt tên khác, sửa `data.annotation_patterns` / `data.annotation_dir` trong `configs/default.yaml`. Chạy bước 00 để kiểm tra trước:

```bash
python scripts/00_inspect_dataset.py
```

**Chưa có nhãn cho video Test?** Pipeline vẫn chạy được (huấn luyện chỉ cần video bình thường; ngưỡng lấy từ 10% video Train giữ lại), chỉ thiếu số liệu AUC/F1. Nên lấy nhãn chính thức của IITB-Corridor (tác giả cung cấp nhãn mức khung hình) để kết quả so sánh được với các bài báo. Nếu không lấy được, dùng công cụ gán nhãn tay:

```bash
python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test --make-template data/IITB-Corridor/test.txt   # CSV mẫu
python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test --class-file data/IITB-Corridor/test.txt     # giao diện xem video
python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test --from-csv segments.csv   # từ CSV
python scripts/08_annotate_videos.py --video-dir data/IITB-Corridor/Test --summary     # thống kê
```

Nếu có `test.txt` (loại hành vi của từng video, khai báo ở `data.test_class_file`), bước 03 còn báo cáo AUC theo từng loại (Chasing, Fighting, CarryObject…). Nhãn được đọc lại mỗi lần đánh giá, nên có thể gán nhãn sau khi đã trích đặc trưng mà không phải tính lại; video chưa gán nhãn tự bị loại khỏi phần đánh giá.

Bước này in số video, độ phân giải, fps, số video Test tìm thấy nhãn, và lưu cách chia tập vào `outputs/splits.json` (mọi bước sau dùng cùng cách chia này).

---

## 5. Chạy

**Kiểm tra nhanh (≈1 phút) trên dữ liệu giả lập** – nên chạy trước để chắc chắn môi trường đúng:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic
python scripts/run_pipeline.py --config configs/synthetic.yaml
python scripts/06_verify_mem_files.py --dir outputs_synthetic/fpga
```

**Chạy thật trên IITB-Corridor:**

```bash
python scripts/run_pipeline.py --jobs 8         # 00 → 04, dùng 8 tiến trình khi trích đặc trưng
python scripts/06_verify_mem_files.py           # PASS = file .mem nhất quán bit-exact
```

Hoặc từng bước:

```bash
python scripts/01_extract_features.py --jobs 8
python scripts/02_train.py
python scripts/03_evaluate.py
python scripts/04_export_fpga.py
```

Ghi đè tham số nhanh không cần sửa YAML:

```bash
python scripts/02_train.py --set model.nu=0.02 model.max_support_vectors=128
python scripts/run_pipeline.py --set features.feature_set=F4
python scripts/02_train.py --set tuning.enabled=true      # tự tìm (ν, γ)
python scripts/02_train.py --set model.type=ls_ocsvm threshold.method=max_train
```

Đặc trưng được cache trong `outputs/features/<hash cấu hình>/`, nên đổi tham số SVM không phải trích xuất lại; đổi tham số tiền xử lý / HS / đặc trưng sẽ tự tạo cache mới. Trích xuất bị ngắt giữa chừng có thể chạy lại và sẽ tiếp tục từ video chưa xong.

Thời gian tham khảo: đặc trưng ≈10 ms/cặp khung ở 160×120 (64 vòng HS) cộng thời gian giải mã video 1080p; với `frame_step: 2` toàn bộ tập Train mất cỡ vài chục phút trên CPU 8 luồng. Huấn luyện OC-SVM chỉ vài giây đến vài phút nhờ nén nguyên mẫu.

---

## 6. Quy trình xử lý ↔ mã nguồn

| Bước | Nội dung | File | Bài báo |
|---|---|---|---|
| 1 | Tách khung xám {I₁…Iₙ}, crop 4:3, resize 160×120, Gaussian | `data/video_reader.py` | §4 |
| 2 | Horn–Schunck: min E = ∬(Iₓu+I_yv+I_t)² + γ²(‖∇u‖²+‖∇v‖²) | `features/optical_flow.py` | (42) |
| 3 | Đạo hàm uₓ,u_y,vₓ,v_y,uₓₓ,u_yy,vₓₓ,v_yy; z_k = F(x,y) theo F1–F4 | `features/covariance.py` | Bảng 1, (43) |
| 4 | μ = (1/n)Σz_k ; C = 1/(n−1)Σ(z_k−μ)(z_k−μ)ᵀ | `features/covariance.py` | (44) |
| 4b | Ánh xạ nhóm Lie → vech(logm C) (Log-Euclidean) | `features/covariance.py` | (45) |
| 5 | One-Class SVM, kernel Gauss | `models/ocsvm.py` | (1)–(3) |
| 5' | LS-OC-SVM thưa, từ điển coherence | `models/ls_ocsvm.py` | (4)–(38) |
| 6 | Ngưỡng, đánh giá AUC mức khung | `models/training.py`, `evaluation/` | (54)–(55) |
| 7 | Lượng tử hóa + Golden Model + xuất file | `quantization/`, `export/` | — |

### Các quyết định thiết kế quan trọng

**Log-Euclidean thay cho kernel nhóm Lie trực tiếp.** Kernel (45) của bài báo cần `log(Cᵢ⁻¹Cⱼ)` cho mọi cặp (mẫu, SV) – quá đắt cho FPGA. Project ánh xạ mỗi C một lần sang `vech(logm C)` (phần tử ngoài đường chéo nhân √2), khi đó ‖vecᵢ − vecⱼ‖² = ‖log Cᵢ − log Cⱼ‖²_F, nên kernel RBF Euclid thông thường chính là kernel Gauss Log-Euclidean – xấp xỉ chuẩn của (45), trùng khớp khi các ma trận giao hoán. FPGA chỉ còn phải tính ‖x − s‖². Với F3: C là 8×8 → véc-tơ 36 chiều (F1/F2: 21, F4: 78).

**Khớp tiền xử lý với ESP32-S3 + OV5640.** Khung được crop giữa về 4:3 rồi đưa về 160×120 xám (QQVGA), `frame_step: 2` mô phỏng ≈15 fps. ESP32 phải dùng đúng các tham số này (đã ghi sẵn trong `ocsvm_params.h`).

**Ngân sách Support Vector.** Số SV quyết định dung lượng BRAM và độ trễ (mỗi SV = một lần tính kernel). Hàng trăm nghìn khung huấn luyện được nén bằng MiniBatchKMeans thành các nguyên mẫu có trọng số; nếu số SV vượt `model.max_support_vectors`, số nguyên mẫu tự giảm và huấn luyện lại.

**Ngưỡng.** Mặc định lấy phân vị 99% điểm số trên 10% video Train giữ lại (≈1% báo động giả trên dữ liệu bình thường). Có thể chọn `zero` (dấu của f) hoặc `max_train` (đúng như bài báo).

**Làm mượt thời gian.** Trung bình trượt nhân quả `temporal_smoothing` khung trên điểm số – thực hiện ở ESP32 sau khi nhận f(x) từ FPGA.

---

## 7. Kết quả đầu ra

```
outputs/
├── splits.json                  # cách chia train/val/test
├── features/<hash>/{train,val,test}/*.npz   # X (véc-tơ), C (ma trận), frame_idx, y
├── models/detector.joblib       # bộ chuẩn hóa + mô hình + ngưỡng
├── reports/
│   ├── dataset_report.json  train_report.json  eval_report.json  quant_report.json
│   └── figures/ roc_test.png  score_hist.png  timelines_test.png
├── logs/*.log
└── fpga/                        # ⇦ NẠP VÀO FPGA / ESP32
    ├── ocsvm_params.vh          # localparam: N_SV, DIM, độ rộng bit, GAMMA_Q, RHO_Q, THR_Q, LUT_SHIFT…
    ├── ocsvm_params.h           # header C cho ESP32: tham số HS, mean/inv_std chuẩn hóa, UART
    ├── ocsvm_model_data.h       # mảng SV/α/LUT dạng C cho OC-SVM phần mềm trên ESP32
    ├── model_float.json         # mô hình số thực đầy đủ (tham chiếu)
    ├── mem/  (dùng với $readmemh)
    │   ├── sv.mem               # N_SV·DIM dòng × DATA_W bit (SV0[0], SV0[1], …, SV1[0], …)
    │   ├── sv_packed.mem        # N_SV dòng, mỗi dòng DIM·DATA_W bit (phần tử 0 ở LSB)
    │   ├── alpha.mem            # N_SV dòng × COEF_W bit (có dấu)
    │   ├── exp_lut.mem          # LUT_SIZE dòng × LUT_W bit, exp(−t) không dấu
    │   ├── scaler_mean.mem / scaler_inv_std.mem   # nếu muốn chuẩn hóa ngay trên FPGA
    │   ├── test_input.mem (+_packed)              # véc-tơ kiểm thử đã lượng tử
    │   ├── test_expected_score.mem                # f_q kỳ vọng (ACC_W bit)
    │   ├── test_expected_label.mem                # cờ bất thường kỳ vọng
    │   ├── test_ground_truth.mem                  # nhãn thật của vector kiểm thử
    │   └── test_kernel_vec0.mem                   # k_q từng SV của vector 0 (gỡ lỗi PE)
    └── coe/ sv.coe alpha.coe exp_lut.coe          # cho Vivado Block Memory Generator
```

### Datapath điểm cố định (mặc định)

| Tín hiệu | Định dạng | Ghi chú |
|---|---|---|
| x, sv | Q5.10 có dấu, 16 bit | x đã chuẩn hóa z-score, bão hòa ±32 |
| diff = x − sv | 17 bit | |
| dist = Σ diff² | 2·DIFF_W + ⌈log₂DIM⌉ bit, Q·20 | khối MAC |
| t = dist · GAMMA_Q | DIST_W + 16 bit | γ không dấu, số bit lẻ tự chọn |
| địa chỉ LUT = t ≫ LUT_SHIFT | 10 bit | t ≥ 16 → k = 0 |
| k = EXP_LUT[addr] | Q0.16 không dấu | exp tại tâm ô → sai số đối xứng |
| α | Q1.14 có dấu | đã chuẩn hóa max\|α\| = 1 |
| acc = Σ α·k, f = acc − RHO_Q | 48 bit, Q·30 | so với THR_Q |

`04_export_fpga.py` báo cáo sai số |f_fixed − f_float|, tỉ lệ trùng nhãn và AUC float/fixed trong `quant_report.json` – số liệu cho mục 2.2 / 3.1 của đề cương. `06_verify_mem_files.py` đọc lại **chỉ** các file `.mem` + `.vh` và tính f_q bằng vòng lặp số nguyên giống phần cứng tuần tự; testbench Verilog nên cho kết quả trùng khớp bit với `test_expected_score.mem`.

Ví dụ nạp trong Verilog:

```verilog
`include "ocsvm_params.vh"
reg signed [DATA_W-1:0] sv_mem   [0:N_SV*DIM-1];
reg signed [COEF_W-1:0] alpha_mem[0:N_SV-1];
reg        [LUT_W-1:0]  exp_lut  [0:LUT_SIZE-1];
initial begin
  $readmemh("sv.mem", sv_mem);
  $readmemh("alpha.mem", alpha_mem);
  $readmemh("exp_lut.mem", exp_lut);
end
```

---

## 8. Phần ESP32-S3 khi demo

Thư mục `firmware/` chứa bản C của bước 1–4 (Gaussian → Horn–Schunck → hiệp biến → logm Jacobi → vech → chuẩn hóa → Q5.10), giao thức UART, bộ làm mượt và một OC-SVM số nguyên chạy trên CPU khớp bit với FPGA (dùng làm baseline khi đo hệ số tăng tốc). Kiểm chứng trên PC trước khi nạp:

```bash
python scripts/07_check_c_port.py     # PASS = firmware C cho ra đúng đầu vào mà mô hình đã học
```

Hướng dẫn build ESP-IDF, giao thức UART và lưu ý về tốc độ khung nằm trong `firmware/README.md`.

**Lưu ý về khác biệt miền dữ liệu.** Mô hình học "chuyển động bình thường" của hành lang IITB. Camera OV5640 ở góc quay và khung cảnh khác sẽ có thống kê chuyển động khác, dễ gây báo động giả. Cách xử lý được khuyến nghị: ghi thêm vài phút video *bình thường* bằng chính ESP32-S3 + OV5640 tại khu vực demo, chép vào một thư mục và huấn luyện lại (hoặc thêm vào `data.train_dir`) – toàn bộ pipeline dùng lại nguyên vẹn. Kết quả trên IITB-Corridor dùng cho phần đánh giá thuật toán trong báo cáo.

---

## 9. Hướng dẫn tinh chỉnh

| Tham số | Ảnh hưởng |
|---|---|
| `features.feature_set` | F1/F2 (21 chiều) nhẹ nhất cho FPGA; F3 (36) cân bằng; F4 (78) giàu thông tin nhất |
| `model.nu` | ν nhỏ → ít SV, biên rộng, ít báo động giả nhưng dễ bỏ sót |
| `model.gamma` | γ lớn → biên sát dữ liệu, nhiều SV; `scale` là điểm khởi đầu tốt |
| `model.max_support_vectors` | 64/128/256… – đánh đổi tài nguyên BRAM/độ trễ với độ chính xác |
| `optical_flow.alpha`, `num_iter` | γ của HS lớn → trường mượt hơn; nhiều vòng lặp → chính xác hơn, chậm hơn trên ESP32 |
| `preprocess.frame_step` | phải khớp tốc độ khung thực tế của ESP32 khi demo |
| `threshold.percentile` | % báo động giả chấp nhận trên dữ liệu bình thường |
| `fpga.data_frac` | giảm nếu nhiều véc-tơ *bình thường* bị bão hòa (xem cảnh báo bước 04) |
| `fpga.exp_lut_addr_bits` | LUT lớn hơn → sai số kernel nhỏ hơn, tốn thêm BRAM |

`tuning.enabled=true` tách 30% video Test làm validation để chọn (ν, γ), các video này bị loại khỏi số liệu báo cáo ở bước 03 để tránh rò rỉ dữ liệu.

---

## 10. Thí nghiệm so sánh đặc trưng (đường cơ sở)

Để chứng minh giá trị của đặc trưng hiệp biến, project có sẵn một đặc trưng đối chứng: **thống kê dòng quang học theo lưới ô** (ý tưởng từ repo sanjeebtiwary/Crowd-Anomaly-Detection). Khung 160×120 được chia lưới 6×8 ô 20×20 px; mỗi ô lấy `[mean u, mean v, mean |v|]` (bản `basic`, 144 chiều) hoặc thêm `max |v|, std |v|` (bản `rich`, 240 chiều); các ô được ghép thành một véc-tơ cho cả khung. Khác repo gốc (mô hình hóa từng ô bằng GMM), ở đây dùng đúng **cùng một OC-SVM** như phương pháp chính, để phép so sánh chỉ khác ở bước trích đặc trưng.

```bash
python scripts/09_compare_features.py --jobs 8 --set model.nu=0.02 model.max_support_vectors=128
```

Script chạy 4 biến thể – `cov` (hiệp biến, cấu hình hiện tại), `grid`, `grid_rich`, `grid_fb` (lưới + Farneback, gần repo gốc nhất) – với cùng cách chia tập, cùng ν/ngân sách SV, cùng quy tắc ngưỡng và làm mượt. Kết quả trong `outputs/experiments/`: `comparison.md` (bảng AUC, khoảng tin cậy 95% bootstrap theo video, EER, AUC theo loại hành vi, AUC float/fixed-point, tỉ lệ véc-tơ bão hòa Q5.10, bộ nhớ SV, MAC/khung, độ trễ FPGA ước tính, thời gian PC), `roc_compare.png`, `per_class_compare.png`. Mô hình chính trong `outputs/models/` **không bị ghi đè**. Cache hiệp biến đã có được dùng lại; mỗi biến thể lưới chỉ trích xuất một lần.

Đặc trưng lưới cũng chạy được với mọi bước khác (`--set features.descriptor=grid_flow`), kể cả xuất file FPGA; riêng firmware ESP32 chỉ cài đặt đặc trưng hiệp biến.

## 11. Kiểm thử

```bash
python -m unittest discover tests -v
```

Bao gồm: HS khôi phục đúng dịch chuyển 1 pixel theo x/y; C khớp `numpy.cov`; tính chất đẳng cự Log-Euclidean; lượng tử hóa/bù hai; Golden Model bám sát mô hình float; ngân sách SV; xuất file.

## Tài liệu tham khảo chính

- T. Wang, J. Chen, Y. Zhou, H. Snoussi, "Online Least Squares One-Class Support Vector Machines-Based Abnormal Visual Event Detection," *Sensors*, 13, 17130–17155, 2013.
- B. Schölkopf et al., "Estimating the Support of a High-Dimensional Distribution," *Neural Computation*, 2001.
- B. K. P. Horn, B. G. Schunck, "Determining Optical Flow," *Artificial Intelligence*, 1981.
- V. Arsigny et al., "Log-Euclidean metrics for fast and simple calculus on diffusion tensors," *MRM*, 2006.
- R. Rodrigues et al., "Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection," WACV 2020 (bộ dữ liệu IITB-Corridor).
