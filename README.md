# Bộ tăng tốc phần cứng phát hiện hành vi bất thường bằng One-Class SVM trên FPGA

**Đồ án tốt nghiệp:** Thiết kế bộ tăng tốc phần cứng phát hiện hành vi bất thường trong giám sát an ninh sử dụng thuật toán One-Class SVM trên FPGA.

Repository triển khai quy trình **huấn luyện trên PC với IITB-Corridor** và **demo bằng ESP32-S3 + OV5640 + FPGA Tang Nano 9K**. Theo `configs/default.yaml` hiện tại, **đặc trưng chính là thống kê dòng quang học Horn–Schunck theo lưới 4×4 (`grid_flow`)**, **không phải ma trận hiệp biến**.

> **Nguồn cấu hình:** `configs/default.yaml` là nguồn xác định cấu hình chạy. Khi thay đổi cấu hình, cần huấn luyện, lượng tử hóa, xuất lại tham số và kiểm chứng tương thích với firmware/RTL.

## 1. Kiến trúc hệ thống

```text
HUẤN LUYỆN TRÊN PC
IITB-Corridor (208 video AVI bình thường)
  → đọc video, lấy khung xám 160×120
  → dòng quang học Horn–Schunck giữa các khung
  → chia trường dòng quang học thành lưới 4 hàng × 4 cột
  → 3 thống kê/ô: [u, v, |v|] (cấu hình basic)
  → véc-tơ đặc trưng 16 × 3 = 48 chiều
  → chuẩn hóa → One-Class SVM (RBF)
  → lượng tử hóa fixed-point → xuất .mem / .vh / .h

DEMO PHẦN CỨNG
OV5640 → ESP32-S3
  → tiền xử lý và trích xuất đặc trưng 4×4 tương thích PC
  → lượng tử hóa véc-tơ 48 chiều
  → UART → Tang Nano 9K (tính hàm quyết định OC-SVM)
  → UART → ESP32-S3 → làm mượt/ngưỡng → LED, buzzer
```

**Phân công tính toán:** ESP32-S3 xử lý ảnh và tạo đặc trưng; FPGA tăng tốc **suy luận OC-SVM**, không huấn luyện mô hình và không trực tiếp xử lý video.

## 2. Đặc trưng dòng quang học lưới 4×4

Với mỗi cặp khung hình liên tiếp, Horn–Schunck ước lượng trường vận tốc `u(x,y)` và `v(x,y)`. Trường dòng quang học được chia thành 16 vùng (4×4). Với `grid_stats: basic`, mỗi vùng được biểu diễn bằng ba thống kê `[u, v, |v|]` theo quy ước của bộ trích xuất trong mã nguồn. Ghép các vùng theo thứ tự nhất quán tạo ra véc-tơ **48 chiều**.

Cấu hình đang chọn:

```yaml
preprocess:
  width: 160
  height: 120
  frame_step: 2
optical_flow:
  method: hs
  alpha: 2.0
  num_iter: 64
features:
  descriptor: grid_flow
  grid_rows: 4
  grid_cols: 4
  grid_stats: basic
  grid_log_scale: 0.5
model:
  type: ocsvm
  nu: 0.02
  gamma: 0.042
  max_support_vectors: 128
```

**Lưu ý:** Trong `default.yaml` vẫn có các tùy chọn `feature_set: F4`, `cov_epsilon`, `log_euclidean`, nhưng chúng **chỉ có hiệu lực khi chọn `descriptor: covariance`**. Chúng không nằm trong pipeline `grid_flow` hiện tại. Một số chú thích cũ trong YAML ghi 144/240 chiều là theo lưới **6×8**; với lưới **4×4**, `basic` có **48 chiều**, còn `rich` có **80 chiều** (5 thống kê/ô).

## 3. Huấn luyện One-Class SVM

Mô hình sử dụng **Scikit-learn `OneClassSVM(kernel="rbf")`**:

\[
f(\mathbf{x})=\sum_{i=1}^{N_{SV}}\alpha_i\exp\left(-\gamma\|\mathbf{x}-\mathbf{s}_i\|^2\right)-\rho.
\]

Điểm quyết định được so sánh với ngưỡng đã hiệu chỉnh để xác định bất thường. Với ngưỡng mặc định của OC-SVM là 0, `f(x) < 0` biểu thị mẫu bất thường; khi sử dụng ngưỡng percentile, quy tắc quyết định phải dùng ngưỡng đã lưu.

Dữ liệu bình thường được nén thành các nguyên mẫu bằng `MiniBatchKMeans`, sau đó huấn luyện OC-SVM có trọng số để kiểm soát số support vectors, tối đa theo cấu hình là **128**. Các support vectors và hệ số sau huấn luyện được lưu ở dạng `float64`; bước xuất FPGA chuyển sang số nguyên fixed-point.

## 4. Cài đặt

Yêu cầu: Python 3.9+ và các thư viện trong `requirements.txt`.

```powershell
cd "D:\New folder\ocsvm_anomaly_fpga"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Chuẩn bị IITB-Corridor

```text
data/
└── IITB-Corridor/
    ├── Train/    # 208 video .avi bình thường (có thể có thư mục con)
    └── Test/     # video kiểm thử và nhãn nếu có
```

Cấu hình mặc định tìm video theo thư mục con, hỗ trợ `.avi`, `.mp4`, `.mov` và các mẫu tên nhãn trong `configs/default.yaml`. Tách 10% video Train làm validation bình thường (`val_video_ratio: 0.10`). **Không sử dụng video bất thường của Test để huấn luyện OC-SVM**; nếu dùng nhãn Test để chọn siêu tham số, phải tách riêng tập tuning và tập báo cáo.

```powershell
python scripts/00_inspect_dataset.py
```

## 6. Chạy pipeline trên PC

**Chạy lần lượt:**

```powershell
python scripts/01_extract_features.py --jobs 4
python scripts/02_train.py
python scripts/03_evaluate.py
python scripts/04_export_fpga.py
python scripts/06_verify_mem_files.py
```

**Hoặc chạy pipeline tổng hợp:**

```powershell
python scripts/run_pipeline.py --jobs 4
python scripts/06_verify_mem_files.py
```

**Chạy thử với dữ liệu giả lập (kiểm tra môi trường, không thay thế đánh giá IITB):**

```powershell
python scripts/make_synthetic_dataset.py --out data/synthetic
python scripts/run_pipeline.py --config configs/synthetic.yaml
python scripts/06_verify_mem_files.py --dir outputs_synthetic/fpga
```

Đánh giá ROC-AUC chỉ có ý nghĩa khi tập kiểm thử có nhãn hợp lệ và chứa cả mẫu bình thường lẫn bất thường. Không công bố AUC nếu chỉ có một lớp nhãn.

## 7. Xuất tham số cho FPGA

Theo `default.yaml`, cấu hình lượng tử hóa gồm:

| Thành phần | Cấu hình |
|---|---|
| Véc-tơ đầu vào và support vectors | 16 bit có dấu, `data_frac: 10` |
| Hệ số α | 16 bit, `coef_frac: 14` |
| Gamma | 16 bit |
| LUT RBF | 1024 phần tử, 16 bit/phần tử |
| Bộ tích lũy nội bộ | 48 bit |
| Số véc-tơ kiểm thử | 200 |

Script `04_export_fpga.py` tạo các tham số, bảng tra kernel và dữ liệu kiểm thử trong `outputs/fpga/`, ví dụ `sv.mem`, `alpha.mem`, `exp_lut.mem`, `ocsvm_params.vh`, `ocsvm_params.h` và các tệp Golden Model (tên, cấu trúc thực tế phụ thuộc phiên bản exporter).

**Quan trọng:** `.mem` được dùng khi xây dựng/mô phỏng RTL, không phải một tệp có thể chép trực tiếp vào FPGA để chạy độc lập. Sau khi thay đổi mô hình hoặc số chiều, cần xây dựng lại bitstream và kiểm tra toàn bộ chuỗi PC → FPGA → ESP32.

## 8. Demo ESP32-S3 + OV5640 + Tang Nano 9K

1. OV5640 thu ảnh xám 160×120.
2. ESP32-S3 tính Horn–Schunck và đặc trưng lưới 4×4, tạo **48 giá trị** theo đúng thứ tự và chuẩn hóa của PC.
3. ESP32-S3 gửi véc-tơ lượng tử hóa qua UART tới FPGA.
4. FPGA tính kernel RBF và điểm quyết định; trả kết quả về ESP32-S3.
5. ESP32-S3 áp dụng làm mượt, ngưỡng cảnh báo và điều khiển LED/buzzer.

Thư mục firmware: `firmware/esp32_features/`, `firmware/esp32_app/`. Hướng dẫn build: `firmware/README.md`.

**Điều kiện bắt buộc trước demo:** xác nhận firmware C thực sự đang trích xuất `grid_flow` **48 chiều**. Nếu firmware vẫn tạo véc-tơ hiệp biến F4 **78 chiều**, không được dùng chung với mô hình 48 chiều vừa train; phải đồng bộ firmware, giao thức UART, tham số và RTL rồi kiểm thử lại. Đặc biệt, các lần kiểm thử FPGA PASS trước đó không tự động xác nhận phiên bản mô hình/lưới mới.

Do khác biệt góc quay và bối cảnh giữa IITB-Corridor và camera demo, có thể xảy ra domain shift và báo động giả. Hiệu năng toàn hệ thống phải được đo thực tế; không mặc định đạt thời gian thực 15 FPS chỉ vì video được lấy mẫu theo `frame_step: 2`.

## 9. Các script và kiểm thử hữu ích

| Script | Chức năng |
|---|---|
| `00_inspect_dataset.py` | Kiểm tra dữ liệu và nhãn |
| `01_extract_features.py` | Video → dòng quang học → đặc trưng |
| `02_train.py` | Huấn luyện OC-SVM |
| `03_evaluate.py` | Đánh giá mô hình |
| `04_export_fpga.py` | Lượng tử hóa và xuất tham số |
| `05_infer_video.py` | Chạy thử video/webcam |
| `06_verify_mem_files.py` | Kiểm tra dữ liệu `.mem` |
| `07_check_c_port.py` | So sánh triển khai C và Python |
| `08_annotate_videos.py` | Hỗ trợ tạo nhãn |
| `09_compare_features.py` | Thí nghiệm so sánh các đặc trưng |
| `16_uart_host_test.py` | Kiểm tra suy luận FPGA qua UART (nếu có trong phiên bản repo) |

```powershell
python -m unittest discover tests -v
python scripts/07_check_c_port.py
```

Các phép so sánh `covariance` và `grid_flow` là **thí nghiệm phụ**, không thay đổi phương pháp chính được chọn trong cấu hình hiện tại.

## 10. Tài liệu tham khảo

- B. Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation, 2001 — cơ sở One-Class SVM.
- B. K. P. Horn và B. G. Schunck, *Determining Optical Flow*, Artificial Intelligence, 1981 — phương pháp ước lượng dòng quang học.
- T. Wang et al., *Online Least Squares One-Class Support Vector Machines-Based Abnormal Visual Event Detection*, Sensors, 2013 — tài liệu tham khảo về phát hiện bất thường bằng thông tin chuyển động; **không phải mô tả chính xác bộ trích xuất lưới 4×4 hiện tại**.
- R. Rodrigues et al., *Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection*, WACV 2020 — tham khảo bộ dữ liệu IITB-Corridor.

---


