# hwmodel – Golden Model số nguyên của phần cứng

Chuẩn tham chiếu **khớp từng bit** cho firmware ESP32 và RTL FPGA, theo `claude/hs_fixed_point_spec.md`.

| File | Nội dung | Khớp với |
|---|---|---|
| `preproc_q.py` | Gauss 5×5 /1024, khung tối (mục 2.2–2.3) | `firmware/esp32_preproc/main/preproc.c` – **đã kiểm chứng trên board** |
| `hs_q.py` | đạo hàm, hệ số a/b, 64 vòng Jacobi, \|v\|, trung bình ô, nén log, z-score (mục 3–8) | RTL khối trích đặc trưng (sắp viết) |
| `hs_export.py` | xuất `hs_params.vh`, bảng log, scaler, véc-tơ kiểm thử từng tầng | testbench RTL |

## Dùng nhanh

```python
from ocsvm_fpga.hwmodel import preproc_q as pq, hs_q as hq
g1, g2 = pq.gauss5_q(raw1), pq.gauss5_q(raw2)          # như ESP32
sc = hq.quantize_scaler(det.scaler.mean, det.scaler.scale)
tr = hq.features_q(g1, g2, scaler=sc)                   # như FPGA
tr.hs.u, tr.hs.v      # Q6.6 sau 64 vòng
tr.mean_q, tr.log_q   # 48 đặc trưng Q5.10 trước / sau nén log
tr.z_q                # 48 giá trị Q5.10 đưa vào OC-SVM (FixedPointOCSVM.decision_q)
```

## Kiểm tra và xuất véc-tơ

```bash
python scripts/20_hs_golden.py --selftest                                   # 9 kiểm tra, không cần dữ liệu
python scripts/20_hs_golden.py --video data/IITB-Corridor/Test/<video>.avi --pairs 300 --sweep
python scripts/20_hs_golden.py --video ... --export 8                       # → outputs/hw_vectors/
```

Với video thật và mô hình `outputs/models/detector.joblib`, báo cáo `outputs/hs_golden/<thời gian>/report.md` cho:
- sai số từng nguồn: số học FPGA, Gauss số nguyên, tổng so với pipeline train;
- sai số theo σ của bộ chuẩn hóa, x_q lệch bao nhiêu LSB, **cờ bất thường trùng bao nhiêu %**;
- bảng quét độ rộng u, v (12–16 bit) và \|v\| floor/round (`--sweep`).

Véc-tơ RTL (`outputs/hw_vectors/case_XX/`): `img1/img2.mem` (ảnh đã Gauss) → `deriv.mem` → `uv_iter1.mem` → `uv_final.mem` → `mag.mem` → `cell_sums.mem` → `feat_mean.mem` → `feat_log.mem` → `feat_z.mem` → `expected.txt` (f_q, cờ). Testbench so từng tầng nên khi sai biết ngay khối nào sai. Định dạng gói bit ghi ở đầu mỗi file và trong `hs_export.py`.
