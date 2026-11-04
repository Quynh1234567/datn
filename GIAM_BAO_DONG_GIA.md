# Giảm báo động giả trên video IITB bình thường

## Vì sao báo nhầm

1. **Người đi ở cột bên phải hoặc góc dưới bên phải.** Lúc train, cột phải (bụi cây) ít người qua, nên σ ở đó rất nhỏ. σ của |v| ở cột phải là 0,024 / 0,063 / 0,098, trong khi ở lối đi là 0,12–0,14. Một người đi qua làm z tăng 2–10 lần, và điểm số chạm đáy −ρ.
2. **Nhiều người đi sát nhau.** Nhiều ô cùng chuyển động, nhiều chiều z cùng lớn, và khoảng cách RBF cộng dồn.
3. **Chỉ một kết quả vượt ngưỡng là bật báo động**, rồi giữ thêm 12 cặp. Vì vậy tỉ lệ vượt ngưỡng 1–5 % thành tỉ lệ báo động 9–21 %.

## Đã sửa gì (bật bằng config, mặc định TẮT nên kết quả cũ không đổi)

| Khóa config | Tác dụng | File |
|---|---|---|
| `model.scaler_std_floor_rel` | σ ≥ k × trung vị σ khi chuẩn hóa | `models/base.py`, `models/training.py` |
| `model.gamma` (đã có) | γ nhỏ hơn thì kernel rộng hơn | – |
| `alarm.on_pairs` | bật cảnh báo khi vượt ngưỡng N cặp liên tiếp | `evaluation/events.py`, `03_evaluate.py`, `05_infer_video.py`, `09_compare_features.py` |
| `scripts/31_tune_false_alarm.py` (mới) | thử nhiều cấu hình trên đặc trưng đã có, xuất bảng so sánh | – |

Phần cứng: RTL **không đổi**. Mức sàn σ và γ chỉ thay nội dung các file ROM (`scaler_inv_q.mem`, SV, α, exp LUT) và các tham số INV_FRAC, γ_q, ρ_q, THR_q trong `hs_params.vh` / `ocsvm_params.vh`.

## Các bước

### 0. Chép file và sao lưu mô hình đang chạy trên board
```
(giải nén zip vào D:\datn\ocsvm_anomaly_fpga, ghi đè)
xcopy /E /I outputs_hw\models  backup\outputs_hw_041126\models
xcopy /E /I outputs_hw\fpga    backup\outputs_hw_041126\fpga
xcopy /E /I outputs_hw\reports backup\outputs_hw_041126\reports
```

### 1. Thử các cấu hình (khoảng 20–40 phút, không ghi đè mô hình)
```
python scripts/31_tune_false_alarm.py --config configs/hw.yaml --floors 0 0.5 0.75 1 --gammas 1 0.5 --on-pairs 1 3 5 --videos 000225 000239 000255 000299 000338 000343
```
Thêm tên 4 video bình thường còn lại vào `--videos`.
- Kết quả nằm ở `outputs_hw/reports/false_alarm_tuning.md`. Dòng đầu là bản gốc.
- Script tự **đề xuất** biến thể có ít báo giả/giờ nhất mà tỉ lệ phát hiện sự kiện giảm không quá 3 điểm %.
- Xem thêm cột AUC (không nên thấp hơn khoảng 0,72) và các cột % báo động của từng video bạn liệt kê.

### 2. Ghi cấu hình đã chọn vào `configs/hw.yaml`
Bỏ dấu `#` ở khối cuối file và sửa thành giá trị script đề xuất, ví dụ:
```yaml
model:
  scaler_std_floor_rel: 1.0
  gamma: 0.021
alarm:
  on_pairs: 3
```

### 3. Train lại, đánh giá, xuất FPGA
```
python scripts/02_train.py        --config configs/hw.yaml
python scripts/03_evaluate.py     --config configs/hw.yaml
python scripts/04_export_fpga.py  --config configs/hw.yaml
```
Kiểm tra trong log:
- 02: dòng `Chuẩn hóa: σ ≥ … → nâng N/48 chiều`; số SV ≤ 128.
- 03: AUC, số sự kiện phát hiện được và báo giả/giờ. So với bản cũ: 0,7348 · 130/143 · 90,1.
- 04: `nhãn trùng với mô hình số thực` ≥ 99,9 %. **Ghi lại `ρ_q` và `THR_q`**, vì chúng đổi so với bản cũ (1 073 408 562 / −1 014 665 947).

### 4. Kiểm tra lại trên video
```
python scripts/05_infer_video.py --config configs/hw.yaml --video data\IITB-Corridor\Test\000343.avi --save-video
```
Chạy cho 10 video bình thường và 2–3 video bất thường thuộc nhóm chuyển động (SuddenRunning, Chasing, Fighting). Mong đợi: % cảnh báo trên video bình thường giảm rõ, video bất thường vẫn báo.

### 5. Đưa lên FPGA (chỉ làm khi bước 3–4 tốt hơn rõ)
```
python scripts/15_export_rtl.py   --config configs/hw.yaml --lanes 8
python scripts/24_hs_rtl_sim.py   --config configs/hw.yaml --tb feat            (phải 100 % PASS)
cd rtl\build && run_sim.bat && cd ..\..                                           (tb_ocsvm_core PASS)
python scripts/25_hs_board_test.py --config configs/hw.yaml --gowin fpga_gowin/hs_board_test
python scripts/26_gowin_system.py  --config configs/hw.yaml
```
1. Xóa thư mục `impl` của từng project, mở Gowin, Run All (Place/Route Option = 2), nạp **External Flash**.
2. Chạy board test: `python scripts/25_hs_board_test.py --config configs/hw.yaml --port COM9`. Phải KHỚP với dòng mong đợi mới.
3. Màn hình demo: trong `demo1_launcher.json`, đổi `"threshold"` thành THR_q mới ở bước 3, hoặc để trống và hiệu chỉnh lại (`calib_sec`). Giá trị cũ −1 070 000 000 **không còn đúng** vì ρ đã đổi.

Nếu Gowin không đi dây được (unrouted nets), giữ bitstream cũ cho demo camera. Phần chứng minh cải thiện thì trình bày bằng kết quả bước 3–4. Đường số nguyên trên PC khớp FPGA từng bit, nên số liệu đó vẫn đúng cho phần cứng.

### Hoàn tác
Thêm lại dấu `#` cho khối cuối `hw.yaml`, rồi chép lại `backup\outputs_hw_041126\*` vào `outputs_hw\`.
