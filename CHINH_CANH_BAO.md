# Chỉnh logic cảnh báo (mô hình và FPGA giữ nguyên)

## Vì sao 000255 vẫn báo
Có 2 đoạn báo động, tổng 8,6 % thời lượng video:
- Khung 98–208 (3,9–8,3 s): **3 người đi hàng ngang** từ sát camera.
- Khung 1738–1822 (69,5–72,9 s): **1 người đi vào từ góc dưới phải**, sát camera.

Điểm cao nhất chỉ 0,105, vừa qua ngưỡng 0,1008 (đáy là 0,1066). Mỗi lần chỉ có 3–6 cặp thật sự vượt ngưỡng. Phần lớn thời gian đỏ là **khoảng giữ**: điểm phải xuống dưới ngưỡng tắt 0,040 (phân vị 85 %) liên tục 1 s mới tắt, nên báo động kéo dài suốt lúc người còn ở gần camera.

## Thêm mới
- `alarm.on_percentile`: ngưỡng BẬT riêng cho cảnh báo, bằng phân vị (100 − p) % điểm bình thường của tập val. Giá trị 2 là ngưỡng hiện tại, 1 là cao hơn. Không đổi mô hình, không đổi THR_Q trên FPGA.
- `scripts/32_tune_alarm.py`: quét ngưỡng bật, số cặp liên tiếp, ngưỡng tắt, thời gian giữ trên mô hình đang dùng. Chạy vài giây.

## Các bước
1. Chép đè file trong zip.
2. Quét:
   `python scripts/32_tune_alarm.py --config configs/hw.yaml --videos 000225 000239 000255 000299 000338 000343`
   Mở `outputs_hw/reports/alarm_tuning.md`, xem dòng **đề xuất** (phát hiện giảm không quá 5 điểm %).
3. Chép khối `alarm:` đề xuất vào `configs/hw.yaml`, thay khối `alarm:` hiện có.
4. Chạy lại `python scripts/03_evaluate.py --config configs/hw.yaml`, rồi `05_infer_video.py --config configs/hw.yaml --video ... --save-video` cho video cần trình bày.
5. Màn hình demo camera: đổi `threshold` trong `demo1_launcher.json` thành giá trị `--thr` mà script in ra, nếu muốn khớp với PC. Bitstream giữ nguyên.
