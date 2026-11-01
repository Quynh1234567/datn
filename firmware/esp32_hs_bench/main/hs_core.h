/* hs_core.h – Khối Horn–Schunck chạy trên CPU (1 luồng), dùng làm MỐC SO SÁNH với bộ tăng tốc FPGA.
 *
 *  hs_*      : số nguyên, KHỚP TỪNG BIT Golden Model (src/ocsvm_fpga/hwmodel/hs_q.py, đặc tả 0.4)
 *              = đúng phép tính của RTL: DERIV → HS_ITER (N vòng Jacobi) → FEAT (48 giá trị z)
 *  hsf_*     : số thực float32, cùng thuật toán (pipeline train số thực), không khớp bit
 *
 *  Cùng một file chạy trên PC (scripts/23_hs_cpu_bench.py nạp qua ctypes) và trên ESP32-S3.
 *  Jacobi tại chỗ với 3 hàng đệm – giống cách RTL dùng line buffer – nên chỉ cần MỘT mảng u, v.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HS_W 160
#define HS_H 120
#define HS_N (HS_W * HS_H)
#define HS_DIM 48

#if defined(_WIN32)
#define HS_API __declspec(dllexport)
#else
#define HS_API
#endif

/* DERIV: 2 ảnh 8 bit (đã Gauss) → Ix4, Iy4, It4 (11 bit), a, b (15 bit, 18 bit lẻ). */
HS_API void hs_deriv(const uint8_t *i1, const uint8_t *i2,
                     int16_t *ix, int16_t *iy, int16_t *it, int16_t *a, int16_t *b);

/* HS_ITER: u, v Q6.6 (13 bit) – hàm tự xóa về 0 rồi chạy n_iter vòng. Trả về số lần bão hòa. */
HS_API int32_t hs_iter(const int16_t *ix, const int16_t *iy, const int16_t *it,
                       const int16_t *a, const int16_t *b, int16_t *u, int16_t *v, int32_t n_iter);

/* FEAT: |v|, tổng 16 ô, trung bình, nén log (bảng 1504 phần tử), z-score → 48 giá trị Q5.10.
 * sums (48), mean_q, log_q có thể NULL nếu không cần. */
HS_API void hs_feat(const int16_t *u, const int16_t *v,
                    const uint16_t *lut, int32_t lut_n,
                    const int16_t *mean_q, const uint16_t *inv_q, int32_t inv_frac,
                    int32_t *sums, int16_t *mean_out, int16_t *log_out, int16_t *z);

/* Bản số thực float32: ws = 4·HS_N float (ix, iy, it, mẫu số). */
HS_API void hsf_deriv(const uint8_t *i1, const uint8_t *i2, float *ws, float alpha);
HS_API void hsf_iter(const float *ws, float *u, float *v, int32_t n_iter);

#ifdef __cplusplus
}
#endif
