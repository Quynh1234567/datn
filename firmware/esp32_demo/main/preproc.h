/* Tiền xử lý khung hình số nguyên trên ESP32-S3 – đặc tả A1, mục 2.2–2.3
 * (claude/hs_fixed_point_spec.md).
 *
 *   Gauss 5×5, σ = 1, tách được, hệ số k = [56, 250, 412, 250, 56] / 1024
 *     h[y][x] = Σ_i k[i]·I[y][x+i−2]                 (lọc ngang, ≤ 261 120 → 18 bit)
 *     s[y][x] = Σ_j k[j]·h[y+j−2][x]                 (lọc dọc,  ≤ 267 386 880 < 2³²)
 *     G[y][x] = (s + 2¹⁹) >> 20                      (làm tròn một lần, luôn ≤ 255)
 *   Biên: REFLECT_101 (mặc định của cv2.GaussianBlur) hoặc REPLICATE.
 *
 *   Khung tối: Σ G < ngưỡng_TB × 19 200 (ngưỡng 15 → 288 000), tính trên ảnh ĐÃ làm mượt
 *   giống src/ocsvm_fpga/features/pipeline.py.
 *
 * C thuần, không phụ thuộc ESP-IDF: cùng một file chạy trên ESP32 và trên PC (gcc),
 * khớp từng bit với Golden Model Python src/ocsvm_fpga/hwmodel/preproc_q.py. */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PP_W 160
#define PP_H 120
#define PP_PIX (PP_W * PP_H)

#define PP_K0 56     /* hệ số ngoài cùng  (±2) */
#define PP_K1 250    /* hệ số kề tâm      (±1) */
#define PP_K2 412    /* hệ số tâm */
#define PP_SHIFT 20  /* 1024 × 1024 */

typedef enum {
    PP_BORDER_REFLECT101 = 0,   /* ...c b | a b c ... (cv2.BORDER_REFLECT_101, np.pad "reflect") */
    PP_BORDER_REPLICATE = 1,    /* ...a a | a b c ... (cv2.BORDER_REPLICATE, np.pad "edge") */
} pp_border_t;

/* Bộ nhớ làm việc: vòng 5 hàng kết quả lọc ngang (5 × 160 × 4 = 3 200 byte) */
typedef struct {
    uint32_t ring[5][PP_W];
} pp_work_t;

/* Làm mượt src (160×120 uint8) → dst (160×120 uint8). src và dst không được trùng nhau.
 * Trả về Σ dst (dùng cho kiểm tra khung tối). */
uint32_t pp_gauss5(const uint8_t *src, uint8_t *dst, pp_border_t border, pp_work_t *work);

/* Khung tối khi độ sáng trung bình của ảnh đã làm mượt < mean_thr (so sánh số nguyên chính xác) */
static inline bool pp_is_dark(uint32_t sum, uint32_t mean_thr)
{
    return sum < mean_thr * (uint32_t)PP_PIX;
}

/* CRC-32 chuẩn (giống zlib.crc32 / binascii.crc32) */
uint32_t pp_crc32(uint32_t crc, const uint8_t *buf, size_t len);

/* Ảnh mẫu tất định cho tự kiểm tra (Python sinh lại y hệt) */
#define PP_N_PATTERNS 10
const char *pp_pattern_name(int id);
void pp_test_pattern(int id, uint8_t *buf);

#ifdef __cplusplus
}
#endif
