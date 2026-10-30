/* Bộ chọn cặp khung theo thời gian (đặc tả A1, mục 2.1).
 *
 * Mỗi khung camera được đưa vào kèm timestamp. Bộ chọn giữ một khung khi khoảng
 * cách tới khung được giữ trước đó gần khoảng cách danh định nhất (target, mặc định
 * 80 ms), rồi đánh giá cặp (khung trước, khung này):
 *   PAIR_OK  : |dt − target| ≤ tol → được tính dòng quang học
 *   PAIR_BAD : lệch quá tol (rớt khung, giật) → bỏ cặp, khung này làm khung đầu cặp mới
 * pacer_reset() ngắt cặp từ bên ngoài (khung tối, lỗi camera) – tương ứng cờ RESET_PAIR.
 *
 * Code C thuần, không phụ thuộc ESP-IDF: dùng lại nguyên văn trong firmware chính
 * và kiểm thử được trên PC (test_pacer.c). */
#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    PACER_SKIP = 0,      /* bỏ qua khung (chưa đến lúc) */
    PACER_FIRST = 1,     /* khung đầu tiên của một chuỗi (chưa có cặp) */
    PACER_PAIR_OK = 2,   /* giữ khung, cặp hợp lệ */
    PACER_PAIR_BAD = 3,  /* giữ khung, cặp bị loại vì sai khoảng cách */
    PACER_NONE = 4,      /* không có khung (lỗi lấy khung) – chỉ dùng khi ghi log */
} pacer_decision_t;

typedef struct {
    int64_t target_us;       /* khoảng cách danh định */
    int64_t tol_us;          /* sai lệch cho phép */
    int64_t period_us;       /* chu kỳ camera ước lượng (trung bình trượt) */
    int64_t last_raw_us;     /* timestamp khung camera gần nhất */
    int64_t last_sel_us;     /* timestamp khung được giữ gần nhất */
    int64_t last_pair_dt_us; /* khoảng cách của cặp vừa tạo */
    bool has_raw;
    bool has_sel;
} frame_pacer_t;

void pacer_init(frame_pacer_t *p, int target_ms, int tol_pct);
void pacer_reset(frame_pacer_t *p);
pacer_decision_t pacer_feed(frame_pacer_t *p, int64_t ts_us);
const char *pacer_name(pacer_decision_t d);
