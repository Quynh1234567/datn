/* Kiểm thử bộ chọn cặp khung trên PC:
 *   gcc -O2 -I ../main test_pacer.c ../main/frame_pacer.c -o test_pacer && ./test_pacer */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "frame_pacer.h"

static int fails;

#define CHECK(cond, ...)                                                                                     \
    do {                                                                                                     \
        if (!(cond)) {                                                                                       \
            printf("  FAIL: " __VA_ARGS__);                                                                  \
            printf("\n");                                                                                    \
            fails++;                                                                                         \
        }                                                                                                    \
    } while (0)

/* Chạy chuỗi khung chu kỳ period_ms; drop[i] = 1 nghĩa là khung i bị mất */
static void run(const char *name, double period_ms, int n, const int *drops, int n_drops, int exp_ok_min,
                int exp_bad, double exp_pair_ms)
{
    frame_pacer_t p;
    pacer_init(&p, 80, 20);
    int ok = 0, bad = 0, skip = 0, first = 0;
    double pair_sum = 0;
    for (int i = 0; i < n; i++) {
        int dropped = 0;
        for (int k = 0; k < n_drops; k++) {
            dropped |= (drops[k] == i);
        }
        if (dropped) {
            continue;
        }
        int64_t ts = (int64_t)llround(i * period_ms * 1000.0);
        pacer_decision_t d = pacer_feed(&p, ts);
        if (d == PACER_PAIR_OK) {
            ok++;
            pair_sum += p.last_pair_dt_us / 1000.0;
        } else if (d == PACER_PAIR_BAD) {
            bad++;
        } else if (d == PACER_SKIP) {
            skip++;
        } else if (d == PACER_FIRST) {
            first++;
        }
    }
    double pair_mean = ok ? pair_sum / ok : 0;
    printf("%-34s ok=%3d bad=%d skip=%3d first=%d pair_mean=%.2f ms\n", name, ok, bad, skip, first, pair_mean);
    CHECK(ok >= exp_ok_min, "%s: ok=%d < %d", name, ok, exp_ok_min);
    CHECK(bad == exp_bad, "%s: bad=%d != %d", name, bad, exp_bad);
    CHECK(fabs(pair_mean - exp_pair_ms) < 1.0, "%s: pair_mean=%.2f != %.2f", name, pair_mean, exp_pair_ms);
}

int main(void)
{
    /* 25 fps: giữ 1 trong 2 khung, mọi cặp đúng 80 ms */
    run("25 fps", 40.0, 201, NULL, 0, 99, 0, 80.0);
    /* 12,5 fps: giữ mọi khung */
    run("12.5 fps", 80.0, 101, NULL, 0, 99, 0, 80.0);
    /* 15 fps: giữ mọi khung, cặp 66,7 ms (lệch 17% < 20% → vẫn hợp lệ, nhưng KHÁC lúc train) */
    run("15 fps", 1000.0 / 15, 101, NULL, 0, 99, 0, 66.67);
    /* 30 fps: giữ 1 trong 2 khung → 66,7 ms */
    run("30 fps", 1000.0 / 30, 201, NULL, 0, 99, 0, 66.67);
    /* 25 fps rớt khung được giữ (khung 10) → một cặp 120 ms bị loại, rồi chạy tiếp */
    int d1[] = {10};
    run("25 fps, rớt 1 khung được giữ", 40.0, 201, d1, 1, 97, 1, 80.0);
    /* 25 fps rớt khung bị bỏ qua (khung 11) → không ảnh hưởng */
    int d2[] = {11};
    run("25 fps, rớt 1 khung bị bỏ qua", 40.0, 201, d2, 1, 99, 0, 80.0);

    /* pacer_reset: khung kế tiếp là khung đầu chuỗi */
    frame_pacer_t p;
    pacer_init(&p, 80, 20);
    pacer_feed(&p, 0);
    pacer_feed(&p, 40000);
    pacer_reset(&p);
    CHECK(pacer_feed(&p, 80000) == PACER_FIRST, "reset không ngắt cặp");
    CHECK(pacer_feed(&p, 120000) == PACER_SKIP, "sau reset phải bỏ khung 40 ms");
    CHECK(pacer_feed(&p, 160000) == PACER_PAIR_OK, "sau reset cặp 80 ms phải hợp lệ");

    printf(fails ? "\n%d lỗi\n" : "\nTẤT CẢ ĐẠT\n", fails);
    return fails != 0;
}
