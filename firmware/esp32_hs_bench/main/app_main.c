/* =============================================================================
 *  ĐO THỜI GIAN KHỐI HORN–SCHUNCK TRÊN ESP32-S3 (240 MHz, 1 lõi)
 *  Mốc so sánh cho bộ tăng tốc FPGA – cùng thuật toán, cùng véc-tơ với testbench RTL.
 *
 *  1. Số nguyên (hs_core.c, khớp từng bit RTL): DERIV → HS_ITER (N vòng) → FEAT, so u, v, z
 *     với test_vectors.h (Golden Model). Chạy với dữ liệu ở SRAM trong (nếu đủ chỗ) và ở PSRAM.
 *  2. Số thực float32 (cùng thuật toán, như pipeline train cũ) – dữ liệu ở PSRAM (461 KB).
 *
 *  test_vectors.h sinh bởi:
 *    python scripts/23_hs_cpu_bench.py --config configs/hw.yaml \
 *           --esp32-header firmware/esp32_hs_bench/main/test_vectors.h
 *
 *  Log:  #INFO …  ·  BENCH impl=<tên> mem=<sram|psram> stage=<deriv|iter|feat> us_min=.. us_med=.. us_max=..
 *        CHECK impl=<tên> mem=.. u_ok=.. v_ok=.. z_ok=.. n_sat=..   ·   bảng tóm tắt   ·   DONE
 * ============================================================================= */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "hs_core.h"

#if __has_include("test_vectors.h")
#include "test_vectors.h"
#else
#error "Thieu test_vectors.h - chay: python scripts/23_hs_cpu_bench.py --config configs/hw.yaml --esp32-header firmware/esp32_hs_bench/main/test_vectors.h"
#endif

#define REPEAT 5
#define MAX_RES 16

typedef struct { const char *impl, *mem, *stage; int64_t mn, med, mx; } res_t;
static res_t g_res[MAX_RES];
static int g_nres;

static int cmp64(const void *a, const void *b)
{
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static void record(const char *impl, const char *mem, const char *stage, int64_t *t, int n)
{
    qsort(t, n, sizeof(int64_t), cmp64);
    res_t r = {impl, mem, stage, t[0], t[n / 2], t[n - 1]};
    if (g_nres < MAX_RES) g_res[g_nres++] = r;
    printf("BENCH impl=%s mem=%s stage=%s us_min=%" PRId64 " us_med=%" PRId64 " us_max=%" PRId64 "\n",
           impl, mem, stage, r.mn, r.med, r.mx);
}

static void *alloc(size_t sz, int psram)
{
    return heap_caps_malloc(sz, (psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT);
}

/* ---------------------------------------------------------------- số nguyên */
static int bench_int(int psram)
{
    const char *mem = psram ? "psram" : "sram";
    int16_t *buf = alloc(7 * HS_N * sizeof(int16_t), psram);         /* ix iy it a b u v = 269 KB */
    if (!buf) {
        printf("#INFO int mem=%s: khong du bo nho (can %u B, con lon nhat %u B) -> bo qua\n", mem,
               (unsigned)(7 * HS_N * sizeof(int16_t)),
               (unsigned)heap_caps_get_largest_free_block((psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL)));
        return 0;
    }
    int16_t *ix = buf, *iy = ix + HS_N, *it = iy + HS_N, *a = it + HS_N, *b = a + HS_N, *u = b + HS_N, *v = u + HS_N;
    int32_t sums[HS_DIM];
    int16_t mq[HS_DIM], lq[HS_DIM], z[HS_DIM];
    int64_t t[REPEAT];
    int32_t n_sat = 0;

    for (int r = 0; r < REPEAT; r++) {
        int64_t t0 = esp_timer_get_time();
        hs_deriv(TV_IMG1, TV_IMG2, ix, iy, it, a, b);
        t[r] = esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    record("c_int", mem, "deriv", t, REPEAT);
    for (int r = 0; r < REPEAT; r++) {
        int64_t t0 = esp_timer_get_time();
        n_sat = hs_iter(ix, iy, it, a, b, u, v, TV_N_ITER);
        t[r] = esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    record("c_int", mem, "iter", t, REPEAT);
    for (int r = 0; r < REPEAT; r++) {
        int64_t t0 = esp_timer_get_time();
        hs_feat(u, v, TV_LUT, TV_LUT_N, TV_MEAN, TV_INV, TV_INV_FRAC, sums, mq, lq, z);
        t[r] = esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    record("c_int", mem, "feat", t, REPEAT);
    const int u_ok = memcmp(u, TV_U, sizeof(TV_U)) == 0, v_ok = memcmp(v, TV_V, sizeof(TV_V)) == 0;
    const int z_ok = memcmp(z, TV_Z, sizeof(TV_Z)) == 0;
    printf("CHECK impl=c_int mem=%s u_ok=%d v_ok=%d z_ok=%d n_sat=%" PRId32 "\n", mem, u_ok, v_ok, z_ok, n_sat);
    heap_caps_free(buf);
    return u_ok && v_ok && z_ok;
}

/* ---------------------------------------------------------------- số thực */
static void bench_float(void)
{
    float *ws = alloc(6 * HS_N * sizeof(float), 1);                  /* ix iy it den u v = 461 KB */
    if (!ws) {
        printf("#INFO float: khong co PSRAM du cho (can 461 KB) -> bo qua\n");
        return;
    }
    float *u = ws + 4 * HS_N, *v = ws + 5 * HS_N;
    int64_t t[REPEAT];
    for (int r = 0; r < REPEAT; r++) {
        int64_t t0 = esp_timer_get_time();
        hsf_deriv(TV_IMG1, TV_IMG2, ws, 2.0f);
        t[r] = esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    record("c_float", "psram", "deriv", t, REPEAT);
    for (int r = 0; r < REPEAT; r++) {
        int64_t t0 = esp_timer_get_time();
        hsf_iter(ws, u, v, TV_N_ITER);
        t[r] = esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    record("c_float", "psram", "iter", t, REPEAT);
    /* so sánh thô với kết quả số nguyên (Q6.6): chỉ để chắc chắn bản float tính đúng thuật toán */
    double dmax = 0;
    for (int p = 0; p < HS_N; p++) {
        double du = u[p] - TV_U[p] / 64.0, dv = v[p] - TV_V[p] / 64.0;
        if (du < 0) du = -du;
        if (dv < 0) dv = -dv;
        if (du > dmax) dmax = du;
        if (dv > dmax) dmax = dv;
    }
    printf("CHECK impl=c_float mem=psram max_diff_vs_int_px=%.4f\n", dmax);
    heap_caps_free(ws);
}

static int64_t find(const char *impl, const char *mem, const char *stage)
{
    for (int i = 0; i < g_nres; i++)
        if (!strcmp(g_res[i].impl, impl) && !strcmp(g_res[i].mem, mem) && !strcmp(g_res[i].stage, stage))
            return g_res[i].med;
    return -1;
}

static void summary(void)
{
    const char *rows[][2] = {{"c_int", "sram"}, {"c_int", "psram"}, {"c_float", "psram"}};
    printf("\n=== TOM TAT (trung vi %d lan, ESP32-S3 %d MHz, 1 loi, N = %d vong, ca %s) ===\n",
           REPEAT, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, TV_N_ITER, TV_CASE);
    printf("%-8s %-6s %10s %12s %10s %10s %10s\n", "impl", "mem", "DERIV ms", "HS_ITER ms", "1 vong ms", "FEAT ms", "Tong ms");
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        int64_t d = find(rows[i][0], rows[i][1], "deriv"), it = find(rows[i][0], rows[i][1], "iter");
        int64_t f = find(rows[i][0], rows[i][1], "feat");
        if (d < 0 || it < 0) continue;
        double tot = (d + it + (f > 0 ? f : 0)) / 1000.0;
        char fs[16] = "-";
        if (f > 0) snprintf(fs, sizeof(fs), "%.2f", f / 1000.0);
        printf("%-8s %-6s %10.2f %12.1f %10.3f %10s %10.1f\n", rows[i][0], rows[i][1], d / 1000.0, it / 1000.0,
               it / 1000.0 / TV_N_ITER, fs, tot);
    }
    printf("FPGA 100 MHz (uoc luong thiet ke): DERIV 0.50 + HS_ITER %.1f + FEAT 0.01 = %.1f ms\n",
           0.43 * TV_N_ITER, 0.51 + 0.43 * TV_N_ITER);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    printf("\n#INFO chip=ESP32-S3 cores=%d rev=%d cpu_mhz=%d case=%s n_iter=%d repeat=%d\n",
           ci.cores, ci.revision, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, TV_CASE, TV_N_ITER, REPEAT);
    printf("#INFO sram_free=%u sram_largest=%u psram_free=%u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    int ok_sram = bench_int(0);
    int ok_psram = bench_int(1);
    bench_float();
    summary();
    printf("KET LUAN: so nguyen khop tung bit voi Golden Model: sram=%s psram=%s\n",
           ok_sram ? "DAT" : "-", ok_psram ? "DAT" : "-");
    printf("DONE\n");
}
