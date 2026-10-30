/* =============================================================================
 *  TIỀN XỬ LÝ TRÊN ESP32-S3: GAUSS 5×5 SỐ NGUYÊN + KHUNG TỐI
 *  Đặc tả A1 – mục 2.2–2.3 (claude/hs_fixed_point_spec.md)
 *
 *  1. Tự kiểm tra trên chip (chưa cần camera): 10 ảnh mẫu × 2 kiểu biên, so CRC với
 *     selftest_vectors.h do Golden Model Python sinh ra → bit-exact.
 *  2. Khởi động OV5640 160×120 xám, đặt 12,5 fps bằng VTS (đã kiểm chứng ở mục 2.1).
 *  3. Benchmark Gauss trên một khung camera (N lần): µs và chu kỳ CPU.
 *  4. Vòng thời gian thực N khung: Gauss → kiểm tra khung tối → bộ chọn cặp 80 ms
 *     (khung tối ngắt cặp). Ghi thời gian, tổng, CRC ảnh gốc và ảnh làm mượt.
 *  5. Gửi vài ảnh GỐC (base64) để PC tính lại bằng Golden Model và so CRC → bit-exact
 *     trên ảnh thật, đồng thời đo sai khác so với cv2.GaussianBlur số thực.
 *
 *  Log (đọc bởi scripts/19_preproc_check.py):
 *    #INFO key=value ...
 *    SELFTEST pattern=<tên> border=<0|1> crc=.. expect=.. sum=.. expect_sum=.. us=.. ok=<0|1>
 *    SELFTEST_RESULT pass=.. fail=..
 *    BENCH iters=.. us_min=.. us_avg=.. us_max=.. cyc_avg=.. src=..
 *    BEGIN_PP n=.. ...
 *    P,idx,ts_us,t_gauss_us,t_loop_us,sum,dark,sel,crc_raw,crc_smooth,mean_raw
 *    END_PP
 *    SUMMARY key=value ...
 *    RAW <idx> 160 120 / base64 / END        SMOOTH <idx> 160 120 / base64 / END
 *    DONE
 * ============================================================================= */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "board_pins.h"

#if defined(CONFIG_IDF_TARGET) && !defined(CONFIG_IDF_TARGET_ESP32S3)
#error "Project dang build cho chip KHAC ESP32-S3. Chay: idf.py set-target esp32s3  (roi build lai)"
#endif
#include "frame_pacer.h"
#include "ov5640_fps.h"
#include "preproc.h"
#include "selftest_vectors.h"

/* Mặc định khi sdkconfig cũ chưa có tùy chọn (muốn đổi: idf.py menuconfig) */
#ifndef CONFIG_PP_XCLK_MHZ
#define CONFIG_PP_XCLK_MHZ 20
#endif
#ifndef CONFIG_PP_TARGET_FPS_X10
#define CONFIG_PP_TARGET_FPS_X10 125
#define CONFIG_PP_AEC_MAX_FOLLOW_VTS 1
#endif
#ifndef CONFIG_PP_AE_LEVEL
#define CONFIG_PP_AE_LEVEL 0
#endif
#ifndef CONFIG_PP_DARK_MEAN
#define CONFIG_PP_DARK_MEAN 15
#endif
#ifndef CONFIG_PP_TARGET_INTERVAL_MS
#define CONFIG_PP_TARGET_INTERVAL_MS 80
#endif
#ifndef CONFIG_PP_TOL_PCT
#define CONFIG_PP_TOL_PCT 20
#endif
#ifndef CONFIG_PP_WARMUP_FRAMES
#define CONFIG_PP_WARMUP_FRAMES 30
#endif
#ifndef CONFIG_PP_BENCH_ITERS
#define CONFIG_PP_BENCH_ITERS 200
#endif
#ifndef CONFIG_PP_N_FRAMES
#define CONFIG_PP_N_FRAMES 300
#endif
#ifndef CONFIG_PP_N_SAMPLES
#define CONFIG_PP_N_SAMPLES 6
#endif
#ifndef CONFIG_PP_EXTRA_LOAD_MS
#define CONFIG_PP_EXTRA_LOAD_MS 0
#endif

#ifdef CONFIG_PP_BORDER_REPLICATE
#define PP_BORDER PP_BORDER_REPLICATE
#else
#define PP_BORDER PP_BORDER_REFLECT101
#endif
#ifdef CONFIG_PP_AEC_MAX_FOLLOW_VTS
#define AEC_FOLLOW true
#else
#define AEC_FOLLOW false
#endif

#define SEL_DARK 5   /* thêm vào pacer_decision_t: khung tối, ngắt cặp */

typedef struct {
    int64_t ts_us;
    uint32_t t_gauss_us;
    uint32_t t_loop_us;
    uint32_t sum;
    uint32_t crc_raw, crc_smooth;
    uint8_t mean_raw;
    uint8_t dark;
    uint8_t sel;
} rec_t;

static pp_work_t s_work;   /* 3,2 KB trong RAM nội */

static void *alloc_big(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(sz, MALLOC_CAP_8BIT);
}

static void *alloc_fast(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : alloc_big(sz);
}

static void print_base64(const char *tag, int idx, const uint8_t *img)
{
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char line[80];
    int li = 0;
    printf("%s %d %d %d\n", tag, idx, PP_W, PP_H);
    for (int i = 0; i < PP_PIX; i += 3) {
        int rem = PP_PIX - i;
        uint32_t v = (uint32_t)img[i] << 16 | (rem > 1 ? (uint32_t)img[i + 1] << 8 : 0) | (rem > 2 ? img[i + 2] : 0);
        line[li++] = tbl[(v >> 18) & 63];
        line[li++] = tbl[(v >> 12) & 63];
        line[li++] = rem > 1 ? tbl[(v >> 6) & 63] : '=';
        line[li++] = rem > 2 ? tbl[v & 63] : '=';
        if (li >= 76) {
            line[li] = 0;
            printf("%s\n", line);
            li = 0;
            vTaskDelay(1);
        }
    }
    if (li) {
        line[li] = 0;
        printf("%s\n", line);
    }
    printf("END\n");
}

/* ------------------------------------------------------------ 1. tự kiểm tra */

static int run_selftest(uint8_t *a, uint8_t *b)
{
    int fails = 0;
    for (size_t i = 0; i < PP_N_VECTORS; i++) {
        const pp_vector_t *v = &PP_VECTORS[i];
        pp_test_pattern(v->pattern, a);
        int64_t t0 = esp_timer_get_time();
        uint32_t sum = pp_gauss5(a, b, (pp_border_t)v->border, &s_work);
        int64_t t1 = esp_timer_get_time();
        uint32_t crc = pp_crc32(0, b, PP_PIX);
        int ok = crc == v->crc && sum == v->sum;
        fails += !ok;
        printf("SELFTEST pattern=%s border=%d crc=%08" PRIX32 " expect=%08" PRIX32 " sum=%" PRIu32
               " expect_sum=%" PRIu32 " us=%d ok=%d\n",
               pp_pattern_name(v->pattern), v->border, crc, v->crc, sum, v->sum, (int)(t1 - t0), ok);
    }
    printf("SELFTEST_RESULT pass=%d fail=%d\n", (int)PP_N_VECTORS - fails, fails);
    return fails;
}

/* ------------------------------------------------------------- 3. benchmark */

static void run_bench(uint8_t *src, uint8_t *dst)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb || fb->len != PP_PIX) {
        printf("BENCH status=FAIL reason=no_frame\n");
        if (fb) {
            esp_camera_fb_return(fb);
        }
        return;
    }
    memcpy(src, fb->buf, PP_PIX);
    esp_camera_fb_return(fb);
    const int n = CONFIG_PP_BENCH_ITERS;
    uint32_t mn = UINT32_MAX, mx = 0;
    uint64_t tot = 0, cyc_tot = 0;
    volatile uint32_t sink = 0;
    for (int i = 0; i < n; i++) {
        uint32_t c0 = esp_cpu_get_cycle_count();
        int64_t t0 = esp_timer_get_time();
        sink += pp_gauss5(src, dst, PP_BORDER, &s_work);
        int64_t t1 = esp_timer_get_time();
        uint32_t c1 = esp_cpu_get_cycle_count();
        uint32_t us = (uint32_t)(t1 - t0);
        mn = us < mn ? us : mn;
        mx = us > mx ? us : mx;
        tot += us;
        cyc_tot += (uint32_t)(c1 - c0);
        if ((i & 31) == 31) {
            vTaskDelay(1);
        }
    }
    printf("BENCH iters=%d us_min=%" PRIu32 " us_avg=%.1f us_max=%" PRIu32 " cyc_avg=%.0f cyc_per_px=%.2f "
           "src=%s border=%d\n",
           n, mn, (double)tot / n, mx, (double)cyc_tot / n, (double)cyc_tot / n / PP_PIX,
           esp_ptr_external_ram(src) ? "psram" : "internal", PP_BORDER);
    (void)sink;
}

/* --------------------------------------------------------- 4. thời gian thực */

static void run_live(rec_t *recs, int n, uint8_t *smooth, uint8_t **samples, int n_samp, int *samp_idx,
                     uint8_t *smooth0)
{
    frame_pacer_t pacer;
    pacer_init(&pacer, CONFIG_PP_TARGET_INTERVAL_MS, CONFIG_PP_TOL_PCT);
    int si = 0;
    for (int i = 0; i < n; i++) {
        rec_t *r = &recs[i];
        memset(r, 0, sizeof(*r));
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t t_start = esp_timer_get_time();
        if (!fb || fb->len != PP_PIX) {
            r->ts_us = t_start;
            r->sel = PACER_NONE;
            pacer_reset(&pacer);
            if (fb) {
                esp_camera_fb_return(fb);
            }
            continue;
        }
        r->ts_us = ov_cam_fb_ts_us(fb);
        int64_t t0 = esp_timer_get_time();
        r->sum = pp_gauss5(fb->buf, smooth, PP_BORDER, &s_work);
        int64_t t1 = esp_timer_get_time();
        r->t_gauss_us = (uint32_t)(t1 - t0);
        r->dark = pp_is_dark(r->sum, CONFIG_PP_DARK_MEAN);
        if (r->dark) {
            pacer_reset(&pacer);          /* khung tối: bỏ khung, ngắt cặp (RESET_PAIR) */
            r->sel = SEL_DARK;
        } else {
            r->sel = (uint8_t)pacer_feed(&pacer, r->ts_us);
        }
        /* ghi chép cho kiểm tra (không thuộc xử lý thật) */
        uint32_t raw_sum = 0;
        for (int k = 0; k < PP_PIX; k++) {
            raw_sum += fb->buf[k];
        }
        r->mean_raw = (uint8_t)(raw_sum / PP_PIX);
        r->crc_raw = pp_crc32(0, fb->buf, PP_PIX);
        r->crc_smooth = pp_crc32(0, smooth, PP_PIX);
        if (si < n_samp && i == samp_idx[si]) {
            memcpy(samples[si], fb->buf, PP_PIX);
            if (si == 0 && smooth0) {
                memcpy(smooth0, smooth, PP_PIX);
            }
            si++;
        }
        esp_camera_fb_return(fb);
        if (CONFIG_PP_EXTRA_LOAD_MS > 0) {
            vTaskDelay(pdMS_TO_TICKS(CONFIG_PP_EXTRA_LOAD_MS));
        }
        r->t_loop_us = (uint32_t)(esp_timer_get_time() - t_start);
    }
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void summarize(const rec_t *recs, int n, int selftest_fail, const fps_result_t *fr)
{
    uint32_t *g = malloc(sizeof(uint32_t) * n), *l = malloc(sizeof(uint32_t) * n);
    double *dt = malloc(sizeof(double) * n);
    int ng = 0, ndt = 0, n_dark = 0, n_ok = 0, n_bad = 0, n_first = 0, n_skip = 0, n_fail = 0;
    int64_t prev = -1;
    for (int i = 0; i < n; i++) {
        const rec_t *r = &recs[i];
        if (r->sel == PACER_NONE) {
            n_fail++;
            prev = -1;
            continue;
        }
        g[ng] = r->t_gauss_us;
        l[ng++] = r->t_loop_us;
        if (prev >= 0) {
            dt[ndt++] = (r->ts_us - prev) / 1000.0;
        }
        prev = r->ts_us;
        switch (r->sel) {
        case SEL_DARK: n_dark++; break;
        case PACER_PAIR_OK: n_ok++; break;
        case PACER_PAIR_BAD: n_bad++; break;
        case PACER_FIRST: n_first++; break;
        default: n_skip++; break;
        }
    }
    qsort(g, ng, sizeof(uint32_t), cmp_u32);
    qsort(l, ng, sizeof(uint32_t), cmp_u32);
    double gs = 0, ls = 0, ds = 0;
    for (int i = 0; i < ng; i++) {
        gs += g[i];
        ls += l[i];
    }
    double dmin = 1e9, dmax = 0;
    int drops = 0;
    for (int i = 0; i < ndt; i++) {
        ds += dt[i];
        dmin = fmin(dmin, dt[i]);
        dmax = fmax(dmax, dt[i]);
    }
    /* chu kỳ tham chiếu: chu kỳ đã đặt bằng VTS, hoặc trung bình đo được */
    double period_ref = CONFIG_PP_TARGET_FPS_X10 > 0 ? 1e4 / CONFIG_PP_TARGET_FPS_X10 : (ndt ? ds / ndt : 0);
    for (int i = 0; i < ndt; i++) {
        drops += dt[i] > 1.5 * period_ref;
    }
    double g_avg = ng ? gs / ng : 0, l_avg = ng ? ls / ng : 0;
    uint32_t g_max = ng ? g[ng - 1] : 0, l_max = ng ? l[ng - 1] : 0, g_p99 = ng ? g[(int)(0.99 * (ng - 1))] : 0;
    double budget = CONFIG_PP_TARGET_INTERVAL_MS * 1000.0;
    printf("SUMMARY n=%d fail=%d selftest_fail=%d gauss_us_avg=%.1f gauss_us_p99=%" PRIu32 " gauss_us_max=%" PRIu32
           " loop_us_avg=%.1f loop_us_max=%" PRIu32 " budget_pct_max=%.2f dt_avg=%.3f dt_min=%.3f dt_max=%.3f "
           "drops=%d dark=%d pair_ok=%d pair_bad=%d first=%d skip=%d fps_ok=%d\n",
           n, n_fail, selftest_fail, g_avg, g_p99, g_max, l_avg, l_max, l_max / budget * 100, ndt ? ds / ndt : 0,
           ndt ? dmin : 0, dmax, drops, n_dark, n_ok, n_bad, n_first, n_skip, fr ? fr->ok : -1);

    printf("\n================ KẾT QUẢ (mục 2.2–2.3) ================\n");
    printf(" Tự kiểm tra bit-exact : %s (%d/%d vector)\n", selftest_fail ? "KHÔNG ĐẠT" : "ĐẠT",
           (int)PP_N_VECTORS - selftest_fail, (int)PP_N_VECTORS);
    printf(" Gauss 5x5 mỗi khung   : TB %.0f µs, p99 %" PRIu32 " µs, max %" PRIu32 " µs\n", g_avg, g_p99, g_max);
    printf(" Xử lý cả vòng / khung : TB %.0f µs, max %" PRIu32 " µs = %.1f%% ngân sách %d ms\n", l_avg, l_max,
           l_max / budget * 100, CONFIG_PP_TARGET_INTERVAL_MS);
    printf(" Khoảng cách khung     : TB %.2f ms, min %.2f, max %.2f\n", ndt ? ds / ndt : 0, ndt ? dmin : 0, dmax);
    printf(" Khung tối (<%d)       : %d  → ngắt cặp, khung kế tiếp bắt đầu cặp mới\n", CONFIG_PP_DARK_MEAN,
           n_dark);
    printf(" Cặp 80 ms hợp lệ/loại : %d / %d (chuỗi mới %d)\n", n_ok, n_bad, n_first);
    printf(" Bit-exact trên ảnh thật: chạy scripts/19_preproc_check.py để kiểm tra\n");
    printf("=======================================================\n\n");
    free(g);
    free(l);
    free(dt);
}

/* ------------------------------------------------------------------- main */

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    printf("\n#INFO test=preproc board=%s idf=%s cores=%d psram_mb=%u\n", CAM_BOARD_NAME, esp_get_idf_version(),
           ci.cores, (unsigned)(esp_psram_is_initialized() ? esp_psram_get_size() >> 20 : 0));
    printf("#INFO cfg xclk_mhz=%d target_fps_x10=%d aec_follow=%d ae_level=%d border=%d dark_mean=%d "
           "target_ms=%d tol_pct=%d n_frames=%d n_samples=%d extra_load_ms=%d kernel=56,250,412,250,56/1024\n",
           CONFIG_PP_XCLK_MHZ, CONFIG_PP_TARGET_FPS_X10, AEC_FOLLOW, CONFIG_PP_AE_LEVEL, PP_BORDER,
           CONFIG_PP_DARK_MEAN, CONFIG_PP_TARGET_INTERVAL_MS, CONFIG_PP_TOL_PCT, CONFIG_PP_N_FRAMES,
           CONFIG_PP_N_SAMPLES, CONFIG_PP_EXTRA_LOAD_MS);

    uint8_t *buf_a = alloc_fast(PP_PIX), *buf_b = alloc_fast(PP_PIX);
    const int n = CONFIG_PP_N_FRAMES;
    rec_t *recs = alloc_big(sizeof(rec_t) * n);
    if (!buf_a || !buf_b || !recs) {
        printf("#INFO error=out_of_memory\n");
        return;
    }

    /* 1. tự kiểm tra trên chip */
    int st_fail = run_selftest(buf_a, buf_b);

    /* 2. camera */
    esp_err_t err = ov_cam_start(CONFIG_PP_XCLK_MHZ, 2, true, false);
    printf("#INFO cam_init=%s err=0x%x\n", err == ESP_OK ? "OK" : "FAIL", err);
    if (err != ESP_OK) {
        printf("DONE\n");
        return;
    }
    sensor_t *s = esp_camera_sensor_get();
    printf("#INFO sensor_pid=0x%04X\n", s ? s->id.PID : 0);
    if (s && CONFIG_PP_AE_LEVEL != 0) {
        s->set_ae_level(s, CONFIG_PP_AE_LEVEL);
    }
    ov_cam_warmup(CONFIG_PP_WARMUP_FRAMES);
    fps_result_t fr = {0};
    if (CONFIG_PP_TARGET_FPS_X10 > 0) {
        fr = ov5640_set_fps(CONFIG_PP_TARGET_FPS_X10, AEC_FOLLOW);
    }

    /* 3. benchmark */
    run_bench(buf_a, buf_b);

    /* 4. thời gian thực */
    int n_samp = CONFIG_PP_N_SAMPLES;
    uint8_t *samples[16] = {0};
    int samp_idx[16];
    for (int k = 0; k < n_samp; k++) {
        samples[k] = alloc_big(PP_PIX);
        if (!samples[k]) {
            n_samp = k;
            break;
        }
        samp_idx[k] = (int)((k + 0.5) * n / (n_samp ? n_samp : 1));   /* rải đều */
    }
    uint8_t *smooth0 = n_samp > 0 ? alloc_big(PP_PIX) : NULL;
    printf("#INFO measuring n=%d (che kín ống kính ~3 s trong lúc đo để thử khung tối)...\n", n);
    run_live(recs, n, buf_b, samples, n_samp, samp_idx, smooth0);

    printf("BEGIN_PP n=%d border=%d dark_mean=%d target_ms=%d tol_pct=%d extra_load_ms=%d\n", n, PP_BORDER,
           CONFIG_PP_DARK_MEAN, CONFIG_PP_TARGET_INTERVAL_MS, CONFIG_PP_TOL_PCT, CONFIG_PP_EXTRA_LOAD_MS);
    printf("P,idx,ts_us,t_gauss_us,t_loop_us,sum,dark,sel,crc_raw,crc_smooth,mean_raw\n");
    for (int i = 0; i < n; i++) {
        const rec_t *r = &recs[i];
        printf("P,%d,%" PRId64 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%u,%u,%08" PRIX32 ",%08" PRIX32 ",%u\n", i,
               r->ts_us, r->t_gauss_us, r->t_loop_us, r->sum, r->dark, r->sel, r->crc_raw, r->crc_smooth,
               r->mean_raw);
        if ((i & 63) == 63) {
            vTaskDelay(1);
        }
    }
    printf("END_PP\n");
    summarize(recs, n, st_fail, CONFIG_PP_TARGET_FPS_X10 > 0 ? &fr : NULL);

    /* 5. ảnh mẫu */
    for (int k = 0; k < n_samp; k++) {
        print_base64("RAW", samp_idx[k], samples[k]);
    }
    if (smooth0) {
        print_base64("SMOOTH", samp_idx[0], smooth0);
    }
    printf("DONE\n");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
