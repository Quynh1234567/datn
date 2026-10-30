/* =============================================================================
 *  KIỂM TRA CAMERA ESP32-S3 + OV5640 VÀ ĐO THỜI GIAN KHUNG
 *  Đặc tả A1 – mục 2.1 (claude/hs_fixed_point_spec.md)
 *
 *  1. In thông tin chip, flash, PSRAM.
 *  2. Khởi tạo OV5640: 160×120 xám (QQVGA, PIXFORMAT_GRAYSCALE), kiểm tra mã cảm biến.
 *  3. (Tùy chọn) quét XCLK × fb_count để tìm cấu hình cho FPS cao và ổn định.
 *  4. Ghi timestamp của N khung (mặc định 500) vào RAM – KHÔNG in trong lúc đo để
 *     không làm sai thời gian – rồi in toàn bộ dạng CSV và bảng tóm tắt.
 *  5. Chạy bộ chọn cặp khung 80 ms ±20% (frame_pacer) trên đúng chuỗi khung đó.
 *  6. Gửi vài ảnh mẫu (base64) để xem bằng scripts/17_cam_grab.py hoặc 18_cam_timing.py.
 *
 *  Định dạng log (đọc bởi scripts/18_cam_timing.py):
 *    #INFO key=value ...
 *    BEGIN_TIMING <nhãn> key=value ...
 *    T,idx,ts_us,get_us,len,w,h,fmt,mean,crc,sel,ok
 *    END_TIMING
 *    SUMMARY <nhãn> key=value ...
 *    SWEEP key=value ...
 *    FRAME 160 120 / <base64> / END
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
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_rom_crc.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "board_pins.h"
#include "frame_pacer.h"

/* Giá trị mặc định khi sdkconfig cũ chưa có tùy chọn (Kconfig.projbuild chưa cập nhật
 * hoặc chưa chạy lại cấu hình). Muốn đổi giá trị: idf.py menuconfig. */
#ifndef CONFIG_CAMT_XCLK_MHZ
#define CONFIG_CAMT_XCLK_MHZ 20
#endif
#ifndef CONFIG_CAMT_FB_COUNT
#define CONFIG_CAMT_FB_COUNT 2
#endif
#ifndef CONFIG_CAMT_WARMUP_FRAMES
#define CONFIG_CAMT_WARMUP_FRAMES 30
#endif
#ifndef CONFIG_CAMT_N_FRAMES
#define CONFIG_CAMT_N_FRAMES 500
#endif
#ifndef CONFIG_CAMT_TARGET_INTERVAL_MS
#define CONFIG_CAMT_TARGET_INTERVAL_MS 80
#endif
#ifndef CONFIG_CAMT_TOL_PCT
#define CONFIG_CAMT_TOL_PCT 20
#endif
#ifndef CONFIG_CAMT_SIM_LOAD_MS
#define CONFIG_CAMT_SIM_LOAD_MS 0
#endif
#ifndef CONFIG_CAMT_SEND_FRAMES
#define CONFIG_CAMT_SEND_FRAMES 3
#endif
#ifndef CONFIG_CAMT_TARGET_FPS_X10
#define CONFIG_CAMT_TARGET_FPS_X10 125      /* 12,5 fps → khung cách nhau 80 ms */
#define CONFIG_CAMT_AEC_MAX_FOLLOW_VTS 1
#endif
#ifndef CONFIG_CAMT_AE_LEVEL
#define CONFIG_CAMT_AE_LEVEL 0
#endif

#define IMG_W 160
#define IMG_H 120
#define IMG_PIX (IMG_W * IMG_H)

#ifdef CONFIG_CAMT_FB_IN_PSRAM
#define FB_LOCATION CAMERA_FB_IN_PSRAM
#else
#define FB_LOCATION CAMERA_FB_IN_DRAM
#endif

#ifdef CONFIG_CAMT_GRAB_LATEST
#define GRAB_LATEST_DEFAULT true
#else
#define GRAB_LATEST_DEFAULT false
#endif

/* Một bản ghi cho mỗi khung, ghi vào RAM trong lúc đo */
typedef struct {
    int64_t ts_us;   /* timestamp driver gắn cho khung (esp_timer, lúc VSYNC) */
    int64_t get_us;  /* thời điểm esp_camera_fb_get() trả về */
    uint32_t len;
    uint32_t crc;    /* CRC32 dữ liệu ảnh – phát hiện khung lặp */
    uint16_t w, h;
    uint8_t fmt;
    uint8_t mean;    /* độ sáng trung bình 0..255 */
    uint8_t sel;     /* pacer_decision_t */
    uint8_t ok;      /* đúng 160×120, 19 200 byte, xám */
} rec_t;

typedef struct {
    int n_rec, n_fail, n_badfmt, n_ok;
    double fps, dt_mean, dt_std, dt_med, dt_min, dt_max, dt_p1, dt_p99; /* ms */
    int n_drop_gaps, n_lost_est, n_dup;
    int mean_min, mean_max, n_dark;
    double mean_avg;
    int n_first, n_pair_ok, n_pair_bad, n_skip;
    double pair_mean, pair_std, pair_min, pair_max; /* ms, chỉ cặp hợp lệ */
} stats_t;

static bool s_cam_on;

/* ------------------------------------------------------------------ hệ thống */

static void print_system_info(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    size_t psram = esp_psram_is_initialized() ? esp_psram_get_size() : 0;
    printf("\n#INFO test=cam_timing board=%s idf=%s\n", CAM_BOARD_NAME, esp_get_idf_version());
    printf("#INFO chip=%s cores=%d rev=%d.%d flash_mb=%" PRIu32 " psram_mb=%u internal_free_kb=%u\n",
           ci.model == CHIP_ESP32S3 ? "ESP32-S3" : "khac", ci.cores, ci.revision / 100, ci.revision % 100,
           flash / (1024 * 1024), (unsigned)(psram / (1024 * 1024)),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    printf("#INFO cfg xclk_mhz=%d fb_count=%d grab=%s fb_in=%s warmup=%d n_frames=%d target_ms=%d "
           "tol_pct=%d load_ms=%d lock_exposure=%d target_fps_x10=%d ae_level=%d\n",
           CONFIG_CAMT_XCLK_MHZ, CONFIG_CAMT_FB_COUNT, GRAB_LATEST_DEFAULT ? "latest" : "when_empty",
#ifdef CONFIG_CAMT_FB_IN_PSRAM
           "psram",
#else
           "dram",
#endif
           CONFIG_CAMT_WARMUP_FRAMES, CONFIG_CAMT_N_FRAMES, CONFIG_CAMT_TARGET_INTERVAL_MS, CONFIG_CAMT_TOL_PCT,
           CONFIG_CAMT_SIM_LOAD_MS,
#ifdef CONFIG_CAMT_LOCK_EXPOSURE
           1,
#else
           0,
#endif
           CONFIG_CAMT_TARGET_FPS_X10, CONFIG_CAMT_AE_LEVEL);
}

/* ------------------------------------------------------------------- camera */

static esp_err_t cam_start(int xclk_mhz, int fb_count, bool grab_latest)
{
    if (s_cam_on) {
        esp_camera_deinit();
        s_cam_on = false;
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    camera_config_t c = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = xclk_mhz * 1000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_GRAYSCALE,
        .frame_size = FRAMESIZE_QQVGA,      /* 160×120 */
        .jpeg_quality = 12,                 /* không dùng với ảnh xám */
        .fb_count = fb_count,
        .fb_location = FB_LOCATION,
        .grab_mode = grab_latest ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY,
    };
    esp_err_t err = esp_camera_init(&c);
    s_cam_on = (err == ESP_OK);
    return err;
}

static bool check_sensor(bool verbose)
{
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        printf("#INFO sensor=NONE\n");
        return false;
    }
    camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);
    bool is_ov5640 = (s->id.PID == OV5640_PID);
    if (verbose) {
        printf("#INFO sensor_pid=0x%04X sensor_name=%s is_ov5640=%d\n", s->id.PID, info ? info->name : "?",
               is_ov5640);
    }
    return is_ov5640;
}

#ifdef CONFIG_CAMT_LOCK_EXPOSURE
static void lock_exposure(void)
{
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_exposure_ctrl(s, 0);   /* tắt tự động phơi sáng, giữ giá trị hiện tại */
        s->set_gain_ctrl(s, 0);       /* tắt tự động độ lợi */
        printf("#INFO exposure_locked=1\n");
    }
}
#endif

static void warmup(int n)
{
    for (int i = 0; i < n; i++) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            esp_camera_fb_return(fb);
        }
    }
}

/* ------------------------------------------------------------- đặt FPS */

static int64_t fb_ts_us(const camera_fb_t *fb)
{
    int64_t ts = (int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec;
    return ts > 0 ? ts : esp_timer_get_time();
}

#if CONFIG_CAMT_TARGET_FPS_X10 > 0
/* Chu kỳ khung trung bình (µs) trên n khoảng */
static double measure_period_us(int n)
{
    int64_t t0 = -1, t1 = -1;
    int got = 0;
    for (int i = 0; i <= n; i++) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            continue;
        }
        int64_t t = fb_ts_us(fb);
        esp_camera_fb_return(fb);
        if (t0 < 0) {
            t0 = t;
        } else {
            t1 = t;
            got++;
        }
    }
    return got > 0 ? (double)(t1 - t0) / got : 0;
}

static int reg16(sensor_t *s, int hi, int lo)
{
    int a = s->get_reg(s, hi, 0xFF), b = s->get_reg(s, lo, 0xFF);
    return (a < 0 || b < 0) ? -1 : ((a << 8) | b);
}

static void wreg16(sensor_t *s, int hi, int lo, int v)
{
    s->set_reg(s, hi, 0xFF, (v >> 8) & 0xFF);
    s->set_reg(s, lo, 0xFF, v & 0xFF);
}

/* FPS = PCLK / (HTS × VTS) → chu kỳ tỉ lệ thuận với VTS: VTS_mới = VTS · T_mục_tiêu / T_đo */
static void set_target_fps(void)
{
    sensor_t *s = esp_camera_sensor_get();
    const double target_us = 1e7 / CONFIG_CAMT_TARGET_FPS_X10;
    int hts = reg16(s, 0x380C, 0x380D), vts = reg16(s, 0x380E, 0x380F);
    double p = measure_period_us(30);
    printf("#INFO fps_ctrl_before hts=%d vts=%d period_ms=%.3f fps=%.3f aec_max60=%d aec_max50=%d night=%d\n", hts,
           vts, p / 1000, p > 0 ? 1e6 / p : 0, reg16(s, 0x3A02, 0x3A03), reg16(s, 0x3A14, 0x3A15),
           (s->get_reg(s, 0x3A00, 0xFF) >> 2) & 1);
    if (vts <= 0 || p <= 0) {
        printf("#INFO fps_ctrl=FAIL reason=khong_doc_duoc_VTS_hoac_chu_ky\n");
        return;
    }
    for (int it = 0; it < 3; it++) {
        int nv = (int)lround(vts * target_us / p);
        if (nv < 130) {
            nv = 130;              /* ít nhất 120 hàng ảnh + lề */
        }
        if (nv > 0xFFFF) {
            nv = 0xFFFF;
        }
        wreg16(s, 0x380E, 0x380F, nv);
#ifdef CONFIG_CAMT_AEC_MAX_FOLLOW_VTS
        wreg16(s, 0x3A02, 0x3A03, nv - 4);   /* phơi sáng tối đa (60 Hz) */
        wreg16(s, 0x3A14, 0x3A15, nv - 4);   /* phơi sáng tối đa (50 Hz) */
#endif
        s->set_reg(s, 0x3A00, 0x04, 0x00);    /* tắt night mode: AEC không tự chèn hàng làm đổi FPS */
        warmup(5);
        double np = measure_period_us(30);
        printf("#INFO fps_ctrl_iter=%d vts=%d->%d period_ms=%.3f fps=%.3f\n", it, vts, nv, np / 1000,
               np > 0 ? 1e6 / np : 0);
        if (np <= 0) {
            break;
        }
        vts = nv;
        p = np;
        if (fabs(p - target_us) / target_us < 0.002) {
            break;
        }
    }
    double err = fabs(p - target_us) / target_us * 100;
    printf("#INFO fps_ctrl=%s target_fps=%.2f period_ms=%.3f err_pct=%.3f vts=%d\n", err < 1 ? "OK" : "APPROX",
           1e6 / target_us, p / 1000, err, vts);
}
#endif

/* ------------------------------------------------------------------ đo đạc */

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double pct(const double *sorted, int n, double p)
{
    if (n <= 0) {
        return 0;
    }
    double k = p * (n - 1);
    int i = (int)k;
    double f = k - i;
    return i + 1 < n ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[n - 1];
}

/* Đo n khung, ghi vào recs; snaps (có thể NULL) nhận tối đa n_snap ảnh rải đều */
static void measure(rec_t *recs, int n, uint8_t **snaps, int n_snap, int load_ms, frame_pacer_t *pacer)
{
    int snap_i = 0;
    int snap_every = n_snap > 0 ? n / n_snap : 0;
    for (int i = 0; i < n; i++) {
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t now = esp_timer_get_time();
        rec_t *r = &recs[i];
        memset(r, 0, sizeof(*r));
        r->get_us = now;
        if (!fb) {
            r->ts_us = now;
            r->sel = PACER_NONE;
            if (pacer) {
                pacer_reset(pacer);     /* mất khung → ngắt cặp */
            }
            continue;
        }
        r->ts_us = fb_ts_us(fb);
        r->len = fb->len;
        r->w = fb->width;
        r->h = fb->height;
        r->fmt = (uint8_t)fb->format;
        r->ok = (fb->width == IMG_W && fb->height == IMG_H && fb->len == IMG_PIX &&
                 fb->format == PIXFORMAT_GRAYSCALE);
        uint32_t nb = fb->len < IMG_PIX ? fb->len : IMG_PIX;
        if (nb > 0) {
            uint32_t sum = 0;
            for (uint32_t k = 0; k < nb; k++) {
                sum += fb->buf[k];
            }
            r->mean = (uint8_t)(sum / nb);
            r->crc = esp_rom_crc32_le(0, fb->buf, nb);
        }
        r->sel = pacer ? (uint8_t)pacer_feed(pacer, r->ts_us) : PACER_SKIP;
        if (snaps && snap_i < n_snap && r->ok && i % snap_every == snap_every / 2) {
            memcpy(snaps[snap_i++], fb->buf, IMG_PIX);
        }
        esp_camera_fb_return(fb);
        if (load_ms > 0) {
            vTaskDelay(pdMS_TO_TICKS(load_ms));   /* giả lập ESP32 bận xử lý */
        }
    }
}

static void compute_stats(const rec_t *recs, int n, stats_t *st)
{
    memset(st, 0, sizeof(*st));
    st->n_rec = n;
    st->mean_min = 255;
    double *dt = malloc(sizeof(double) * n);
    double *pd = malloc(sizeof(double) * n);
    int n_dt = 0, n_pd = 0;
    int64_t prev_ts = -1, first_ts = -1, last_ts = -1, prev_sel_ts = -1;
    uint32_t prev_crc = 0;
    double mean_sum = 0;
    for (int i = 0; i < n; i++) {
        const rec_t *r = &recs[i];
        if (r->sel == PACER_NONE) {
            st->n_fail++;
            prev_ts = -1;
            prev_sel_ts = -1;
            continue;
        }
        if (!r->ok) {
            st->n_badfmt++;
        }
        st->n_ok++;
        if (first_ts < 0) {
            first_ts = r->ts_us;
        }
        last_ts = r->ts_us;
        if (prev_ts >= 0) {
            dt[n_dt++] = (r->ts_us - prev_ts) / 1000.0;
            if (r->crc == prev_crc) {
                st->n_dup++;
            }
        }
        prev_ts = r->ts_us;
        prev_crc = r->crc;
        mean_sum += r->mean;
        if (r->mean < st->mean_min) {
            st->mean_min = r->mean;
        }
        if (r->mean > st->mean_max) {
            st->mean_max = r->mean;
        }
        if (r->mean < 15) {
            st->n_dark++;
        }
        switch (r->sel) {
        case PACER_SKIP: st->n_skip++; break;
        case PACER_FIRST: st->n_first++; prev_sel_ts = r->ts_us; break;
        case PACER_PAIR_OK:
            st->n_pair_ok++;
            if (prev_sel_ts >= 0) {
                pd[n_pd++] = (r->ts_us - prev_sel_ts) / 1000.0;
            }
            prev_sel_ts = r->ts_us;
            break;
        case PACER_PAIR_BAD: st->n_pair_bad++; prev_sel_ts = r->ts_us; break;
        default: break;
        }
    }
    st->mean_avg = st->n_ok ? mean_sum / st->n_ok : 0;
    if (n_dt > 0) {
        double s = 0, s2 = 0;
        for (int i = 0; i < n_dt; i++) {
            s += dt[i];
            s2 += dt[i] * dt[i];
        }
        st->dt_mean = s / n_dt;
        st->dt_std = sqrt(fmax(0, s2 / n_dt - st->dt_mean * st->dt_mean));
        qsort(dt, n_dt, sizeof(double), cmp_double);
        st->dt_min = dt[0];
        st->dt_max = dt[n_dt - 1];
        st->dt_med = pct(dt, n_dt, 0.5);
        st->dt_p1 = pct(dt, n_dt, 0.01);
        st->dt_p99 = pct(dt, n_dt, 0.99);
        for (int i = 0; i < n_dt; i++) {
            if (dt[i] > 1.5 * st->dt_med) {
                st->n_drop_gaps++;
                st->n_lost_est += (int)lround(dt[i] / st->dt_med) - 1;
            }
        }
        if (last_ts > first_ts) {
            st->fps = n_dt * 1e6 / (double)(last_ts - first_ts);
        }
    }
    if (n_pd > 0) {
        double s = 0, s2 = 0;
        st->pair_min = 1e9;
        for (int i = 0; i < n_pd; i++) {
            s += pd[i];
            s2 += pd[i] * pd[i];
            st->pair_min = fmin(st->pair_min, pd[i]);
            st->pair_max = fmax(st->pair_max, pd[i]);
        }
        st->pair_mean = s / n_pd;
        st->pair_std = sqrt(fmax(0, s2 / n_pd - st->pair_mean * st->pair_mean));
    }
    free(dt);
    free(pd);
}

static void print_records(const char *label, const rec_t *recs, int n, int xclk, int fbc, bool latest,
                          int load_ms)
{
    printf("BEGIN_TIMING %s n=%d xclk_mhz=%d fb_count=%d grab=%s target_ms=%d tol_pct=%d load_ms=%d\n", label, n,
           xclk, fbc, latest ? "latest" : "when_empty", CONFIG_CAMT_TARGET_INTERVAL_MS, CONFIG_CAMT_TOL_PCT,
           load_ms);
    printf("T,idx,ts_us,get_us,len,w,h,fmt,mean,crc,sel,ok\n");
    for (int i = 0; i < n; i++) {
        const rec_t *r = &recs[i];
        printf("T,%d,%" PRId64 ",%" PRId64 ",%" PRIu32 ",%u,%u,%u,%u,%08" PRIX32 ",%u,%u\n", i, r->ts_us, r->get_us,
               r->len, r->w, r->h, r->fmt, r->mean, r->crc, r->sel, r->ok);
        if ((i & 63) == 63) {
            vTaskDelay(1);   /* nhường CPU cho task idle khi in dài */
        }
    }
    printf("END_TIMING\n");
}

static void print_summary(const char *label, const stats_t *st)
{
    printf("SUMMARY %s n=%d ok=%d fail=%d badfmt=%d fps=%.3f dt_mean=%.3f dt_std=%.3f dt_med=%.3f dt_min=%.3f "
           "dt_max=%.3f dt_p1=%.3f dt_p99=%.3f drop_gaps=%d lost_est=%d dup=%d bright_min=%d bright_avg=%.1f "
           "bright_max=%d dark=%d first=%d pair_ok=%d pair_bad=%d skip=%d pair_mean=%.3f pair_std=%.3f "
           "pair_min=%.3f pair_max=%.3f\n",
           label, st->n_rec, st->n_ok, st->n_fail, st->n_badfmt, st->fps, st->dt_mean, st->dt_std, st->dt_med,
           st->dt_min, st->dt_max, st->dt_p1, st->dt_p99, st->n_drop_gaps, st->n_lost_est, st->n_dup, st->mean_min,
           st->mean_avg, st->mean_max, st->n_dark, st->n_first, st->n_pair_ok, st->n_pair_bad, st->n_skip,
           st->pair_mean, st->pair_std, st->pair_min, st->pair_max);
}

static void print_human(const stats_t *st)
{
    const double target = CONFIG_CAMT_TARGET_INTERVAL_MS;
    printf("\n================ KẾT QUẢ ĐO (mục 2.1) ================\n");
    printf(" Khung ghi được      : %d / %d   (lỗi lấy khung %d, sai định dạng %d)\n", st->n_ok, st->n_rec,
           st->n_fail, st->n_badfmt);
    printf(" FPS thực tế         : %.2f khung/s\n", st->fps);
    printf(" Khoảng cách khung   : TB %.2f ms, trung vị %.2f, độ lệch %.2f, min %.2f, max %.2f\n", st->dt_mean,
           st->dt_med, st->dt_std, st->dt_min, st->dt_max);
    printf(" Phân vị 1%% / 99%%    : %.2f / %.2f ms\n", st->dt_p1, st->dt_p99);
    printf(" Rớt khung           : %d chỗ hở, ước lượng mất %d khung; khung lặp (CRC trùng) %d\n", st->n_drop_gaps,
           st->n_lost_est, st->n_dup);
    printf(" Độ sáng TB          : min %d, TB %.1f, max %d; khung tối (<15) %d\n", st->mean_min, st->mean_avg,
           st->mean_max, st->n_dark);
    printf(" Cặp khung %d ms ±%d%%: hợp lệ %d, bị loại %d, chuỗi mới %d\n", CONFIG_CAMT_TARGET_INTERVAL_MS,
           CONFIG_CAMT_TOL_PCT, st->n_pair_ok, st->n_pair_bad, st->n_first);
    printf(" Khoảng cách cặp     : TB %.2f ms, độ lệch %.2f, min %.2f, max %.2f\n", st->pair_mean, st->pair_std,
           st->pair_min, st->pair_max);
    int k = st->dt_med > 0 ? (int)lround(target / st->dt_med) : 0;
    if (k < 1) {
        k = 1;
    }
    double natural = k * st->dt_med;
    printf(" Gợi ý               : lấy 1 trong %d khung → khoảng cách tự nhiên %.1f ms (lệch %+.1f%% so với %.0f ms)\n",
           k, natural, (natural - target) / target * 100, target);
    bool cam_ok = st->n_fail == 0 && st->n_badfmt == 0 && st->n_ok == st->n_rec;
    int n_pairs = st->n_pair_ok + st->n_pair_bad;
    double bad_pct = n_pairs ? 100.0 * st->n_pair_bad / n_pairs : 100.0;
    bool match = st->pair_mean > 0 && fabs(st->pair_mean - target) <= 0.05 * target;
    printf(" Camera              : %s\n", cam_ok ? "ĐẠT (160x120 xám, không lỗi)" : "KHÔNG ĐẠT – xem các dòng lỗi");
    int n_frames_seen = st->n_ok + st->n_lost_est;
    double drop_pct = n_frames_seen ? 100.0 * st->n_lost_est / n_frames_seen : 0;
    double nat_dev = fabs(natural - target) / target * 100;
    const char *why = bad_pct < 5 ? "(ĐẠT, < 5%)"
                      : (drop_pct < 2 && nat_dev > CONFIG_CAMT_TOL_PCT)
                          ? "(KHÔNG ĐẠT – chu kỳ camera không chia hết cho khoảng cách danh định, không phải rớt khung)"
                          : "(KHÔNG ĐẠT – rớt khung nhiều)";
    printf(" Cặp bị loại         : %.1f%% %s\n", bad_pct, why);
    printf(" So với mô hình 80ms : %s\n",
           match ? "KHỚP – dùng được mô hình IITB"
                 : "KHÔNG KHỚP – train lại ở khoảng cách đo được (hoặc chỉnh XCLK/FPS)");
    printf("======================================================\n\n");
}

/* --------------------------------------------------------------- ảnh mẫu */

static void print_base64_frame(const uint8_t *img)
{
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char line[80];
    int li = 0;
    printf("FRAME %d %d\n", IMG_W, IMG_H);
    for (int i = 0; i < IMG_PIX; i += 3) {
        uint32_t v = (uint32_t)img[i] << 16;
        int rem = IMG_PIX - i;
        if (rem > 1) {
            v |= (uint32_t)img[i + 1] << 8;
        }
        if (rem > 2) {
            v |= img[i + 2];
        }
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

/* ------------------------------------------------------------ quét cấu hình */

#ifdef CONFIG_CAMT_SWEEP
static void run_sweep(rec_t *recs)
{
    const int xclks[] = {10, 16, 20, 24};
    const int fbs[] = {1, 2};
    const int n = 150;
    printf("\n#INFO sweep_begin n=%d\n", n);
    for (size_t a = 0; a < sizeof(xclks) / sizeof(xclks[0]); a++) {
        for (size_t b = 0; b < sizeof(fbs) / sizeof(fbs[0]); b++) {
            esp_err_t err = cam_start(xclks[a], fbs[b], true);
            if (err != ESP_OK || !check_sensor(false)) {
                printf("SWEEP xclk_mhz=%d fb_count=%d status=FAIL err=0x%x\n", xclks[a], fbs[b], err);
                continue;
            }
            warmup(20);
            frame_pacer_t p;
            pacer_init(&p, CONFIG_CAMT_TARGET_INTERVAL_MS, CONFIG_CAMT_TOL_PCT);
            measure(recs, n, NULL, 0, 0, &p);
            stats_t st;
            compute_stats(recs, n, &st);
            printf("SWEEP xclk_mhz=%d fb_count=%d status=OK fps=%.3f dt_med=%.3f dt_std=%.3f drop_gaps=%d "
                   "badfmt=%d fail=%d bright_avg=%.1f\n",
                   xclks[a], fbs[b], st.fps, st.dt_med, st.dt_std, st.n_drop_gaps, st.n_badfmt, st.n_fail,
                   st.mean_avg);
        }
    }
    printf("#INFO sweep_end\n");
}
#endif

/* --------------------------------------------------------------------- main */

static void *alloc_any(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : malloc(sz);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));   /* chờ console ổn định */
    print_system_info();

    const int n = CONFIG_CAMT_N_FRAMES;
    rec_t *recs = alloc_any(sizeof(rec_t) * n);
    uint8_t *snaps[10] = {0};
    int n_snap = CONFIG_CAMT_SEND_FRAMES;
    for (int i = 0; i < n_snap; i++) {
        snaps[i] = alloc_any(IMG_PIX);
        if (!snaps[i]) {
            n_snap = i;
            break;
        }
    }
    if (!recs) {
        printf("#INFO error=out_of_memory need_bytes=%u\n", (unsigned)(sizeof(rec_t) * n));
        return;
    }

#ifdef CONFIG_CAMT_SWEEP
    run_sweep(recs);
#endif

    const int xclk = CONFIG_CAMT_XCLK_MHZ, fbc = CONFIG_CAMT_FB_COUNT;
    const bool latest = GRAB_LATEST_DEFAULT;
    esp_err_t err = cam_start(xclk, fbc, latest);
    printf("#INFO cam_init=%s err=0x%x\n", err == ESP_OK ? "OK" : "FAIL", err);
    if (err != ESP_OK) {
        printf("#INFO hint=kiem_tra_so_do_chan(board_pins.h),cap_camera,nguon_3V3,thu_XCLK_10MHz\n");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(5000));
            printf("#INFO cam_init=FAIL err=0x%x\n", err);
        }
    }
    bool is_ov5640 = check_sensor(true);
    if (!is_ov5640) {
        printf("#INFO warning=cam_bien_khong_phai_OV5640\n");
    }

    /* Kiểm tra khung đầu tiên */
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
        printf("#INFO first_frame w=%u h=%u len=%u fmt=%d expect=160x120x19200_gray(%d)\n", (unsigned)fb->width,
               (unsigned)fb->height, (unsigned)fb->len, (int)fb->format, (int)PIXFORMAT_GRAYSCALE);
        esp_camera_fb_return(fb);
    } else {
        printf("#INFO first_frame=NONE\n");
    }

    if (CONFIG_CAMT_AE_LEVEL != 0) {
        sensor_t *s = esp_camera_sensor_get();
        s->set_ae_level(s, CONFIG_CAMT_AE_LEVEL);
        printf("#INFO ae_level=%d\n", CONFIG_CAMT_AE_LEVEL);
    }
#if CONFIG_CAMT_TARGET_FPS_X10 > 0
    warmup(CONFIG_CAMT_WARMUP_FRAMES);
    set_target_fps();
#endif

    int run = 0;
    do {
        char label[24];
        snprintf(label, sizeof(label), "run%d", run);
        warmup(run == 0 ? CONFIG_CAMT_WARMUP_FRAMES : 5);
#ifdef CONFIG_CAMT_LOCK_EXPOSURE
        if (run == 0) {
            lock_exposure();
            warmup(10);
        }
#endif
        frame_pacer_t pacer;
        pacer_init(&pacer, CONFIG_CAMT_TARGET_INTERVAL_MS, CONFIG_CAMT_TOL_PCT);
        printf("#INFO measuring n=%d (khong in trong luc do)...\n", n);
        int64_t t0 = esp_timer_get_time();
        measure(recs, n, run == 0 ? snaps : NULL, run == 0 ? n_snap : 0, CONFIG_CAMT_SIM_LOAD_MS, &pacer);
        int64_t t1 = esp_timer_get_time();
        printf("#INFO measure_wall_ms=%.1f\n", (t1 - t0) / 1000.0);

        stats_t st;
        compute_stats(recs, n, &st);
        print_records(label, recs, n, xclk, fbc, latest, CONFIG_CAMT_SIM_LOAD_MS);
        print_summary(label, &st);
        print_human(&st);
        if (run == 0) {
            for (int i = 0; i < n_snap; i++) {
                print_base64_frame(snaps[i]);
            }
        }
        run++;
        if (run > 0) {
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
#ifdef CONFIG_CAMT_REPEAT
    } while (true);
#else
    } while (false);
#endif
    printf("DONE\n");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
