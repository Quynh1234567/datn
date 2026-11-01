/* =============================================================================
 *  DEMO 1 – ESP32-S3: OV5640 160×120 xám 12,5 fps → Gauss 5×5 số nguyên → SPI → Tang Nano 20K
 *
 *  Mỗi khung được bộ chọn cặp (frame_pacer, 80 ms) giữ lại:
 *    - Gauss /1024 (khớp từng bit Golden Model, đã kiểm trên board ở mục 2.2–2.3)
 *    - khung tối → không gửi, ngắt cặp
 *    - gửi 19 200 byte qua SPI (5 giao dịch A5|cờ|chunk|seq|dữ liệu); cờ RESET_PAIR khi khung này
 *      bắt đầu chuỗi mới (khung đầu, sau khung tối, sau khi rớt khung)
 *  FPGA tính Horn–Schunck + OC-SVM và tự báo kết quả qua UART của nó (BL616 → PC).
 *
 *  Log (console, 921 600 baud – đọc bởi scripts/27_demo_monitor.py --esp):
 *    #INFO …
 *    F seq=<n> ts=<µs> dt=<ms×10> g=<µs Gauss> s=<µs SPI> fl=<cờ> d=<quyết định pacer>
 *    D ts=<µs>                       (khung tối, không gửi)
 *    PV <seq> 80 60 / base64 / END   (ảnh xem trước 80×60 mỗi DEMO_PREVIEW_EVERY khung)
 * ============================================================================= */
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#if defined(CONFIG_IDF_TARGET) && !defined(CONFIG_IDF_TARGET_ESP32S3)
#error "Project dang build cho chip KHAC ESP32-S3. Chay: idf.py set-target esp32s3  (roi build lai)"
#endif

#include "board_pins.h"
#include "frame_pacer.h"
#include "ov5640_fps.h"
#include "preproc.h"
#include "spi_link.h"

#ifndef CONFIG_DEMO_PREVIEW_EVERY
#define CONFIG_DEMO_PREVIEW_EVERY 6
#endif
#ifdef CONFIG_PP_AEC_MAX_FOLLOW_VTS
#define AEC_FOLLOW true
#else
#define AEC_FOLLOW false
#endif

#define PV_W 80
#define PV_H 60

static pp_work_t s_work;
static QueueHandle_t s_pv_q;               /* hàng đợi 1 phần tử: ảnh xem trước mới nhất */
typedef struct {
    uint32_t seq;
    uint8_t px[PV_W * PV_H];
} preview_t;

/* In ảnh xem trước (chạy ở lõi 1, ưu tiên thấp – không chặn vòng camera) */
static void preview_task(void *arg)
{
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static preview_t pv;
    static char line[96];
    for (;;) {
        if (xQueueReceive(s_pv_q, &pv, portMAX_DELAY) != pdTRUE) continue;
        printf("PV %" PRIu32 " %d %d\n", pv.seq, PV_W, PV_H);
        int li = 0;
        const int n = PV_W * PV_H;
        for (int i = 0; i < n; i += 3) {
            uint32_t v = (uint32_t)pv.px[i] << 16 | (uint32_t)pv.px[i + 1] << 8 | pv.px[i + 2];   /* 4800 chia hết cho 3 */
            line[li++] = tbl[(v >> 18) & 63];
            line[li++] = tbl[(v >> 12) & 63];
            line[li++] = tbl[(v >> 6) & 63];
            line[li++] = tbl[v & 63];
            if (li >= 76) { line[li] = 0; puts(line); li = 0; }
        }
        if (li) { line[li] = 0; puts(line); }
        puts("END");
    }
}

static void make_preview(const uint8_t *img, uint32_t seq)
{
    static preview_t pv;
    pv.seq = seq;
    for (int y = 0; y < PV_H; y++)
        for (int x = 0; x < PV_W; x++) {
            const uint8_t *p = img + (2 * y) * PP_W + 2 * x;
            pv.px[y * PV_W + x] = (uint8_t)((p[0] + p[1] + p[PP_W] + p[PP_W + 1] + 2) >> 2);
        }
    xQueueOverwrite(s_pv_q, &pv);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(300));
    printf("\n#INFO demo=1 board=%s spi_sclk=%d spi_mosi=%d spi_cs=%d spi_hz=%d preview_every=%d\n",
           CAM_BOARD_NAME, CONFIG_DEMO_SPI_SCLK, CONFIG_DEMO_SPI_MOSI, CONFIG_DEMO_SPI_CS, CONFIG_DEMO_SPI_HZ,
           CONFIG_DEMO_PREVIEW_EVERY);

    esp_err_t err = link_init(CONFIG_DEMO_SPI_SCLK, CONFIG_DEMO_SPI_MOSI, CONFIG_DEMO_SPI_CS, CONFIG_DEMO_SPI_HZ);
    printf("#INFO spi_init=%s\n", esp_err_to_name(err));
    if (err != ESP_OK) return;

    err = ov_cam_start(CONFIG_PP_XCLK_MHZ, 2, true, false);
    printf("#INFO cam_init=%s\n", esp_err_to_name(err));
    if (err != ESP_OK) return;
    if (CONFIG_PP_TARGET_FPS_X10 > 0) {
        fps_result_t fr = ov5640_set_fps(CONFIG_PP_TARGET_FPS_X10, AEC_FOLLOW);
        printf("#INFO fps_ok=%d period_us=%.0f\n", fr.ok, fr.period_after_us);
    }
    ov_cam_warmup(CONFIG_PP_WARMUP_FRAMES);

    uint8_t *smooth = heap_caps_malloc(PP_PIX, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!smooth) { printf("#INFO loi=khong_du_ram\n"); return; }
    if (CONFIG_DEMO_PREVIEW_EVERY > 0) {
        s_pv_q = xQueueCreate(1, sizeof(preview_t));
        xTaskCreatePinnedToCore(preview_task, "preview", 4096, NULL, 1, NULL, 1);
    }

    frame_pacer_t pacer;
    pacer_init(&pacer, CONFIG_PP_TARGET_INTERVAL_MS, CONFIG_PP_TOL_PCT);
    uint32_t seq = 0, n_sent = 0, n_dark = 0, n_err = 0;
    bool need_reset = true;
    int64_t t_stat = esp_timer_get_time();
    printf("#INFO bat_dau=1\n");

    for (;;) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { pacer_reset(&pacer); need_reset = true; continue; }
        if (fb->len < PP_PIX) { esp_camera_fb_return(fb); continue; }
        int64_t ts = ov_cam_fb_ts_us(fb);
        pacer_decision_t d = pacer_feed(&pacer, ts);
        if (d == PACER_SKIP) { esp_camera_fb_return(fb); continue; }

        int64_t t0 = esp_timer_get_time();
        uint32_t sum = pp_gauss5(fb->buf, smooth, PP_BORDER_REFLECT101, &s_work);
        uint32_t t_g = (uint32_t)(esp_timer_get_time() - t0);
        esp_camera_fb_return(fb);

        if (pp_is_dark(sum, CONFIG_PP_DARK_MEAN)) {        /* khung tối: không gửi, ngắt cặp */
            pacer_reset(&pacer);
            need_reset = true;
            n_dark++;
            printf("D ts=%" PRId64 "\n", ts);
            continue;
        }
        uint8_t flags = 0;
        if (need_reset || d != PACER_PAIR_OK) flags |= LINK_FLAG_RESET_PAIR;
        seq++;
        uint32_t t_s = 0;
        err = link_send_frame(smooth, (uint8_t)seq, flags, &t_s);
        if (err != ESP_OK) { n_err++; need_reset = true; continue; }
        need_reset = false;
        n_sent++;
        printf("F seq=%" PRIu32 " ts=%" PRId64 " dt=%" PRId64 " g=%" PRIu32 " s=%" PRIu32 " fl=%u d=%s\n",
               seq, ts, pacer.last_pair_dt_us / 100, t_g, t_s, flags, pacer_name(d));
        if (CONFIG_DEMO_PREVIEW_EVERY > 0 && (seq % CONFIG_DEMO_PREVIEW_EVERY) == 0) make_preview(smooth, seq);

        int64_t now = esp_timer_get_time();
        if (now - t_stat > 10000000) {                       /* thống kê mỗi 10 s */
            printf("#INFO gui=%" PRIu32 " toi=%" PRIu32 " loi_spi=%" PRIu32 "\n", n_sent, n_dark, n_err);
            t_stat = now;
        }
    }
}
