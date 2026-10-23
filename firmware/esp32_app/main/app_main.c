/* ============================================================================
 * app_main.c - Demo: OV5640 -> ESP32-S3 (dac trung) -> UART -> FPGA (OC-SVM)
 *
 * MAU (template) cho ESP-IDF v5.x + thanh phan espressif/esp32-camera.
 * - Chan camera ben duoi theo so do ESP32-S3-EYE / Freenove ESP32-S3-WROOM CAM.
 *   KIEM TRA LAI theo board cua ban.
 * - Chan UART noi FPGA: UART_TX_PIN / UART_RX_PIN.
 * - USE_FPGA = 0: chay OC-SVM ngay tren CPU (baseline do toc do / khi chua co FPGA).
 * ==========================================================================*/
#include <stdio.h>
#include <string.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ocsvm_features.h"
#include "ocsvm_soft.h"

#define USE_FPGA        1
#define FPGA_UART       UART_NUM_1
#define UART_TX_PIN     GPIO_NUM_43
#define UART_RX_PIN     GPIO_NUM_44
#define UART_BAUD       921600
#define ALARM_LED_PIN   GPIO_NUM_2

static const char *TAG = "ocsvm";

static void *psram_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }

static esp_err_t camera_init(void)
{
    camera_config_t c = {
        .pin_pwdn = -1, .pin_reset = -1, .pin_xclk = 15,
        .pin_sccb_sda = 4, .pin_sccb_scl = 5,
        .pin_d7 = 16, .pin_d6 = 17, .pin_d5 = 18, .pin_d4 = 12,
        .pin_d3 = 10, .pin_d2 = 8, .pin_d1 = 9, .pin_d0 = 11,
        .pin_vsync = 6, .pin_href = 7, .pin_pclk = 13,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_GRAYSCALE,     /* khop huan luyen: anh xam   */
        .frame_size = FRAMESIZE_QQVGA,           /* 160x120 = FRAME_W x FRAME_H */
        .jpeg_quality = 12, .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };
    return esp_camera_init(&c);
}

static void uart_init(void)
{
    const uart_config_t cfg = {
        .baud_rate = UART_BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(FPGA_UART, 1024, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(FPGA_UART, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(FPGA_UART, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

/* Gui x_q, cho f_q tu FPGA. Tra ve 0 neu thanh cong. */
static int fpga_classify(const int16_t *xq, int64_t *f_q, uint8_t *flag)
{
    uint8_t req[OFE_REQ_LEN], resp[OFE_RESP_LEN];
    size_t n = ofe_uart_build_request(xq, req);
    uart_flush_input(FPGA_UART);
    uart_write_bytes(FPGA_UART, (const char *)req, n);
    int got = uart_read_bytes(FPGA_UART, resp, OFE_RESP_LEN, pdMS_TO_TICKS(50));
    return got == OFE_RESP_LEN ? ofe_uart_parse_response(resp, got, f_q, flag) : -3;
}

void app_main(void)
{
    ESP_ERROR_CHECK(camera_init());
    uart_init();
    gpio_set_direction(ALARM_LED_PIN, GPIO_MODE_OUTPUT);

    static ofe_ctx_t ctx;
    static ofe_result_t res;
    static ofe_smoother_t sm;
    if (ofe_init(&ctx, psram_alloc) != 0) { ESP_LOGE(TAG, "Khong du PSRAM"); return; }
    ofe_smoother_init(&sm);

    uint32_t frame_no = 0;
    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) continue;
        if (fb->width != FRAME_W || fb->height != FRAME_H) {
            ESP_LOGE(TAG, "Sai kich thuoc %dx%d", fb->width, fb->height);
            esp_camera_fb_return(fb);
            continue;
        }
        /* Luu y: khi chay thuc, toc do khung do thoi gian xu ly quyet dinh;
         * FRAME_STEP mo phong ~15 fps luc huan luyen - xem README firmware. */
        int64_t t0 = esp_timer_get_time();
        int ready = ofe_process_frame(&ctx, fb->buf, &res);
        esp_camera_fb_return(fb);
        if (!ready) continue;
        int64_t t1 = esp_timer_get_time();

        int64_t f_q = 0; uint8_t flag = 0;
#if USE_FPGA
        if (fpga_classify(res.xq, &f_q, &flag) != 0) {
            ESP_LOGW(TAG, "FPGA khong phan hoi - dung OC-SVM phan mem");
            f_q = ocsvm_soft_decision_q(res.xq);
            flag = (uint8_t)ocsvm_soft_flag(f_q);
        }
#else
        f_q = ocsvm_soft_decision_q(res.xq);
        flag = (uint8_t)ocsvm_soft_flag(f_q);
#endif
        int64_t t2 = esp_timer_get_time();
        double f_s = ofe_smoother_push(&sm, ofe_fq_to_float(f_q));
        int alarm = ofe_is_alarm(f_s);
        gpio_set_level(ALARM_LED_PIN, alarm);
        ESP_LOGI(TAG, "#%lu feat=%lldus svm=%lldus f=%.4f f_s=%.4f raw=%d ALARM=%d motion=%.3f",
                 (unsigned long)frame_no++, t1 - t0, t2 - t1, ofe_fq_to_float(f_q), f_s, flag, alarm,
                 res.mean_motion);
    }
}
