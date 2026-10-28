/* ============================================================================
 * camtest_main.c - Kiem tra board ESP32-S3 + camera OV5640
 *
 *  1. In thong tin chip, flash, PSRAM (xac nhan dung loai board)
 *  2. Khoi tao camera dung cau hinh he thong: anh XAM 160x120 (QQVGA)
 *  3. Doc ma cam bien (OV5640 = PID 0x5640)
 *  4. Chup lien tuc: in fps va do sang trung binh moi giay
 *  5. Moi 5 s gui 1 khung anh dang base64 qua cong console:
 *        FRAME <w> <h>
 *        <base64 ...>
 *        END
 *     -> scripts/17_cam_grab.py tren PC giai ma, luu PNG va hien anh
 *
 *  Chan camera theo bo tri pho bien ESP32-S3-WROOM CAM (Freenove / ESP32-S3-EYE):
 *  KIEM TRA LAI theo tai lieu board cua ban.
 * ==========================================================================*/
#include <stdio.h>
#include <string.h>
#include "esp_camera.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "mbedtls/base64.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define DUMP_EVERY_US   (5 * 1000 * 1000)

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
        .pixel_format = PIXFORMAT_GRAYSCALE,
        .frame_size = FRAMESIZE_QQVGA,
        .jpeg_quality = 12, .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };
    return esp_camera_init(&c);
}

static void print_board_info(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    printf("\n===== THONG TIN BOARD =====\n");
    printf("Chip   : %s, %d nhan, revision %d.%d\n",
           ci.model == CHIP_ESP32S3 ? "ESP32-S3" : "KHONG PHAI ESP32-S3",
           ci.cores, ci.revision / 100, ci.revision % 100);
    printf("Flash  : %lu MB\n", (unsigned long)(flash / (1024 * 1024)));
    printf("PSRAM  : %u KB\n", (unsigned)(esp_psram_get_size() / 1024));
    printf("Heap PSRAM con trong: %u KB\n", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

static void dump_frame(const camera_fb_t *fb, unsigned char *b64, size_t b64_cap)
{
    size_t olen = 0;
    if (mbedtls_base64_encode(b64, b64_cap, &olen, fb->buf, fb->len) != 0) {
        printf("LOI: bo dem base64 qua nho\n");
        return;
    }
    printf("FRAME %d %d\n", fb->width, fb->height);
    for (size_t i = 0; i < olen; i += 76) {
        size_t n = (olen - i) < 76 ? (olen - i) : 76;
        fwrite(b64 + i, 1, n, stdout);
        fputc('\n', stdout);
    }
    printf("END\n");
    fflush(stdout);
}

void app_main(void)
{
    print_board_info();
    esp_err_t err = camera_init();
    if (err != ESP_OK) {
        printf("LOI khoi tao camera: 0x%x (%s)\n", err, esp_err_to_name(err));
        printf("-> Kiem tra cap FPC cam dung chieu, so do chan camera, nguon 3.3 V.\n");
        return;
    }
    sensor_t *s = esp_camera_sensor_get();
    printf("Camera : PID = 0x%04X %s\n", s->id.PID,
           s->id.PID == 0x5640 ? "(OV5640 - DUNG)" : "(KHONG PHAI OV5640!)");
    printf("Cau hinh: anh xam 160x120 (giong luc huan luyen)\n===========================\n");

    const size_t b64_cap = 4 * ((160 * 120 + 2) / 3) + 4;
    unsigned char *b64 = heap_caps_malloc(b64_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int frames = 0;
    int64_t t_sec = esp_timer_get_time(), t_dump = t_sec;
    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { printf("LOI: khong lay duoc khung anh\n"); vTaskDelay(pdMS_TO_TICKS(100)); continue; }
        frames++;
        int64_t now = esp_timer_get_time();
        if (now - t_sec >= 1000000) {
            uint32_t sum = 0;
            for (size_t i = 0; i < fb->len; ++i) sum += fb->buf[i];
            printf("fps = %d | %dx%d | %u byte | do sang TB = %lu\n", frames, fb->width, fb->height,
                   (unsigned)fb->len, (unsigned long)(sum / (fb->len ? fb->len : 1)));
            frames = 0;
            t_sec = now;
        }
        if (b64 && now - t_dump >= DUMP_EVERY_US) {
            dump_frame(fb, b64, b64_cap);
            t_dump = now;
        }
        esp_camera_fb_return(fb);
    }
}
