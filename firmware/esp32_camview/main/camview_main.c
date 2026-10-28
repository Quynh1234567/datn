/* ============================================================================
 * camview_main.c - Xem camera OV5640 MAU, do phan giai cao, truc tiep qua Wi-Fi
 *
 *  ESP32-S3 phat Wi-Fi rieng (khong can router):
 *      SSID     : OCSVM-CAM
 *      Mat khau : 12345678
 *  Ket noi dien thoai / laptop vao mang tren, mo trinh duyet:  http://192.168.4.1
 *
 *  Anh JPEG mau 640x480 (VGA), truyen dang MJPEG -> muot hon nhieu so voi gui qua
 *  cong COM. Chuong trinh nay CHI DE XEM/QUAY DEMO; mo hinh van dung anh xam
 *  160x120 trong esp32_app.
 *  Chan camera theo bo tri ESP32-S3-WROOM CAM (Freenove / ESP32-S3-EYE).
 * ==========================================================================*/
#include <stdio.h>
#include <string.h>
#include "esp_camera.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define AP_SSID       "OCSVM-CAM"
#define AP_PASS       "12345678"
#define FRAME_SIZE    FRAMESIZE_VGA      /* FRAMESIZE_QVGA (320x240) neu muon fps cao hon */
#define JPEG_QUALITY  12                 /* 10..20: so nho = net hon, file lon hon */
#define BOUNDARY      "ocsvmframe"

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
        .pixel_format = PIXFORMAT_JPEG,          /* anh MAU nen JPEG ngay trong cam bien */
        .frame_size = FRAME_SIZE,
        .jpeg_quality = JPEG_QUALITY, .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };
    return esp_camera_init(&c);
}

static void wifi_start_ap(void)
{
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        e = nvs_flash_init();
    }
    ESP_ERROR_CHECK(e);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t ap = {0};
    strncpy((char *)ap.ap.ssid, AP_SSID, sizeof(ap.ap.ssid));
    strncpy((char *)ap.ap.password, AP_PASS, sizeof(ap.ap.password));
    ap.ap.ssid_len = strlen(AP_SSID);
    ap.ap.channel = 6;
    ap.ap.max_connection = 2;
    ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());
    printf("Wi-Fi: ket noi vao '%s' (mat khau %s), mo http://192.168.4.1\n", AP_SSID, AP_PASS);
}

static esp_err_t index_handler(httpd_req_t *req)
{
    static const char html[] =
        "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>OV5640</title><style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;"
        "text-align:center}img{max-width:100%;height:auto}</style></head><body>"
        "<p>ESP32-S3 + OV5640 &ndash; anh mau truc tiep</p><img src='/stream'></body></html>";
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    char hdr[96];
    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" BOUNDARY);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    int64_t t0 = 0; int frames = 0;
    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        int n = snprintf(hdr, sizeof(hdr),
                         "\r\n--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                         (unsigned)fb->len);
        esp_err_t r = httpd_resp_send_chunk(req, hdr, n);
        if (r == ESP_OK) r = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        unsigned len = (unsigned)fb->len;
        esp_camera_fb_return(fb);
        if (r != ESP_OK) break;                        /* trinh duyet da dong */
        frames++;
        int64_t now = esp_timer_get_time();
        if (now - t0 >= 1000000) {
            printf("stream: %d fps, JPEG %u byte/khung\n", frames, len);
            frames = 0; t0 = now;
        }
    }
    return ESP_OK;
}

void app_main(void)
{
    if (camera_init() != ESP_OK) {
        printf("LOI khoi tao camera - kiem tra cap FPC va so do chan\n");
        return;
    }
    sensor_t *s = esp_camera_sensor_get();
    printf("Camera PID = 0x%04X%s\n", s->id.PID, s->id.PID == 0x5640 ? " (OV5640)" : "");
    if (s->id.PID == 0x5640) {
        s->set_vflip(s, 0);                            /* doi 0/1 neu anh bi lat */
        s->set_hmirror(s, 0);
    }
    wifi_start_ap();
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t srv = NULL;
    ESP_ERROR_CHECK(httpd_start(&srv, &cfg));
    httpd_uri_t idx = {.uri = "/", .method = HTTP_GET, .handler = index_handler};
    httpd_uri_t stm = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler};
    httpd_register_uri_handler(srv, &idx);
    httpd_register_uri_handler(srv, &stm);
}
