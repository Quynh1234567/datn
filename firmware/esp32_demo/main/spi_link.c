#include "spi_link.h"

#include <string.h>

#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#define CHUNK 4096
#define FRAME_BYTES 19200

static spi_device_handle_t s_dev;
static uint8_t *s_buf;   /* bộ đệm DMA trong RAM nội: 4 byte header + 4096 byte */

esp_err_t link_init(int sclk, int mosi, int cs, int hz)
{
    spi_bus_config_t bus = {
        .mosi_io_num = mosi, .miso_io_num = -1, .sclk_io_num = sclk,
        .quadwp_io_num = -1, .quadhd_io_num = -1, .max_transfer_sz = CHUNK + 8,
    };
    esp_err_t err = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO);
    if (err != ESP_OK) return err;
    spi_device_interface_config_t dev = {
        .mode = 0, .clock_speed_hz = hz, .spics_io_num = cs, .queue_size = 2,
    };
    err = spi_bus_add_device(SPI2_HOST, &dev, &s_dev);
    if (err != ESP_OK) return err;
    s_buf = heap_caps_malloc(CHUNK + 8, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    return s_buf ? ESP_OK : ESP_ERR_NO_MEM;
}

esp_err_t link_send_frame(const uint8_t *img, uint8_t seq, uint8_t flags, uint32_t *us)
{
    int64_t t0 = esp_timer_get_time();
    for (int c = 0; c * CHUNK < FRAME_BYTES; c++) {
        int n = FRAME_BYTES - c * CHUNK;
        if (n > CHUNK) n = CHUNK;
        s_buf[0] = 0xA5;
        s_buf[1] = flags;
        s_buf[2] = (uint8_t)c;
        s_buf[3] = seq;
        memcpy(s_buf + 4, img + c * CHUNK, n);
        spi_transaction_t t = { .length = (size_t)(n + 4) * 8, .tx_buffer = s_buf };
        esp_err_t err = spi_device_polling_transmit(s_dev, &t);
        if (err != ESP_OK) return err;
    }
    if (us) *us = (uint32_t)(esp_timer_get_time() - t0);
    return ESP_OK;
}
