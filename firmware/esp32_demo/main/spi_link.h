/* spi_link.h – gửi ảnh 160×120 sang Tang Nano 20K qua SPI (giao thức của rtl/src/sys/spi_frame_rx.v)
 *   mỗi giao dịch: A5 | cờ | chunk | seq | ≤ 4096 byte ảnh   (5 giao dịch / khung)
 *   cờ bit0 = RESET_PAIR: khung này bắt đầu chuỗi mới, FPGA không ghép với khung trước. */
#pragma once
#include <stdint.h>
#include "esp_err.h"

#define LINK_FLAG_RESET_PAIR 0x01

esp_err_t link_init(int sclk, int mosi, int cs, int hz);
/* Gửi 19 200 byte; trả thời gian truyền (µs) qua *us (có thể NULL). */
esp_err_t link_send_frame(const uint8_t *img, uint8_t seq, uint8_t flags, uint32_t *us);
