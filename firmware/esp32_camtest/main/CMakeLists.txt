idf_component_register(SRCS "camtest_main.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp32-camera mbedtls esp_psram spi_flash esp_timer)
