idf_component_register(SRCS "app_main.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp32_features esp32-camera driver esp_timer)
