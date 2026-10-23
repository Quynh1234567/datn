# Project ESP-IDF (>= v5.1) cho ESP32-S3 + OV5640
cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS ${CMAKE_CURRENT_LIST_DIR}/../esp32_features)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ocsvm_demo)
