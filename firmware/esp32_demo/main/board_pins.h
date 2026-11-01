/* Sơ đồ chân camera DVP cho các board ESP32-S3 phổ biến.
 * KIỂM TRA LẠI theo sơ đồ nguyên lý board của bạn trước khi nạp.
 * Chọn board bằng: idf.py menuconfig → Camera timing test → Board camera. */
#pragma once
#include "sdkconfig.h"

#if defined(CONFIG_PP_BOARD_S3_EYE)
/* ESP32-S3-EYE, Freenove ESP32-S3-WROOM CAM và đa số board "ESP32-S3-CAM N16R8" */
#define CAM_BOARD_NAME "ESP32-S3-EYE/Freenove"
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7     16   /* Y9 */
#define CAM_PIN_D6     17   /* Y8 */
#define CAM_PIN_D5     18   /* Y7 */
#define CAM_PIN_D4     12   /* Y6 */
#define CAM_PIN_D3     10   /* Y5 */
#define CAM_PIN_D2      8   /* Y4 */
#define CAM_PIN_D1      9   /* Y3 */
#define CAM_PIN_D0     11   /* Y2 */
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13

#elif defined(CONFIG_PP_BOARD_XIAO_S3)
#define CAM_BOARD_NAME "XIAO-ESP32S3-Sense"
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   10
#define CAM_PIN_SIOD   40
#define CAM_PIN_SIOC   39
#define CAM_PIN_D7     48
#define CAM_PIN_D6     11
#define CAM_PIN_D5     12
#define CAM_PIN_D4     14
#define CAM_PIN_D3     16
#define CAM_PIN_D2     18
#define CAM_PIN_D1     17
#define CAM_PIN_D0     15
#define CAM_PIN_VSYNC  38
#define CAM_PIN_HREF   47
#define CAM_PIN_PCLK   13

#else
/* Board tự khai báo: sửa các giá trị dưới đây */
#define CAM_BOARD_NAME "custom"
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7     16
#define CAM_PIN_D6     17
#define CAM_PIN_D5     18
#define CAM_PIN_D4     12
#define CAM_PIN_D3     10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0     11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13
#endif
