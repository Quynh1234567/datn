/* ============================================================================
 * ocsvm_features.h - Trich xuat dac trung tren ESP32-S3 (C thuan, khong phu thuoc OS)
 *
 * Tai hien CHINH XAC buoc 1-4 cua pipeline Python (src/ocsvm_fpga/features):
 *   anh xam uint8 160x120 (OV5640 QQVGA)
 *   -> Gaussian 5x5 (bien REFLECT_101 nhu OpenCV)
 *   -> Horn-Schunck (bien REPLICATE, so vong lap co dinh)
 *   -> dao ham sai phan trung tam (nhu numpy.gradient)
 *   -> vector diem anh F1..F4 -> ma tran hiep bien C (d x d)
 *   -> logm(C) bang Jacobi -> vech (ngoai cheo x sqrt 2)
 *   -> chuan hoa (mean, inv_std) -> luong tu Q(DATA_FRAC) int16 -> gui FPGA
 *
 * Moi tham so lay tu ocsvm_params.h (sinh boi scripts/04_export_fpga.py).
 * File ASCII-only.
 * ==========================================================================*/
#ifndef OCSVM_FEATURES_H
#define OCSVM_FEATURES_H

#include <stddef.h>
#include <stdint.h>
#include "ocsvm_params.h"

#if defined(OCSVM_DESCRIPTOR) && OCSVM_DESCRIPTOR != 0
#error "Firmware chi ho tro dac trung hiep bien (descriptor = covariance). Xuat lai ocsvm_params.h voi cau hinh mac dinh."
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OFE_NPIX (FRAME_W * FRAME_H)

/* Bo dem lam viec: 9 anh float (~9 x 75 KB voi 160x120) -> tren ESP32-S3 nen
 * cap phat trong PSRAM: heap_caps_malloc(n, MALLOC_CAP_SPIRAM). */
typedef struct {
    float *prev, *cur;          /* khung truoc / khung hien tai da lam muot     */
    float *ix, *iy, *it;        /* dao ham anh                                   */
    float *u, *v, *ua, *va;     /* dong quang hoc + trung binh lan can           */
    int    has_prev;
} ofe_ctx_t;

typedef void *(*ofe_alloc_fn)(size_t bytes);

/* Ket qua cho mot cap khung */
typedef struct {
    float   vec[OCSVM_DIM];     /* vech(logm C) - chua chuan hoa                  */
    int16_t xq[OCSVM_DIM];      /* vector da chuan hoa + luong tu -> gui FPGA     */
    float   cov[FEAT_D * FEAT_D];
    float   mean_motion;        /* trung binh |(u,v)|                            */
    int     n_saturated;        /* so phan tu bi bao hoa khi luong tu            */
} ofe_result_t;

int  ofe_init(ofe_ctx_t *ctx, ofe_alloc_fn alloc);   /* 0 = OK                   */
void ofe_reset(ofe_ctx_t *ctx);                      /* quen khung truoc         */

/* Dua mot khung xam uint8 FRAME_W x FRAME_H vao.
 * Tra ve 1 neu da co ket qua (tu khung thu 2 tro di), 0 neu chua.
 * Nguoi goi tu bo khung theo FRAME_STEP. */
int  ofe_process_frame(ofe_ctx_t *ctx, const uint8_t *gray, ofe_result_t *out);

/* Cac buoc rieng le (phuc vu kiem thu) */
void ofe_gaussian_blur(const uint8_t *src, float *dst, float *tmp);
void ofe_horn_schunck(ofe_ctx_t *ctx);
void ofe_covariance(const float *u, const float *v, double *cov /* d*d */);
void ofe_spd_logm(const double *c, double *logc, int d);
void ofe_vech_scaled(const double *m, int d, float *vec);
int  ofe_quantize(const float *vec, int16_t *xq);

/* --------------------------- giao tiep UART voi FPGA ------------------------
 * ESP32 -> FPGA : 0xAA 0x55 | DIM x int16 LE | XOR(payload)
 * FPGA -> ESP32 : 0x5A | f_q (ACC_W/8 byte, LE, bu hai) | flag | XOR(f_q..flag) */
#define OFE_REQ_LEN  (2 + 2 * OCSVM_DIM + 1)
#define OFE_ACC_BYTES (OCSVM_ACC_W / 8)
#define OFE_RESP_LEN  (1 + OFE_ACC_BYTES + 1 + 1)

size_t ofe_uart_build_request(const int16_t *xq, uint8_t *buf);
int    ofe_uart_parse_response(const uint8_t *buf, size_t len, int64_t *f_q, uint8_t *flag);

/* --------------------------- hau xu ly ------------------------------------ */
double ofe_fq_to_float(int64_t f_q);           /* f_q -> f(x) so thuc            */

typedef struct {
    double win[OCSVM_SMOOTH_WIN > 0 ? OCSVM_SMOOTH_WIN : 1];
    int    n, pos;
    double sum;
} ofe_smoother_t;

void   ofe_smoother_init(ofe_smoother_t *s);
double ofe_smoother_push(ofe_smoother_t *s, double f);   /* tra ve f lam muot     */
int    ofe_is_alarm(double f_smoothed);                  /* 1 = bat thuong        */

#ifdef __cplusplus
}
#endif
#endif /* OCSVM_FEATURES_H */
