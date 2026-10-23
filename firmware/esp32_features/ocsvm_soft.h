/* ocsvm_soft.h - OC-SVM so nguyen chay tren CPU (bit-exact voi FPGA va Golden Model).
 * Dung lam baseline phan mem de do he so tang toc cua FPGA, va de kiem tra cheo
 * ket qua FPGA tra ve qua UART. ASCII-only. */
#ifndef OCSVM_SOFT_H
#define OCSVM_SOFT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int64_t ocsvm_soft_decision_q(const int16_t *xq);   /* f_q = sum alpha*k - rho  */
int     ocsvm_soft_flag(int64_t f_q);               /* 1 = bat thuong            */
#ifdef __cplusplus
}
#endif
#endif
