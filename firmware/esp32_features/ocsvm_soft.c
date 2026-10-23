/* ocsvm_soft.c - datapath giong het RTL: dist -> *GAMMA_Q -> >>LUT_SHIFT -> LUT -> MAC */
#include "ocsvm_soft.h"
#include "ocsvm_params.h"
#include "ocsvm_model_data.h"

int64_t ocsvm_soft_decision_q(const int16_t *xq)
{
    int64_t acc = 0;
    for (int i = 0; i < OCSVM_N_SV; ++i) {
        const int16_t *sv = &OCSVM_SV_Q[i * OCSVM_DIM];
        int64_t dist = 0;
        for (int j = 0; j < OCSVM_DIM; ++j) {
            const int32_t diff = (int32_t)xq[j] - (int32_t)sv[j];
            dist += (int64_t)diff * diff;
        }
        const int64_t idx = (dist * OCSVM_GAMMA_Q) >> OCSVM_LUT_SHIFT;
        const int64_t k = idx < OCSVM_LUT_SIZE ? (int64_t)OCSVM_EXP_LUT[idx] : 0;
        acc += (int64_t)OCSVM_ALPHA_Q[i] * k;
    }
    return acc - OCSVM_RHO_Q;
}

int ocsvm_soft_flag(int64_t f_q)
{
#if OCSVM_DECISION_MODE == 1
    return (f_q < 0 ? -f_q : f_q) > OCSVM_THR_Q;
#else
    return f_q < OCSVM_THR_Q;
#endif
}
