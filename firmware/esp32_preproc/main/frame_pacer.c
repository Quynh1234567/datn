#include "frame_pacer.h"

#include <string.h>

void pacer_init(frame_pacer_t *p, int target_ms, int tol_pct)
{
    memset(p, 0, sizeof(*p));
    p->target_us = (int64_t)target_ms * 1000;
    p->tol_us = p->target_us * tol_pct / 100;
}

void pacer_reset(frame_pacer_t *p)
{
    p->has_sel = false;          /* giữ ước lượng chu kỳ camera */
}

pacer_decision_t pacer_feed(frame_pacer_t *p, int64_t ts_us)
{
    /* Ước lượng chu kỳ camera, bỏ qua khoảng bất thường (rớt nhiều khung, timestamp lùi) */
    if (p->has_raw) {
        int64_t d = ts_us - p->last_raw_us;
        if (d > 0 && d < 4 * p->target_us) {
            p->period_us = p->period_us ? (3 * p->period_us + d) / 4 : d;
        }
    }
    p->last_raw_us = ts_us;
    p->has_raw = true;

    if (!p->has_sel) {
        p->has_sel = true;
        p->last_sel_us = ts_us;
        return PACER_FIRST;
    }

    int64_t dt = ts_us - p->last_sel_us;
    /* Giữ khung đầu tiên có dt ≥ target − T/2: khung có dt gần target nhất khi không rớt khung */
    if (dt < p->target_us - p->period_us / 2) {
        return PACER_SKIP;
    }
    p->last_sel_us = ts_us;
    p->last_pair_dt_us = dt;
    int64_t err = dt - p->target_us;
    if (err < 0) {
        err = -err;
    }
    return err <= p->tol_us ? PACER_PAIR_OK : PACER_PAIR_BAD;
}

const char *pacer_name(pacer_decision_t d)
{
    switch (d) {
    case PACER_SKIP: return "skip";
    case PACER_FIRST: return "first";
    case PACER_PAIR_OK: return "pair_ok";
    case PACER_PAIR_BAD: return "pair_bad";
    default: return "none";
    }
}
