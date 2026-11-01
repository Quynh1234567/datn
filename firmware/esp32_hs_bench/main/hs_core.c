/* hs_core.c – xem hs_core.h. Mọi công thức theo claude/hs_fixed_point_spec.md bản 0.4. */
#include "hs_core.h"
#include <string.h>

#define COEF_FRAC 18
#define COEF_MAX 16383          /* a, b: 15 bit có dấu */
#define ALPHA2_16 64            /* 16·α², α = 2 */
#define FU 6                    /* u, v Q6.6 */
#define UV_MIN (-4096)
#define UV_MAX 4095
#define AVG_MUL 10923
#define AVG_SHIFT 17
#define CELL_H 30
#define CELL_W 40
#define CELL_MUL 13981
#define CELL_SHIFT 20
#define LOG_A_LIMIT 2048
#define D16_MAX 32767

static inline int32_t sat_i32(int32_t x, int32_t lo, int32_t hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* ------------------------------------------------------------------ DERIV */
static inline int16_t coef_div(int32_t n, int32_t d16)
{
    uint32_t an = (uint32_t)(n < 0 ? -n : n);
    int32_t q = (int32_t)(((an << COEF_FRAC) + (uint32_t)(d16 >> 1)) / (uint32_t)d16);
    if (n < 0) q = -q;
    return (int16_t)sat_i32(q, -COEF_MAX - 1, COEF_MAX);
}

void hs_deriv(const uint8_t *i1, const uint8_t *i2,
              int16_t *ix, int16_t *iy, int16_t *it, int16_t *a, int16_t *b)
{
    for (int y = 0; y < HS_H; y++) {
        const int y1 = (y + 1 < HS_H) ? y + 1 : y;               /* biên dưới replicate */
        const uint8_t *p0 = i1 + y * HS_W, *p1 = i1 + y1 * HS_W;
        const uint8_t *q0 = i2 + y * HS_W, *q1 = i2 + y1 * HS_W;
        for (int x = 0; x < HS_W; x++) {
            const int x1 = (x + 1 < HS_W) ? x + 1 : x;           /* biên phải replicate */
            const int32_t a00 = p0[x], a01 = p0[x1], a10 = p1[x], a11 = p1[x1];
            const int32_t b00 = q0[x], b01 = q0[x1], b10 = q1[x], b11 = q1[x1];
            const int32_t gx = (a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10);
            const int32_t gy = (a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01);
            const int32_t gt = (b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11);
            const int32_t d16 = ALPHA2_16 + gx * gx + gy * gy;
            const int k = y * HS_W + x;
            ix[k] = (int16_t)gx;
            iy[k] = (int16_t)gy;
            it[k] = (int16_t)gt;
            a[k] = coef_div(gx, d16);
            b[k] = coef_div(gy, d16);
        }
    }
}

/* ----------------------------------------------------------------- HS_ITER */
int32_t hs_iter(const int16_t *ix, const int16_t *iy, const int16_t *it,
                const int16_t *a, const int16_t *b, int16_t *u, int16_t *v, int32_t n_iter)
{
    /* rb_u[r % 3] giữ hàng r của vòng TRƯỚC (Jacobi): hàng y−1, y, y+1 – như 2 line buffer của RTL */
    static int16_t rb_u[3][HS_W], rb_v[3][HS_W];
    int32_t n_sat = 0;
    memset(u, 0, sizeof(int16_t) * HS_N);
    memset(v, 0, sizeof(int16_t) * HS_N);
    for (int32_t k = 0; k < n_iter; k++) {
        memcpy(rb_u[0], u, sizeof(rb_u[0]));
        memcpy(rb_v[0], v, sizeof(rb_v[0]));
        for (int y = 0; y < HS_H; y++) {
            if (y + 1 < HS_H) {                                   /* nạp hàng y+1 (còn là giá trị cũ) */
                memcpy(rb_u[(y + 1) % 3], u + (y + 1) * HS_W, sizeof(rb_u[0]));
                memcpy(rb_v[(y + 1) % 3], v + (y + 1) * HS_W, sizeof(rb_v[0]));
            }
            const int16_t *mu = rb_u[y % 3], *mv = rb_v[y % 3];
            const int16_t *uu = y > 0 ? rb_u[(y - 1) % 3] : mu, *uv = y > 0 ? rb_v[(y - 1) % 3] : mv;
            const int16_t *du = y + 1 < HS_H ? rb_u[(y + 1) % 3] : mu;
            const int16_t *dv = y + 1 < HS_H ? rb_v[(y + 1) % 3] : mv;
            const int base = y * HS_W;
            for (int x = 0; x < HS_W; x++) {
                const int xl = x > 0 ? x - 1 : x, xr = x + 1 < HS_W ? x + 1 : x;
                const int32_t eu = uu[x] + du[x] + mu[xl] + mu[xr];
                const int32_t cu = uu[xl] + uu[xr] + du[xl] + du[xr];
                const int32_t ev = uv[x] + dv[x] + mv[xl] + mv[xr];
                const int32_t cv = uv[xl] + uv[xr] + dv[xl] + dv[xr];
                const int32_t au = ((2 * eu + cu) * AVG_MUL + (1 << (AVG_SHIFT - 1))) >> AVG_SHIFT;
                const int32_t av = ((2 * ev + cv) * AVG_MUL + (1 << (AVG_SHIFT - 1))) >> AVG_SHIFT;
                const int p = base + x;
                const int32_t n = ix[p] * au + iy[p] * av + ((int32_t)it[p] << FU);
                int32_t nu = au - (int32_t)(((int64_t)a[p] * n + (1 << (COEF_FRAC - 1))) >> COEF_FRAC);
                int32_t nv = av - (int32_t)(((int64_t)b[p] * n + (1 << (COEF_FRAC - 1))) >> COEF_FRAC);
                if (nu < UV_MIN || nu > UV_MAX) { n_sat++; nu = sat_i32(nu, UV_MIN, UV_MAX); }
                if (nv < UV_MIN || nv > UV_MAX) { n_sat++; nv = sat_i32(nv, UV_MIN, UV_MAX); }
                u[p] = (int16_t)nu;                                /* ghi đè tại chỗ: hàng y cũ đã ở rb */
                v[p] = (int16_t)nv;
            }
        }
    }
    return n_sat;
}

/* -------------------------------------------------------------------- FEAT */
static uint32_t isqrt32(uint32_t x)                               /* floor(√x) */
{
    uint32_t r = 0, bit = 1u << 30;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return r;
}

void hs_feat(const int16_t *u, const int16_t *v,
             const uint16_t *lut, int32_t lut_n,
             const int16_t *mean_q, const uint16_t *inv_q, int32_t inv_frac,
             int32_t *sums, int16_t *mean_out, int16_t *log_out, int16_t *z)
{
    int32_t s[HS_DIM];
    memset(s, 0, sizeof(s));
    for (int y = 0; y < HS_H; y++) {
        const int row = (y / CELL_H) * 4;
        for (int x = 0; x < HS_W; x++) {
            const int p = y * HS_W + x, c = (row + x / CELL_W) * 3;
            const int32_t uu = u[p], vv = v[p];
            const uint32_t s4 = 4u * (uint32_t)(uu * uu + vv * vv);
            s[c] += uu;
            s[c + 1] += vv;
            s[c + 2] += (int32_t)((isqrt32(s4) + 1) >> 1);       /* |v| = round(√(u²+v²)) */
        }
    }
    for (int d = 0; d < HS_DIM; d++) {
        const int32_t m = sat_i32((int32_t)(((int64_t)s[d] * CELL_MUL + (1 << (CELL_SHIFT - 1))) >> CELL_SHIFT),
                                  -32768, 32767);
        const int32_t ax = m < 0 ? -m : m;
        int32_t addr = ax < LOG_A_LIMIT ? (ax >> 1) : 1024 + ((ax - LOG_A_LIMIT) >> 6);
        if (addr > lut_n - 1) addr = lut_n - 1;
        const int32_t g = m > 0 ? lut[addr] : (m < 0 ? -(int32_t)lut[addr] : 0);
        if (sums) sums[d] = s[d];
        if (mean_out) mean_out[d] = (int16_t)m;
        if (log_out) log_out[d] = (int16_t)g;
        if (z) {
            const int64_t t = (int64_t)(g - mean_q[d]) * inv_q[d];
            z[d] = (int16_t)sat_i32((int32_t)((t + ((int64_t)1 << (inv_frac - 1))) >> inv_frac), -32768, 32767);
        }
    }
}

/* ------------------------------------------------------------ số thực float */
void hsf_deriv(const uint8_t *i1, const uint8_t *i2, float *ws, float alpha)
{
    float *fx = ws, *fy = ws + HS_N, *ft = ws + 2 * HS_N, *den = ws + 3 * HS_N;
    for (int y = 0; y < HS_H; y++) {
        const int y1 = (y + 1 < HS_H) ? y + 1 : y;
        for (int x = 0; x < HS_W; x++) {
            const int x1 = (x + 1 < HS_W) ? x + 1 : x;
            const float a00 = i1[y * HS_W + x], a01 = i1[y * HS_W + x1], a10 = i1[y1 * HS_W + x], a11 = i1[y1 * HS_W + x1];
            const float b00 = i2[y * HS_W + x], b01 = i2[y * HS_W + x1], b10 = i2[y1 * HS_W + x], b11 = i2[y1 * HS_W + x1];
            const int k = y * HS_W + x;
            fx[k] = 0.25f * ((a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10));
            fy[k] = 0.25f * ((a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01));
            ft[k] = 0.25f * ((b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11));
            den[k] = alpha * alpha + fx[k] * fx[k] + fy[k] * fy[k];
        }
    }
}

void hsf_iter(const float *ws, float *u, float *v, int32_t n_iter)
{
    static float rb_u[3][HS_W], rb_v[3][HS_W];
    const float *fx = ws, *fy = ws + HS_N, *ft = ws + 2 * HS_N, *den = ws + 3 * HS_N;
    const float k12 = 1.0f / 12.0f;
    memset(u, 0, sizeof(float) * HS_N);
    memset(v, 0, sizeof(float) * HS_N);
    for (int32_t k = 0; k < n_iter; k++) {
        memcpy(rb_u[0], u, sizeof(rb_u[0]));
        memcpy(rb_v[0], v, sizeof(rb_v[0]));
        for (int y = 0; y < HS_H; y++) {
            if (y + 1 < HS_H) {
                memcpy(rb_u[(y + 1) % 3], u + (y + 1) * HS_W, sizeof(rb_u[0]));
                memcpy(rb_v[(y + 1) % 3], v + (y + 1) * HS_W, sizeof(rb_v[0]));
            }
            const float *mu = rb_u[y % 3], *mv = rb_v[y % 3];
            const float *uu = y > 0 ? rb_u[(y - 1) % 3] : mu, *uv = y > 0 ? rb_v[(y - 1) % 3] : mv;
            const float *du = y + 1 < HS_H ? rb_u[(y + 1) % 3] : mu, *dv = y + 1 < HS_H ? rb_v[(y + 1) % 3] : mv;
            for (int x = 0; x < HS_W; x++) {
                const int xl = x > 0 ? x - 1 : x, xr = x + 1 < HS_W ? x + 1 : x;
                const float au = (2.0f * (uu[x] + du[x] + mu[xl] + mu[xr]) + (uu[xl] + uu[xr] + du[xl] + du[xr])) * k12;
                const float av = (2.0f * (uv[x] + dv[x] + mv[xl] + mv[xr]) + (uv[xl] + uv[xr] + dv[xl] + dv[xr])) * k12;
                const int p = y * HS_W + x;
                const float der = (fx[p] * au + fy[p] * av + ft[p]) / den[p];
                u[p] = au - fx[p] * der;
                v[p] = av - fy[p] * der;
            }
        }
    }
}
