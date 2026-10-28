/* ============================================================================
 * ocsvm_features.c - Cai dat C cua buoc 1-4 (xem ocsvm_features.h)
 * Da kiem chung voi Python bang scripts/07_check_c_port.py.
 * ==========================================================================*/
#include "ocsvm_features.h"

#include <math.h>
#include <string.h>

#define W FRAME_W
#define H FRAME_H
#define IDX(y, x) ((y) * W + (x))

/* ------------------------------------------------------------------ tien ich */
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* BORDER_REFLECT_101 cua OpenCV: -1 -> 1, n -> n-2 */
static inline int reflect101(int i, int n)
{
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

int ofe_init(ofe_ctx_t *ctx, ofe_alloc_fn alloc)
{
    float **bufs[] = {&ctx->prev, &ctx->cur, &ctx->ix, &ctx->iy, &ctx->it,
                      &ctx->u, &ctx->v, &ctx->ua, &ctx->va};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
        *bufs[i] = (float *)alloc(sizeof(float) * OFE_NPIX);
        if (*bufs[i] == NULL) return -1;
    }
    ctx->has_prev = 0;
    return 0;
}

void ofe_reset(ofe_ctx_t *ctx) { ctx->has_prev = 0; }

/* ------------------------------------------------- Buoc 1: lam muot Gaussian */
void ofe_gaussian_blur(const uint8_t *src, float *dst, float *tmp)
{
#if GAUSS_KSIZE > 1
    const int r = GAUSS_KSIZE / 2;
    float k[GAUSS_KSIZE];
    double sum = 0.0;
    for (int i = 0; i < GAUSS_KSIZE; ++i) {
        double x = (double)(i - r);
        k[i] = (float)exp(-x * x / (2.0 * (double)GAUSS_SIGMA * (double)GAUSS_SIGMA));
        sum += k[i];
    }
    for (int i = 0; i < GAUSS_KSIZE; ++i) k[i] = (float)(k[i] / sum);
    /* loc ngang */
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) acc += k[i + r] * (float)src[IDX(y, reflect101(x + i, W))];
            tmp[IDX(y, x)] = acc;
        }
    /* loc doc */
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) acc += k[i + r] * tmp[IDX(reflect101(y + i, H), x)];
            dst[IDX(y, x)] = acc;
        }
#else
    (void)tmp;
    for (int i = 0; i < OFE_NPIX; ++i) dst[i] = (float)src[i];
#endif
}

/* ------------------------------------------------ Buoc 2: Horn-Schunck (42) */
static void image_derivatives(const float *a, const float *b, float *ix, float *iy, float *it)
{
    const float s = 1.0f / HS_INTENSITY_SCALE;
    for (int y = 0; y < H; ++y) {
        const int y1 = y + 1 < H ? y + 1 : H - 1;               /* bien replicate */
        for (int x = 0; x < W; ++x) {
            const int x1 = x + 1 < W ? x + 1 : W - 1;
            const float a00 = a[IDX(y, x)] * s, a01 = a[IDX(y, x1)] * s;
            const float a10 = a[IDX(y1, x)] * s, a11 = a[IDX(y1, x1)] * s;
            const float b00 = b[IDX(y, x)] * s, b01 = b[IDX(y, x1)] * s;
            const float b10 = b[IDX(y1, x)] * s, b11 = b[IDX(y1, x1)] * s;
            ix[IDX(y, x)] = 0.25f * ((a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10));
            iy[IDX(y, x)] = 0.25f * ((a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01));
            it[IDX(y, x)] = 0.25f * ((b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11));
        }
    }
}

/* Trung binh lan can HS: canh 1/6, goc 1/12, bien replicate */
static void neighbor_average(const float *f, float *out)
{
    const float c = 1.0f / 12.0f, e = 1.0f / 6.0f;
    for (int y = 0; y < H; ++y) {
        const int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);
        for (int x = 0; x < W; ++x) {
            const int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
            out[IDX(y, x)] = c * f[IDX(ym, xm)] + e * f[IDX(ym, x)] + c * f[IDX(ym, xp)]
                           + e * f[IDX(y, xm)]                     + e * f[IDX(y, xp)]
                           + c * f[IDX(yp, xm)] + e * f[IDX(yp, x)] + c * f[IDX(yp, xp)];
        }
    }
}

void ofe_horn_schunck(ofe_ctx_t *ctx)
{
    image_derivatives(ctx->prev, ctx->cur, ctx->ix, ctx->iy, ctx->it);
    memset(ctx->u, 0, sizeof(float) * OFE_NPIX);
    memset(ctx->v, 0, sizeof(float) * OFE_NPIX);
    const float a2 = HS_ALPHA * HS_ALPHA;
    for (int k = 0; k < HS_NUM_ITER; ++k) {
        neighbor_average(ctx->u, ctx->ua);
        neighbor_average(ctx->v, ctx->va);
        for (int i = 0; i < OFE_NPIX; ++i) {
            const float gx = ctx->ix[i], gy = ctx->iy[i];
            const float der = (gx * ctx->ua[i] + gy * ctx->va[i] + ctx->it[i]) / (a2 + gx * gx + gy * gy);
            ctx->u[i] = ctx->ua[i] - gx * der;
            ctx->v[i] = ctx->va[i] - gy * der;
        }
    }
}

/* ------------------------------------ Buoc 3: dao ham (giong numpy.gradient) */
static inline float grad_x(const float *f, int y, int x)
{
    if (x == 0) return f[IDX(y, 1)] - f[IDX(y, 0)];
    if (x == W - 1) return f[IDX(y, W - 1)] - f[IDX(y, W - 2)];
    return 0.5f * (f[IDX(y, x + 1)] - f[IDX(y, x - 1)]);
}
static inline float grad_y(const float *f, int y, int x)
{
    if (y == 0) return f[IDX(1, x)] - f[IDX(0, x)];
    if (y == H - 1) return f[IDX(H - 1, x)] - f[IDX(H - 2, x)];
    return 0.5f * (f[IDX(y + 1, x)] - f[IDX(y - 1, x)]);
}
static inline float grad_xx(const float *f, int y, int x)
{
    if (x == 0) return grad_x(f, y, 1) - grad_x(f, y, 0);
    if (x == W - 1) return grad_x(f, y, W - 1) - grad_x(f, y, W - 2);
    return 0.5f * (grad_x(f, y, x + 1) - grad_x(f, y, x - 1));
}
static inline float grad_yy(const float *f, int y, int x)
{
    if (y == 0) return grad_y(f, 1, x) - grad_y(f, 0, x);
    if (y == H - 1) return grad_y(f, H - 1, x) - grad_y(f, H - 2, x);
    return 0.5f * (grad_y(f, y + 1, x) - grad_y(f, y - 1, x));
}

/* z = F(x, y) theo Bang 1 */
static void pixel_feature(const float *u, const float *v, int y, int x, double *z)
{
#if NORMALIZE_COORDS
    z[0] = (double)y / H;
    z[1] = (double)x / W;
#else
    z[0] = y;
    z[1] = x;
#endif
    z[2] = u[IDX(y, x)];
    z[3] = v[IDX(y, x)];
#if FEATURE_SET_ID == 1
    z[4] = grad_x(u, y, x); z[5] = grad_y(u, y, x);
#elif FEATURE_SET_ID == 2
    z[4] = grad_x(v, y, x); z[5] = grad_y(v, y, x);
#else
    z[4] = grad_x(u, y, x); z[5] = grad_y(u, y, x);
    z[6] = grad_x(v, y, x); z[7] = grad_y(v, y, x);
#if FEATURE_SET_ID == 4
    z[8] = grad_xx(u, y, x); z[9] = grad_yy(u, y, x);
    z[10] = grad_xx(v, y, x); z[11] = grad_yy(v, y, x);
#endif
#endif
}

static inline int keep_pixel(const float *u, const float *v, int y, int x, int use_mask)
{
    if (!use_mask) return 1;
    const float m = hypotf(u[IDX(y, x)], v[IDX(y, x)]);
    return m > MOTION_THRESHOLD;
}

/* ------------------------------------------- Buoc 4: ma tran hiep bien (44) */
void ofe_covariance(const float *u, const float *v, double *cov)
{
    const int d = FEAT_D;
    int use_mask = 0;
    if (MOTION_THRESHOLD > 0.f) {
        int cnt = 0;
        for (int y = PIXEL_BORDER; y < H - PIXEL_BORDER; y += PIXEL_STRIDE)
            for (int x = PIXEL_BORDER; x < W - PIXEL_BORDER; x += PIXEL_STRIDE)
                cnt += keep_pixel(u, v, y, x, 1);
        use_mask = cnt >= MIN_MOTION_PIXELS;
    }
    double mu[FEAT_D] = {0}, z[FEAT_D];
    long n = 0;
    for (int y = PIXEL_BORDER; y < H - PIXEL_BORDER; y += PIXEL_STRIDE)       /* luot 1: mu */
        for (int x = PIXEL_BORDER; x < W - PIXEL_BORDER; x += PIXEL_STRIDE) {
            if (!keep_pixel(u, v, y, x, use_mask)) continue;
            pixel_feature(u, v, y, x, z);
            for (int i = 0; i < d; ++i) mu[i] += z[i];
            ++n;
        }
    for (int i = 0; i < d; ++i) mu[i] /= (double)n;
    memset(cov, 0, sizeof(double) * d * d);
    for (int y = PIXEL_BORDER; y < H - PIXEL_BORDER; y += PIXEL_STRIDE)       /* luot 2: C */
        for (int x = PIXEL_BORDER; x < W - PIXEL_BORDER; x += PIXEL_STRIDE) {
            if (!keep_pixel(u, v, y, x, use_mask)) continue;
            pixel_feature(u, v, y, x, z);
            for (int i = 0; i < d; ++i) z[i] -= mu[i];
            for (int i = 0; i < d; ++i)
                for (int j = i; j < d; ++j) cov[i * d + j] += z[i] * z[j];
        }
    const double inv = 1.0 / (double)(n > 1 ? n - 1 : 1);
    for (int i = 0; i < d; ++i)
        for (int j = i; j < d; ++j) {
            double c = cov[i * d + j] * inv;
            if (i == j) c += (double)COV_EPSILON;
            cov[i * d + j] = cov[j * d + i] = c;
        }
}

/* --------------------------------- logm ma tran SPD: Jacobi tri rieng (d<=12) */
static void jacobi_eig(double *a, int n, double *w, double *V)
{
    for (int i = 0; i < n * n; ++i) V[i] = 0.0;
    for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int p = 0; p < n; ++p) {
            diag += a[p * n + p] * a[p * n + p];
            for (int q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        }
        if (off <= 1e-30 * diag) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (fabs(apq) < 1e-300) continue;
                const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {                 /* A <- A J (cot) */
                    const double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {                 /* A <- J^T A (hang) */
                    const double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {                 /* V <- V J */
                    const double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < n; ++i) w[i] = a[i * n + i];
}

void ofe_spd_logm(const double *c, double *logc, int d)
{
    double a[FEAT_D * FEAT_D], V[FEAT_D * FEAT_D], w[FEAT_D];
    memcpy(a, c, sizeof(double) * d * d);
    jacobi_eig(a, d, w, V);
    for (int i = 0; i < d; ++i) w[i] = log(w[i] > 1e-12 ? w[i] : 1e-12);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += V[i * d + k] * w[k] * V[j * d + k];
            logc[i * d + j] = s;
        }
}

void ofe_vech_scaled(const double *m, int d, float *vec)
{
    int k = 0;
    for (int i = 0; i < d; ++i)
        for (int j = i; j < d; ++j)
            vec[k++] = (float)(i == j ? m[i * d + j] : m[i * d + j] * 1.4142135623730951);
}

int ofe_quantize(const float *vec, int16_t *xq)
{
    const long lo = -(1L << (OCSVM_DATA_W - 1)), hi = (1L << (OCSVM_DATA_W - 1)) - 1;
    int sat = 0;
    for (int i = 0; i < OCSVM_DIM; ++i) {
        const double z = ((double)vec[i] - (double)OCSVM_SCALER_MEAN[i]) * (double)OCSVM_SCALER_INV_STD[i];
        long q = (long)nearbyint(z * (double)(1L << OCSVM_DATA_FRAC));   /* lam tron ve chan nhu numpy */
        if (q < lo) { q = lo; ++sat; }
        if (q > hi) { q = hi; ++sat; }
        xq[i] = (int16_t)q;
    }
    return sat;
}

#if OCSVM_DESCRIPTOR == 1
/* ------------------------------------------------- dac trung luoi (grid) */
/* Giong src/ocsvm_fpga/features/grid_flow.py: o (r, c) theo hang, trong moi o
 * [mean u, mean v, mean |v|] (+ max |v| khi GRID_NSTATS >= 4, + std |v| khi GRID_NSTATS == 5)
 * (+ mean |I2 - I1| khi GRID_DIFF). Bien o lay tu GRID_Y_EDGES / GRID_X_EDGES (co the khong deu). */
void ofe_grid_descriptor(const float *u, const float *v, const float *prev, const float *cur, float *vec)
{
#ifndef GRID_DIFF
#define GRID_DIFF 0
#endif
    static const int ye[GRID_ROWS + 1] = GRID_Y_EDGES;
    static const int xe[GRID_COLS + 1] = GRID_X_EDGES;
#ifndef GRID_NSTATS
#define GRID_NSTATS (GRID_STATS_RICH ? 5 : 3)                /* header cu */
#endif
    const int ns = GRID_NSTATS + GRID_DIFF;             /* so gia tri moi o */
    for (int r = 0; r < GRID_ROWS; ++r)
        for (int c = 0; c < GRID_COLS; ++c) {
            double su = 0, sv = 0, sm = 0, mx = 0, sd = 0;
            int n = 0;
            for (int y = ye[r]; y < ye[r + 1]; ++y)
                for (int x = xe[c]; x < xe[c + 1]; ++x) {
                    int i = y * FRAME_W + x;
                    float m = hypotf(u[i], v[i]);
                    su += u[i]; sv += v[i]; sm += m;
                    if (GRID_DIFF) sd += fabsf(cur[i] - prev[i]);
                    if (m > mx) mx = m;
                    ++n;
                }
            float *o = vec + (r * GRID_COLS + c) * ns;
            double mu = sm / n;
            o[0] = (float)(su / n);
            o[1] = (float)(sv / n);
            o[2] = (float)mu;
            if (GRID_NSTATS >= 4) o[3] = (float)mx;
            if (GRID_NSTATS == 5) {
                double ss = 0;
                for (int y = ye[r]; y < ye[r + 1]; ++y)
                    for (int x = xe[c]; x < xe[c + 1]; ++x) {
                        double d = hypotf(u[y * FRAME_W + x], v[y * FRAME_W + x]) - mu;
                        ss += d * d;
                    }
                o[4] = (float)sqrt(ss / n);              /* do lech chuan tong the, nhu numpy.std */
            }
            if (GRID_DIFF) o[GRID_NSTATS] = (float)(sd / n); /* mean |I2 - I1| */
        }
#if defined(GRID_LOG_SCALE)
    if (GRID_LOG_SCALE > 0.0f) {                         /* nen bien do: c*sign(x)*log(1+|x|/c) */
        const int dim = GRID_ROWS * GRID_COLS * ns;
        for (int i = 0; i < dim; ++i) {
            float x = vec[i], ax = fabsf(x);
            vec[i] = (x < 0 ? -1.0f : 1.0f) * GRID_LOG_SCALE * log1pf(ax / GRID_LOG_SCALE);
        }
    }
#endif
}
#endif

/* ---------------------------------------------------------- toan bo 1 khung */
int ofe_process_frame(ofe_ctx_t *ctx, const uint8_t *gray, ofe_result_t *out)
{
#if defined(SKIP_DARK_BELOW)
    if (SKIP_DARK_BELOW > 0.f) {                                       /* khung den -> ngat cap */
        uint32_t s = 0;
        for (int i = 0; i < OFE_NPIX; ++i) s += gray[i];
        if ((float)s / OFE_NPIX < SKIP_DARK_BELOW) { ctx->has_prev = 0; return 0; }
    }
#endif
    float *t = ctx->prev; ctx->prev = ctx->cur; ctx->cur = t;          /* doi bo dem */
    ofe_gaussian_blur(gray, ctx->cur, ctx->ua);                        /* ua lam tam  */
    if (!ctx->has_prev) { ctx->has_prev = 1; return 0; }

    ofe_horn_schunck(ctx);
    double mot = 0.0;
    for (int i = 0; i < OFE_NPIX; ++i) mot += hypotf(ctx->u[i], ctx->v[i]);
    out->mean_motion = (float)(mot / OFE_NPIX);

#if OCSVM_DESCRIPTOR == 1
    memset(out->cov, 0, sizeof(out->cov));
    ofe_grid_descriptor(ctx->u, ctx->v, ctx->prev, ctx->cur, out->vec);
    out->n_saturated = ofe_quantize(out->vec, out->xq);
    return 1;
#else
    double cov[FEAT_D * FEAT_D], m[FEAT_D * FEAT_D];
    ofe_covariance(ctx->u, ctx->v, cov);
    for (int i = 0; i < FEAT_D * FEAT_D; ++i) out->cov[i] = (float)cov[i];
#if LOG_EUCLIDEAN
    ofe_spd_logm(cov, m, FEAT_D);
#else
    memcpy(m, cov, sizeof(m));
#endif
    ofe_vech_scaled(m, FEAT_D, out->vec);
    out->n_saturated = ofe_quantize(out->vec, out->xq);
    return 1;
#endif
}

/* ------------------------------------------------------------------- UART */
size_t ofe_uart_build_request(const int16_t *xq, uint8_t *buf)
{
    size_t n = 0;
    uint8_t chk = 0;
    buf[n++] = UART_SYNC0;
    buf[n++] = UART_SYNC1;
    for (int i = 0; i < OCSVM_DIM; ++i) {
        const uint16_t w = (uint16_t)xq[i];
        buf[n++] = (uint8_t)(w & 0xFF);
        buf[n++] = (uint8_t)(w >> 8);
        chk ^= (uint8_t)(w & 0xFF) ^ (uint8_t)(w >> 8);
    }
    buf[n++] = chk;
    return n;
}

int ofe_uart_parse_response(const uint8_t *buf, size_t len, int64_t *f_q, uint8_t *flag)
{
    if (len < OFE_RESP_LEN || buf[0] != UART_RESP_SYNC) return -1;
    uint8_t chk = 0;
    uint64_t raw = 0;
    for (int i = 0; i < OFE_ACC_BYTES; ++i) {
        raw |= (uint64_t)buf[1 + i] << (8 * i);
        chk ^= buf[1 + i];
    }
    chk ^= buf[1 + OFE_ACC_BYTES];
    if (chk != buf[2 + OFE_ACC_BYTES]) return -2;
    if (OCSVM_ACC_W < 64 && (raw >> (OCSVM_ACC_W - 1)) & 1)                /* mo rong dau */
        raw |= ~((1ULL << OCSVM_ACC_W) - 1);
    *f_q = (int64_t)raw;
    *flag = buf[1 + OFE_ACC_BYTES];
    return 0;
}

/* -------------------------------------------------------------- hau xu ly */
double ofe_fq_to_float(int64_t f_q)
{
    return ldexp((double)f_q, -OCSVM_ACC_FRAC) / OCSVM_COEF_SCALE;
}

void ofe_smoother_init(ofe_smoother_t *s) { memset(s, 0, sizeof(*s)); }

double ofe_smoother_push(ofe_smoother_t *s, double f)
{
    const int k = OCSVM_SMOOTH_WIN > 0 ? OCSVM_SMOOTH_WIN : 1;
    const double val = OCSVM_DECISION_MODE == 1 ? fabs(f) : f;   /* lam muot tren mien diem so */
    if (s->n == k) s->sum -= s->win[s->pos]; else s->n++;
    s->win[s->pos] = val;
    s->sum += val;
    s->pos = (s->pos + 1) % k;
    return s->sum / s->n;
}

void ofe_alarm_init(ofe_alarm_t *a) { a->active = 0; a->below = 0; }

int ofe_alarm_update(ofe_alarm_t *a, double f_smoothed)
{
    /* f_smoothed tu ofe_smoother_push: mode 1 da la |f|, mode 0 la f → diem so = -f */
    const double score = OCSVM_DECISION_MODE == 1 ? f_smoothed : -f_smoothed;
    if (!a->active) {
        if (score > OCSVM_ALARM_ON_SCORE) { a->active = 1; a->below = 0; }
    } else {
        a->below = (score < OCSVM_ALARM_OFF_SCORE) ? a->below + 1 : 0;
        if (a->below > OCSVM_ALARM_HOLD) a->active = 0;
    }
    return a->active;
}

int ofe_is_alarm(double f_smoothed)
{
    return OCSVM_DECISION_MODE == 1 ? (f_smoothed > OCSVM_THR_FLOAT) : (f_smoothed < OCSVM_THR_FLOAT);
}
