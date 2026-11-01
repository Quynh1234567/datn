#include "preproc.h"

#include <string.h>

/* Chỉ số sau khi xử lý biên, i ∈ [−2, n+1] */
static inline int map_idx(int i, int n, pp_border_t b)
{
    if (i < 0) {
        return b == PP_BORDER_REFLECT101 ? -i : 0;
    }
    if (i >= n) {
        return b == PP_BORDER_REFLECT101 ? 2 * n - 2 - i : n - 1;
    }
    return i;
}

/* Lọc ngang một hàng: h[x] = 56·(p[x−2]+p[x+2]) + 250·(p[x−1]+p[x+1]) + 412·p[x] */
static void hpass_row(const uint8_t *p, uint32_t *h, pp_border_t b)
{
    /* hai cột biên mỗi bên: dùng bảng chỉ số */
    static const int edge_x[4] = {0, 1, PP_W - 2, PP_W - 1};
    for (int e = 0; e < 4; e++) {
        int x = edge_x[e];
        uint32_t m2 = p[map_idx(x - 2, PP_W, b)], m1 = p[map_idx(x - 1, PP_W, b)];
        uint32_t p1 = p[map_idx(x + 1, PP_W, b)], p2 = p[map_idx(x + 2, PP_W, b)];
        h[x] = PP_K0 * (m2 + p2) + PP_K1 * (m1 + p1) + PP_K2 * (uint32_t)p[x];
    }
    /* phần trong: không rẽ nhánh */
    for (int x = 2; x < PP_W - 2; x++) {
        h[x] = PP_K0 * ((uint32_t)p[x - 2] + p[x + 2]) + PP_K1 * ((uint32_t)p[x - 1] + p[x + 1]) +
               PP_K2 * (uint32_t)p[x];
    }
}

uint32_t pp_gauss5(const uint8_t *src, uint8_t *dst, pp_border_t border, pp_work_t *work)
{
    uint32_t (*ring)[PP_W] = work->ring;
    int next_row = 0;          /* hàng nguồn kế tiếp cần lọc ngang */
    uint32_t total = 0;
    const uint32_t half = 1u << (PP_SHIFT - 1);

    for (int y = 0; y < PP_H; y++) {
        /* bảo đảm đã lọc ngang tới hàng y+2 (giới hạn trong ảnh) */
        int need = y + 2 < PP_H ? y + 2 : PP_H - 1;
        while (next_row <= need) {
            hpass_row(src + (size_t)next_row * PP_W, ring[next_row % 5], border);
            next_row++;
        }
        /* 5 hàng nguồn sau khi xử lý biên – luôn nằm trong cửa sổ [y−2, y+2] của vòng */
        const uint32_t *r0 = ring[map_idx(y - 2, PP_H, border) % 5];
        const uint32_t *r1 = ring[map_idx(y - 1, PP_H, border) % 5];
        const uint32_t *r2 = ring[y % 5];
        const uint32_t *r3 = ring[map_idx(y + 1, PP_H, border) % 5];
        const uint32_t *r4 = ring[map_idx(y + 2, PP_H, border) % 5];
        uint8_t *out = dst + (size_t)y * PP_W;
        uint32_t row_sum = 0;
        for (int x = 0; x < PP_W; x++) {
            uint32_t s = PP_K0 * (r0[x] + r4[x]) + PP_K1 * (r1[x] + r3[x]) + PP_K2 * r2[x];
            uint32_t g = (s + half) >> PP_SHIFT;   /* ≤ 255, không cần kẹp */
            out[x] = (uint8_t)g;
            row_sum += g;
        }
        total += row_sum;
    }
    return total;
}

/* ------------------------------------------------------------------ CRC-32 */

static uint32_t crc_table[256];
static bool crc_ready;

static void crc_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
        crc_table[i] = c;
    }
    crc_ready = true;
}

uint32_t pp_crc32(uint32_t crc, const uint8_t *buf, size_t len)
{
    if (!crc_ready) {
        crc_init();
    }
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = crc_table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

/* ------------------------------------------------------------- ảnh mẫu */

static const char *const names[PP_N_PATTERNS] = {
    "zeros", "ones255", "hramp", "vramp", "checker1", "noise12345", "noise777", "impulses", "blocks8", "xor",
};

const char *pp_pattern_name(int id)
{
    return (id >= 0 && id < PP_N_PATTERNS) ? names[id] : "?";
}

void pp_test_pattern(int id, uint8_t *buf)
{
    uint32_t s = id == 5 ? 12345u : 777u;   /* LCG: s = s·1103515245 + 12345, byte = s >> 24 */
    memset(buf, 0, PP_PIX);
    for (int y = 0; y < PP_H; y++) {
        for (int x = 0; x < PP_W; x++) {
            uint8_t v = 0;
            switch (id) {
            case 1: v = 255; break;
            case 2: v = (uint8_t)(x * 255 / (PP_W - 1)); break;
            case 3: v = (uint8_t)(y * 255 / (PP_H - 1)); break;
            case 4: v = ((x + y) & 1) ? 255 : 0; break;
            case 5:
            case 6:
                s = s * 1103515245u + 12345u;
                v = (uint8_t)(s >> 24);
                break;
            case 8: v = (((x / 8) + (y / 8)) & 1) ? 200 : 30; break;
            case 9: v = (uint8_t)((x ^ y) & 0xFF); break;
            default: break;
            }
            buf[y * PP_W + x] = v;
        }
    }
    if (id == 7) {   /* xung sáng ở góc, sát biên và tâm – kiểm tra xử lý biên */
        static const int pts[][2] = {{0, 0}, {0, PP_W - 1}, {PP_H - 1, 0}, {PP_H - 1, PP_W - 1},
                                     {1, 1}, {PP_H - 2, PP_W - 2}, {PP_H / 2, PP_W / 2}, {0, 80}, {60, 0}};
        for (size_t k = 0; k < sizeof(pts) / sizeof(pts[0]); k++) {
            buf[pts[k][0] * PP_W + pts[k][1]] = 255;
        }
    }
}
