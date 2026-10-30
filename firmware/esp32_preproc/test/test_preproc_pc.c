/* Chạy preproc.c trên PC để so với Golden Model Python (scripts/19_preproc_check.py --selftest).
 *
 *   gcc -O2 -I ../main test_preproc_pc.c ../main/preproc.c -o test_preproc_pc
 *   ./test_preproc_pc vectors                      # tự kiểm tra với selftest_vectors.h
 *   ./test_preproc_pc run in.bin out.bin <border>  # in.bin: int32 n + n×19200 byte
 *                                                  # out.bin: n×19200 byte + n×uint32 tổng */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preproc.h"
#include "selftest_vectors.h"

static int run_vectors(void)
{
    static uint8_t img[PP_PIX], out[PP_PIX];
    pp_work_t work;
    int fails = 0;
    for (size_t i = 0; i < PP_N_VECTORS; i++) {
        const pp_vector_t *v = &PP_VECTORS[i];
        pp_test_pattern(v->pattern, img);
        uint32_t sum = pp_gauss5(img, out, (pp_border_t)v->border, &work);
        uint32_t crc = pp_crc32(0, out, PP_PIX);
        int ok = crc == v->crc && sum == v->sum;
        fails += !ok;
        printf("%-11s border=%d crc=%08X expect=%08X sum=%u expect=%u %s\n", pp_pattern_name(v->pattern), v->border,
               crc, v->crc, sum, v->sum, ok ? "OK" : "FAIL");
    }
    printf("VECTORS pass=%d fail=%d\n", (int)PP_N_VECTORS - fails, fails);
    return fails != 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "vectors") == 0) {
        return run_vectors();
    }
    if (argc != 5 || strcmp(argv[1], "run") != 0) {
        fprintf(stderr, "dùng: %s vectors | run in.bin out.bin <0|1>\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[2], "rb");
    FILE *fo = fopen(argv[3], "wb");
    if (!fi || !fo) {
        perror("fopen");
        return 2;
    }
    int32_t n = 0;
    if (fread(&n, 4, 1, fi) != 1 || n <= 0) {
        fprintf(stderr, "file vào lỗi\n");
        return 2;
    }
    uint8_t *img = malloc(PP_PIX), *out = malloc((size_t)n * PP_PIX);
    uint32_t *sums = malloc(sizeof(uint32_t) * n);
    pp_work_t work;
    for (int i = 0; i < n; i++) {
        if (fread(img, 1, PP_PIX, fi) != PP_PIX) {
            fprintf(stderr, "thiếu dữ liệu ảnh %d\n", i);
            return 2;
        }
        sums[i] = pp_gauss5(img, out + (size_t)i * PP_PIX, (pp_border_t)atoi(argv[4]), &work);
    }
    fwrite(out, 1, (size_t)n * PP_PIX, fo);
    fwrite(sums, 4, n, fo);
    fclose(fi);
    fclose(fo);
    printf("đã xử lý %d ảnh\n", n);
    return 0;
}
