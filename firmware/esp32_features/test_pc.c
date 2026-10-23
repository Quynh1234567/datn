/* test_pc.c - chay thu vien tren PC de so sanh voi Python (scripts/07_check_c_port.py)
 *   gcc -O2 -I. -I<outputs/fpga> test_pc.c ocsvm_features.c ocsvm_soft.c -lm -o test_pc
 *   ./test_pc frames.bin result.bin
 * frames.bin: int32 n, int32 w, int32 h, n*w*h byte
 * result.bin: moi cap khung: float vec[DIM] | int16 xq[DIM] | int64 f_q | uint8 flag */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ocsvm_features.h"
#include "ocsvm_soft.h"

static void *pc_alloc(size_t n) { return malloc(n); }

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s frames.bin result.bin\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("fopen"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, 4, 3, fi) != 3 || hdr[1] != FRAME_W || hdr[2] != FRAME_H) {
        fprintf(stderr, "Sai kich thuoc khung (can %dx%d)\n", FRAME_W, FRAME_H);
        return 1;
    }
    ofe_ctx_t ctx;
    if (ofe_init(&ctx, pc_alloc)) { fprintf(stderr, "het bo nho\n"); return 1; }
    uint8_t *frame = malloc(OFE_NPIX);
    ofe_result_t r;
    double t_feat = 0, t_svm = 0;
    int pairs = 0, uart_ok = 1;
    for (int f = 0; f < hdr[0]; ++f) {
        if (fread(frame, 1, OFE_NPIX, fi) != OFE_NPIX) break;
        double t0 = now_ms();
        int ready = ofe_process_frame(&ctx, frame, &r);
        double t1 = now_ms();
        if (!ready) continue;
        int64_t fq = ocsvm_soft_decision_q(r.xq);
        uint8_t flag = (uint8_t)ocsvm_soft_flag(fq);
        double t2 = now_ms();
        t_feat += t1 - t0; t_svm += t2 - t1; ++pairs;

        /* tu kiem tra UART: dong goi yeu cau + gia lap phan hoi FPGA roi giai ma */
        uint8_t req[OFE_REQ_LEN], resp[OFE_RESP_LEN], chk = 0;
        size_t n = ofe_uart_build_request(r.xq, req);
        uart_ok &= (n == OFE_REQ_LEN && req[0] == UART_SYNC0);
        resp[0] = UART_RESP_SYNC;
        for (int i = 0; i < OFE_ACC_BYTES; ++i) { resp[1 + i] = (uint8_t)((uint64_t)fq >> (8 * i)); chk ^= resp[1 + i]; }
        resp[1 + OFE_ACC_BYTES] = flag; chk ^= flag;
        resp[2 + OFE_ACC_BYTES] = chk;
        int64_t fq2; uint8_t flag2;
        uart_ok &= (ofe_uart_parse_response(resp, sizeof resp, &fq2, &flag2) == 0 && fq2 == fq && flag2 == flag);

        fwrite(r.vec, sizeof(float), OCSVM_DIM, fo);
        fwrite(r.xq, sizeof(int16_t), OCSVM_DIM, fo);
        fwrite(&fq, sizeof(int64_t), 1, fo);
        fwrite(&flag, 1, 1, fo);
    }
    fclose(fi); fclose(fo);
    printf("pairs=%d feat_ms=%.3f svm_ms=%.4f uart_ok=%d\n", pairs,
           pairs ? t_feat / pairs : 0, pairs ? t_svm / pairs : 0, uart_ok);
    return 0;
}
