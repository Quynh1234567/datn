/* SINH TỰ ĐỘNG bởi src/ocsvm_fpga/hwmodel/preproc_q.py – vectors_header(). KHÔNG SỬA TAY. */
#pragma once
#include <stdint.h>

typedef struct {
    uint8_t pattern;
    uint8_t border;   /* 0 = reflect101, 1 = replicate */
    uint32_t crc;     /* CRC-32 ảnh đã làm mượt */
    uint32_t sum;     /* Σ ảnh đã làm mượt */
} pp_vector_t;

static const pp_vector_t PP_VECTORS[] = {
    {0, 0, 0xF2B78FFBu, 0u},   /* zeros, reflect101 */
    {0, 1, 0xF2B78FFBu, 0u},   /* zeros, replicate */
    {1, 0, 0xB68B55DDu, 4896000u},   /* ones255, reflect101 */
    {1, 1, 0xB68B55DDu, 4896000u},   /* ones255, replicate */
    {2, 0, 0x21B75CD1u, 2438640u},   /* hramp, reflect101 */
    {2, 1, 0x894840DBu, 2438520u},   /* hramp, replicate */
    {3, 0, 0x655EA951u, 2439680u},   /* vramp, reflect101 */
    {3, 1, 0xCA05649Au, 2439840u},   /* vramp, replicate */
    {4, 0, 0x725E6E45u, 2448000u},   /* checker1, reflect101 */
    {4, 1, 0xFADBECC5u, 2448000u},   /* checker1, replicate */
    {5, 0, 0x5ABEE691u, 2450496u},   /* noise12345, reflect101 */
    {5, 1, 0x20ABF926u, 2450181u},   /* noise12345, replicate */
    {6, 0, 0x1F6BEC1Eu, 2450777u},   /* noise777, reflect101 */
    {6, 1, 0xC4C45AC3u, 2452055u},   /* noise777, replicate */
    {7, 0, 0x4C99A383u, 1897u},   /* impulses, reflect101 */
    {7, 1, 0xF7399B9Fu, 2381u},   /* impulses, replicate */
    {8, 0, 0x081DC451u, 2208000u},   /* blocks8, reflect101 */
    {8, 1, 0x081DC451u, 2208000u},   /* blocks8, replicate */
    {9, 0, 0x7D608783u, 1698432u},   /* xor, reflect101 */
    {9, 1, 0x7D608783u, 1698432u},   /* xor, replicate */
};
#define PP_N_VECTORS (sizeof(PP_VECTORS) / sizeof(PP_VECTORS[0]))
