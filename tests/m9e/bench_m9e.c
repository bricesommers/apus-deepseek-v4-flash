/*
 * tests/m9e/bench_m9e.c — M9e microbenchmark at the real routed-expert
 * shapes. Interleaved same-process A/B: `ref` is a bench-local verbatim
 * copy of the pre-M9e (M9a/M9d, git HEAD) 4-row-block kernel body;
 * `new` is the production apus_fp4_gemm_mt. Alternating reps, best-of-7.
 * Shapes: w1/w3 (O=2048, K=4096), w2 (O=4096, K=2048), M in
 * {1,4,6,8,12,16,24,32}. Informational only (no checks); the gates are
 * test_m9e + the m2..m9d suites.
 */
#define APUS_FP4_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fp4.h"

#ifndef __ARM_NEON
/* M12a-1: the pre-M9e reference kernel is a NEON body — ARM-only bench.
 * The x86 scalar kernels get their own bench with the M12a-2 AVX2 work. */
int main(void) {
    fprintf(stderr, "bench_m9e: NEON kernel benchmark (ARM-only)\n");
    return 0;
}
#else

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* --- pre-M9e reference kernel (verbatim HEAD apus_fp4_gemm_rows_neon body,
 *     4-row m-blocks), so the A/B runs in one process -------------------- */
typedef struct {
    const uint8_t *w, *ws;
    const float *as;
    const float *scratch;
    float *out;
    size_t M, O, K;
} RefJob;

static void ref_rows(void *vjob, size_t o0, size_t o1) {
    const RefJob *j = vjob;
    const uint8_t *w = j->w;
    const uint8_t *ws = j->ws;
    const float *as = j->as;
    const float16_t *acts16 = (const float16_t *)j->scratch;
    float *out = j->out;
    size_t M = j->M, O = j->O, K = j->K;
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = w + o * (K / 2);
            const uint8_t *sp = ws + o * nb;
            float32x4_t total[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                    vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (size_t kb = 0; kb < nb; kb++) {
                float16x8_t k[4];
                apus_fp4_expand32_f16_neon(wp + kb * 16, k);
                float32x4_t sb = vdupq_n_f32(apus_ue8m0_f32(sp[kb]));
                for (size_t r = 0; r < mc; r++) {
                    const float16_t *ap = acts16 + (m0 + r) * K
                                        + kb * APUS_FP4_GROUP;
                    float32x4_t acc = apus_fp4_dot32_f16_neon(ap, k);
                    float32x4_t t = vmulq_f32(acc, vdupq_n_f32(
                        0.5f * as[(m0 + r) * nab + kb / 4]));
                    total[r] = vaddq_f32(total[r], vmulq_f32(t, sb));
                }
            }
            for (size_t r = 0; r < mc; r++)
                out[(m0 + r) * O + o] = apus_fp4_hsum4_neon(total[r]);
        }
    }
}

static void ref_gemm_mt(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    float16_t *acts16 = (float16_t *)scratch;
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_dequant_f16_all(acodes + m * K, acts16 + m * K, K);
    RefJob j = { w, ws, as, scratch, out, M, O, K };
    apus_pool_run(O, ref_rows, &j);
}

/* --- A/B driver ----------------------------------------------------------*/

static void bench_shape(size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2));
    uint8_t *ws = malloc(O * nb);
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(120 + (rng_u64() % 16));   /* sane scales */
    static const size_t MS[] = { 1, 4, 6, 8, 12, 16, 24, 32 };
    for (int mi = 0; mi < 8; mi++) {
        size_t M = MS[mi];
        uint8_t *acodes = malloc(M * K);
        float *as = malloc(M * nab * sizeof(float));
        float *scratch = malloc(M * K * sizeof(float));
        float *o_ref = malloc(M * O * sizeof(float));
        float *o_new = malloc(M * O * sizeof(float));
        for (size_t i = 0; i < M * K; i++) acodes[i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < M * nab; i++) as[i] = 0.25f;
        double gf = 2.0 * (double)M * O * K / 1e9;
        double b_ref = 1e30, b_new = 1e30;
        for (int r = 0; r < 7; r++) {
            double t0 = now_s();
            ref_gemm_mt(w, ws, acodes, as, scratch, o_ref, M, O, K);
            double dt = now_s() - t0;
            if (dt < b_ref) b_ref = dt;
            t0 = now_s();
            apus_fp4_gemm_mt(w, ws, acodes, as, scratch, o_new, M, O, K);
            dt = now_s() - t0;
            if (dt < b_new) b_new = dt;
        }
        int same = memcmp(o_ref, o_new, M * O * sizeof(float)) == 0;
        printf("  O=%-5zu K=%-5zu M=%-3zu ref %8.3f ms (%6.1f GF/s)"
               "  new %8.3f ms (%6.1f GF/s)  %5.2fx  %s\n",
               O, K, M, b_ref * 1e3, gf / b_ref, b_new * 1e3, gf / b_new,
               b_ref / b_new, same ? "bitwise" : "DIFFERS!");
        free(acodes); free(as); free(scratch); free(o_ref); free(o_new);
    }
    free(w); free(ws);
}

int main(void) {
    printf("bench_m9e: pre-M9e 4-row kernel vs current apus_fp4_gemm_mt"
           " (interleaved, best of 7, APUS_THREADS=%d)\n",
           apus_pool_threads());
    printf("w1/w3 (O=2048, K=4096):\n");
    bench_shape(2048, 4096);
    printf("w2 (O=4096, K=2048):\n");
    bench_shape(4096, 2048);
    return 0;
}
#endif /* __ARM_NEON */
