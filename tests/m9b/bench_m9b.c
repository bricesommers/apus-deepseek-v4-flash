/*
 * tests/m9b/bench_m9b.c — M9b GEMM microbenchmark: M9a NEON kernels vs the
 * Accelerate (AMX) BLAS path at the real model shapes, across batch M.
 *
 * Shapes (DeepSeek-V4-Flash):
 *   fp8: wq_b 32768x1024, shared-expert w1/w3 2048x4096, w2 4096x2048
 *   fp4: expert w1/w3 2048x4096, expert w2 4096x2048
 * M: 2, 4 (M8 spec-verify batch), 8, 32, 128, 256, 512 (prefill).
 * Three rows per (shape, M): NEON single-thread (gemm_neon), NEON threaded
 * (gemm_mt, the engine's decode/small-M path), BLAS single vecLib thread
 * (gemm_blas, the M>=256 dispatch target). Best-of-3, GFLOP/s.
 */
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#include "fp4.h"
#include "fp8.h"
#include "blas.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef __ARM_NEON
/* M12a-1: NEON-vs-Accelerate microbenchmark, ARM/macOS-only. The x86 scalar
 * kernels get their own bench with the M12a-2 AVX2 work. */
int main(void) {
    fprintf(stderr, "bench_m9b: NEON/BLAS kernel benchmark (ARM/macOS-only)\n");
    return 0;
}
#else

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef void (*gemm_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                        const float *, float *, float *,
                        size_t, size_t, size_t);

static double bench_one(gemm_fn fn, const uint8_t *w, const uint8_t *ws,
                        const uint8_t *codes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K, double budget) {
    fn(w, ws, codes, as, scratch, out, M, O, K);   /* warmup */
    double t0 = now_s();
    fn(w, ws, codes, as, scratch, out, M, O, K);
    double one = now_s() - t0;
    size_t reps = (size_t)(budget / (one > 1e-9 ? one : 1e-9));
    if (reps < 2) reps = 2;
    if (reps > 100) reps = 100;
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        double a = now_s();
        for (size_t r = 0; r < reps; r++)
            fn(w, ws, codes, as, scratch, out, M, O, K);
        double per = (now_s() - a) / (double)reps;
        if (per < best) best = per;
    }
    return best;
}

static void bench_fp8(const char *name, size_t O, size_t K, size_t M) {
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    size_t nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * K), *ws = malloc(nbo * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *out = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < nbo * nb; i++)
        ws[i] = (uint8_t)(120 + rng_u64() % 12);
    for (size_t i = 0; i < K; i++)
        x[i] = (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                       - 2.0);
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nab);
    double fl = 2.0 * (double)M * (double)O * (double)K;
    double tn = bench_one(apus_fp8_gemm_neon, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    double tm = bench_one(apus_fp8_gemm_mt, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    double tb = bench_one(apus_fp8_gemm_blas, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    printf("  fp8 %-8s O=%5zu K=%4zu M=%-3zu  neon %8.3f ms %6.1f GF | "
           "mt %8.3f ms %6.1f GF | blas %8.3f ms %6.1f GF  (%.1fx vs mt)\n",
           name, O, K, M, tn * 1e3, fl / tn / 1e9, tm * 1e3, fl / tm / 1e9,
           tb * 1e3, fl / tb / 1e9, tm / tb);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch); free(out);
}

static void bench_fp4(const char *name, size_t O, size_t K, size_t M) {
    size_t nb = K / 32, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2)), *ws = malloc(O * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *out = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(120 + rng_u64() % 12);
    for (size_t i = 0; i < K; i++)
        x[i] = (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                       - 2.0);
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nab);
    double fl = 2.0 * (double)M * (double)O * (double)K;
    double tn = bench_one(apus_fp4_gemm_neon, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    double tm = bench_one(apus_fp4_gemm_mt, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    double tb = bench_one(apus_fp4_gemm_blas, w, ws, codes, as, scratch, out,
                          M, O, K, 0.3);
    printf("  fp4 %-8s O=%5zu K=%4zu M=%-3zu  neon %8.3f ms %6.1f GF | "
           "mt %8.3f ms %6.1f GF | blas %8.3f ms %6.1f GF  (%.1fx vs mt)\n",
           name, O, K, M, tn * 1e3, fl / tn / 1e9, tm * 1e3, fl / tm / 1e9,
           tb * 1e3, fl / tb / 1e9, tm / tb);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch); free(out);
}

int main(void) {
    /* gemm_mt must show the NEON pool path here (the >=256 dispatch would
     * route it to BLAS); the BLAS rows call apus_*_gemm_blas directly. */
    setenv("APUS_NO_BLAS", "1", 1);
    printf("bench_m9b: NEON (M9a) vs Accelerate BLAS, hot shapes x batch M\n");
    printf("  (APUS_NO_BLAS=1 set for the mt rows; blas rows call the BLAS "
           "kernels directly; cutoff M>=%d)\n", APUS_BLAS_M_MIN);
    const size_t Ms[] = {2, 4, 8, 32, 128, 256, 512};
    for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++) {
        size_t M = Ms[i];
        bench_fp8("wq_b", 32768, 1024, M);
        bench_fp8("sh_w1w3", 2048, 4096, M);
        bench_fp8("sh_w2", 4096, 2048, M);
        bench_fp4("ex_w1w3", 2048, 4096, M);
        bench_fp4("ex_w2", 4096, 2048, M);
    }
    return 0;
}
#endif /* __ARM_NEON */
