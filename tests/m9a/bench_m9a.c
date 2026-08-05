/*
 * tests/m9a/bench_m9a.c — isolate the fp8/fp4 NEON GEMM inner loops at the
 * real model shapes (M9a). Single-threaded, calls the NEON kernels directly
 * (the *_gemm_neon_rows hot path shares the same row body byte-for-byte, so
 * these numbers track the decode hot path minus pool dispatch).
 *
 * Shapes (DeepSeek-V4-Flash):
 *   fp8: wq_b 32768x1024, wq_a 1024x4096, shared-expert w1/w3 2048x4096,
 *        shared-expert w2 4096x2048
 *   fp4: expert w1/w3 2048x4096, expert w2 4096x2048
 * Reports us/call, effective GB/s (weight bytes streamed) and GFLOP/s.
 */
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#include "fp4.h"
#include "fp8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef __ARM_NEON
/* M12a-1: NEON-kernel microbenchmark, ARM-only. The x86 scalar kernels get
 * their own bench with the M12a-2 AVX2 work. */
int main(void) {
    fprintf(stderr, "bench_m9a: NEON kernel benchmark (ARM-only)\n");
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

typedef void (*fp8_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                       const float *, float *, float *, size_t, size_t);
typedef void (*fp8_gemm_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                            const float *, float *, float *,
                            size_t, size_t, size_t);

static double calib_reps(double one) {
    size_t reps = (size_t)(0.4 / (one > 1e-9 ? one : 1e-9));
    if (reps < 3) reps = 3;
    if (reps > 500) reps = 500;
    return (double)reps;
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

    double best = 1e30;
    if (M == 1) {
        fp8_fn fn = apus_fp8_gemv_neon;
        fn(w, ws, codes, as, scratch, out, O, K);
        double t0 = now_s();
        fn(w, ws, codes, as, scratch, out, O, K);
        double reps = calib_reps(now_s() - t0);
        for (int t = 0; t < 3; t++) {
            double a = now_s();
            for (size_t r = 0; r < (size_t)reps; r++)
                fn(w, ws, codes, as, scratch, out, O, K);
            double per = (now_s() - a) / reps;
            if (per < best) best = per;
        }
    } else {
        fp8_gemm_fn fn = apus_fp8_gemm_neon;
        fn(w, ws, codes, as, scratch, out, M, O, K);
        double t0 = now_s();
        fn(w, ws, codes, as, scratch, out, M, O, K);
        double reps = calib_reps(now_s() - t0);
        for (int t = 0; t < 3; t++) {
            double a = now_s();
            for (size_t r = 0; r < (size_t)reps; r++)
                fn(w, ws, codes, as, scratch, out, M, O, K);
            double per = (now_s() - a) / reps;
            if (per < best) best = per;
        }
    }
    double bytes = (double)O * ((double)K + (double)nb) * (M == 1 ? 1.0 : 1.0);
    double flops = 2.0 * (double)M * (double)O * (double)K;
    printf("  fp8 %-16s O=%5zu K=%4zu M=%zu  %9.3f us  %7.2f GB/s  %7.2f GFLOP/s\n",
           name, O, K, M, best * 1e6, bytes / best / 1e9, flops / best / 1e9);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch); free(out);
}

typedef void (*fp4_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                       const float *, float *, float *, size_t, size_t);
typedef void (*fp4_gemm_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                            const float *, float *, float *,
                            size_t, size_t, size_t);

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

    double best = 1e30;
    if (M == 1) {
        fp4_fn fn = apus_fp4_gemv_neon;
        fn(w, ws, codes, as, scratch, out, O, K);
        double t0 = now_s();
        fn(w, ws, codes, as, scratch, out, O, K);
        double reps = calib_reps(now_s() - t0);
        for (int t = 0; t < 3; t++) {
            double a = now_s();
            for (size_t r = 0; r < (size_t)reps; r++)
                fn(w, ws, codes, as, scratch, out, O, K);
            double per = (now_s() - a) / reps;
            if (per < best) best = per;
        }
    } else {
        fp4_gemm_fn fn = apus_fp4_gemm_neon;
        fn(w, ws, codes, as, scratch, out, M, O, K);
        double t0 = now_s();
        fn(w, ws, codes, as, scratch, out, M, O, K);
        double reps = calib_reps(now_s() - t0);
        for (int t = 0; t < 3; t++) {
            double a = now_s();
            for (size_t r = 0; r < (size_t)reps; r++)
                fn(w, ws, codes, as, scratch, out, M, O, K);
            double per = (now_s() - a) / reps;
            if (per < best) best = per;
        }
    }
    double bytes = (double)O * ((double)K / 2.0 + (double)nb);
    double flops = 2.0 * (double)M * (double)O * (double)K;
    printf("  fp4 %-16s O=%5zu K=%4zu M=%zu  %9.3f us  %7.2f GB/s  %7.2f GFLOP/s\n",
           name, O, K, M, best * 1e6, bytes / best / 1e9, flops / best / 1e9);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch); free(out);
}

int main(void) {
#ifdef __ARM_NEON
    printf("bench_m9a: fp8/fp4 NEON GEMM inner-loop benchmark (single thread)\n");
    bench_fp8("wq_b", 32768, 1024, 1);
    bench_fp8("wq_b", 32768, 1024, 4);
    bench_fp8("wq_a", 1024, 4096, 1);
    bench_fp8("sh_w1w3", 2048, 4096, 1);
    bench_fp8("sh_w2", 4096, 2048, 1);
    bench_fp4("ex_w1w3", 2048, 4096, 1);
    bench_fp4("ex_w1w3", 2048, 4096, 4);
    bench_fp4("ex_w2", 4096, 2048, 1);
    bench_fp4("ex_w2", 4096, 2048, 4);
#else
    printf("bench_m9a: NEON not available on this host\n");
#endif
    return 0;
}
#endif /* __ARM_NEON */
