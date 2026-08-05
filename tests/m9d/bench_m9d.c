/*
 * tests/m9d/bench_m9d.c — M9d microbenchmark at the real model shapes.
 * Times the pre-M9d paths (pool-partitioned scalar f32 rows; NEON wo_a
 * rows) against the M9d BLAS dispatches, single process, best-of-3.
 * Informational only (no checks); the gates are test_m9d + the m2..m9c
 * suites.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_MHC_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "attn.h"
#include "blas.h"

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
static float rng_float(void) {
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 2.0
                   - 1.0);
}

static void bench_f32(size_t M, size_t O, size_t K) {
    float *w = malloc(O * K * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *out = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = apus_bf16_round(rng_float());
    for (size_t i = 0; i < M * K; i++) x[i] = apus_bf16_round(rng_float());
    double gf = 2.0 * (double)M * O * K / 1e9;
    /* pre-M9d: pool-partitioned scalar rows (the pinned path below the
     * cutoff, which also ran at every M before M9d) */
    ApusF32LinearJob job = { w, x, out, (int)M, (int)K, (int)O, 0 };
    double best_s = 1e30, best_b = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_s();
        apus_pool_run(M * O, apus_f32_linear_rows_scalar, &job);
        double dt = now_s() - t0;
        if (dt < best_s) best_s = dt;
        t0 = now_s();
        apus_f32_gemm_blas(w, x, out, M, O, K);
        dt = now_s() - t0;
        if (dt < best_b) best_b = dt;
    }
    printf("f32  M=%-3zu O=%-5zu K=%-5zu scalar-pool %8.2f ms (%6.1f GF/s)"
           "  blas %7.2f ms (%7.1f GF/s)  %5.1fx\n",
           M, O, K, best_s * 1e3, gf / best_s, best_b * 1e3, gf / best_b,
           best_s / best_b);
    free(w); free(x); free(out);
}

static void bench_woa(size_t M, size_t G, size_t ol, size_t sub) {
    size_t hd = G * sub;
    uint16_t *wa = malloc(G * ol * sub * sizeof(uint16_t));
    float *o = malloc(M * hd * sizeof(float));
    float *y = malloc(M * G * ol * sizeof(float));
    for (size_t i = 0; i < G * ol * sub; i++) wa[i] = (uint16_t)(rng_u64() >> 16);
    for (size_t i = 0; i < M * hd; i++) o[i] = apus_bf16_round(rng_float());
    double gf = 2.0 * (double)M * G * ol * sub / 1e9;
    ApusWoAJob job = { wa, o, y, (int)M, (int)G, (int)ol, (int)sub,
                       (int)hd };
    double best_n = 1e30, best_b = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_s();
        apus_pool_run(M * G * ol, apus_woa_rows, &job);
        double dt = now_s() - t0;
        if (dt < best_n) best_n = dt;
        t0 = now_s();
        apus_woa_gemm_blas(wa, o, y, M, G, ol, sub, hd);
        dt = now_s() - t0;
        if (dt < best_b) best_b = dt;
    }
    printf("woa  M=%-3zu G=%zu ol=%-4zu sub=%-4zu neon-pool %7.2f ms (%6.1f GF/s)"
           "  blas %7.2f ms (%7.1f GF/s)  %5.1fx\n",
           M, G, ol, sub, best_n * 1e3, gf / best_n, best_b * 1e3,
           gf / best_b, best_n / best_b);
    free(wa); free(o); free(y);
}

int main(void) {
    printf("bench_m9d: pre-M9d pooled rows vs M9d BLAS (best of 3,"
           " APUS_THREADS=%d)\n", apus_pool_threads());
    /* real shapes: compressor wkv/wgate (ratio 4 + indexer), router,
     * idx_wproj, compressor (ratio 128) */
    bench_f32(512, 1024, 4096);
    bench_f32(512, 256, 4096);
    bench_f32(512, 64, 4096);
    bench_f32(512, 512, 4096);
    bench_f32(256, 1024, 4096);
    /* grouped wo_a o-proj (real: G=8, ol=1024, sub=4096) */
    bench_woa(512, 8, 1024, 4096);
    bench_woa(256, 8, 1024, 4096);
    return 0;
}
