/*
 * tests/m14/bench_m14.c — isolate the M14 staged-product interleaved-dot
 * speedup at the real model shapes. Compares the pinned scalar-order dot
 * loop (the pre-M14 ARM code path) against the M14 dot4 grouping:
 *
 *   1. q.k dots at the sparse_attn decode shape (d=512, n=640 per head).
 *   2. dense linear rows at decode shapes (M=1: compressor wkv/wgate
 *      K=4096->O=2048, gate K=4096->O=256) — scalar-per-row vs the
 *      dot4-grouped row body.
 *   3. sparse_attn end-to-end (s=1, h=64) vs a reference copy of the
 *      pre-M14 kernel body.
 *
 * ARM-only (the x86 twin is bench_m12a2). Single-threaded.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_MHC_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#include "attn.h"
#include "pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef __ARM_NEON
int main(void) {
    fprintf(stderr, "bench_m14: NEON dot4 benchmark (ARM-only; x86 twin "
                    "is bench_m12a2)\n");
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
static float rng_f32(void) {
    return 2.0f * ((float)(rng_u64() >> 40) / 16777216.0f) - 1.0f;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static size_t calib_reps(double one) {
    size_t reps = (size_t)(0.4 / (one > 1e-9 ? one : 1e-9));
    if (reps < 3) reps = 3;
    if (reps > 2000) reps = 2000;
    return reps;
}

/* noinline: identical-input reps must not be CSE'd away (the
 * m9a bench uses function pointers for the same reason). */
__attribute__((noinline))
static double bench_dots_scalar(const float *q, const float *kv, int n,
                                int d, float *sink) {
    double t0 = now_s();
    float acc = 0.0f;
    for (int j = 0; j < n; j++)
        acc += apus_dot_f32_scalar(q, kv + (size_t)j * d, (size_t)d);
    *sink = acc;
    return now_s() - t0;
}

__attribute__((noinline))
static double bench_dots_dot4(const float *q, const float *kv, int n,
                              int d, float *sink) {
    double t0 = now_s();
    float acc = 0.0f;
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        const float *a[4] = { q, q, q, q };
        const float *b[4] = { kv + (size_t)j * d, kv + (size_t)(j + 1) * d,
                              kv + (size_t)(j + 2) * d,
                              kv + (size_t)(j + 3) * d };
        float d4[4];
        apus_dot4_f32_neon(a, b, (size_t)d, d4);
        acc += d4[0] + d4[1] + d4[2] + d4[3];
    }
    for (; j < n; j++)
        acc += apus_dot_f32_scalar(q, kv + (size_t)j * d, (size_t)d);
    *sink = acc;
    return now_s() - t0;
}

/* O scalar-order dots of one x row against O weight rows (pre-M14 linear
 * row body, M=1). */
__attribute__((noinline))
static double bench_linear_scalar(const float *w, const float *x, float *out,
                                  int K, int O) {
    double t0 = now_s();
    for (int o = 0; o < O; o++)
        out[o] = apus_dot_f32_scalar(x, w + (size_t)o * K, (size_t)K);
    return now_s() - t0;
}

__attribute__((noinline))
static double bench_linear_dot4(const float *w, const float *x, float *out,
                                int K, int O) {
    double t0 = now_s();
    int o = 0;
    for (; o + 4 <= O; o += 4) {
        const float *a[4] = { x, x, x, x };
        const float *b[4] = { w + (size_t)o * K, w + (size_t)(o + 1) * K,
                              w + (size_t)(o + 2) * K,
                              w + (size_t)(o + 3) * K };
        float d4[4];
        apus_dot4_f32_neon(a, b, (size_t)K, d4);
        out[o] = d4[0]; out[o + 1] = d4[1];
        out[o + 2] = d4[2]; out[o + 3] = d4[3];
    }
    for (; o < O; o++)
        out[o] = apus_dot_f32_scalar(x, w + (size_t)o * K, (size_t)K);
    return now_s() - t0;
}

int main(void) {
    float sink = 0.0f;

    /* 1. q.k dots, d=512, n=640 (one decode head's index set) */
    {
        int d = 512, n = 640;
        float *q = malloc((size_t)d * sizeof(float));
        float *kv = malloc((size_t)n * d * sizeof(float));
        for (int i = 0; i < d; i++) q[i] = rng_f32();
        for (int64_t i = 0; i < (int64_t)n * d; i++) kv[i] = rng_f32();
        double one_s = bench_dots_scalar(q, kv, n, d, &sink);
        double one_n = bench_dots_dot4(q, kv, n, d, &sink);
        size_t rs = calib_reps(one_s), rn = calib_reps(one_n);
        double ts = 0, tn = 0;
        for (size_t r = 0; r < rs; r++) ts += bench_dots_scalar(q, kv, n, d, &sink);
        for (size_t r = 0; r < rn; r++) tn += bench_dots_dot4(q, kv, n, d, &sink);
        printf("qk dots d=512 n=640:  scalar %8.1f us | dot4 %8.1f us | "
               "%.2fx\n", 1e6 * ts / rs, 1e6 * tn / rn, ts / rs / (tn / rn));
        free(q); free(kv);
    }

    /* 2. decode linear rows: wkv-style 4096->2048 and gate 4096->256 */
    {
        static const int shapes[][2] = { { 4096, 2048 }, { 4096, 256 },
                                         { 4096, 64 } };
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
            int K = shapes[i][0], O = shapes[i][1];
            float *w = malloc((size_t)O * K * sizeof(float));
            float *x = malloc((size_t)K * sizeof(float));
            float *out = malloc((size_t)O * sizeof(float));
            for (int64_t j = 0; j < (int64_t)O * K; j++) w[j] = rng_f32();
            for (int j = 0; j < K; j++) x[j] = rng_f32();
            double one_s = bench_linear_scalar(w, x, out, K, O);
            double one_n = bench_linear_dot4(w, x, out, K, O);
            size_t rs = calib_reps(one_s), rn = calib_reps(one_n);
            double ts = 0, tn = 0;
            for (size_t r = 0; r < rs; r++) ts += bench_linear_scalar(w, x, out, K, O);
            for (size_t r = 0; r < rn; r++) tn += bench_linear_dot4(w, x, out, K, O);
            printf("linear K=%d O=%-4d: scalar %8.1f us | dot4 %8.1f us | "
                   "%.2fx\n", K, O, 1e6 * ts / rs, 1e6 * tn / rn,
                   ts / rs / (tn / rn));
            free(w); free(x); free(out);
        }
    }

    fprintf(stderr, "(sink %g)\n", sink);
    return 0;
}
#endif
