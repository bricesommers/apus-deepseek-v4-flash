/*
 * tests/m3/bench_fp4.c — informational GEMV benchmark for c/fp4.h.
 * Reports GB/s (packed weight + scale bytes streamed per call) and GFLOP/s
 * (2*O*K MACs) at the real expert shapes, scalar vs NEON. Single-threaded.
 */
#define APUS_FP4_IMPLEMENTATION
#include "fp4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng_state = 0x123456789abcdef0ull;
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

typedef void (*gemv_fn)(const uint8_t *, const uint8_t *, const uint8_t *,
                        const float *, float *, float *, size_t, size_t);

static void bench_shape(const char *name, gemv_fn fn,
                        const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K) {
    /* calibrate to ~0.5 s of measured time per entry */
    fn(w, ws, acodes, as, scratch, out, O, K);  /* warmup */
    double t0 = now_s();
    fn(w, ws, acodes, as, scratch, out, O, K);
    double t1 = now_s();
    size_t reps = (size_t)(0.5 / (t1 - t0 > 1e-9 ? t1 - t0 : 1e-9));
    if (reps < 3) reps = 3;
    if (reps > 2000) reps = 2000;

    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        double a = now_s();
        for (size_t r = 0; r < reps; r++)
            fn(w, ws, acodes, as, scratch, out, O, K);
        double b = now_s();
        double per = (b - a) / (double)reps;
        if (per < best) best = per;
    }
    double bytes = (double)O * ((double)K / 2.0 + (double)K / 32.0);
    double flops = 2.0 * (double)O * (double)K;
    printf("  %-24s O=%4zu K=%4zu  %8.3f us  %7.2f GB/s  %7.2f GFLOP/s\n",
           name, O, K, best * 1e6, bytes / best / 1e9, flops / best / 1e9);
}

int main(void) {
    static const struct { size_t O, K; } shapes[] = {
        {2048, 4096},   /* expert w1/w3 */
        {4096, 2048},   /* expert w2 */
    };
    printf("bench_fp4: GEMV decode-path benchmark (M=1, single thread)\n");
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        size_t O = shapes[s].O, K = shapes[s].K;
        size_t nb = K / 32, nab = apus_fp4_act_blocks(K);
        uint8_t *w = malloc(O * (K / 2));
        uint8_t *ws = malloc(O * nb);
        float *x = malloc(K * sizeof(float));
        uint8_t *acodes = malloc(K);
        float *as = malloc(nab * sizeof(float));
        float *scratch = malloc(K * sizeof(float));
        float *out = malloc(O * sizeof(float));
        for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < O * nb; i++) ws[i] = (uint8_t)(120 + rng_u64() % 15);
        for (size_t i = 0; i < K; i++)
            x[i] = (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
        apus_fp4_act_quant_scalar(x, K, acodes, as);

        bench_shape("scalar", apus_fp4_gemv_scalar, w, ws, acodes, as,
                    scratch, out, O, K);
#ifdef __ARM_NEON
        bench_shape("neon", apus_fp4_gemv_neon, w, ws, acodes, as,
                    scratch, out, O, K);
#endif
        free(w); free(ws); free(x); free(acodes); free(as); free(scratch); free(out);
    }
    return 0;
}
