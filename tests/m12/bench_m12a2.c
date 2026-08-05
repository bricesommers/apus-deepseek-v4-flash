/*
 * tests/m12/bench_m12a2.c — M12a-2 microbench: AVX2 vs scalar at the real
 * model shapes. NOT a gate. Under linux/amd64 emulation (Rosetta on the
 * M1 host) the absolute numbers are indicative only — the real x86
 * numbers are M12c's. The scalar column is the M12a-1 fallback (the
 * public scalar kernels for fp8/fp4; verbatim local replicas for
 * woa/head/linear, whose dispatch is inside the row workers).
 *
 * Run from the repository root (`make bench-m12a2`).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_MHC_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION

#include "model.h"
#include "sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define REPS 3

int main(void) {
    printf("bench_m12a2: AVX2 vs scalar (linux/amd64 EMULATED — indicative "
           "only)\n");
    printf("  cpu: avx2=%d fma=%d f16c=%d\n",
           __builtin_cpu_supports("avx2"), __builtin_cpu_supports("fma"),
           __builtin_cpu_supports("f16c"));
    if (!apus_x86_have_avx2()) {
        printf("  no AVX2 on this CPU — nothing to bench\n");
        return 0;
    }
    printf("%-26s %6s %12s %12s %8s\n", "kernel", "M", "scalar", "avx2",
           "speedup");

    /* --- fp8 / fp4 GEMV + GEMM at the real shapes --- */
    struct { size_t O, K; } shapes[] = {
        { 32768, 1024 },   /* wq_b (fp8) */
        { 2048, 4096 },    /* shared/expert w1,w3 */
        { 4096, 2048 },    /* shared/expert w2 */
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s].O, K = shapes[s].K;
        size_t nb8 = apus_fp8_blocks(K), nb4 = K / APUS_FP4_GROUP;
        size_t nab = apus_fp4_act_blocks(K);
        size_t M = 4;
        uint8_t *w8 = malloc(O * K), *ws8 = malloc(((O + 127) / 128) * nb8);
        uint8_t *w4 = malloc(O * (K / 2)), *ws4 = malloc(O * nb4);
        uint8_t *acodes = malloc(M * K);
        float *as = malloc(M * nab * sizeof(float));
        float *scratch = malloc(M * K * sizeof(float));
        float *out = malloc(M * O * sizeof(float));
        for (size_t i = 0; i < O * K; i++) w8[i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < ((O + 127) / 128) * nb8; i++)
            ws8[i] = (uint8_t)(96 + rng_u64() % 64);
        for (size_t i = 0; i < O * (K / 2); i++) w4[i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < O * nb4; i++)
            ws4[i] = (uint8_t)(96 + rng_u64() % 64);
        for (size_t i = 0; i < M * K; i++) acodes[i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < M * nab; i++)
            as[i] = 0x1p-4f;

        double ts = 1e30, ta = 1e30, t0;
        char name[64];

        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_fp8_gemv_scalar(w8, ws8, acodes, as, scratch, out, O, K);
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_fp8_gemv_avx2(w8, ws8, acodes, as, scratch, out, O, K);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        snprintf(name, sizeof name, "fp8 gemv %zux%zu", O, K);
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n", name, 1,
               ts * 1e3, ta * 1e3, ts / ta);

        if (O * K <= ((size_t)8 << 20)) {   /* keep the emulated run sane */
            ts = ta = 1e30;
            for (int r = 0; r < REPS; r++) {
                t0 = now_s();
                apus_fp8_gemm_scalar(w8, ws8, acodes, as, scratch, out,
                                     M, O, K);
                if (now_s() - t0 < ts) ts = now_s() - t0;
            }
            for (int r = 0; r < REPS; r++) {
                t0 = now_s();
                apus_fp8_gemm_avx2(w8, ws8, acodes, as, scratch, out,
                                   M, O, K);
                if (now_s() - t0 < ta) ta = now_s() - t0;
            }
            snprintf(name, sizeof name, "fp8 gemm %zux%zu", O, K);
            printf("%-26s %6zu %10.2f ms %10.2f ms %7.2fx\n", name, M,
                   ts * 1e3, ta * 1e3, ts / ta);
        }

        ts = ta = 1e30;
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_fp4_gemv_scalar(w4, ws4, acodes, as, scratch, out, O, K);
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_fp4_gemv_avx2(w4, ws4, acodes, as, scratch, out, O, K);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        snprintf(name, sizeof name, "fp4 gemv %zux%zu", O, K);
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n", name, 1,
               ts * 1e3, ta * 1e3, ts / ta);

        if (O * K <= ((size_t)8 << 20)) {
            ts = ta = 1e30;
            for (int r = 0; r < REPS; r++) {
                t0 = now_s();
                apus_fp4_gemm_scalar(w4, ws4, acodes, as, scratch, out,
                                     M, O, K);
                if (now_s() - t0 < ts) ts = now_s() - t0;
            }
            for (int r = 0; r < REPS; r++) {
                t0 = now_s();
                apus_fp4_gemm_avx2(w4, ws4, acodes, as, scratch, out,
                                   M, O, K);
                if (now_s() - t0 < ta) ta = now_s() - t0;
            }
            snprintf(name, sizeof name, "fp4 gemm %zux%zu", O, K);
            printf("%-26s %6zu %10.2f ms %10.2f ms %7.2fx\n", name, M,
                   ts * 1e3, ta * 1e3, ts / ta);
        }
        free(w8); free(ws8); free(w4); free(ws4);
        free(acodes); free(as); free(scratch); free(out);
    }

    /* --- woa rows (dispatched) vs verbatim scalar --- */
    {
        int s = 1, G = 8, ol = 128, sub = 512, hd = G * sub;
        size_t rows = (size_t)s * G * ol;
        uint16_t *wa = malloc((size_t)G * ol * sub * sizeof(uint16_t));
        float *o = malloc((size_t)s * hd * sizeof(float));
        float *y = malloc(rows * sizeof(float));
        for (size_t i = 0; i < (size_t)G * ol * sub; i++)
            wa[i] = apus_bf16_bits(rng_float());
        for (int i = 0; i < s * hd; i++) o[i] = apus_bf16_round(rng_float());
        ApusWoAJob job = { wa, o, y, s, G, ol, sub, hd };
        double ts = 1e30, ta = 1e30, t0;
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            /* verbatim pre-M12a-2 scalar body */
            size_t gl = (size_t)G * ol;
            for (size_t rr = 0; rr < rows; rr++) {
                size_t t = rr / gl, rem = rr % gl;
                size_t g = rem / (size_t)ol, jj = rem % (size_t)ol;
                const float *og = o + t * (size_t)hd + g * (size_t)sub;
                const uint16_t *wr = wa + (g * (size_t)ol + jj) * (size_t)sub;
                float dot = 0.0f;
                for (size_t k = 0; k < (size_t)sub; k++)
                    dot += og[k] * apus_bf16_f32(wr[k]);
                y[rr] = apus_bf16_round(dot);
            }
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_woa_rows(&job, 0, rows);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n",
               "woa rows G*ol=1024 sub=512", 1, ts * 1e3, ta * 1e3, ts / ta);
        free(wa); free(o); free(y);
    }

    /* --- head gemv (BF16, dispatched) vs verbatim scalar --- */
    {
        int64_t O = 4096, K = 7168;
        uint16_t *w = malloc((size_t)O * K * sizeof(uint16_t));
        float *x = malloc((size_t)K * sizeof(float));
        float *out = malloc((size_t)O * sizeof(float));
        for (int64_t i = 0; i < O * K; i++) w[i] = apus_bf16_bits(rng_float());
        for (int64_t i = 0; i < K; i++) x[i] = rng_float();
        ApusStTensor t;
        memset(&t, 0, sizeof t);
        t.dtype = APUS_ST_BF16;
        t.data = w;
        double ts = 1e30, ta = 1e30, t0;
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            for (int64_t o = 0; o < O; o++) {
                const uint16_t *wr = w + o * K;
                float acc = 0.0f;
                for (int64_t k = 0; k < K; k++)
                    acc += apus_bf16_f32(wr[k]) * x[k];
                out[o] = acc;
            }
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_head_gemv(&t, x, out, O, K);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n",
               "head gemv bf16 4096x7168", 1, ts * 1e3, ta * 1e3, ts / ta);
        free(w); free(x); free(out);
    }

    /* --- f32/bf16 linear (dispatched) vs verbatim scalar --- */
    {
        int M = 1, K = 7168, O = 2048;
        float *w = malloc((size_t)O * K * sizeof(float));
        float *x = malloc((size_t)M * K * sizeof(float));
        float *out = malloc((size_t)M * O * sizeof(float));
        for (int64_t i = 0; i < (int64_t)O * K; i++) w[i] = rng_float();
        for (int64_t i = 0; i < (int64_t)M * K; i++) x[i] = rng_float();
        double ts = 1e30, ta = 1e30, t0;
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            for (int o = 0; o < O; o++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += x[k] * w[(size_t)o * K + k];
                out[o] = acc;
            }
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_f32_linear(w, x, out, M, K, O);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n",
               "f32 linear 2048x7168", M, ts * 1e3, ta * 1e3, ts / ta);
        ts = ta = 1e30;
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            for (int o = 0; o < O; o++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += apus_bf16_round(x[k]) * w[(size_t)o * K + k];
                out[o] = apus_bf16_round(acc);
            }
            if (now_s() - t0 < ts) ts = now_s() - t0;
        }
        for (int r = 0; r < REPS; r++) {
            t0 = now_s();
            apus_bf16_linear(w, x, out, M, K, O);
            if (now_s() - t0 < ta) ta = now_s() - t0;
        }
        printf("%-26s %6d %10.2f ms %10.2f ms %7.2fx\n",
               "bf16 linear 2048x7168", M, ts * 1e3, ta * 1e3, ts / ta);
        free(w); free(x); free(out);
    }
    return 0;
}
