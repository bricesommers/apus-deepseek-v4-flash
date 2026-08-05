/*
 * tests/m9a/test_m9a.c — hard gates for the M9a ILP reorder of the fp8/fp4
 * NEON GEMM inner loops (c/fp8.h, c/fp4.h).
 *
 *   1. EXACT-CONVERSION PROOF: the M9a F16-path E4M3 expand matches the
 *      scalar apus_e4m3_dequant_f32 on ALL 256 codes (bitwise).
 *   2. CANONICAL-ORDER PROOF: a scalar model of the M9a accumulation order
 *      (4 independent accumulators, fixed combine ((a0+a1)+(a2+a3)) then
 *      ((s0+s1)+(s2+s3)), fmaf everywhere the kernel fuses) reproduces the
 *      NEON gemv/gemm outputs BITWISE — fp8 (incl. partial blocks and a
 *      <16 tail) and fp4 (this also proves the FMLAL path computes exactly
 *      the documented order).
 *   3. FMLAL vs f32 anchor: apus_fp4_dot32_f16_neon bitwise ==
 *      apus_fp4_dot32_neon on random blocks (direct unit check).
 *   4. THREADING CONTRACT (m6c class): gemm_mt bitwise == the single-thread
 *      anchor kernel (NEON on ARM; the normative scalar kernels on x86,
 *      M12a-1 — the mt threaded rows mirror them step for step); the
 *      process digest is diffed across APUS_THREADS=1/4/8 by the Makefile.
 *   5. TOLERANCE CLASS: FP64 truth with the m3/m4a esc metric, err/esc < 2e-5
 *      (the accepted scalar-vs-NEON reorder class).
 *   6. M-INDEPENDENCE: a row's value at M=1..5 is bitwise identical
 *      (prefill/decode consistency through the mc==4 fast path and the
 *      per-row fallback).
 *
 * Run from the repository root.
 */
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#include "fp4.h"
#include "fp8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static long checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0xB5297A4D9E3779B9ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-2, 2) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
}

static uint64_t g_digest = 0xCBF29CE484222325ull;
static void digest_f32(const float *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, p + i, 4);
        for (int b = 0; b < 4; b++) {
            g_digest ^= (u >> (8 * b)) & 0xFFu;
            g_digest *= 0x100000001B3ull;
        }
    }
}

/* =========================================================================*/
/* 1. exhaustive E4M3 conversion: F16-path expand == scalar dequant, bitwise */
static void test_e4m3_expand_exhaustive(void) {
#ifdef __ARM_NEON
    uint8_t codes[16];
    float expect[16];
    long bad = 0;
    for (int base = 0; base < 256; base += 16) {
        for (int i = 0; i < 16; i++) {
            codes[i] = (uint8_t)(base + i);
            expect[i] = apus_e4m3_dequant_f32(codes[i]);
        }
        float32x4_t v[4];
        apus_e4m3_expand16_neon(codes, v);
        float got[16];
        for (int j = 0; j < 4; j++) vst1q_f32(got + 4 * j, v[j]);
        for (int i = 0; i < 16; i++) {
            uint32_t a, b;
            memcpy(&a, &got[i], 4);
            memcpy(&b, &expect[i], 4);
            if (a != b) {
                if (bad < 5)
                    fprintf(stderr, "  e4m3 code %02x: neon=%a scalar=%a\n",
                            codes[i], got[i], expect[i]);
                bad++;
            }
        }
    }
    CHECK(bad == 0, "e4m3 F16-path expand: %ld/256 codes not bitwise", bad);
    printf("  e4m3 expand: 256/256 codes bitwise vs scalar dequant\n");
#else
    CHECK(1, "no NEON (placeholder)");
#endif
}

/* =========================================================================*/
/* Scalar models of the M9a canonical order (the semantic anchor for the
 * bitwise proof — straight-line C, fmaf exactly where the kernel fuses).
 * ARM-only: on x86 (M12a-1) the anchor is the normative scalar kernel
 * itself (apus_fp{4,8}_gemv_scalar), which the mt threaded rows mirror
 * step for step — see the FP*_ANCHOR macros below. */
#ifdef __ARM_NEON

/* fp8 block dot: acc[j][l] over 16-chunks, combine, scalar tail, all fmaf. */
static float model_fp8_dot(const float *ap, const uint8_t *wp, size_t len) {
    float acc[4][4] = {{0}};
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 4; l++)
                acc[j][l] = fmaf(apus_e4m3_dequant_f32(wp[i + 4 * j + l]),
                                 ap[i + 4 * j + l], acc[j][l]);
    float s[4];
    for (int l = 0; l < 4; l++)
        s[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    float dot = (s[0] + s[1]) + (s[2] + s[3]);
    for (; i < len; i++)
        dot = fmaf(ap[i], apus_e4m3_dequant_f32(wp[i]), dot);
    return dot;
}

/* fp8 row: scale product first, fold fmaf (mirrors the CPU kernel's
 * contracted `total += dot * sc`). */
static void model_fp8_row(const uint8_t *wp_all, const uint8_t *ws,
                          const float *adeq, const float *as, float *out,
                          size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = wp_all + o * K;
        const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
        float total = 0.0f;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * APUS_FP8_GROUP;
            size_t hi = lo + APUS_FP8_GROUP;
            if (hi > K) hi = K;
            float dot = model_fp8_dot(adeq + lo, wp + lo, hi - lo);
            float sc = as[kb] * apus_ue8m0_f32(sp[kb]);
            total = fmaf(dot, sc, total);
        }
        out[o] = total;
    }
}

/* fp4 row: dot in the canonical order, (dot*sa)*sb in two ROUNDED steps
 * (volatile temps block any mul+add contraction — the kernel keeps the two
 * roundings with explicit vmulq/vaddq). */
static void model_fp4_row(const uint8_t *wp_all, const uint8_t *ws,
                          const float *adeq, const float *as, float *out,
                          size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = wp_all + o * (K / 2);
        const uint8_t *sp = ws + o * nb;
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t kb = 0; kb < nb; kb++) {
            const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
            const float *a = adeq + kb * APUS_FP4_GROUP;
            float c2[32];
            for (int i = 0; i < 16; i++) {
                c2[2 * i]     = apus_fp4_lut[p[i] & 0x0F] * 2.0f;
                c2[2 * i + 1] = apus_fp4_lut[p[i] >> 4] * 2.0f;
            }
            float acc[4][4] = {{0}};
            for (int j = 0; j < 4; j++)
                for (int l = 0; l < 4; l++) {
                    acc[j][l] = fmaf(c2[4 * j + l], a[4 * j + l], acc[j][l]);
                    acc[j][l] = fmaf(c2[16 + 4 * j + l], a[16 + 4 * j + l],
                                     acc[j][l]);
                }
            volatile float sa05 = 0.5f * as[kb / 4];
            volatile float sb = apus_ue8m0_f32(sp[kb]);
            for (int l = 0; l < 4; l++) {
                float s = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
                volatile float t = s * sa05;
                volatile float prod = t * sb;
                total[l] += prod;
            }
        }
        out[o] = (total[0] + total[1]) + (total[2] + total[3]);
    }
}
#endif /* __ARM_NEON */

/* Kernel under test (single-thread anchor): NEON on ARM, the normative
 * scalar kernels elsewhere (M12a-1 x86 — zero numerics change: the scalar
 * kernels are the semantic anchor the NEON kernels are pinned against). */
#ifdef __ARM_NEON
#define FP8_GEMV_ANCHOR apus_fp8_gemv_neon
#define FP8_GEMM_ANCHOR apus_fp8_gemm_neon
#define FP4_GEMV_ANCHOR apus_fp4_gemv_neon
#define FP4_GEMM_ANCHOR apus_fp4_gemm_neon
#else
#define FP8_GEMV_ANCHOR apus_fp8_gemv_scalar
#define FP8_GEMM_ANCHOR apus_fp8_gemm_scalar
#define FP4_GEMV_ANCHOR apus_fp4_gemv_scalar
#define FP4_GEMM_ANCHOR apus_fp4_gemm_scalar
#endif

/* =========================================================================*/
/* 2/4/5/6. fp8 battery: model bitwise, mt bitwise, esc bound, M-indep */
static void test_fp8_shape(size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    uint8_t *w = malloc(O * K), *ws = malloc(nbo * nb);
    for (size_t i = 0; i < O * K; i++) w[i] = (uint8_t)(rng_u64() >> 56);
    for (size_t i = 0; i < nbo * nb; i++)
        ws[i] = (uint8_t)(120 + (rng_u64() >> 62) % 12);   /* 2^-7..2^4ish */
    enum { MM = 5 };
    uint8_t *codes = malloc(MM * K);
    float *as = malloc(MM * nb * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *adeq = malloc(K * sizeof(float));
    float *scratch = malloc(MM * K * sizeof(float));
    float *o_neon = malloc(MM * O * sizeof(float));
    float *o_mt = malloc(MM * O * sizeof(float));
    float *o_model = malloc(O * sizeof(float));
    long bit_bad = 0;
    double worst_esc = 0;
    for (int m = 0; m < MM; m++) {
        for (size_t i = 0; i < K; i++) x[i] = apus_bf16_round(rng_float());
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nb);
    }
    /* anchor (bitwise) vs gemv at M=1 */
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(codes[i]);
#ifdef __ARM_NEON
    model_fp8_row(w, ws, adeq, as, o_model, O, K);
#else
    FP8_GEMV_ANCHOR(w, ws, codes, as, scratch, o_model, O, K);
#endif
    FP8_GEMV_ANCHOR(w, ws, codes, as, scratch, o_neon, O, K);
    if (memcmp(o_model, o_neon, O * sizeof(float)) != 0) {
        bit_bad++;
        for (size_t o = 0; o < O; o++)
            if (memcmp(o_model + o, o_neon + o, 4) != 0 && bit_bad < 4)
                fprintf(stderr, "  fp8 gemv row %zu: anchor=%a kernel=%a\n",
                        o, o_model[o], o_neon[o]);
    }
    CHECK(bit_bad == 0, "fp8 gemv != anchor bitwise (O=%zu K=%zu)",
          O, K);
    /* gemm at M=1..5: every row bitwise vs the anchor (M-independence) */
    for (size_t M = 1; M <= MM; M++) {
        FP8_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, M, O, K);
        for (size_t m = 0; m < M; m++) {
            for (size_t i = 0; i < K; i++)
                adeq[i] = apus_e4m3_dequant_f32(codes[m * K + i]);
#ifdef __ARM_NEON
            model_fp8_row(w, ws, adeq, as + m * nb, o_model, O, K);
#else
            FP8_GEMV_ANCHOR(w, ws, codes + m * K, as + m * nb, scratch,
                            o_model, O, K);
#endif
            CHECK(memcmp(o_model, o_neon + m * O, O * sizeof(float)) == 0,
                  "fp8 gemm row %zu != anchor at M=%zu (O=%zu K=%zu)",
                  m, M, O, K);
        }
    }
    /* mt bitwise == single-thread */
    apus_fp8_gemm_mt(w, ws, codes, as, scratch, o_mt, MM, O, K);
    FP8_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, MM, O, K);
    CHECK(memcmp(o_mt, o_neon, MM * O * sizeof(float)) == 0,
          "fp8 gemm_mt != single-thread anchor (O=%zu K=%zu)", O, K);
    /* esc bound vs FP64 truth (row 0) */
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(codes[i]);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * K;
        const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
        double t64 = 0.0, esc = 0.0;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * APUS_FP8_GROUP, hi = lo + APUS_FP8_GROUP;
            if (hi > K) hi = K;
            double dot = 0.0, adot = 0.0;
            for (size_t i = lo; i < hi; i++) {
                double p = (double)adeq[i] *
                           (double)apus_e4m3_dequant_f32(wp[i]);
                dot += p;
                adot += fabs(p);
            }
            double sc = (double)as[kb] * (double)apus_ue8m0_f32(sp[kb]);
            t64 += dot * sc;
            esc += adot * sc;
        }
        double e = fabs((double)o_neon[o] - t64);
        double r = e / (esc > 1e-30 ? esc : 1e-30);
        if (r > worst_esc) worst_esc = r;
    }
    CHECK(worst_esc < 2e-5, "fp8 err/esc %.3g >= 2e-5 (O=%zu K=%zu)",
          worst_esc, O, K);
    printf("  fp8 O=%-4zu K=%-4zu: model/gemv/gemm M=1..%d bitwise, mt "
           "bitwise, err/esc=%.3g\n", O, K, MM, worst_esc);
    digest_f32(o_mt, MM * O);
    free(w); free(ws); free(codes); free(as); free(x); free(adeq);
    free(scratch); free(o_neon); free(o_mt); free(o_model);
}

/* =========================================================================*/
/* 3. FMLAL vs f32 anchor, direct */
static void test_fp4_fmlal_anchor(void) {
#ifdef __ARM_NEON
    uint8_t packed[16];
    float af[32];
    float16_t ah[32];
    long bad = 0;
    for (int trial = 0; trial < 4096; trial++) {
        for (int i = 0; i < 16; i++) packed[i] = (uint8_t)(rng_u64() >> 56);
        for (int i = 0; i < 32; i++) {
            /* acts through the exact E4M3 grid (what the kernels see) */
            af[i] = apus_e4m3_dequant_f32((uint8_t)(rng_u64() >> 56));
            ah[i] = (float16_t)af[i];
        }
        float32x4_t wv[8];
        apus_fp4_expand32_neon(packed, wv);
        float32x4_t ref = apus_fp4_dot32_neon(af, wv);
        float16x8_t k[4];
        apus_fp4_expand32_f16_neon(packed, k);
        float32x4_t got = apus_fp4_dot32_f16_neon(ah, k);
        float rf[4], gt[4];
        vst1q_f32(rf, ref);
        vst1q_f32(gt, got);
        for (int l = 0; l < 4; l++) {
            uint32_t ua, ub;
            memcpy(&ua, &rf[l], 4);
            memcpy(&ub, &gt[l], 4);
            if (ua != ub) bad++;
        }
    }
    CHECK(bad == 0, "fp4 FMLAL dot != f32 anchor: %ld lane mismatches", bad);
    printf("  fp4 FMLAL dot32: 4096 blocks x 4 lanes bitwise == f32 anchor\n");
#else
    CHECK(1, "no NEON (placeholder)");
#endif
}

/* =========================================================================*/
/* 2/4/5/6. fp4 battery (same structure as fp8) */
static void test_fp4_shape(size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2)), *ws = malloc(O * nb);
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)(rng_u64() >> 56);
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(120 + (rng_u64() >> 62) % 12);
    enum { MM = 5 };
    uint8_t *codes = malloc(MM * K);
    float *as = malloc(MM * nab * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *adeq = malloc(K * sizeof(float));
    float *scratch = malloc(MM * K * sizeof(float));
    float *o_neon = malloc(MM * O * sizeof(float));
    float *o_mt = malloc(MM * O * sizeof(float));
    float *o_model = malloc(O * sizeof(float));
    double worst_esc = 0;
    for (int m = 0; m < MM; m++) {
        for (size_t i = 0; i < K; i++) x[i] = apus_bf16_round(rng_float());
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nab);
    }
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(codes[i]);
#ifdef __ARM_NEON
    model_fp4_row(w, ws, adeq, as, o_model, O, K);
#else
    FP4_GEMV_ANCHOR(w, ws, codes, as, scratch, o_model, O, K);
#endif
    FP4_GEMV_ANCHOR(w, ws, codes, as, scratch, o_neon, O, K);
    CHECK(memcmp(o_model, o_neon, O * sizeof(float)) == 0,
          "fp4 gemv != anchor bitwise (O=%zu K=%zu)", O, K);
    for (size_t M = 1; M <= MM; M++) {
        FP4_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, M, O, K);
        for (size_t m = 0; m < M; m++) {
            for (size_t i = 0; i < K; i++)
                adeq[i] = apus_e4m3_dequant_f32(codes[m * K + i]);
#ifdef __ARM_NEON
            model_fp4_row(w, ws, adeq, as + m * nab, o_model, O, K);
#else
            FP4_GEMV_ANCHOR(w, ws, codes + m * K, as + m * nab, scratch,
                            o_model, O, K);
#endif
            CHECK(memcmp(o_model, o_neon + m * O, O * sizeof(float)) == 0,
                  "fp4 gemm row %zu != anchor at M=%zu (O=%zu K=%zu)",
                  m, M, O, K);
        }
    }
    apus_fp4_gemm_mt(w, ws, codes, as, scratch, o_mt, MM, O, K);
    FP4_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, MM, O, K);
    CHECK(memcmp(o_mt, o_neon, MM * O * sizeof(float)) == 0,
          "fp4 gemm_mt != single-thread anchor (O=%zu K=%zu)", O, K);
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(codes[i]);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * (K / 2);
        const uint8_t *sp = ws + o * nb;
        double t64 = 0.0, esc = 0.0;
        for (size_t kb = 0; kb < nb; kb++) {
            const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
            const float *a = adeq + kb * APUS_FP4_GROUP;
            double dot = 0.0, adot = 0.0;
            for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                double p0 = (double)a[2 * i] *
                            (double)apus_fp4_lut[p[i] & 0x0F];
                double p1 = (double)a[2 * i + 1] *
                            (double)apus_fp4_lut[p[i] >> 4];
                dot += p0 + p1;
                adot += fabs(p0) + fabs(p1);
            }
            double sc = (double)as[kb / 4] * (double)apus_ue8m0_f32(sp[kb]);
            t64 += dot * sc;
            esc += adot * sc;
        }
        double e = fabs((double)o_neon[o] - t64);
        double r = e / (esc > 1e-30 ? esc : 1e-30);
        if (r > worst_esc) worst_esc = r;
    }
    CHECK(worst_esc < 2e-5, "fp4 err/esc %.3g >= 2e-5 (O=%zu K=%zu)",
          worst_esc, O, K);
    printf("  fp4 O=%-4zu K=%-4zu: model/gemv/gemm M=1..%d bitwise, mt "
           "bitwise, err/esc=%.3g\n", O, K, MM, worst_esc);
    digest_f32(o_mt, MM * O);
    free(w); free(ws); free(codes); free(as); free(x); free(adeq);
    free(scratch); free(o_neon); free(o_mt); free(o_model);
}

int main(void) {
    printf("test_m9a: M9a ILP reorder gates (canonical order, bitwise "
           "anchor, threading, tolerance class)\n");
    test_e4m3_expand_exhaustive();
    test_fp4_fmlal_anchor();
    /* full blocks, partial 64-block, partial 32-block, <16 tail, odd O */
    test_fp8_shape(128, 128);
    test_fp8_shape(192, 448);
    test_fp8_shape(96, 160);
    test_fp8_shape(64, 136);
    test_fp8_shape(100, 1024);
    test_fp4_shape(64, 64);
    test_fp4_shape(192, 256);
    test_fp4_shape(100, 128);
    test_fp4_shape(96, 4096);
    fprintf(stderr, "checks=%ld failures=%d\n", checks, failures);
    /* stdout is diffed across APUS_THREADS=1/4/8 by the Makefile */
    printf("digest=%016llx\n", (unsigned long long)g_digest);
    if (failures) {
        printf("RESULT: FAIL (%d failures)\n", failures);
        return 1;
    }
    printf("RESULT: ok (%ld checks)\n", checks);
    return 0;
}
