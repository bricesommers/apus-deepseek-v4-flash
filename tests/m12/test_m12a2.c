/*
 * tests/m12/test_m12a2.c — hard gates for the M12a-2 AVX2 x86 kernels
 * (c/x86.h + the dispatch sites in c/fp4.h, c/fp8.h, c/attn.h, c/model.h).
 *
 * THE CONTRACT: every AVX2 kernel is BITWISE IDENTICAL to the normative
 * scalar kernel it replaces (c/x86.h: exact expansion, staged exact
 * products, scalar sequential summation order, no FMA). This suite pins
 * that bitwise identity on every path, plus the tolerance-class gates
 * (FP64 truth, esc metric) the scalar anchor itself is held to, plus the
 * thread-count-independence digest (diffed across APUS_THREADS=1/4/8 by
 * the Makefile), plus the "AVX2 path was taken HERE" probe.
 *
 *   1. PROBE: report cpu support; after the battery the AVX2 hit counter
 *      must be > 0 when this CPU has AVX2 (and APUS_X86_DISABLE is unset).
 *   2. EXHAUSTIVE E4M3 EXPAND: all 256 codes x both AVX2 expand variants
 *      (F16C and integer) == apus_e4m3_dequant_f32 bitwise.
 *   3. FP8/FP4 GEMV/GEMM/mt/grouped: AVX2 == scalar BITWISE on a shape
 *      sweep (incl. partial blocks, odd O, odd tails, the real shapes
 *      32768x1024 / 2048x4096 / 4096x2048) x a UE8M0 scale-byte sweep
 *      (incl. byte 0 = 2^-127 subnormal); all 256 E4M3 codes covered.
 *   4. FP64 TRUTH: err/esc < 2e-5 (the m3/m4a tolerance class).
 *   5. M-INDEPENDENCE: a row's value at M=1..5 is bitwise identical.
 *   6. woa / head_gemv (F32+BF16) / f32_linear / bf16_linear /
 *      sparse_attn: dispatched kernels == local verbatim scalar
 *      references BITWISE.
 *
 * Off x86-64 (APUS_X86 == 0) the suite compiles to a trivial pass — the
 * macOS battery includes it to keep the target list platform-uniform.
 *
 * Run from the repository root.
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

/* ---- digest state (FNV-1a; diffed across APUS_THREADS) ---- */
static uint64_t fnv = 0xcbf29ce484222325ull;

/* =========================================================================*/
#if APUS_X86

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0xC2B2AE3D27D4EB4Full;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-2, 2) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
}

static void digest(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { fnv ^= b[i]; fnv *= 0x100000001b3ull; }
}

static long bitdiff(const float *a, const float *b, size_t n) {
    long bad = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) bad++;
    }
    return bad;
}

/* ---- 1. exhaustive E4M3 expand, both variants ---------------------------*/
static void test_expand(void) {
    uint8_t codes[16];
    float of[16], oi[16];
    long badf = 0, badi = 0;
    for (int base = 0; base < 256; base += 16) {
        for (int i = 0; i < 16; i++) codes[i] = (uint8_t)(base + i);
        apus_e4m3_expand16_f16c_x86(codes, of);
        apus_e4m3_expand16_int_x86(codes, oi);
        for (int i = 0; i < 16; i++) {
            float s = apus_e4m3_dequant_f32(codes[i]);
            uint32_t bs, bf, bi;
            memcpy(&bs, &s, 4);
            memcpy(&bf, &of[i], 4);
            memcpy(&bi, &oi[i], 4);
            if (bs != bf) badf++;
            if (bs != bi) badi++;
        }
    }
    CHECK(badf == 0, "E4M3 expand F16C: %ld/256 codes differ from scalar", badf);
    CHECK(badi == 0, "E4M3 expand int: %ld/256 codes differ from scalar", badi);
    /* FP4 nibble expand: all 256 packed bytes at both byte positions vs
     * the scalar LUT construction (bitwise). */
    long bad4 = 0;
    for (int byte = 0; byte < 256; byte++) {
        uint8_t pk[16] = {0};
        float got[32];
        for (int pos = 0; pos < 16; pos++) {
            memset(pk, 0, sizeof pk);
            pk[pos] = (uint8_t)byte;
            apus_fp4_expand32_x86(pk, got);
            for (int i = 0; i < 16; i++) {
                float e0 = apus_fp4_lut[pk[i] & 0x0F];
                float e1 = apus_fp4_lut[pk[i] >> 4];
                uint32_t g0, g1, r0, r1;
                memcpy(&g0, &got[2 * i], 4);     memcpy(&r0, &e0, 4);
                memcpy(&g1, &got[2 * i + 1], 4); memcpy(&r1, &e1, 4);
                if (g0 != r0 || g1 != r1) bad4++;
            }
        }
    }
    CHECK(bad4 == 0, "FP4 expand: %ld mismatches vs scalar LUT", bad4);
}

/* ---- 2. fp8 GEMV/GEMM/mt bitwise + FP64 truth ---------------------------*/

static void fp8_fill(uint8_t *w, uint8_t *ws, uint8_t *acodes, float *as,
                     size_t O, size_t K, int cover_all, int scale_byte) {
    size_t nb = apus_fp8_blocks(K);
    size_t nws = ((O + APUS_FP8_GROUP - 1) / APUS_FP8_GROUP) * nb;
    for (size_t i = 0; i < O * K; i++)
        w[i] = cover_all && i < 256 ? (uint8_t)i : (uint8_t)rng_u64();
    for (size_t i = 0; i < nws; i++)
        ws[i] = scale_byte >= 0 ? (uint8_t)scale_byte
                                : (uint8_t)(64 + rng_u64() % 128);
    for (size_t i = 0; i < K; i++)
        acodes[i] = cover_all && i < 256 ? (uint8_t)(255 - i)
                                         : (uint8_t)rng_u64();
    for (size_t i = 0; i < nb; i++)
        as[i] = ldexpf(1.0f, (int)(rng_u64() % 41) - 20);
}

static double fp8_err_esc(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          const float *out, size_t O, size_t K, double *escm) {
    size_t nb = apus_fp8_blocks(K);
    double worst = 0.0;
    *escm = 0.0;
    for (size_t o = 0; o < O; o++) {
        double ref = 0.0, esc = 0.0;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * APUS_FP8_GROUP, hi = lo + APUS_FP8_GROUP;
            if (hi > K) hi = K;
            double dot = 0.0;
            for (size_t i = lo; i < hi; i++)
                dot += (double)apus_e4m3_dequant_f32(acodes[i])
                     * (double)apus_e4m3_dequant_f32(w[o * K + i]);
            double sc = (double)as[kb]
                      * (double)apus_ue8m0_f32(ws[(o / APUS_FP8_GROUP) * nb + kb]);
            ref += dot * sc;
            esc += fabs(dot * sc);
        }
        double err = fabs((double)out[o] - ref);
        if (esc > 0 && err / esc > worst) worst = err / esc;
        if (esc > *escm) *escm = esc;
    }
    return worst;
}

static void test_fp8_shape(size_t O, size_t K, int cover_all,
                           int scale_byte, int do_fp64) {
    size_t nb = apus_fp8_blocks(K);
    uint8_t *w = malloc(O * K);
    uint8_t *ws = malloc(((O + 127) / 128) * nb);
    uint8_t *acodes = malloc(5 * K);
    float *as = malloc(5 * nb * sizeof(float));
    float *scratch = malloc(5 * K * sizeof(float));
    float *o_ref = malloc(5 * O * sizeof(float));
    float *o_got = malloc(5 * O * sizeof(float));
    fp8_fill(w, ws, acodes, as, O, K, cover_all, scale_byte);
    for (size_t m = 1; m < 5; m++) {   /* extra act rows for the GEMM */
        for (size_t i = 0; i < K; i++) acodes[m * K + i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < nb; i++)
            as[m * nb + i] = ldexpf(1.0f, (int)(rng_u64() % 41) - 20);
    }
    /* GEMV: AVX2 == scalar bitwise */
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, o_ref, O, K);
    apus_fp8_gemv_avx2(w, ws, acodes, as, scratch, o_got, O, K);
    long bad = bitdiff(o_ref, o_got, O);
    CHECK(bad == 0, "fp8 gemv O=%zu K=%zu sb=%d: %ld/%zu outputs differ",
          O, K, scale_byte, bad, O);
    if (O * K > (size_t)16 << 20) {
        /* light mode for the 32 MB weight stream (emulation cost): GEMV
         * (above) + one threaded M=1 GEMM; the M sweep and M-independence
         * are covered by the smaller shapes. */
        memset(o_got, 0, O * sizeof(float));
        apus_fp8_gemm_mt(w, ws, acodes, as, scratch, o_got, 1, O, K);
        bad = bitdiff(o_ref, o_got, O);
        CHECK(bad == 0, "fp8 gemm_mt M=1 O=%zu K=%zu: %ld differ", O, K, bad);
        digest(o_got, O * sizeof(float));
        free(w); free(ws); free(acodes); free(as);
        free(scratch); free(o_ref); free(o_got);
        return;
    }
    /* GEMM at several M: AVX2 == scalar bitwise */
    for (size_t M = 1; M <= 5; M += (M == 3 ? 2 : 1)) {   /* 1,2,3,5 */
        apus_fp8_gemm_scalar(w, ws, acodes, as, scratch, o_ref, M, O, K);
        apus_fp8_gemm_avx2(w, ws, acodes, as, scratch, o_got, M, O, K);
        bad = bitdiff(o_ref, o_got, M * O);
        CHECK(bad == 0, "fp8 gemm M=%zu O=%zu K=%zu: %ld differ", M, O, K, bad);
        /* mt (dispatched) == scalar bitwise */
        memset(o_got, 0, M * O * sizeof(float));
        apus_fp8_gemm_mt(w, ws, acodes, as, scratch, o_got, M, O, K);
        bad = bitdiff(o_ref, o_got, M * O);
        CHECK(bad == 0, "fp8 gemm_mt M=%zu O=%zu K=%zu: %ld differ",
              M, O, K, bad);
    }
    /* M-independence: row 3 of the M=5 run == the M=1 run of that row */
    apus_fp8_gemm_avx2(w, ws, acodes + 3 * K, as + 3 * nb, scratch,
                       o_got, 1, O, K);
    apus_fp8_gemm_avx2(w, ws, acodes, as, scratch, o_ref, 5, O, K);
    bad = bitdiff(o_ref + 3 * O, o_got, O);
    CHECK(bad == 0, "fp8 M-independence O=%zu K=%zu: %ld differ", O, K, bad);
    if (do_fp64) {
        apus_fp8_gemv_avx2(w, ws, acodes, as, scratch, o_got, O, K);
        double escm;
        double e = fp8_err_esc(w, ws, acodes, as, o_got, O, K, &escm);
        CHECK(e < 2e-5, "fp8 gemv O=%zu K=%zu err/esc %.3e >= 2e-5", O, K, e);
        printf("  fp8 O=%-6zu K=%-5zu sb=%-4d err/esc=%.2e\n",
               O, K, scale_byte, e);
    }
    digest(o_got, O * sizeof(float));
    free(w); free(ws); free(acodes); free(as);
    free(scratch); free(o_ref); free(o_got);
}

/* ---- 3. fp4 GEMV/GEMM/grouped bitwise + FP64 truth ----------------------*/

static void fp4_fill(uint8_t *w, uint8_t *ws, uint8_t *acodes, float *as,
                     size_t O, size_t K, int scale_byte) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = scale_byte >= 0 ? (uint8_t)scale_byte
                                : (uint8_t)(64 + rng_u64() % 128);
    for (size_t i = 0; i < K; i++)
        acodes[i] = i < 256 ? (uint8_t)(255 - i) : (uint8_t)rng_u64();
    for (size_t i = 0; i < nab; i++)
        as[i] = ldexpf(1.0f, (int)(rng_u64() % 41) - 20);
}

static double fp4_err_esc(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          const float *out, size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    double worst = 0.0;
    for (size_t o = 0; o < O; o++) {
        double ref = 0.0, esc = 0.0;
        for (size_t kb = 0; kb < nb; kb++) {
            const uint8_t *p = w + o * (K / 2) + kb * (APUS_FP4_GROUP / 2);
            double dot = 0.0;
            for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                dot += (double)apus_e4m3_dequant_f32(
                           acodes[kb * APUS_FP4_GROUP + 2 * i])
                     * (double)apus_fp4_lut[p[i] & 0x0F];
                dot += (double)apus_e4m3_dequant_f32(
                           acodes[kb * APUS_FP4_GROUP + 2 * i + 1])
                     * (double)apus_fp4_lut[p[i] >> 4];
            }
            double c = (dot * (double)as[kb / 4])
                     * (double)apus_ue8m0_f32(ws[o * nb + kb]);
            ref += c;
            esc += fabs(c);
        }
        double err = fabs((double)out[o] - ref);
        if (esc > 0 && err / esc > worst) worst = err / esc;
    }
    return worst;
}

static void test_fp4_shape(size_t O, size_t K, int scale_byte, int do_fp64) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2));
    uint8_t *ws = malloc(O * nb);
    uint8_t *acodes = malloc(5 * K);
    float *as = malloc(5 * nab * sizeof(float));
    float *scratch = malloc(5 * K * sizeof(float));
    float *o_ref = malloc(5 * O * sizeof(float));
    float *o_got = malloc(5 * O * sizeof(float));
    fp4_fill(w, ws, acodes, as, O, K, scale_byte);
    for (size_t m = 1; m < 5; m++) {
        for (size_t i = 0; i < K; i++) acodes[m * K + i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < nab; i++)
            as[m * nab + i] = ldexpf(1.0f, (int)(rng_u64() % 41) - 20);
    }
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, o_ref, O, K);
    apus_fp4_gemv_avx2(w, ws, acodes, as, scratch, o_got, O, K);
    long bad = bitdiff(o_ref, o_got, O);
    CHECK(bad == 0, "fp4 gemv O=%zu K=%zu sb=%d: %ld/%zu differ",
          O, K, scale_byte, bad, O);
    for (size_t M = 1; M <= 5; M += (M == 3 ? 2 : 1)) {
        apus_fp4_gemm_scalar(w, ws, acodes, as, scratch, o_ref, M, O, K);
        apus_fp4_gemm_avx2(w, ws, acodes, as, scratch, o_got, M, O, K);
        bad = bitdiff(o_ref, o_got, M * O);
        CHECK(bad == 0, "fp4 gemm M=%zu O=%zu K=%zu: %ld differ", M, O, K, bad);
        memset(o_got, 0, M * O * sizeof(float));
        apus_fp4_gemm_mt(w, ws, acodes, as, scratch, o_got, M, O, K);
        bad = bitdiff(o_ref, o_got, M * O);
        CHECK(bad == 0, "fp4 gemm_mt M=%zu O=%zu K=%zu: %ld differ",
              M, O, K, bad);
    }
    /* M-independence */
    apus_fp4_gemm_avx2(w, ws, acodes + 3 * K, as + 3 * nab, scratch,
                       o_got, 1, O, K);
    apus_fp4_gemm_avx2(w, ws, acodes, as, scratch, o_ref, 5, O, K);
    bad = bitdiff(o_ref + 3 * O, o_got, O);
    CHECK(bad == 0, "fp4 M-independence O=%zu K=%zu: %ld differ", O, K, bad);
    if (do_fp64) {
        apus_fp4_gemv_avx2(w, ws, acodes, as, scratch, o_got, O, K);
        double e = fp4_err_esc(w, ws, acodes, as, o_got, O, K);
        CHECK(e < 2e-5, "fp4 gemv O=%zu K=%zu err/esc %.3e >= 2e-5", O, K, e);
        printf("  fp4 O=%-6zu K=%-5zu sb=%-4d err/esc=%.2e\n",
               O, K, scale_byte, e);
    }
    digest(o_got, O * sizeof(float));
    free(w); free(ws); free(acodes); free(as);
    free(scratch); free(o_ref); free(o_got);
}

/* Grouped GEMM (M9e): dispatched grouped call == per-entry scalar GEMM. */
static void test_fp4_grouped(void) {
    const size_t O = 64, K = 512;
    const size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    const size_t Ms[3] = { 1, 2, 3 }, sumM = 6;
    uint8_t *w = malloc(3 * O * (K / 2));
    uint8_t *ws = malloc(3 * O * nb);
    uint8_t *acodes = malloc(sumM * K);
    float *as = malloc(sumM * nab * sizeof(float));
    float *scratch = malloc(sumM * K * sizeof(float));
    float *outs = malloc(3 * 3 * O * sizeof(float));   /* max M=3 per entry */
    float *refs = malloc(3 * 3 * O * sizeof(float));
    ApusFp4GemmEnt ents[3];
    size_t m0 = 0;
    for (int e = 0; e < 3; e++) {
        for (size_t i = 0; i < O * (K / 2); i++)
            w[e * O * (K / 2) + i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < O * nb; i++)
            ws[e * O * nb + i] = (uint8_t)(64 + rng_u64() % 128);
        ents[e].w = w + e * O * (K / 2);
        ents[e].ws = ws + e * O * nb;
        ents[e].out = outs + e * 3 * O;
        ents[e].m0 = m0;
        ents[e].M = Ms[e];
        m0 += Ms[e];
    }
    for (size_t i = 0; i < sumM * K; i++) acodes[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < sumM * nab; i++)
        as[i] = ldexpf(1.0f, (int)(rng_u64() % 41) - 20);
    memset(outs, 0, 3 * 3 * O * sizeof(float));   /* pad rows: deterministic */
    apus_fp4_gemm_mt_grouped(ents, 3, acodes, as, scratch, O, K);
    long bad = 0;
    for (int e = 0; e < 3; e++) {
        apus_fp4_gemm_scalar(ents[e].w, ents[e].ws,
                             acodes + ents[e].m0 * K,
                             as + ents[e].m0 * nab,
                             scratch, refs + e * 3 * O, ents[e].M, O, K);
        bad += bitdiff(refs + e * 3 * O, ents[e].out, ents[e].M * O);
    }
    CHECK(bad == 0, "fp4 grouped == per-entry scalar: %ld differ", bad);
    digest(outs, 3 * 3 * O * sizeof(float));
    free(w); free(ws); free(acodes); free(as);
    free(scratch); free(outs); free(refs);
}

/* ---- 4. woa_rows --------------------------------------------------------*/

/* Verbatim pre-M12a-2 scalar row body (the x86 anchor). */
static void ref_woa_rows(const ApusWoAJob *j, size_t r0, size_t r1) {
    size_t gl = (size_t)j->G * j->ol;
    for (size_t r = r0; r < r1; r++) {
        size_t t = r / gl, rr = r % gl;
        size_t g = rr / (size_t)j->ol, jj = rr % (size_t)j->ol;
        const float *og = j->o + t * (size_t)j->hd + g * (size_t)j->sub;
        const uint16_t *wr = j->wa + (g * (size_t)j->ol + jj) * (size_t)j->sub;
        float dot = 0.0f;
        for (size_t k = 0; k < (size_t)j->sub; k++)
            dot += og[k] * apus_bf16_f32(wr[k]);
        j->y[t * gl + rr] = apus_bf16_round(dot);
    }
}

static void test_woa(int s, int G, int ol, int sub) {
    int hd = G * sub;
    size_t rows = (size_t)s * G * ol;
    uint16_t *wa = malloc((size_t)G * ol * sub * sizeof(uint16_t));
    float *o = malloc((size_t)s * hd * sizeof(float));
    float *y1 = malloc(rows * sizeof(float));
    float *y2 = malloc(rows * sizeof(float));
    for (size_t i = 0; i < (size_t)G * ol * sub; i++)
        wa[i] = apus_bf16_bits(rng_float());
    for (int i = 0; i < s * hd; i++) o[i] = apus_bf16_round(rng_float());
    ApusWoAJob j1 = { wa, o, y1, s, G, ol, sub, hd };
    ApusWoAJob j2 = { wa, o, y2, s, G, ol, sub, hd };
    apus_pool_run(rows, apus_woa_rows, &j1);   /* dispatched (AVX2 here) */
    ref_woa_rows(&j2, 0, rows);
    long bad = bitdiff(y1, y2, rows);
    CHECK(bad == 0, "woa s=%d G=%d ol=%d sub=%d: %ld/%zu differ",
          s, G, ol, sub, bad, rows);
    digest(y1, rows * sizeof(float));
    free(wa); free(o); free(y1); free(y2);
}

/* ---- 5. head_gemv (F32 + BF16) ------------------------------------------*/

static void test_head(int64_t O, int64_t K, int bf16) {
    void *w = malloc((size_t)O * K * (bf16 ? 2 : 4));
    float *x = malloc((size_t)K * sizeof(float));
    float *o1 = malloc((size_t)O * sizeof(float));
    float *o2 = malloc((size_t)O * sizeof(float));
    for (int64_t i = 0; i < K; i++) x[i] = rng_float();
    ApusStTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = bf16 ? APUS_ST_BF16 : APUS_ST_F32;
    t.data = w;
    if (bf16) {
        for (int64_t i = 0; i < O * K; i++)
            ((uint16_t *)w)[i] = apus_bf16_bits(rng_float());
    } else {
        for (int64_t i = 0; i < O * K; i++)
            ((float *)w)[i] = rng_float();
    }
    apus_head_gemv(&t, x, o1, O, K);   /* dispatched */
    for (int64_t o = 0; o < O; o++) {  /* verbatim scalar reference */
        float acc = 0.0f;
        if (bf16) {
            const uint16_t *wr = (const uint16_t *)w + o * K;
            for (int64_t k = 0; k < K; k++)
                acc += apus_bf16_f32(wr[k]) * x[k];
        } else {
            const float *wr = (const float *)w + o * K;
            for (int64_t k = 0; k < K; k++)
                acc += wr[k] * x[k];
        }
        o2[o] = acc;
    }
    long bad = bitdiff(o1, o2, (size_t)O);
    CHECK(bad == 0, "head_gemv %s O=%lld K=%lld: %ld differ",
          bf16 ? "bf16" : "f32", (long long)O, (long long)K, bad);
    digest(o1, (size_t)O * sizeof(float));
    free(w); free(x); free(o1); free(o2);
}

/* ---- 6. f32_linear / bf16_linear ----------------------------------------*/

static void test_linear(int M, int K, int O, int bf16) {
    float *w = malloc((size_t)O * K * sizeof(float));
    float *x = malloc((size_t)M * K * sizeof(float));
    float *o1 = malloc((size_t)M * O * sizeof(float));
    float *o2 = malloc((size_t)M * O * sizeof(float));
    for (int64_t i = 0; i < (int64_t)O * K; i++) w[i] = rng_float();
    for (int64_t i = 0; i < (int64_t)M * K; i++) x[i] = rng_float();
    if (bf16)
        apus_bf16_linear(w, x, o1, M, K, O);
    else
        apus_f32_linear(w, x, o1, M, K, O);
    /* verbatim scalar reference (bf16_linear pre-rounds the input once) */
    for (int m = 0; m < M; m++)
        for (int o = 0; o < O; o++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float xv = bf16 ? apus_bf16_round(x[(size_t)m * K + k])
                                : x[(size_t)m * K + k];
                acc += xv * w[(size_t)o * K + k];
            }
            o2[(size_t)m * O + o] = bf16 ? apus_bf16_round(acc) : acc;
        }
    long bad = bitdiff(o1, o2, (size_t)M * O);
    CHECK(bad == 0, "%s_linear M=%d K=%d O=%d: %ld differ",
          bf16 ? "bf16" : "f32", M, K, O, bad);
    digest(o1, (size_t)M * O * sizeof(float));
    free(w); free(x); free(o1); free(o2);
}

/* ---- 7. sparse_attn ------------------------------------------------------*/

/* Verbatim scalar replica of apus_sparse_attn (the x86 anchor: q.k dots
 * sequential, P*V as mul+add, two roundings). */
static void ref_sparse_attn(const float *q, const float *sink, int h, int d,
                            const int32_t *idxs, int idxw, int s, float scale,
                            const float *kv_a, int64_t na,
                            const float *kv_b, float *o) {
    int32_t *idsf = malloc((size_t)s * idxw * sizeof(int32_t));
    int *ns = malloc((size_t)s * sizeof(int));
    for (int t = 0; t < s; t++) {
        int n = 0;
        for (int jj = 0; jj < idxw; jj++)
            if (idxs[(size_t)t * idxw + jj] >= 0)
                idsf[(size_t)t * idxw + n++] = idxs[(size_t)t * idxw + jj];
        ns[t] = n;
    }
    float *sc = malloc((size_t)idxw * sizeof(float));
    float *p = malloc((size_t)idxw * sizeof(float));
    for (int t = 0; t < s; t++) {
        for (int hh = 0; hh < h; hh++) {
            int n = ns[t];
            float *ov = o + ((size_t)t * h + hh) * d;
            if (n == 0) {
                memset(ov, 0, (size_t)d * sizeof(float));
                continue;
            }
            const int32_t *ids = idsf + (size_t)t * idxw;
            const float *qv = q + ((size_t)t * h + hh) * d;
            float mx = -INFINITY;
            for (int jj = 0; jj < n; jj++) {
                const float *kv = ids[jj] < na
                    ? kv_a + (size_t)ids[jj] * d
                    : kv_b + (size_t)(ids[jj] - na) * d;
                float dot = 0.0f;
                for (int k = 0; k < d; k++) dot += qv[k] * kv[k];
                sc[jj] = dot * scale;
                if (sc[jj] > mx) mx = sc[jj];
            }
            float sum = 0.0f;
            for (int jj = 0; jj < n; jj++) {
                p[jj] = apus_bf16_round(expf(sc[jj] - mx));
                sum += p[jj];
            }
            for (int k = 0; k < d; k++) ov[k] = 0.0f;
            for (int jj = 0; jj < n; jj++) {
                const float *kv = ids[jj] < na
                    ? kv_a + (size_t)ids[jj] * d
                    : kv_b + (size_t)(ids[jj] - na) * d;
                for (int k = 0; k < d; k++) ov[k] += p[jj] * kv[k];
            }
            float denom = sum + expf(sink[hh] - mx);
            for (int k = 0; k < d; k++)
                ov[k] = apus_bf16_round(ov[k] / denom);
        }
    }
    free(idsf); free(ns); free(sc); free(p);
}

static void test_sparse(int s, int h, int d, int idxw, int64_t na, int64_t nb2) {
    float *q = malloc((size_t)s * h * d * sizeof(float));
    float *sink = malloc((size_t)h * sizeof(float));
    int32_t *idxs = malloc((size_t)s * idxw * sizeof(int32_t));
    float *kv_a = malloc((size_t)na * d * sizeof(float));
    float *kv_b = malloc((size_t)nb2 * d * sizeof(float));
    float *o1 = malloc((size_t)s * h * d * sizeof(float));
    float *o2 = malloc((size_t)s * h * d * sizeof(float));
    for (size_t i = 0; i < (size_t)s * h * d; i++) q[i] = rng_float();
    for (int i = 0; i < h; i++) sink[i] = rng_float();
    for (size_t i = 0; i < (size_t)na * d; i++) kv_a[i] = rng_float();
    for (size_t i = 0; i < (size_t)nb2 * d; i++) kv_b[i] = rng_float();
    for (int t = 0; t < s; t++)
        for (int jj = 0; jj < idxw; jj++) {
            /* mix valid ids (both KV sources) and masked -1 entries */
            uint64_t r = rng_u64();
            idxs[(size_t)t * idxw + jj] =
                (r % 4 == 0) ? -1 : (int32_t)(r % (uint64_t)(na + nb2));
        }
    apus_sparse_attn(q, sink, h, d, idxs, idxw, s, 0.25f, kv_a, na, kv_b, o1);
    ref_sparse_attn(q, sink, h, d, idxs, idxw, s, 0.25f, kv_a, na, kv_b, o2);
    long bad = bitdiff(o1, o2, (size_t)s * h * d);
    CHECK(bad == 0, "sparse s=%d h=%d d=%d idxw=%d: %ld differ",
          s, h, d, idxw, bad);
    digest(o1, (size_t)s * h * d * sizeof(float));
    free(q); free(sink); free(idxs); free(kv_a); free(kv_b); free(o1); free(o2);
}

/* =========================================================================*/
int main(void) {
    printf("test_m12a2: AVX2 kernels vs scalar anchors (bitwise contract)\n");
    printf("  cpu: avx2=%d fma=%d f16c=%d  APUS_X86_DISABLE=%s\n",
           __builtin_cpu_supports("avx2"), __builtin_cpu_supports("fma"),
           __builtin_cpu_supports("f16c"),
           getenv("APUS_X86_DISABLE") ? "set" : "unset");
    if (!apus_x86_have_avx2()) {
        /* non-AVX2 x86-64 (or APUS_X86_DISABLE set): the AVX2 kernels must
         * not execute — the engine dispatches to scalar. The AVX2-direct
         * checks are skipped (m9a NEON-section pattern); the scalar paths
         * are the M12a-1 battery's business. */
        printf("  no AVX2 dispatch on this machine — AVX2-direct checks "
               "skipped (placeholder)\n");
        CHECK(1, "placeholder");
        printf("digest %016llx\n", (unsigned long long)fnv);
        printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    }

    test_expand();

    /* fp8: shape sweep (partial blocks, odd O, scale-row boundary, real
     * shapes) — mixed scales, FP64 truth on the moderate shapes */
    test_fp8_shape(1, 128, 1, -1, 1);
    test_fp8_shape(3, 96, 1, -1, 1);
    test_fp8_shape(17, 100, 1, -1, 1);
    test_fp8_shape(5, 1000, 0, -1, 1);
    test_fp8_shape(64, 4096, 0, -1, 1);
    test_fp8_shape(129, 1032, 0, -1, 0);
    test_fp8_shape(32768, 1024, 0, -1, 0);   /* wq_b */
    test_fp8_shape(2048, 4096, 0, -1, 0);    /* shared w1/w3 */
    test_fp8_shape(4096, 2048, 0, -1, 0);    /* shared w2 */
    /* fp8: UE8M0 scale-byte sweep (incl. byte 0 = 2^-127 subnormal) with
     * all 256 E4M3 codes in both weights and acts */
    static const int sweep[] = { 0, 1, 2, 63, 126, 127, 128, 129, 190, 254 };
    for (size_t i = 0; i < sizeof sweep / sizeof sweep[0]; i++)
        test_fp8_shape(5, 256, 1, sweep[i], 0);

    /* fp4: same structure */
    test_fp4_shape(1, 32, -1, 1);
    test_fp4_shape(3, 96, -1, 1);
    test_fp4_shape(17, 160, -1, 1);
    test_fp4_shape(5, 992, -1, 1);     /* 31 groups: 3 chunks + 7 rest */
    test_fp4_shape(33, 256, -1, 1);    /* exactly 8 groups */
    test_fp4_shape(64, 4096, -1, 1);
    test_fp4_shape(2048, 4096, -1, 0);
    test_fp4_shape(4096, 2048, -1, 0);
    for (size_t i = 0; i < sizeof sweep / sizeof sweep[0]; i++)
        test_fp4_shape(5, 256, sweep[i], 0);
    test_fp4_grouped();

    /* woa / head / linear / sparse (odd tails everywhere) */
    test_woa(2, 3, 5, 390);            /* 30 rows: 7 dot4 groups + 2 rest */
    test_woa(1, 4, 64, 512);
    test_woa(1, 2, 16, 7);             /* sub < 8: scalar tail only */
    test_head(33, 1000, 0);            /* F32, K%8=0? 1000%8=0 -> add odd */
    test_head(33, 1003, 0);            /* F32, K%8=3 */
    test_head(33, 1003, 1);            /* BF16 */
    test_head(128, 7168, 1);           /* real LM-head K */
    test_linear(3, 1003, 33, 0);
    test_linear(3, 1003, 33, 1);
    test_linear(1, 4096, 128, 1);
    test_sparse(2, 3, 132, 8, 16, 8);  /* d%8=4 tail, mixed ids, some -1 */
    test_sparse(1, 2, 576, 4, 4, 0);   /* kv_a only */

    /* the AVX2 path must actually have been taken on this machine (the
     * raw count is thread-count dependent — print only the boolean) */
    CHECK(apus_x86_avx2_hits() > 0,
          "AVX2 supported but no AVX2 kernel was dispatched (hits=%lu)",
          apus_x86_avx2_hits());
    printf("  avx2 path taken: %s\n", apus_x86_avx2_hits() > 0 ? "yes" : "no");

    printf("digest %016llx\n", (unsigned long long)fnv);
    printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

/* =========================================================================*/
#else  /* !APUS_X86: nothing to test off x86-64 (macOS uniform target) */

int main(void) {
    printf("test_m12a2: APUS_X86 == 0 on this platform — no AVX2 kernels, "
           "trivial pass\n");
    printf("digest %016llx\n", (unsigned long long)fnv);
    printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
    return 0;
}

#endif /* APUS_X86 */
