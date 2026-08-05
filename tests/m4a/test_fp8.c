/*
 * tests/m4a/test_fp8.c — hard-gate tests for c/fp8.h (FP8-E4M3 dense
 * blockwise-128x128 GEMV/GEMM).
 *
 *   1. E4M3 exhaustive: quant(dequant(c)) round trip over all 256 codes
 *      (NaN codes collapse to +-448, documented); NEON 16-code expand
 *      bitwise-equal to scalar dequant at every position.
 *   2. Activation quant vs numpy golden (bitwise codes + scales).
 *   3. Golden GEMM/GEMV vs reference fp8_gemm semantics (gen_golden.py,
 *      float64): error measured against the per-output error scale esc.
 *   4. Shape sweep incl. the real dense shapes ([1024,4096], [32768,1024],
 *      [512,4096], [8192,4096], [4096,8192]) plus odd/small/partial-block
 *      shapes: scalar vs NEON vs in-test FP64 ground truth.
 *   5. Edge cases: K=128 single block, zero blocks, extreme scale bytes
 *      (0 = 2^-127 subnormal, 254, 255 = inf), +-448 saturation codes,
 *      zero activations.
 *
 * Run from the repository root (golden fixtures under tests/m4a/golden/).
 */
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
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

static int g_verbose = 0;

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-4, 4) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}
static uint8_t rng_byte(void) { return (uint8_t)(rng_u64() >> 56); }

static uint32_t f32bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static int ulp_diff(float a, float b) {
    if (a == b) return 0;
    if (isnan(a) && isnan(b)) return 0;
    int32_t ia, ib;
    uint32_t ua = f32bits(a), ub = f32bits(b);
    ia = (ua & 0x80000000u) ? (int32_t)(0x80000000u - ua) : (int32_t)ua;
    ib = (ub & 0x80000000u) ? (int32_t)(0x80000000u - ub) : (int32_t)ub;
    int64_t d = (int64_t)ia - (int64_t)ib;
    if (d < 0) d = -d;
    return d > 0x7FFFFFFF ? 0x7FFFFFFF : (int)d;
}

/* ---- FP64 ground truth mirroring fp8_gemm semantics:
 *      total += dot_kb * (sa[kb] * sb[o/128, kb])   in double,
 *      esc = sum |dot*sc| per output. ---- */
static void truth_gemm_f64(const uint8_t *w, const uint8_t *ws,
                           const uint8_t *acodes, const float *as,
                           double *out, double *esc,
                           size_t M, size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    double *ad = malloc(K * sizeof(double));
    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++)
            ad[k] = (double)apus_e4m3_dequant_f32(acodes[m * K + k]);
        for (size_t o = 0; o < O; o++) {
            const uint8_t *wp = w + o * K;
            const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
            double total = 0.0, scale = 0.0;
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * APUS_FP8_GROUP;
                size_t hi = lo + APUS_FP8_GROUP;
                if (hi > K) hi = K;
                double dot = 0.0, adot = 0.0;
                for (size_t i = lo; i < hi; i++) {
                    double p = ad[i] * (double)apus_e4m3_dequant_f32(wp[i]);
                    dot += p;
                    adot += fabs(p);
                }
                double sc = (double)as[m * nb + kb] *
                            (double)apus_ue8m0_f32(sp[kb]);
                total += dot * sc;
                /* honest FP32 error scale: accumulation rounding scales
                 * with sum|products|, not |dot| (intra-block cancellation
                 * can make |dot| arbitrarily small) */
                scale += adot * sc;
            }
            out[m * O + o] = total;
            if (esc) esc[m * O + o] = scale;
        }
    }
    free(ad);
}

/* ---- file loading ---- */
static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

/* =========================================================================*/
/* 1. E4M3 exhaustive */
static void test_e4m3_exhaustive(void) {
    long rt_bad = 0, neon_bad = 0;
    for (int c = 0; c < 256; c++) {
        float v = apus_e4m3_dequant_f32((uint8_t)c);
        uint8_t q = apus_e4m3_quant_f32(v);
        /* NaN codes 0x7F/0xFF decode as +-480 here and requantize to the
         * finite max +-448 (0x7E/0xFE) — documented behavior */
        uint8_t want = (c == 0x7F) ? 0x7E : (c == 0xFF) ? 0xFE : (uint8_t)c;
        if (q != want) rt_bad++;
    }
    CHECK(rt_bad == 0, "e4m3 quant(dequant) round trip: %ld mismatches", rt_bad);

#ifdef __ARM_NEON
    /* NEON expand vs scalar dequant, all codes at all 16 positions */
    for (int c = 0; c < 256; c++) {
        for (int pos = 0; pos < 16; pos++) {
            uint8_t buf[16];
            memset(buf, 0, sizeof(buf));
            buf[pos] = (uint8_t)c;
            float32x4_t v[4];
            apus_e4m3_expand16_neon(buf, v);
            float got[16];
            vst1q_f32(got, v[0]); vst1q_f32(got + 4, v[1]);
            vst1q_f32(got + 8, v[2]); vst1q_f32(got + 12, v[3]);
            for (int i = 0; i < 16; i++) {
                float want = apus_e4m3_dequant_f32(buf[i]);
                if (f32bits(got[i]) != f32bits(want)) {
                    if (neon_bad < 5)
                        fprintf(stderr, "  expand16 mismatch c=%02x pos=%d i=%d: %a vs %a\n",
                                c, pos, i, got[i], want);
                    neon_bad++;
                }
            }
        }
    }
    CHECK(neon_bad == 0, "e4m3 NEON expand vs scalar: %ld mismatches", neon_bad);
#endif
    if (g_verbose) printf("  e4m3 exhaustive: 256 codes, round trip + expand ok\n");
}

/* =========================================================================*/
/* 2+3. golden fixtures */
static unsigned char *g_w_codes, *g_w_scales, *g_act_x, *g_act_codes,
    *g_act_scales, *g_out, *g_esc;
static size_t g_M, g_O, g_K;

static int load_golden(void) {
    size_t len;
    unsigned char *man = read_file("tests/m4a/golden/fp8_manifest.txt", &len);
    if (!man) { fprintf(stderr, "golden manifest missing — run gen_golden.py\n"); return 0; }
    if (sscanf((char *)man, "M=%zu\nO=%zu\nK=%zu", &g_M, &g_O, &g_K) != 3) {
        fprintf(stderr, "bad manifest\n"); free(man); return 0;
    }
    free(man);
    size_t l1, l2, l3, l4, l5, l6, l7;
    g_w_codes    = read_file("tests/m4a/golden/fp8_w_codes.bin", &l1);
    g_w_scales   = read_file("tests/m4a/golden/fp8_w_scales.bin", &l2);
    g_act_x      = read_file("tests/m4a/golden/fp8_act_x.bin", &l3);
    g_act_codes  = read_file("tests/m4a/golden/fp8_act_codes.bin", &l4);
    g_act_scales = read_file("tests/m4a/golden/fp8_act_scales.bin", &l5);
    g_out        = read_file("tests/m4a/golden/fp8_out.bin", &l6);
    g_esc        = read_file("tests/m4a/golden/fp8_esc.bin", &l7);
    if (!g_w_codes || !g_w_scales || !g_act_x || !g_act_codes ||
        !g_act_scales || !g_out || !g_esc) {
        fprintf(stderr, "golden fixtures missing\n"); return 0;
    }
    size_t nb = apus_fp8_blocks(g_K), oh = (g_O + 127) / 128;
    if (l1 != g_O * g_K || l2 != oh * nb || l3 != g_M * g_K * 4 ||
        l4 != g_M * g_K || l5 != g_M * nb * 4 || l6 != g_M * g_O * 8 ||
        l7 != g_M * g_O * 8) {
        fprintf(stderr, "golden fixture size mismatch\n"); return 0;
    }
    return 1;
}

static void test_golden_act_quant(void) {
    uint8_t *codes = malloc(g_M * g_K);
    float *scales = malloc(g_M * apus_fp8_blocks(g_K) * sizeof(float));
    long badc = 0, bads = 0;
    size_t nb = apus_fp8_blocks(g_K);
    for (size_t m = 0; m < g_M; m++) {
        apus_fp8_act_quant_scalar((const float *)g_act_x + m * g_K, g_K,
                                  codes + m * g_K, scales + m * nb);
        for (size_t k = 0; k < g_K; k++)
            if (codes[m * g_K + k] != g_act_codes[m * g_K + k]) badc++;
        for (size_t b = 0; b < nb; b++)
            if (f32bits(scales[m * nb + b]) !=
                f32bits(((const float *)g_act_scales)[m * nb + b])) bads++;
    }
    CHECK(badc == 0, "golden act quant: %ld/%zu code mismatches", badc, g_M * g_K);
    CHECK(bads == 0, "golden act quant: %ld scale mismatches", bads);
    free(codes); free(scales);
}

static void test_golden_gemm(void) {
    float *out_s = malloc(g_M * g_O * sizeof(float));
    float *out_n = malloc(g_M * g_O * sizeof(float));
    float *scratch = malloc(g_M * g_K * sizeof(float));
    const double *truth = (const double *)g_out;
    const double *esc = (const double *)g_esc;

    apus_fp8_gemm_scalar(g_w_codes, g_w_scales, g_act_codes,
                         (const float *)g_act_scales, scratch, out_s,
                         g_M, g_O, g_K);
#ifdef __ARM_NEON
    apus_fp8_gemm_neon(g_w_codes, g_w_scales, g_act_codes,
                       (const float *)g_act_scales, scratch, out_n,
                       g_M, g_O, g_K);
#endif
    double max_rel = 0, max_sn = 0;
    int max_ulp = 0;
    for (size_t i = 0; i < g_M * g_O; i++) {
        double e = esc[i] > 1e-30 ? esc[i] : 1e-30;
        double r = fabs((double)out_s[i] - truth[i]) / e;
        if (r > max_rel) max_rel = r;
#ifdef __ARM_NEON
        double rn = fabs((double)out_n[i] - truth[i]) / e;
        if (rn > max_rel) max_rel = rn;
        double sn = fabs((double)out_s[i] - (double)out_n[i]) / e;
        if (sn > max_sn) max_sn = sn;
        if (fabs(truth[i]) >= 0.25 * esc[i]) {
            int u = ulp_diff(out_s[i], out_n[i]);
            if (u > max_ulp) max_ulp = u;
        }
#endif
    }
    printf("  golden gemm (reference fp8_gemm semantics, M=%zu O=%zu K=%zu): "
           "max err/esc=%.3g scalar-vs-NEON=%.3g ulp=%d\n",
           g_M, g_O, g_K, max_rel, max_sn, max_ulp);
    CHECK(max_rel < 2e-5, "golden gemm err/esc %.3g >= 2e-5", max_rel);

    float *outv = malloc(g_O * sizeof(float));
    apus_fp8_gemv_scalar(g_w_codes, g_w_scales, g_act_codes,
                         (const float *)g_act_scales, scratch, outv, g_O, g_K);
    double mv = 0;
    for (size_t o = 0; o < g_O; o++) {
        double e = esc[o] > 1e-30 ? esc[o] : 1e-30;
        double r = fabs((double)outv[o] - truth[o]) / e;
        if (r > mv) mv = r;
    }
    printf("  golden gemv row0: err/esc=%.3g\n", mv);
    CHECK(mv < 2e-5, "golden gemv err/esc %.3g >= 2e-5", mv);
    free(outv); free(out_s); free(out_n); free(scratch);
}

/* =========================================================================*/
/* 4. shape sweep: scalar vs NEON vs FP64 truth.
 * Tolerance is a fraction of the per-output error scale
 * esc = sum_kb |dot*sa*sb| (FP32 accumulation error scales with the sum of
 * absolute contributions, not with |out|, which cancellation drives to 0).
 * K=8192 -> 64 block accumulations + 128-term dots: linear bound
 * ~ (64+128)*2^-24 ~= 1.1e-5, so 2e-5 leaves ~2x headroom. */
#define TOL_NORM 2e-5

static void run_shape(size_t O, size_t K, size_t M,
                      double *gmax_rel, double *gmax_sn, int *gmax_ulp) {
    size_t nb = apus_fp8_blocks(K), oh = (O + APUS_FP8_GROUP - 1) / APUS_FP8_GROUP;
    uint8_t *w = malloc(O * K);
    uint8_t *ws = malloc(oh * nb);
    float *x = malloc(K * sizeof(float));
    for (size_t i = 0; i < O * K; i++) {
        uint8_t b = rng_byte();
        w[i] = (uint8_t)(b & 0x7F) == 0x7F ? 0x7E : b;  /* avoid NaN codes */
    }
    for (size_t i = 0; i < oh * nb; i++) {
        uint8_t r = rng_byte();
        ws[i] = (i % 41 == 0) ? 1 : (uint8_t)(118 + (r % 15));
    }
    for (size_t o = 0; o < O; o += 13)   /* zero out a few weight rows */
        memset(w + o * K, 0, K < 64 ? K : 64);

    uint8_t *acodes = malloc(M * K);
    float *as = malloc(M * nb * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *out_s = malloc(M * O * sizeof(float));
    float *out_n = malloc(M * O * sizeof(float));
    double *truth = malloc(M * O * sizeof(double));
    double *esc = malloc(M * O * sizeof(double));

    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++) x[k] = rng_float();
        apus_fp8_act_quant_scalar(x, K, acodes + m * K, as + m * nb);
    }
    apus_fp8_gemm_scalar(w, ws, acodes, as, scratch, out_s, M, O, K);
#ifdef __ARM_NEON
    apus_fp8_gemm_neon(w, ws, acodes, as, scratch, out_n, M, O, K);
#endif
    truth_gemm_f64(w, ws, acodes, as, truth, esc, M, O, K);

    int mu = 0;
    double mr = 0, msn = 0;
    for (size_t i = 0; i < M * O; i++) {
        double e = esc[i] > 1e-30 ? esc[i] : 1e-30;
        double r = fabs((double)out_s[i] - truth[i]) / e;
        if (r > mr) mr = r;
#ifdef __ARM_NEON
        double rn = fabs((double)out_n[i] - truth[i]) / e;
        if (rn > mr) mr = rn;
        double sn = fabs((double)out_s[i] - (double)out_n[i]) / e;
        if (sn > msn) msn = sn;
        if (fabs(truth[i]) >= 0.25 * esc[i]) {
            int u = ulp_diff(out_s[i], out_n[i]);
            if (u > mu) mu = u;
        }
#endif
    }
    if (mr > *gmax_rel) *gmax_rel = mr;
    if (msn > *gmax_sn) *gmax_sn = msn;
    if (mu > *gmax_ulp) *gmax_ulp = mu;
    if (g_verbose || O >= 1024)
        printf("  gemm O=%zu K=%zu M=%zu: err/esc=%.3g s-vs-n=%.3g ulp=%d\n",
               O, K, M, mr, msn, mu);
    CHECK(mr < TOL_NORM, "gemm O=%zu K=%zu M=%zu err/esc %.3g", O, K, M, mr);
    CHECK(msn < TOL_NORM, "gemm O=%zu K=%zu M=%zu scalar-vs-NEON %.3g", O, K, M, msn);

    /* GEMV on row 0 */
    float *outv = malloc(O * sizeof(float));
    double *truthv = malloc(O * sizeof(double));
    double *escv = malloc(O * sizeof(double));
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, outv, O, K);
#ifdef __ARM_NEON
    float *outvn = malloc(O * sizeof(float));
    apus_fp8_gemv_neon(w, ws, acodes, as, scratch, outvn, O, K);
#endif
    truth_gemm_f64(w, ws, acodes, as, truthv, escv, 1, O, K);
    double mv = 0, mvn = 0;
    int mvu = 0;
    for (size_t o = 0; o < O; o++) {
        double e = escv[o] > 1e-30 ? escv[o] : 1e-30;
        double r = fabs((double)outv[o] - truthv[o]) / e;
        if (r > mv) mv = r;
#ifdef __ARM_NEON
        double rn = fabs((double)outvn[o] - truthv[o]) / e;
        if (rn > mv) mv = rn;
        double sn = fabs((double)outv[o] - (double)outvn[o]) / e;
        if (sn > mvn) mvn = sn;
        if (fabs(truthv[o]) >= 0.25 * escv[o]) {
            int u = ulp_diff(outv[o], outvn[o]);
            if (u > mvu) mvu = u;
        }
#endif
    }
    if (mv > *gmax_rel) *gmax_rel = mv;
    if (mvn > *gmax_sn) *gmax_sn = mvn;
    if (mvu > *gmax_ulp) *gmax_ulp = mvu;
    CHECK(mv < TOL_NORM, "gemv O=%zu K=%zu err/esc %.3g", O, K, mv);
    CHECK(mvn < TOL_NORM, "gemv O=%zu K=%zu scalar-vs-NEON %.3g", O, K, mvn);

    free(outv); free(truthv); free(escv);
#ifdef __ARM_NEON
    free(outvn);
#endif
    free(acodes); free(as); free(scratch);
    free(out_s); free(out_n); free(truth); free(esc);
    free(w); free(ws); free(x);
}

static void test_shapes(void) {
    /* real dense shapes (GEMV + small-M GEMM) */
    static const struct { size_t O, K; } real_shapes[] = {
        {1024, 4096}, {32768, 1024}, {512, 4096}, {8192, 4096}, {4096, 8192},
    };
    /* small/odd/partial-block shapes */
    static const struct { size_t O, K; } odd_shapes[] = {
        {1, 128}, {3, 128}, {5, 384}, {17, 448}, {130, 256}, {64, 384}, {200, 512},
    };
    double gmax_rel = 0, gmax_sn = 0;
    int gmax_ulp = 0;

    for (size_t i = 0; i < sizeof(real_shapes) / sizeof(real_shapes[0]); i++) {
        run_shape(real_shapes[i].O, real_shapes[i].K, 1, &gmax_rel, &gmax_sn, &gmax_ulp);
        run_shape(real_shapes[i].O, real_shapes[i].K, 2, &gmax_rel, &gmax_sn, &gmax_ulp);
    }
    for (size_t i = 0; i < sizeof(odd_shapes) / sizeof(odd_shapes[0]); i++) {
        static const size_t Ms[] = {1, 2, 5};
        for (size_t j = 0; j < sizeof(Ms) / sizeof(Ms[0]); j++)
            run_shape(odd_shapes[i].O, odd_shapes[i].K, Ms[j],
                      &gmax_rel, &gmax_sn, &gmax_ulp);
    }
    printf("  shape sweep: max err/esc vs f64 = %.3g (tol %.1g), "
           "scalar-vs-NEON = %.3g, max ulp well-conditioned = %d\n",
           gmax_rel, TOL_NORM, gmax_sn, gmax_ulp);
}

/* =========================================================================*/
/* 5. edge cases */
static void test_edges(void) {
    uint8_t w[128], ws[1];
    float x[128], out[1], scratch[128];
    uint8_t acodes[128];
    float as[1];
    double truth[1];

    /* K=128 single block, +448 saturation codes */
    memset(w, 0x7E, 128);
    ws[0] = 127;                 /* scale 1 */
    for (int i = 0; i < 128; i++) x[i] = 1.0f;
    apus_fp8_act_quant_scalar(x, 128, acodes, as);
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 128);
    truth_gemm_f64(w, ws, acodes, as, truth, NULL, 1, 1, 128);
    CHECK(fabs((double)out[0] - truth[0]) <= 1e-6 * fabs(truth[0]),
          "edge +448 sat: got %a want %a", out[0], (float)truth[0]);
    CHECK(out[0] > 0.0f, "edge +448 sat sign");

    memset(w, 0xFE, 128);        /* -448 everywhere */
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(out[0] < 0.0f, "edge -448 sat sign: got %a", out[0]);

    /* all-zero weight block with huge scale stays exactly 0 */
    memset(w, 0x00, 128);
    ws[0] = 254;
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(out[0] == 0.0f, "edge zero block: got %a", out[0]);
#ifdef __ARM_NEON
    apus_fp8_gemv_neon(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(out[0] == 0.0f, "edge zero block NEON: got %a", out[0]);
#endif

    /* scale byte 0 (2^-127 FP32 subnormal): exact tiny result */
    memset(w, 0x7E, 128);
    ws[0] = 0;
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 128);
    truth_gemm_f64(w, ws, acodes, as, truth, NULL, 1, 1, 128);
    CHECK(out[0] == (float)truth[0],
          "edge scale byte 0: got %a want %a", out[0], (float)truth[0]);
#ifdef __ARM_NEON
    {
        float outn[1];
        apus_fp8_gemv_neon(w, ws, acodes, as, scratch, outn, 1, 128);
        double e = fabs(truth[0]) > 0 ? fabs(truth[0]) : 1.0;
        CHECK(fabs((double)outn[0] - truth[0]) / e < 1e-6,
              "edge scale byte 0 NEON: got %a want %a", outn[0], (float)truth[0]);
    }
#endif

    /* scale byte 255 (2^128 -> inf): nonzero codes -> +-inf, scalar==NEON */
    ws[0] = 255;
    float outs[1];
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, outs, 1, 128);
    CHECK(isinf(outs[0]), "edge scale 255: expected inf, got %a", outs[0]);
#ifdef __ARM_NEON
    apus_fp8_gemv_neon(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(f32bits(out[0]) == f32bits(outs[0]),
          "edge scale 255 NEON!=scalar: %a vs %a", out[0], outs[0]);
#endif

    /* zero activations -> amax floor 1e-4 path, result exactly 0 */
    memset(x, 0, sizeof(x));
    ws[0] = 127;
    apus_fp8_act_quant_scalar(x, 128, acodes, as);
    apus_fp8_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(out[0] == 0.0f, "edge zero act: got %a", out[0]);
#ifdef __ARM_NEON
    apus_fp8_gemv_neon(w, ws, acodes, as, scratch, out, 1, 128);
    CHECK(out[0] == 0.0f, "edge zero act NEON: got %a", out[0]);
#endif

    /* weight scale tile boundary: rows 127/128 use different scale rows
     * (O=129 -> weight scale grid [2, 1]) */
    {
        uint8_t *w2 = malloc(129 * 128), ws2[2];
        memset(w2, 0x38, 129 * 128);   /* code 0x38 = e=7,m=0 -> 1.0 */
        ws2[0] = 127;                   /* row-tile 0: scale 1 */
        ws2[1] = 129;                   /* row-tile 1: scale 4 */
        float out2[129];
        for (int i = 0; i < 128; i++) x[i] = 1.0f;
        apus_fp8_act_quant_scalar(x, 128, acodes, as);
        apus_fp8_gemv_scalar(w2, ws2, acodes, as, scratch, out2, 129, 128);
        double t2[129];
        truth_gemm_f64(w2, ws2, acodes, as, t2, NULL, 1, 129, 128);
        CHECK(out2[127] == (float)t2[127] && out2[128] == (float)t2[128],
              "edge tile boundary rows 127/128: %a %a want %a %a",
              out2[127], out2[128], (float)t2[127], (float)t2[128]);
        CHECK(fabsf(out2[128] - 4.0f * out2[127]) <= 1e-6f * fabsf(out2[128]),
              "edge tile boundary ratio: %a vs 4*%a", out2[128], out2[127]);
        free(w2);
    }
}

int main(int argc, char **argv) {
    g_verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    printf("test_fp8: FP8-E4M3 blockwise dense kernel hard-gate tests\n");
#ifdef __ARM_NEON
    printf("  NEON paths: enabled\n");
#else
    printf("  NEON paths: NOT compiled (scalar only)\n");
#endif

    test_e4m3_exhaustive();

    if (load_golden()) {
        test_golden_act_quant();
        test_golden_gemm();
    } else {
        failures++;
        fprintf(stderr, "FAIL: could not load golden fixtures\n");
    }

    test_shapes();
    test_edges();

    printf("test_fp8: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
