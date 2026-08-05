/*
 * tests/m3/test_fp4.c — hard-gate tests for c/fp4.h (MXFP4 kernel).
 *
 *   1. Exhaustive dequant: all 256 packed byte values x representative scale
 *      exponents; scalar vs NEON bitwise-identical; both exact vs the LUT
 *      formula evaluated in double and rounded once to f32.
 *   2. FP8-E4M3 activation quant vs numpy golden (bitwise codes + scales).
 *   3. GEMV/GEMM across shapes (incl. 2048x4096, 4096x2048, odd/small, K=96):
 *      scalar vs NEON (ulp diff reported), both vs in-test FP64 ground truth.
 *   4. Golden test vs reference semantics ported in gen_golden.py (quant rule
 *      + fp4_gemm accumulation); max error measured and asserted.
 *   5. Edge cases: K=32, all-zero blocks, all-negative, +-6 saturation.
 *
 * Run from the repository root (golden fixtures under tests/m3/golden/).
 */
#define APUS_FP4_IMPLEMENTATION
#include "fp4.h"

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

/* ---- FP64 ground truth for the normative path (mirrors fp4_gemm semantics
 *      with FP32 act codes / FP4 LUT codes, evaluated in double).
 *      Also returns, per output, the error scale
 *          esc = sum_kb |dot_kb * sa * sb|
 *      FP32 accumulation rounding is proportional to esc (not to |out|,
 *      which can be near zero after cancellation), so tolerances are
 *      expressed as fractions of esc. ---- */
static double e4m3_val(uint8_t c) { return (double)apus_e4m3_dequant_f32(c); }

static void truth_gemm_f64(const uint8_t *w, const uint8_t *ws,
                           const uint8_t *acodes, const float *as,
                           double *out, double *esc, size_t M, size_t O, size_t K) {
    size_t nb = K / 32, nab = apus_fp4_act_blocks(K);
    for (size_t m = 0; m < M; m++) {
        for (size_t o = 0; o < O; o++) {
            double total = 0.0, scale = 0.0;
            for (size_t kb = 0; kb < nb; kb++) {
                double dot = 0.0;
                const uint8_t *p = w + o * (K / 2) + kb * 16;
                for (size_t i = 0; i < 16; i++) {
                    dot += e4m3_val(acodes[m * K + kb * 32 + 2 * i]) *
                           (double)apus_fp4_lut[p[i] & 0x0F];
                    dot += e4m3_val(acodes[m * K + kb * 32 + 2 * i + 1]) *
                           (double)apus_fp4_lut[p[i] >> 4];
                }
                double term = (dot * (double)as[m * nab + kb / 4]) *
                              (double)apus_ue8m0_f32(ws[o * nb + kb]);
                total += term;
                scale += fabs(term);
            }
            out[m * O + o] = total;
            if (esc) esc[m * O + o] = scale;
        }
    }
}

static void truth_gemv_f32_f64(const uint8_t *w, const uint8_t *ws,
                               const float *x, double *out, double *esc,
                               size_t O, size_t K) {
    size_t nb = K / 32;
    for (size_t o = 0; o < O; o++) {
        double acc = 0.0, scale = 0.0;
        for (size_t kb = 0; kb < nb; kb++) {
            double s = (double)apus_ue8m0_f32(ws[o * nb + kb]);
            const uint8_t *p = w + o * (K / 2) + kb * 16;
            for (size_t i = 0; i < 16; i++) {
                double t0 = (double)x[kb * 32 + 2 * i] *
                            ((double)apus_fp4_lut[p[i] & 0x0F] * s);
                double t1 = (double)x[kb * 32 + 2 * i + 1] *
                            ((double)apus_fp4_lut[p[i] >> 4] * s);
                acc += t0 + t1;
                scale += fabs(t0) + fabs(t1);
            }
        }
        out[o] = acc;
        if (esc) esc[o] = scale;
    }
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
/* 1. exhaustive dequant */
static void test_dequant_exhaustive(void) {
    static const uint8_t scale_bytes[] = {0, 1, 2, 63, 126, 127, 128, 129, 190, 254};
    enum { NS = sizeof(scale_bytes) };
    /* one row of 32 elements = 16 bytes; each byte value placed at every
     * byte position (16 rotations) so both nibble positions are covered
     * at all offsets */
    uint8_t packed[16], scales[1];
    float out_s[32], out_n[32];
    int tested = 0;
    for (int si = 0; si < NS; si++) {
        scales[0] = scale_bytes[si];
        double sd = ldexp(1.0, (int)scales[0] - 127);
        for (int byte = 0; byte < 256; byte++) {
            for (int pos = 0; pos < 16; pos++) {
                for (int i = 0; i < 16; i++) packed[i] = 0x00;
                packed[pos] = (uint8_t)byte;
                apus_fp4_dequant_row_scalar(packed, scales, out_s, 32);
#ifdef __ARM_NEON
                apus_fp4_dequant_row_neon(packed, scales, out_n, 32);
#endif
                for (int k = 0; k < 32; k++) {
                    int nib = (k % 2 == 0) ? (packed[k / 2] & 0x0F) : (packed[k / 2] >> 4);
                    float expect = (float)((double)apus_fp4_lut[nib] * sd);
                    if (f32bits(out_s[k]) != f32bits(expect)) {
                        CHECK(0, "dequant scalar scale=%u byte=%02x k=%d: got %a want %a",
                              scales[0], byte, k, out_s[k], expect);
                    }
#ifdef __ARM_NEON
                    if (f32bits(out_n[k]) != f32bits(out_s[k])) {
                        CHECK(0, "dequant NEON!=scalar scale=%u byte=%02x k=%d: %a vs %a",
                              scales[0], byte, k, out_n[k], out_s[k]);
                    }
#endif
                }
                tested++;
            }
        }
    }
    /* scale byte 255 (2^128 -> inf): nonzero codes -> +-inf, scalar==NEON */
    scales[0] = 255;
    for (int byte = 1; byte < 256; byte++) {
        memset(packed, byte, 16);
        apus_fp4_dequant_row_scalar(packed, scales, out_s, 32);
#ifdef __ARM_NEON
        apus_fp4_dequant_row_neon(packed, scales, out_n, 32);
#endif
        for (int k = 0; k < 32; k++) {
            int nib = (k % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            if (nib & 0x7) {
                CHECK(isinf(out_s[k]), "dequant scale=255 byte=%02x k=%d: expected inf, got %a",
                      byte, k, out_s[k]);
            }
#ifdef __ARM_NEON
            CHECK(f32bits(out_n[k]) == f32bits(out_s[k]),
                  "dequant scale=255 NEON!=scalar byte=%02x k=%d", byte, k);
#endif
        }
        tested++;
    }
    if (g_verbose) printf("  dequant exhaustive: %d rows x 32 elems checked\n", tested);
    CHECK(1, "dequant exhaustive done");
}

/* block boundary: elements 31/32 use different scales */
static void test_dequant_block_boundary(void) {
    uint8_t packed[32] = {0}, scales[2] = {127, 129};
    float out[64];
    packed[15] = 0x02;   /* elements 30,31: lo=2 -> 1.0 */
    packed[16] = 0x02;   /* elements 32,33 */
    apus_fp4_dequant_row_scalar(packed, scales, out, 64);
    CHECK(out[30] == 1.0f && out[31] == 0.0f, "block boundary elem 30/31: %a %a",
          out[30], out[31]);
    CHECK(out[32] == 4.0f && out[33] == 0.0f, "block boundary elem 32/33: %a %a",
          out[32], out[33]);
}

/* =========================================================================*/
/* 2 + 4. golden fixtures */
static unsigned char *g_w_packed, *g_w_scales, *g_act_x, *g_act_codes,
    *g_act_scales, *g_w_deq, *g_out;
static size_t g_M, g_O, g_K;

static int load_golden(void) {
    size_t len;
    unsigned char *man = read_file("tests/m3/golden/manifest.txt", &len);
    if (!man) { fprintf(stderr, "golden manifest missing — run gen_golden.py\n"); return 0; }
    if (sscanf((char *)man, "M=%zu\nO=%zu\nK=%zu", &g_M, &g_O, &g_K) != 3) {
        fprintf(stderr, "bad manifest\n"); free(man); return 0;
    }
    free(man);
    size_t l1, l2, l3, l4, l5, l6, l7;
    g_w_packed   = read_file("tests/m3/golden/w_packed.bin", &l1);
    g_w_scales   = read_file("tests/m3/golden/w_scales.bin", &l2);
    g_act_x      = read_file("tests/m3/golden/act_x.bin", &l3);
    g_act_codes  = read_file("tests/m3/golden/act_codes.bin", &l4);
    g_act_scales = read_file("tests/m3/golden/act_scales.bin", &l5);
    g_w_deq      = read_file("tests/m3/golden/w_deq.bin", &l6);
    g_out        = read_file("tests/m3/golden/out.bin", &l7);
    if (!g_w_packed || !g_w_scales || !g_act_x || !g_act_codes ||
        !g_act_scales || !g_w_deq || !g_out) {
        fprintf(stderr, "golden fixtures missing\n"); return 0;
    }
    if (l1 != g_O * (g_K / 2) || l2 != g_O * (g_K / 32) ||
        l3 != g_M * g_K * 4 || l4 != g_M * g_K ||
        l5 != g_M * apus_fp4_act_blocks(g_K) * 4 || l6 != g_O * g_K * 4 ||
        l7 != g_M * g_O * 8) {
        fprintf(stderr, "golden fixture size mismatch\n"); return 0;
    }
    return 1;
}

static void test_golden_dequant(void) {
    float *row = malloc(g_K * sizeof(float));
#ifdef __ARM_NEON
    float *rown = malloc(g_K * sizeof(float));
#endif
    long bad = 0;
    for (size_t o = 0; o < g_O; o++) {
        const uint8_t *p = g_w_packed + o * (g_K / 2);
        const uint8_t *s = g_w_scales + o * (g_K / 32);
        const float *want = (const float *)g_w_deq + o * g_K;
        apus_fp4_dequant_row_scalar(p, s, row, g_K);
        for (size_t k = 0; k < g_K; k++)
            if (f32bits(row[k]) != f32bits(want[k])) bad++;
#ifdef __ARM_NEON
        apus_fp4_dequant_row_neon(p, s, rown, g_K);
        for (size_t k = 0; k < g_K; k++)
            if (f32bits(rown[k]) != f32bits(want[k])) bad++;
#endif
    }
    CHECK(bad == 0, "golden dequant: %ld mismatched floats (want bitwise 0)", bad);
#ifdef __ARM_NEON
    free(rown);
#endif
    free(row);
}

static void test_golden_act_quant(void) {
    uint8_t *codes = malloc(g_M * g_K);
    float *scales = malloc(g_M * apus_fp4_act_blocks(g_K) * sizeof(float));
    long badc = 0, bads = 0;
    size_t nab = apus_fp4_act_blocks(g_K);
    for (size_t m = 0; m < g_M; m++) {
        apus_fp4_act_quant_scalar((const float *)g_act_x + m * g_K, g_K,
                                  codes + m * g_K, scales + m * nab);
        for (size_t k = 0; k < g_K; k++)
            if (codes[m * g_K + k] != g_act_codes[m * g_K + k]) {
                if (badc < 5)
                    fprintf(stderr, "  act code mismatch m=%zu k=%zu: got %02x want %02x (x=%a)\n",
                            m, k, codes[m * g_K + k], g_act_codes[m * g_K + k],
                            ((const float *)g_act_x)[m * g_K + k]);
                badc++;
            }
        for (size_t b = 0; b < nab; b++)
            if (f32bits(scales[m * nab + b]) !=
                f32bits(((const float *)g_act_scales)[m * nab + b])) bads++;
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

    apus_fp4_gemm_scalar(g_w_packed, g_w_scales, g_act_codes,
                         (const float *)g_act_scales, scratch, out_s,
                         g_M, g_O, g_K);
#ifdef __ARM_NEON
    apus_fp4_gemm_neon(g_w_packed, g_w_scales, g_act_codes,
                       (const float *)g_act_scales, scratch, out_n,
                       g_M, g_O, g_K);
#endif
    double max_abs = 0, max_rel = 0;
    int max_ulp = 0;
    for (size_t i = 0; i < g_M * g_O; i++) {
        double d = fabs((double)out_s[i] - truth[i]);
        double r = d / (fabs(truth[i]) > 1.0 ? fabs(truth[i]) : 1.0);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
#ifdef __ARM_NEON
        int u = ulp_diff(out_s[i], out_n[i]);
        if (u > max_ulp) max_ulp = u;
        double dn = fabs((double)out_n[i] - truth[i]);
        double rn = dn / (fabs(truth[i]) > 1.0 ? fabs(truth[i]) : 1.0);
        if (dn > max_abs) max_abs = dn;
        if (rn > max_rel) max_rel = rn;
#endif
    }
    printf("  golden gemm (reference semantics, M=%zu O=%zu K=%zu): "
           "max_abs=%.3g max_rel=%.3g scalar-vs-NEON max_ulp=%d\n",
           g_M, g_O, g_K, max_abs, max_rel, max_ulp);
    CHECK(max_rel < 1e-5, "golden gemm max_rel %.3g >= 1e-5", max_rel);
    /* GEMV path on row 0 must agree with GEMM row 0 bitwise (same code path
     * structure) — checked implicitly via the shape sweep; here just compare
     * GEMV scalar against golden truth too. */
    float *outv = malloc(g_O * sizeof(float));
    apus_fp4_gemv_scalar(g_w_packed, g_w_scales, g_act_codes,
                         (const float *)g_act_scales, scratch, outv, g_O, g_K);
    double mv = 0;
    for (size_t o = 0; o < g_O; o++) {
        double d = fabs((double)outv[o] - truth[o]);
        double r = d / (fabs(truth[o]) > 1.0 ? fabs(truth[o]) : 1.0);
        if (r > mv) mv = r;
    }
    printf("  golden gemv row0: max_rel=%.3g\n", mv);
    CHECK(mv < 1e-5, "golden gemv max_rel %.3g >= 1e-5", mv);
    free(outv); free(out_s); free(out_n); free(scratch);
}

/* =========================================================================*/
/* 3. shape sweep: scalar vs NEON vs FP64 truth */
/* Tolerances for the shape sweep, expressed as a fraction of the per-output
 * error scale esc = sum of |block contributions| (see truth_gemm_f64).
 * FP32 accumulation of n terms costs at most ~n*eps (eps = 2^-24 = 6.0e-8);
 * with K=4096 there are 128 blocks, so 2e-5 leaves ~2.5x headroom over the
 * linear bound. The single-accumulator f32 diagnostic path sums all K
 * products into one accumulator: 4096*eps = 2.4e-4, hence 1e-3. */
#define TOL_NORM 2e-5
#define TOL_DIAG 1e-3

static void test_shapes(void) {
    static const struct { size_t O, K; } shapes[] = {
        {1, 32}, {3, 32}, {5, 96}, {17, 160}, {64, 256},
        {128, 512}, {2048, 4096}, {4096, 2048},
    };
    static const size_t Ms[] = {1, 2, 4, 7};
    int gmax_ulp = 0;
    double gmax_rel = 0, gmax_rel32 = 0, gmax_sn = 0;

    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        size_t O = shapes[si].O, K = shapes[si].K;
        size_t nb = K / 32, nab = apus_fp4_act_blocks(K);
        uint8_t *w = malloc(O * (K / 2));
        uint8_t *ws = malloc(O * nb);
        float *x = malloc(K * sizeof(float));
        for (size_t i = 0; i < O * (K / 2); i++) w[i] = rng_byte();
        for (size_t i = 0; i < O * nb; i++) {
            /* sane exponents plus extremes; some all-zero blocks */
            uint8_t r = rng_byte();
            ws[i] = (i % 37 == 0) ? 1 : (uint8_t)(120 + (r % 15));
        }
        for (size_t o = 0; o < O; o += 11)   /* zero out a few weight blocks */
            memset(w + o * (K / 2), 0, K / 2 > 16 ? 16 : K / 2);

        for (size_t mi = 0; mi < sizeof(Ms) / sizeof(Ms[0]); mi++) {
            size_t M = Ms[mi];
            uint8_t *acodes = malloc(M * K);
            float *as = malloc(M * nab * sizeof(float));
            float *scratch = malloc(M * K * sizeof(float));
            float *out_s = malloc(M * O * sizeof(float));
            float *out_n = malloc(M * O * sizeof(float));
            double *truth = malloc(M * O * sizeof(double));
            double *esc = malloc(M * O * sizeof(double));

            for (size_t m = 0; m < M; m++) {
                for (size_t k = 0; k < K; k++) x[k] = rng_float();
                apus_fp4_act_quant_scalar(x, K, acodes + m * K, as + m * nab);
            }
            apus_fp4_gemm_scalar(w, ws, acodes, as, scratch, out_s, M, O, K);
#ifdef __ARM_NEON
            apus_fp4_gemm_neon(w, ws, acodes, as, scratch, out_n, M, O, K);
#endif
            truth_gemm_f64(w, ws, acodes, as, truth, esc, M, O, K);

            int mu = 0;
            double mr = 0, msn = 0;
            for (size_t i = 0; i < M * O; i++) {
                double t = truth[i], e = esc[i] > 1e-30 ? esc[i] : 1e-30;
                double r = fabs((double)out_s[i] - t) / e;
                if (r > mr) mr = r;
#ifdef __ARM_NEON
                double rn = fabs((double)out_n[i] - t) / e;
                if (rn > mr) mr = rn;
                double sn = fabs((double)out_s[i] - (double)out_n[i]) / e;
                if (sn > msn) msn = sn;
                /* ulp only meaningful away from cancellation; report for
                 * well-conditioned outputs (|t| >= esc/4) */
                if (fabs(t) >= 0.25 * esc[i]) {
                    int u = ulp_diff(out_s[i], out_n[i]);
                    if (u > mu) mu = u;
                }
#endif
            }
            if (mu > gmax_ulp) gmax_ulp = mu;
            if (mr > gmax_rel) gmax_rel = mr;
            if (msn > gmax_sn) gmax_sn = msn;
            if (g_verbose || O >= 2048)
                printf("  gemm O=%zu K=%zu M=%zu: err/esc=%.3g s-vs-n=%.3g ulp(s,n)=%d\n",
                       O, K, M, mr, msn, mu);
            CHECK(mr < TOL_NORM, "gemm O=%zu K=%zu M=%zu err/esc %.3g", O, K, M, mr);
            CHECK(msn < TOL_NORM, "gemm O=%zu K=%zu M=%zu scalar-vs-NEON %.3g",
                  O, K, M, msn);

            /* GEMV on row 0 agrees with GEMM row 0 within the same tolerance */
            float *outv = malloc(O * sizeof(float));
            double *truthv = malloc(O * sizeof(double));
            double *escv = malloc(O * sizeof(double));
            apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, outv, O, K);
#ifdef __ARM_NEON
            float *outvn = malloc(O * sizeof(float));
            apus_fp4_gemv_neon(w, ws, acodes, as, scratch, outvn, O, K);
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
            if (mv > gmax_rel) gmax_rel = mv;
            if (mvn > gmax_sn) gmax_sn = mvn;
            if (mvu > gmax_ulp) gmax_ulp = mvu;
            CHECK(mv < TOL_NORM, "gemv O=%zu K=%zu err/esc %.3g", O, K, mv);
            CHECK(mvn < TOL_NORM, "gemv O=%zu K=%zu scalar-vs-NEON %.3g", O, K, mvn);
            free(outv); free(truthv); free(escv);
#ifdef __ARM_NEON
            free(outvn);
#endif
            /* diagnostic FP32 path vs f64 */
            {
                float *xf = malloc(K * sizeof(float));
                for (size_t k = 0; k < K; k++) xf[k] = rng_float();
                float *od = malloc(O * sizeof(float));
                double *td = malloc(O * sizeof(double));
                double *ed = malloc(O * sizeof(double));
                apus_fp4_gemv_f32_scalar(w, ws, xf, od, O, K);
                truth_gemv_f32_f64(w, ws, xf, td, ed, O, K);
                double md = 0;
                for (size_t o = 0; o < O; o++) {
                    double e = ed[o] > 1e-30 ? ed[o] : 1e-30;
                    double r = fabs((double)od[o] - td[o]) / e;
                    if (r > md) md = r;
                }
                if (md > gmax_rel32) gmax_rel32 = md;
                CHECK(md < TOL_DIAG, "gemv_f32 O=%zu K=%zu err/esc %.3g", O, K, md);
                free(xf); free(od); free(td); free(ed);
            }
            free(acodes); free(as); free(scratch);
            free(out_s); free(out_n); free(truth); free(esc);
        }
        free(w); free(ws); free(x);
    }
    printf("  shape sweep: max err/esc vs f64 = %.3g (normative, tol %.1g), "
           "%.3g (f32 diag, tol %.1g)\n", gmax_rel, TOL_NORM, gmax_rel32, TOL_DIAG);
    printf("  shape sweep: scalar-vs-NEON max diff/esc = %.3g, "
           "max ulp on well-conditioned outputs = %d\n", gmax_sn, gmax_ulp);
}

/* =========================================================================*/
/* 5. edge cases */
static void test_edges(void) {
    /* K=32 single block, +-6 saturation codes, all-negative, zero block */
    uint8_t w[16], ws[1];
    float x[32], out[1], scratch[32];
    uint8_t acodes[32];
    float as[1];
    double truth[1];

    memset(w, 0x77, 16);       /* every element +6.0 */
    ws[0] = 127;               /* scale 1 */
    for (int i = 0; i < 32; i++) x[i] = 1.0f;
    apus_fp4_act_quant_scalar(x, 32, acodes, as);
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 32);
    truth_gemm_f64(w, ws, acodes, as, truth, NULL, 1, 1, 32);
    CHECK(fabs((double)out[0] - truth[0]) <= 1e-6 * fabs(truth[0]),
          "edge +6 sat: got %a want %a", out[0], (float)truth[0]);
    CHECK(out[0] > 0.0f, "edge +6 sat sign");

    memset(w, 0xFF, 16);       /* every element -6.0 */
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 32);
    CHECK(out[0] < 0.0f, "edge -6 sat sign: got %a", out[0]);

    memset(w, 0x00, 16);       /* all-zero block */
    ws[0] = 200;               /* huge scale on zero codes must stay 0 */
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 32);
    CHECK(out[0] == 0.0f, "edge zero block: got %a", out[0]);
#ifdef __ARM_NEON
    apus_fp4_gemv_neon(w, ws, acodes, as, scratch, out, 1, 32);
    CHECK(out[0] == 0.0f, "edge zero block NEON: got %a", out[0]);
#endif

    /* all-negative activations against mixed weights */
    for (int i = 0; i < 32; i++) x[i] = -2.5f;
    apus_fp4_act_quant_scalar(x, 32, acodes, as);
    memset(w, 0x10, 16);       /* lo=0 (0), hi=1 (0.5) */
    ws[0] = 128;               /* scale 2 */
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 32);
    truth_gemm_f64(w, ws, acodes, as, truth, NULL, 1, 1, 32);
    CHECK(fabs((double)out[0] - truth[0]) <= 1e-6 * (fabs(truth[0]) + 1.0),
          "edge all-neg act: got %a want %a", out[0], (float)truth[0]);

    /* zero activations -> amax floor path (scale from 1e-4), result 0 */
    memset(x, 0, sizeof(x));
    apus_fp4_act_quant_scalar(x, 32, acodes, as);
    apus_fp4_gemv_scalar(w, ws, acodes, as, scratch, out, 1, 32);
    CHECK(out[0] == 0.0f, "edge zero act: got %a", out[0]);
}

int main(int argc, char **argv) {
    g_verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    printf("test_fp4: MXFP4 kernel hard-gate tests\n");
#ifdef __ARM_NEON
    printf("  NEON paths: enabled\n");
#else
    printf("  NEON paths: NOT compiled (scalar only)\n");
#endif

    test_dequant_exhaustive();
    test_dequant_block_boundary();

    if (load_golden()) {
        test_golden_dequant();
        test_golden_act_quant();
        test_golden_gemm();
    } else {
        failures++;
        fprintf(stderr, "FAIL: could not load golden fixtures\n");
    }

    test_shapes();
    test_edges();

    printf("test_fp4: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
