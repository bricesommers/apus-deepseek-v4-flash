/*
 * tests/m4a/test_mhc.c — hard-gate tests for c/mhc.h (mHC residual stream:
 * pre/post/comb generation, Sinkhorn-20, apply step, hc_head collapse).
 *
 *   1. Golden mixes/pre/post/comb/ypre/ypost/yhead vs gen_golden.py float64
 *      port of model.py hc_pre/hc_post/hc_head + kernel.py
 *      hc_split_sinkhorn (exact iteration order and eps placement).
 *   2. Sinkhorn iteration-for-iteration: all 40 normalization stages of
 *      gen_golden.py's float64 port reproduced step-by-step with the
 *      exposed step functions; full driver bitwise-equal to the stepwise
 *      sequence; doubly-stochastic check (col sums to ~1, matching golden).
 *   3. Random real-shape test (d=4096, n=4) and generic-n (n=3) test:
 *      scalar vs NEON vs in-test float64 recomputation.
 *   4. Edge cases: zero state (rsqrt(0+eps)), huge/tiny states, sigmoid
 *      saturation (base = +-30), dominant comb logits (max subtraction),
 *      uniform logits.
 *
 * Tolerances: absolute, against magnitudes documented in README.md
 * (probabilities are O(1); mixes and y are O(1..100)).
 *
 * Run from the repository root (golden fixtures under tests/m4a/golden/).
 */
#define APUS_MHC_IMPLEMENTATION
#include "mhc.h"

#include <math.h>
#include <stdint.h>
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
static uint64_t rng_state = 0xB5297A4D9E3779B9ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* approx N(0,1)-ish in (-4, 4) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}

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

static double max_abs_diff_f64(const float *got, const double *want, size_t n,
                               double scale) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]) / scale;
        if (d > m) m = d;
    }
    return m;
}

/* =========================================================================*/
/* golden fixture state */
static float *g_x4, *g_fn, *g_scale, *g_base, *g_f, *g_hfn, *g_hscale,
    *g_hbase, *g_sk_logits;
static double *g_mixes, *g_pre, *g_post, *g_comb, *g_ypre, *g_ypost,
    *g_yhead, *g_sk_stages;
static size_t g_T, g_D, g_N, g_T2, g_nstages;
static int g_iters;
static float g_norm_eps, g_hc_eps;

static int load_golden(void) {
    size_t len;
    unsigned char *man = read_file("tests/m4a/golden/mhc_manifest.txt", &len);
    if (!man) { fprintf(stderr, "mhc manifest missing — run gen_golden.py\n"); return 0; }
    double ne = 0, he = 0;
    if (sscanf((char *)man, "T=%zu\nD=%zu\nN=%zu\nT2=%zu\nITERS=%d\nNSTAGES=%zu\n"
                            "NORM_EPS=%lf\nHC_EPS=%lf",
               &g_T, &g_D, &g_N, &g_T2, &g_iters, &g_nstages, &ne, &he) != 8) {
        fprintf(stderr, "bad mhc manifest\n"); free(man); return 0;
    }
    free(man);
    g_norm_eps = (float)ne; g_hc_eps = (float)he;
    size_t n = g_N, d = g_D, t = g_T, t2 = g_T2, nmix = (2 + n) * n;
    size_t l[16];
    g_x4     = (float *)read_file("tests/m4a/golden/mhc_x4.bin", &l[0]);
    g_fn     = (float *)read_file("tests/m4a/golden/mhc_fn.bin", &l[1]);
    g_scale  = (float *)read_file("tests/m4a/golden/mhc_scale.bin", &l[2]);
    g_base   = (float *)read_file("tests/m4a/golden/mhc_base.bin", &l[3]);
    g_mixes  = (double *)read_file("tests/m4a/golden/mhc_mixes.bin", &l[4]);
    g_pre    = (double *)read_file("tests/m4a/golden/mhc_pre.bin", &l[5]);
    g_post   = (double *)read_file("tests/m4a/golden/mhc_post.bin", &l[6]);
    g_comb   = (double *)read_file("tests/m4a/golden/mhc_comb.bin", &l[7]);
    g_ypre   = (double *)read_file("tests/m4a/golden/mhc_ypre.bin", &l[8]);
    g_f      = (float *)read_file("tests/m4a/golden/mhc_f.bin", &l[9]);
    g_ypost  = (double *)read_file("tests/m4a/golden/mhc_ypost.bin", &l[10]);
    g_hfn    = (float *)read_file("tests/m4a/golden/mhc_hfn.bin", &l[11]);
    g_hscale = (float *)read_file("tests/m4a/golden/mhc_hscale.bin", &l[12]);
    g_hbase  = (float *)read_file("tests/m4a/golden/mhc_hbase.bin", &l[13]);
    g_yhead  = (double *)read_file("tests/m4a/golden/mhc_yhead.bin", &l[14]);
    g_sk_logits = (float *)read_file("tests/m4a/golden/mhc_sk_logits.bin", &l[15]);
    size_t l16;
    g_sk_stages = (double *)read_file("tests/m4a/golden/mhc_sk_stages.bin", &l16);
    if (!g_x4 || !g_fn || !g_scale || !g_base || !g_mixes || !g_pre ||
        !g_post || !g_comb || !g_ypre || !g_f || !g_ypost || !g_hfn ||
        !g_hscale || !g_hbase || !g_yhead || !g_sk_logits || !g_sk_stages) {
        fprintf(stderr, "mhc golden fixtures missing\n"); return 0;
    }
    if (l[0] != t * n * d * 4 || l[1] != nmix * n * d * 4 || l[2] != 12 ||
        l[3] != nmix * 4 || l[4] != t * nmix * 8 || l[5] != t * n * 8 ||
        l[6] != t * n * 8 || l[7] != t * n * n * 8 || l[8] != t * d * 8 ||
        l[9] != t * d * 4 || l[10] != t * n * d * 8 || l[11] != n * n * d * 4 ||
        l[12] != 4 || l[13] != n * 4 || l[14] != t * d * 8 ||
        l[15] != t2 * n * n * 4 || l16 != t2 * g_nstages * n * n * 8) {
        fprintf(stderr, "mhc golden fixture size mismatch\n"); return 0;
    }
    return 1;
}

/* =========================================================================*/
/* 1. golden pre/post/comb/ypre/ypost/yhead */
static void test_golden_maps(void) {
    size_t n = g_N, d = g_D, nmix = (2 + n) * n;
    float *mixes = malloc(nmix * sizeof(float));
    float *pre = malloc(n * sizeof(float));
    float *post = malloc(n * sizeof(float));
    float *comb = malloc(n * n * sizeof(float));
    float *y = malloc(d * sizeof(float));
    float *y4 = malloc(n * d * sizeof(float));
    float *mixesn = malloc(nmix * sizeof(float));
    float *pren = malloc(n * sizeof(float));
    float *postn = malloc(n * sizeof(float));
    float *combn = malloc(n * n * sizeof(float));
    float *yn = malloc(d * sizeof(float));
    float *y4n = malloc(n * d * sizeof(float));

    double m_mix = 0, m_pre = 0, m_post = 0, m_comb = 0, m_ypre = 0,
           m_ypost = 0, m_yhead = 0, m_sn = 0;
    int m_ulp = 0;

    for (size_t t = 0; t < g_T; t++) {
        const float *x4 = g_x4 + t * n * d;
        apus_mhc_prepost_scalar(x4, d, n, g_fn, g_scale, g_base,
                                g_norm_eps, g_hc_eps, g_iters,
                                pre, post, comb, mixes);
        apus_mhc_collapse_scalar(x4, pre, y, d, n);
        m_mix  = fmax(m_mix,  max_abs_diff_f64(mixes, g_mixes + t * nmix, nmix, 1.0));
        m_pre  = fmax(m_pre,  max_abs_diff_f64(pre, g_pre + t * n, n, 1.0));
        m_post = fmax(m_post, max_abs_diff_f64(post, g_post + t * n, n, 1.0));
        m_comb = fmax(m_comb, max_abs_diff_f64(comb, g_comb + t * n * n, n * n, 1.0));
        m_ypre = fmax(m_ypre, max_abs_diff_f64(y, g_ypre + t * d, d, 1.0));

        apus_mhc_apply_scalar(g_f + t * d, x4, post, comb, y4, d, n);
        m_ypost = fmax(m_ypost, max_abs_diff_f64(y4, g_ypost + t * n * d, n * d, 1.0));

        apus_mhc_head_scalar(x4, d, n, g_hfn, g_hscale[0], g_hbase,
                             g_norm_eps, g_hc_eps, y, mixes);
        m_yhead = fmax(m_yhead, max_abs_diff_f64(y, g_yhead + t * d, d, 1.0));

#ifdef __ARM_NEON
        apus_mhc_prepost_neon(x4, d, n, g_fn, g_scale, g_base,
                              g_norm_eps, g_hc_eps, g_iters,
                              pren, postn, combn, mixesn);
        apus_mhc_collapse_neon(x4, pren, yn, d, n);
        double r;
        r = max_abs_diff_f64(mixesn, g_mixes + t * nmix, nmix, 1.0);
        if (r > m_mix) m_mix = r;
        r = max_abs_diff_f64(pren, g_pre + t * n, n, 1.0);
        if (r > m_pre) m_pre = r;
        r = max_abs_diff_f64(postn, g_post + t * n, n, 1.0);
        if (r > m_post) m_post = r;
        r = max_abs_diff_f64(combn, g_comb + t * n * n, n * n, 1.0);
        if (r > m_comb) m_comb = r;
        r = max_abs_diff_f64(yn, g_ypre + t * d, d, 1.0);
        if (r > m_ypre) m_ypre = r;
        apus_mhc_apply_neon(g_f + t * d, x4, postn, combn, y4n, d, n);
        r = max_abs_diff_f64(y4n, g_ypost + t * n * d, n * d, 1.0);
        if (r > m_ypost) m_ypost = r;
        apus_mhc_head_neon(x4, d, n, g_hfn, g_hscale[0], g_hbase,
                           g_norm_eps, g_hc_eps, yn, mixesn);
        r = max_abs_diff_f64(yn, g_yhead + t * d, d, 1.0);
        if (r > m_yhead) m_yhead = r;
        /* scalar-vs-NEON */
        for (size_t i = 0; i < n * n; i++) {
            double dd = fabs((double)comb[i] - (double)combn[i]);
            if (dd > m_sn) m_sn = dd;
            int u = ulp_diff(comb[i], combn[i]);
            if (u > m_ulp) m_ulp = u;
        }
        for (size_t i = 0; i < n * d; i++) {
            double dd = fabs((double)y4[i] - (double)y4n[i]);
            if (dd > m_sn) m_sn = dd;
        }
#endif
    }
    printf("  golden maps: mixes=%.3g pre=%.3g post=%.3g comb=%.3g "
           "ypre=%.3g ypost=%.3g yhead=%.3g (max abs vs f64)\n",
           m_mix, m_pre, m_post, m_comb, m_ypre, m_ypost, m_yhead);
#ifdef __ARM_NEON
    printf("  golden maps scalar-vs-NEON: comb/ypost max abs diff=%.3g, "
           "comb max ulp=%d\n", m_sn, m_ulp);
#endif
    /* FP32 vs float64 port: exp/sigmoid/dot rounding only. The comb matrix
     * passes through 40 normalization stages, so allow 1e-4. */
    CHECK(m_mix < 1e-4, "golden mixes abs err %.3g", m_mix);
    CHECK(m_pre < 1e-5, "golden pre abs err %.3g", m_pre);
    CHECK(m_post < 1e-5, "golden post abs err %.3g", m_post);
    CHECK(m_comb < 1e-4, "golden comb abs err %.3g", m_comb);
    CHECK(m_ypre < 1e-3, "golden ypre abs err %.3g", m_ypre);
    CHECK(m_ypost < 1e-3, "golden ypost abs err %.3g", m_ypost);
    CHECK(m_yhead < 1e-3, "golden yhead abs err %.3g", m_yhead);

    free(mixes); free(pre); free(post); free(comb); free(y); free(y4);
    free(mixesn); free(pren); free(postn); free(combn); free(yn); free(y4n);
}

/* =========================================================================*/
/* 2. Sinkhorn iteration-for-iteration + doubly stochastic */
static void test_sinkhorn_stages(void) {
    size_t n = g_N;
    float *c = malloc(n * n * sizeof(float));
    float *cfull = malloc(n * n * sizeof(float));
    double max_stage = 0, max_col_dev = 0, max_row_dev = 0;

    for (size_t t = 0; t < g_T2; t++) {
        memcpy(c, g_sk_logits + t * n * n, n * n * sizeof(float));
        memcpy(cfull, c, n * n * sizeof(float));
        const double *stages = g_sk_stages + t * g_nstages * n * n;
        size_t s = 0;
        /* reference order: row softmax+eps; col; (iters-1)x(row; col) */
        apus_mhc_row_softmax_eps(c, n, g_hc_eps);
        double d0 = max_abs_diff_f64(c, stages + s * n * n, n * n, 1.0);
        if (d0 > max_stage) max_stage = d0;
        s++;
        apus_mhc_norm_cols_eps(c, n, g_hc_eps);
        d0 = max_abs_diff_f64(c, stages + s * n * n, n * n, 1.0);
        if (d0 > max_stage) max_stage = d0;
        s++;
        for (int it = 1; it < g_iters; it++) {
            apus_mhc_norm_rows_eps(c, n, g_hc_eps);
            d0 = max_abs_diff_f64(c, stages + s * n * n, n * n, 1.0);
            if (d0 > max_stage) max_stage = d0;
            s++;
            apus_mhc_norm_cols_eps(c, n, g_hc_eps);
            d0 = max_abs_diff_f64(c, stages + s * n * n, n * n, 1.0);
            if (d0 > max_stage) max_stage = d0;
            s++;
        }
        CHECK(s == g_nstages, "stage count %zu != %zu", s, g_nstages);

        /* full driver must reproduce the stepwise sequence bitwise */
        apus_mhc_sinkhorn(cfull, n, g_iters, g_hc_eps);
        long bbad = 0;
        for (size_t i = 0; i < n * n; i++)
            if (f32bits(c[i]) != f32bits(cfull[i])) bbad++;
        CHECK(bbad == 0, "sinkhorn driver vs stepwise t=%zu: %ld bit diffs", t, bbad);

        /* doubly stochastic: col sums == golden col sums; both ~1 up to the
         * eps-in-denominator effect (~1e-6). Row sums are the penultimate
         * quantity (last op is a col norm) — compare against golden too. */
        for (size_t k = 0; k < n; k++) {
            double cs = 0, csg = 0, rs = 0, rsg = 0;
            for (size_t j = 0; j < n; j++) {
                cs += c[j * n + k];
                csg += stages[(g_nstages - 1) * n * n + j * n + k];
                rs += c[k * n + j];
                rsg += stages[(g_nstages - 1) * n * n + k * n + j];
            }
            double dev = fabs(cs - 1.0);
            if (dev > max_col_dev) max_col_dev = dev;
            double rd = fabs(rs - rsg), cd = fabs(cs - csg);
            if (rd > max_row_dev) max_row_dev = rd;
            if (cd > max_row_dev) max_row_dev = cd;
        }
    }
    printf("  sinkhorn stages: max per-stage abs err vs f64 = %.3g "
           "(%zu stages x %zu cases)\n", max_stage, g_nstages, g_T2);
    printf("  sinkhorn stochasticity: max |colsum-1| = %.3g, "
           "max |sums-golden| = %.3g\n", max_col_dev, max_row_dev);
    CHECK(max_stage < 1e-4, "sinkhorn per-stage err %.3g", max_stage);
    CHECK(max_col_dev < 1e-4, "sinkhorn col sums deviate %.3g from 1", max_col_dev);
    CHECK(max_row_dev < 1e-4, "sinkhorn sums vs golden %.3g", max_row_dev);
    free(c); free(cfull);
}

/* =========================================================================*/
/* 3. random real-shape + generic-n, scalar vs NEON vs in-test f64 */
static void run_random(size_t d, size_t n, double *worst) {
    size_t nmix = (2 + n) * n, nx = n * d;
    float *x4 = malloc(nx * sizeof(float));
    float *fn = malloc(nmix * nx * sizeof(float));
    float *scale = malloc(3 * sizeof(float));
    float *base = malloc(nmix * sizeof(float));
    float *hfn = malloc(n * nx * sizeof(float));
    float *hbase = malloc(n * sizeof(float));
    float *mixes = malloc(nmix * sizeof(float));
    float *pre = malloc(n * sizeof(float));
    float *post = malloc(n * sizeof(float));
    float *comb = malloc(n * n * sizeof(float));
    float *pren = malloc(n * sizeof(float));
    float *postn = malloc(n * sizeof(float));
    float *combn = malloc(n * n * sizeof(float));
    float *y = malloc(d * sizeof(float));
    float *yn = malloc(d * sizeof(float));
    float *f = malloc(d * sizeof(float));
    float *y4 = malloc(nx * sizeof(float));
    float *y4n = malloc(nx * sizeof(float));
    float norm_eps = 1e-6f, hc_eps = 1e-6f;

    for (size_t i = 0; i < nx; i++) x4[i] = rng_float();
    for (size_t i = 0; i < nmix * nx; i++) fn[i] = rng_float() * 0.02f;
    for (int i = 0; i < 3; i++) scale[i] = 1.0f + 0.25f * rng_float();
    for (size_t i = 0; i < nmix; i++) base[i] = rng_float() * 0.5f;
    for (size_t i = 0; i < n * nx; i++) hfn[i] = rng_float() * 0.02f;
    for (size_t i = 0; i < n; i++) hbase[i] = rng_float() * 0.5f;
    for (size_t i = 0; i < d; i++) f[i] = rng_float();

    apus_mhc_prepost_scalar(x4, d, n, fn, scale, base, norm_eps, hc_eps, 20,
                            pre, post, comb, mixes);
    apus_mhc_collapse_scalar(x4, pre, y, d, n);
    apus_mhc_apply_scalar(f, x4, post, comb, y4, d, n);
#ifdef __ARM_NEON
    float *mixesn = malloc(nmix * sizeof(float));
    apus_mhc_prepost_neon(x4, d, n, fn, scale, base, norm_eps, hc_eps, 20,
                          pren, postn, combn, mixesn);
    apus_mhc_collapse_neon(x4, pren, yn, d, n);
    apus_mhc_apply_neon(f, x4, postn, combn, y4n, d, n);
#endif

    /* in-test f64 recomputation of pre/collapse/apply from the C pre/post/
     * comb (isolates FP32 rounding of the vector ops, not the mixes) */
    double w = 0;
    for (size_t i = 0; i < d; i++) {
        double acc = 0;
        for (size_t j = 0; j < n; j++) acc += (double)pre[j] * x4[j * d + i];
        double dd = fabs(acc - (double)y[i]);
        if (dd > w) w = dd;
    }
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < d; i++) {
            double acc = (double)post[j] * f[i];
            for (size_t k = 0; k < n; k++)
                acc += (double)comb[k * n + j] * x4[k * d + i];
            double dd = fabs(acc - (double)y4[j * d + i]);
            if (dd > w) w = dd;
        }
#ifdef __ARM_NEON
    double wsn = 0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)pre[i] - (double)pren[i]);
        if (dd > wsn) wsn = dd;
        dd = fabs((double)post[i] - (double)postn[i]);
        if (dd > wsn) wsn = dd;
    }
    for (size_t i = 0; i < n * n; i++) {
        double dd = fabs((double)comb[i] - (double)combn[i]);
        if (dd > wsn) wsn = dd;
    }
    for (size_t i = 0; i < nx; i++) {
        double dd = fabs((double)y4[i] - (double)y4n[i]);
        if (dd > wsn) wsn = dd;
    }
    if (g_verbose)
        printf("  random d=%zu n=%zu: vec-op err=%.3g scalar-vs-NEON=%.3g\n",
               d, n, w, wsn);
    CHECK(wsn < 1e-4, "random d=%zu n=%zu scalar-vs-NEON %.3g", d, n, wsn);
    free(mixesn);
#endif
    if (w > *worst) *worst = w;
    CHECK(w < 1e-4, "random d=%zu n=%zu vec-op err %.3g", d, n, w);

    /* head at this shape */
    float *yh = malloc(d * sizeof(float));
    apus_mhc_head_scalar(x4, d, n, hfn, 1.0f, hbase, norm_eps, hc_eps, yh, mixes);
#ifdef __ARM_NEON
    float *yhn = malloc(d * sizeof(float));
    apus_mhc_head_neon(x4, d, n, hfn, 1.0f, hbase, norm_eps, hc_eps, yhn, mixes);
    double wh = 0;
    for (size_t i = 0; i < d; i++) {
        double dd = fabs((double)yh[i] - (double)yhn[i]);
        if (dd > wh) wh = dd;
    }
    CHECK(wh < 1e-4, "random head d=%zu n=%zu scalar-vs-NEON %.3g", d, n, wh);
    free(yhn);
#endif
    free(yh);

    free(x4); free(fn); free(scale); free(base); free(hfn); free(hbase);
    free(mixes); free(pre); free(post); free(comb);
    free(pren); free(postn); free(combn);
    free(y); free(yn); free(f); free(y4); free(y4n);
}

static void test_random_shapes(void) {
    double worst = 0;
    run_random(4096, 4, &worst);   /* real hidden size */
    run_random(64, 4, &worst);
    run_random(48, 3, &worst);     /* generic n */
    run_random(8, 2, &worst);
    printf("  random shapes: max vec-op err vs f64 = %.3g\n", worst);
}

/* =========================================================================*/
/* 4. edge cases */
static void test_edges(void) {
    size_t n = 4, d = 32, nmix = 24, nx = n * d;
    float *x4 = calloc(nx, sizeof(float));
    float *fn = malloc(nmix * nx * sizeof(float));
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float base[24];
    float mixes[24], pre[4], post[4], comb[16], y[32];
    for (size_t i = 0; i < nmix * nx; i++) fn[i] = rng_float() * 0.02f;
    for (int i = 0; i < 24; i++) base[i] = 0.0f;

    /* zero state: rsqrt(0 + 1e-6f) == 1/sqrtf(1e-6f) (~999.99994, not 1000:
     * 1e-6f is not exactly 1e-6), mixes all exactly 0 */
    float rwant = 1.0f / sqrtf(1e-6f);
    float r = apus_mhc_rsqrt_scalar(x4, nx, 1e-6f);
    CHECK(r == rwant, "zero state rsqrt: got %a want %a", r, rwant);
#ifdef __ARM_NEON
    CHECK(apus_mhc_rsqrt_neon(x4, nx, 1e-6f) == rwant, "zero state rsqrt NEON");
#endif
    apus_mhc_prepost_scalar(x4, d, n, fn, scale, base, 1e-6f, 1e-6f, 20,
                            pre, post, comb, mixes);
    for (int i = 0; i < 24; i++)
        CHECK(mixes[i] == 0.0f, "zero state mixes[%d] = %a", i, mixes[i]);
    for (int i = 0; i < 4; i++) {
        CHECK(fabsf(pre[i] - (0.5f + 1e-6f)) < 1e-9f, "zero state pre[%d]=%a", i, pre[i]);
        CHECK(fabsf(post[i] - 1.0f) < 1e-7f, "zero state post[%d]=%a", i, post[i]);
    }
    /* comb from all-zero logits -> uniform 1/4 up to eps effects */
    for (int i = 0; i < 16; i++)
        CHECK(fabsf(comb[i] - 0.25f) < 1e-5f, "zero state comb[%d]=%a", i, comb[i]);

    /* sigmoid saturation: base +-30 -> pre == 1+eps / 0+eps */
    base[0] = 30.0f; base[1] = -30.0f;
    apus_mhc_prepost_scalar(x4, d, n, fn, scale, base, 1e-6f, 1e-6f, 20,
                            pre, post, comb, mixes);
    CHECK(pre[0] == 1.0f + 1e-6f, "saturated pre hi: %a", pre[0]);
    CHECK(pre[1] >= 1e-6f && pre[1] < 1e-6f + 2e-13f, "saturated pre lo: %a", pre[1]);
    base[0] = base[1] = 0.0f;

    /* huge state: no overflow in rsqrt or mixes */
    for (size_t i = 0; i < nx; i++) x4[i] = 1e30f;
    r = apus_mhc_rsqrt_scalar(x4, nx, 1e-6f);
    CHECK(isfinite(r) && r < 1e-29f, "huge state rsqrt: %a", r);
    apus_mhc_prepost_scalar(x4, d, n, fn, scale, base, 1e-6f, 1e-6f, 20,
                            pre, post, comb, mixes);
    int ok = 1;
    for (int i = 0; i < 24; i++) ok &= isfinite(mixes[i]);
    for (int i = 0; i < 16; i++) ok &= isfinite(comb[i]) && comb[i] >= 0.0f;
    CHECK(ok, "huge state: non-finite mixes/comb");

    /* dominant comb logit: max subtraction keeps exp finite */
    {
        float c[16] = {0};
        c[0] = 80.0f;
        apus_mhc_sinkhorn(c, n, 20, 1e-6f);
        ok = 1;
        for (int i = 0; i < 16; i++) ok &= isfinite(c[i]) && c[i] >= 0.0f;
        CHECK(ok && c[0] > 0.9f, "dominant logit sinkhorn: c[0]=%a", c[0]);
        /* all-equal large logits -> uniform */
        for (int i = 0; i < 16; i++) c[i] = 50.0f;
        apus_mhc_sinkhorn(c, n, 20, 1e-6f);
        ok = 1;
        for (int i = 0; i < 16; i++)
            ok &= fabsf(c[i] - 0.25f) < 1e-5f;
        CHECK(ok, "uniform large logits sinkhorn");
        /* generic n=3 sinkhorn still doubly stochastic-ish */
        float c3[9] = {1, -2, 0.5f, 3, 0, -1, 2, 1, -0.5f};
        apus_mhc_sinkhorn(c3, 3, 20, 1e-6f);
        for (int k = 0; k < 3; k++) {
            float cs = 0;
            for (int j = 0; j < 3; j++) cs += c3[j * 3 + k];
            CHECK(fabsf(cs - 1.0f) < 1e-4f, "n=3 col sum %d = %a", k, cs);
        }
    }

    /* collapse/apply d not multiple of 4 (scalar tail parity) */
    {
        float xs[4 * 7], pr[4] = {0.1f, 0.2f, 0.3f, 0.4f}, yy[7], yy2[7];
        for (int i = 0; i < 28; i++) xs[i] = rng_float();
        apus_mhc_collapse_scalar(xs, pr, yy, 7, 4);
#ifdef __ARM_NEON
        apus_mhc_collapse_neon(xs, pr, yy2, 7, 4);
        for (int i = 0; i < 7; i++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += (double)pr[j] * xs[j * 7 + i];
            CHECK(fabs((double)yy2[i] - acc) < 1e-6 * (fabs(acc) + 1.0),
                  "collapse tail NEON i=%d: %a want ~%a", i, yy2[i], (float)acc);
        }
#endif
        for (int i = 0; i < 7; i++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += (double)pr[j] * xs[j * 7 + i];
            CHECK(fabs((double)yy[i] - acc) < 1e-6 * (fabs(acc) + 1.0),
                  "collapse tail scalar i=%d: %a want ~%a", i, yy[i], (float)acc);
        }
    }

    (void)y;
    free(x4); free(fn);
}

int main(int argc, char **argv) {
    g_verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    printf("test_mhc: mHC (Sinkhorn-20 residual stream) hard-gate tests\n");
#ifdef __ARM_NEON
    printf("  NEON paths: enabled\n");
#else
    printf("  NEON paths: NOT compiled (scalar only)\n");
#endif

    if (load_golden()) {
        test_golden_maps();
        test_sinkhorn_stages();
    } else {
        failures++;
        fprintf(stderr, "FAIL: could not load golden fixtures\n");
    }

    test_random_shapes();
    test_edges();

    printf("test_mhc: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
