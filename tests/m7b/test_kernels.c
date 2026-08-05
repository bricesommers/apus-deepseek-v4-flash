/*
 * tests/m7b/test_kernels.c — kernel-level verification of the Metal backend
 * (c/backend_metal.mm) against the CPU c/fp8.h / c/attn.h paths.
 *
 *   1. Act quant: GPU codes + scales BITWISE vs apus_fp4_act_quant_scalar
 *      (same kernel.py rule; the shader must be exact, not close).
 *   2. Raw FP8 GEMM (apus_metal_fp8_gemm) vs in-test FP64 truth with the
 *      m4a esc metric (err/esc < 2e-5) and vs apus_fp8_gemm_neon —
 *      since M9a the shader accumulates in the CPU NEON canonical order,
 *      so the comparison is BITWISE (asserted).
 *      Battery: real dense shapes + mini-model shapes + odd/partial ones.
 *   3. Full fp8 linear (BF16 in/out) vs the CPU composition: mostly bitwise
 *      (differences only where a near-exact FP32 tie crosses a BF16 RNE
 *      boundary), max 1 BF16 ulp.
 *   4. f32 linear (router-gate / bf16-linear / wo_a flag variants) vs the
 *      CPU loops — near-bitwise (fma contraction only).
 *   5. BF16 head GEMV vs scalar CPU — near-bitwise.
 *   6. RMSNorm vs apus_rms_norm — small (parallel sum order), reported.
 *   7. Fail-soft: with APUS_METAL_DENSE_MB tiny, oversized weights return
 *      "unsupported" and the CPU result is produced by the fallback.
 *
 * Exit 0 iff all checks pass. Run from the repository root.
 */
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp8.h"
#include "attn.h"
#include "backend_metal.h"

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

/* ---- deterministic PRNG (splitmix64, same as tests/m4a) ---- */
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

/* ---- FP64 truth mirroring fp8_gemm semantics + esc (tests/m4a) ---- */
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
                scale += adot * sc;
            }
            out[m * O + o] = total;
            if (esc) esc[m * O + o] = scale;
        }
    }
    free(ad);
}

/* =========================================================================*/
/* 1. act quant bitwise */
static void test_act_quant(void) {
    struct { int K; } shapes[] = {{128}, {256}, {384}, {4096}, {448}, {64}};
    long bad_codes = 0, bad_scales = 0, total = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        int K = shapes[s].K;
        for (int m = 0; m < 4; m++) {
            float *x = malloc((size_t)K * sizeof(float));
            for (int i = 0; i < K; i++) {
                x[i] = rng_float();
                if (m == 1) x[i] = 0.0f;             /* amax floor path */
                if (m == 2) x[i] *= 1e-6f;           /* tiny */
                if (m == 3 && (i & 15) == 0) x[i] *= 1e4f; /* outliers */
            }
            size_t nb = apus_fp8_blocks((size_t)K);
            uint8_t *cc = malloc((size_t)K), *cg = malloc((size_t)K);
            float *sc = malloc(nb * sizeof(float)), *sg = malloc(nb * sizeof(float));
            /* the GPU entry point BF16-rounds the input inside (same front
             * end as apus_fp8_linear) — the CPU reference gets the rounded
             * values explicitly */
            float *xb = malloc((size_t)K * sizeof(float));
            for (int i = 0; i < K; i++) xb[i] = apus_bf16_round(x[i]);
            apus_fp4_act_quant_scalar(xb, (size_t)K, cc, sc);
            free(xb);
            int rc = apus_metal_fp8_act_quant(x, 1, K, cg, sg);
            CHECK(rc == 0, "act_quant rc=%d K=%d", rc, K);
            if (rc == 0) {
                for (int i = 0; i < K; i++) {
                    total++;
                    if (cc[i] != cg[i]) {
                        if (bad_codes < 5)
                            fprintf(stderr,
                                    "  act code mismatch K=%d m=%d i=%d: cpu=%02x gpu=%02x (x=%a)\n",
                                    K, m, i, cc[i], cg[i], x[i]);
                        bad_codes++;
                    }
                }
                for (size_t b = 0; b < nb; b++)
                    if (f32bits(sc[b]) != f32bits(sg[b])) {
                        if (bad_scales < 5)
                            fprintf(stderr,
                                    "  act scale mismatch K=%d m=%d b=%zu: cpu=%a gpu=%a\n",
                                    K, m, b, sc[b], sg[b]);
                        bad_scales++;
                    }
            }
            free(x); free(cc); free(cg); free(sc); free(sg);
        }
    }
    CHECK(bad_codes == 0, "act quant codes: %ld/%ld mismatches (want bitwise)",
          bad_codes, total);
    CHECK(bad_scales == 0, "act quant scales: %ld mismatches (want bitwise)",
          bad_scales);
    printf("  act quant: %ld codes bitwise (codes bad=%ld, scales bad=%ld)\n",
           total, bad_codes, bad_scales);
}

/* =========================================================================*/
/* 2. raw GEMM vs FP64 truth + vs NEON */
static void test_gemm_shape(size_t M, size_t O, size_t K,
                            double *worst_esc, double *worst_neon) {
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    /* weights are deliberately NOT freed: the backend caches weight buffers
     * by pointer (stable-pointer invariant, c/backend_metal.h), and a freed
     * address reused by a later malloc would alias a stale GPU buffer. The
     * engine never frees weights before apus_metal_disable, so this only
     * matters in tests. */
    uint8_t *w = malloc(O * K), *ws = malloc(nbo * nb);
    uint8_t *ac = malloc(M * K);
    float *as = malloc(M * nb * sizeof(float));
    for (size_t i = 0; i < O * K; i++) {
        uint8_t b = rng_byte();
        if ((b & 0x7F) == 0x7F) b ^= 1;    /* keep NaN codes out of the sweep */
        w[i] = b;
    }
    for (size_t i = 0; i < nbo * nb; i++) {
        /* sane UE8M0 range (2^-31..2^33): keeps FP32 outputs finite so the
         * esc comparison is meaningful; extreme bytes 0/254/255 are covered
         * by the edge tests */
        ws[i] = (uint8_t)(96 + rng_byte() % 65);
    }
    for (size_t m = 0; m < M; m++) {
        float *x = malloc(K * sizeof(float));
        for (size_t k = 0; k < K; k++) x[k] = rng_float();
        apus_fp4_act_quant_scalar(x, K, ac + m * K, as + m * nb);
        free(x);
    }
    float *og = malloc(M * O * sizeof(float));
    float *oc = malloc(M * O * sizeof(float));
    double *ot = malloc(M * O * sizeof(double));
    double *esc = malloc(M * O * sizeof(double));
    int rc = apus_metal_fp8_gemm(w, ws, ac, as, og, (int)M, (int)O, (int)K);
    CHECK(rc == 0, "metal gemm rc=%d M=%zu O=%zu K=%zu", rc, M, O, K);
    if (rc) goto out;
#ifdef __ARM_NEON
    {
        float *scratch = malloc(M * K * sizeof(float));
        apus_fp8_gemm_neon(w, ws, ac, as, scratch, oc, M, O, K);
        free(scratch);
    }
#else
    {
        float *scratch = malloc(M * K * sizeof(float));
        apus_fp8_gemm_scalar(w, ws, ac, as, scratch, oc, M, O, K);
        free(scratch);
    }
#endif
    truth_gemm_f64(w, ws, ac, as, ot, esc, M, O, K);
    double me = 0, mn = 0;
    size_t nbit_neon = 0;
    for (size_t i = 0; i < M * O; i++) {
        if (isinf(ot[i])) continue;   /* finite-scale battery: not expected */
        double e = fabs((double)og[i] - ot[i]);
        double sc = esc[i] > 1e-30 ? esc[i] : 1e-30;
        if (e / sc > me) me = e / sc;
        double d = fabs((double)og[i] - (double)oc[i]);
        double ds = d / sc;
        if (ds > mn) mn = ds;
        if (f32bits(og[i]) != f32bits(oc[i])) nbit_neon++;
    }
    if (me > *worst_esc) *worst_esc = me;
    if (mn > *worst_neon) *worst_neon = mn;
    CHECK(me < 2e-5, "gemm M=%zu O=%zu K=%zu err/esc %.3e >= 2e-5", M, O, K, me);
    /* M9a: the GPU fp8_dot accumulates in the exact CPU NEON canonical order
     * and fp8_fold mirrors the CPU's contracted fmadd fold — the raw GEMM
     * outputs must be BITWISE equal to the CPU NEON kernel. */
    CHECK(nbit_neon == 0,
          "gemm M=%zu O=%zu K=%zu: %zu outputs not bitwise vs CPU NEON (M9a)",
          M, O, K, nbit_neon);
    printf("  gemm M=%-2zu O=%-6zu K=%-6zu err/esc=%.2e gpu-vs-cpu=%.2e "
           "(bitwise %zu/%zu)\n",
           M, O, K, me, mn, M * O - nbit_neon, M * O);
out:
    free(ac); free(as);   /* activations are staged per call; weights leak */
    free(og); free(oc); free(ot); free(esc);
}

static void test_gemm_battery(void) {
    double we = 0, wn = 0;
    /* real dense shapes (wq_a, wq_b, wkv, wo_b, shared w1/w3, w2, idx wq_b) */
    test_gemm_shape(1, 1024, 4096, &we, &wn);
    test_gemm_shape(1, 4096, 4096, &we, &wn);   /* downscaled wq_b proxy */
    test_gemm_shape(2, 512, 4096, &we, &wn);
    test_gemm_shape(1, 2048, 4096, &we, &wn);
    test_gemm_shape(1, 4096, 2048, &we, &wn);
    test_gemm_shape(3, 1024, 1024, &we, &wn);
    /* mini-model (tests/m5) shapes */
    test_gemm_shape(1, 128, 256, &we, &wn);
    test_gemm_shape(5, 512, 128, &we, &wn);
    test_gemm_shape(1, 256, 256, &we, &wn);
    /* odd / partial-block shapes */
    test_gemm_shape(1, 1, 128, &we, &wn);
    test_gemm_shape(2, 3, 128, &we, &wn);
    test_gemm_shape(1, 5, 384, &we, &wn);
    test_gemm_shape(2, 17, 448, &we, &wn);
    test_gemm_shape(5, 130, 256, &we, &wn);
    test_gemm_shape(1, 64, 384, &we, &wn);
    test_gemm_shape(2, 200, 512, &we, &wn);
    test_gemm_shape(9, 333, 640, &we, &wn);    /* chunking (>8 rows) */
    printf("  gemm battery: worst err/esc=%.3e (assert <2e-5), "
           "worst gpu-vs-cpu/esc=%.3e\n", we, wn);
}

/* edge cases: scale byte 0 (2^-127 subnormal), 255 (inf), zero blocks */
static void test_gemm_edges(void) {
    size_t K = 256, O = 130, nb = apus_fp8_blocks(K);
    size_t nbo = (O + 127) / 128;
    uint8_t *w = calloc(O * K, 1), *ws = malloc(nbo * nb);
    uint8_t *ac = malloc(K);
    float as[2];
    float *x = malloc(K * sizeof(float));
    for (size_t i = 0; i < K; i++) x[i] = rng_float();
    apus_fp4_act_quant_scalar(x, K, ac, as);
    free(x);
    float *og = malloc(O * sizeof(float)), *oc = malloc(O * sizeof(float));

    /* all-zero weights with scale byte 254 -> exact zeros */
    memset(ws, 254, nbo * nb);
    CHECK(apus_metal_fp8_gemm(w, ws, ac, as, og, 1, (int)O, (int)K) == 0,
          "edge: gemm rc");
    float *scratch = malloc(K * sizeof(float));
    apus_fp8_gemv_scalar(w, ws, ac, as, scratch, oc, O, K);
    free(scratch);
    int bad = 0;
    for (size_t i = 0; i < O; i++)
        if (f32bits(og[i]) != f32bits(oc[i])) bad++;
    CHECK(bad == 0 && og[0] == 0.0f, "edge: zero weights byte 254: %d", bad);

    /* scale byte 0 = 2^-127 subnormal (exact), nonzero weights */
    for (size_t i = 0; i < O * K; i++) w[i] = 0x08;  /* code: 2^-6 */
    memset(ws, 0, nbo * nb);
    CHECK(apus_metal_fp8_gemm(w, ws, ac, as, og, 1, (int)O, (int)K) == 0,
          "edge: gemm rc byte0");
    {
        float *scr = malloc(K * sizeof(float));
        apus_fp8_gemv_scalar(w, ws, ac, as, scr, oc, O, K);
        free(scr);
    }
    bad = 0;
    double md = 0;
    for (size_t i = 0; i < O; i++) {
        if (f32bits(og[i]) != f32bits(oc[i])) bad++;
        double d = fabs((double)og[i] - (double)oc[i]);
        if (d > md) md = d;
    }
    CHECK(md < 1e-30, "edge: scale byte 0 (2^-127): maxdiff %.3e", md);
    printf("  edges: zero/254 bitwise ok; byte-0 maxdiff=%.3e (%d rows differ)\n",
           md, bad);

    /* scale byte 255 = +inf; zero dot rows -> 0*inf = NaN on BOTH paths */
    memset(w, 0, O * K);
    memset(ws, 255, nbo * nb);
    CHECK(apus_metal_fp8_gemm(w, ws, ac, as, og, 1, (int)O, (int)K) == 0,
          "edge: gemm rc byte255");
    {
        float *scr = malloc(K * sizeof(float));
        apus_fp8_gemv_scalar(w, ws, ac, as, scr, oc, O, K);
        free(scr);
    }
    bad = 0;
    for (size_t i = 0; i < O; i++) {
        int gn = isnan(og[i]), cn = isnan(oc[i]);
        if (gn != cn || (!gn && f32bits(og[i]) != f32bits(oc[i]))) bad++;
    }
    CHECK(bad == 0, "edge: scale byte 255 NaN parity: %d", bad);

    free(ac); free(og); free(oc);   /* weights not freed (pointer cache) */
}

/* =========================================================================*/
/* 3. full fp8 linear vs CPU composition (BF16-rounded outputs) */
static void test_full_linear(void) {
    struct { int M, O, K; } shapes[] = {
        {1, 1024, 4096}, {1, 512, 4096}, {3, 2048, 2048}, {1, 256, 256},
        {2, 130, 384}, {16, 1024, 512},
    };
    long total = 0, nbit = 0;
    double worst_rel = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        int M = shapes[s].M, O = shapes[s].O, K = shapes[s].K;
        size_t nb = apus_fp8_blocks((size_t)K), nbo = ((size_t)O + 127) / 128;
        ApusFp8W w;
        uint8_t *wc = malloc((size_t)O * K), *wsc = malloc(nbo * nb);
        for (int i = 0; i < O * K; i++) {
            uint8_t b = rng_byte();
            if ((b & 0x7F) == 0x7F) b ^= 1;
            wc[i] = b;
        }
        for (size_t i = 0; i < nbo * nb; i++)
            wsc[i] = (uint8_t)(96 + rng_byte() % 65);
        w.codes = wc; w.scales = wsc; w.O = O; w.K = K;
        float *x = malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) x[i] = rng_float();
        float *og = malloc((size_t)M * O * sizeof(float));
        float *oc = malloc((size_t)M * O * sizeof(float));
        /* CPU composition exactly as apus_fp8_linear does it (hooks stay
         * unset in this binary, so apus_fp8_linear IS the CPU path) */
        apus_fp8_linear(&w, x, oc, M, K, O);
        int rc = apus_metal_fp8_linear(&w, x, og, M, K, O);
        CHECK(rc == 0, "full linear rc=%d", rc);
        if (rc == 0)
            for (int i = 0; i < M * O; i++) {
                total++;
                if (f32bits(og[i]) != f32bits(oc[i])) {
                    nbit++;
                    double a = og[i], b = oc[i];
                    double sc = fabs(b) > 1e-30 ? fabs(b) : 1e-30;
                    double r = fabs(a - b) / sc;
                    if (r > worst_rel) worst_rel = r;
                }
            }
        free(x); free(og); free(oc);   /* weights not freed (cache) */
    }
    /* BF16 output rounding: a GPU/CPU FP32 difference can only flip an RNE
     * boundary -> at most 1 bf16 ulp ~= 2^-8 relative, and rare. */
    CHECK(worst_rel < 0.01, "full linear worst rel %.3e (1 bf16 ulp class)",
          worst_rel);
    printf("  full linear: bitwise %ld/%ld (diffs %ld, worst rel %.2e)\n",
           total - nbit, total, nbit, worst_rel);
    CHECK(nbit * 1000 <= total, "full linear: >0.1%% non-bitwise (%ld/%ld)",
          nbit, total);
}

/* =========================================================================*/
/* 4. f32 linear flag variants */
static void test_f32_linear(void) {
    struct { int M, O, K, flags; const char *name; } cases[] = {
        {1, 256, 4096, 0, "router gate (no rounding)"},
        {2, 64, 256, APUS_HOOK_R_IN | APUS_HOOK_R_OUT, "bf16 linear"},
        {1, 1024, 512, APUS_HOOK_R_OUT, "wo_a group (R_OUT)"},
        {1, 1024, 512, APUS_HOOK_R_OUT | APUS_HOOK_W_BF16,
         "wo_a bf16 store (R_OUT|W_BF16)"},
        {3, 8, 256, 0, "mini gate"},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int M = cases[c].M, O = cases[c].O, K = cases[c].K, fl = cases[c].flags;
        /* the W_BF16 case draws from a private stream so the shared rng
         * sequence (and every later test's data) is unchanged by its
         * addition */
        uint64_t rng_saved = rng_state;
        if (fl & APUS_HOOK_W_BF16) rng_state = 0xB16B00B5DEADBEEFull;
        float *w = malloc((size_t)O * K * sizeof(float));
        uint16_t *wb = malloc((size_t)O * K * sizeof(uint16_t));
        float *x = malloc((size_t)M * K * sizeof(float));
        float *og = malloc((size_t)M * O * sizeof(float));
        float *oc = malloc((size_t)M * O * sizeof(float));
        for (int i = 0; i < O * K; i++) {
            w[i] = apus_bf16_round(rng_float());
            wb[i] = apus_bf16_bits(w[i]);
        }
        for (int i = 0; i < M * K; i++) x[i] = rng_float();
        for (int m = 0; m < M; m++)
            for (int o = 0; o < O; o++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    float xv = x[(size_t)m * K + k];
                    if (fl & APUS_HOOK_R_IN) xv = apus_bf16_round(xv);
                    acc += xv * w[(size_t)o * K + k];
                }
                oc[(size_t)m * O + o] =
                    (fl & APUS_HOOK_R_OUT) ? apus_bf16_round(acc) : acc;
            }
        int rc = apus_metal_f32_linear(
            (fl & APUS_HOOK_W_BF16) ? (const float *)wb : w,
            x, og, M, K, O, fl);
        CHECK(rc == 0, "f32 linear rc=%d (%s)", rc, cases[c].name);
        long nbit = 0, total = (long)M * O;
        double worst = 0, sc = 0;
        if (rc == 0)
            for (int i = 0; i < M * O; i++) {
                if (f32bits(og[i]) != f32bits(oc[i])) nbit++;
                double d = fabs((double)og[i] - (double)oc[i]);
                if (d > worst) worst = d;
                if (fabs((double)oc[i]) > sc) sc = fabs((double)oc[i]);
            }
        printf("  f32 %-28s bitwise %ld/%ld worst abs %.3e (scale %.1e)\n",
               cases[c].name, total - nbit, total, worst, sc);
        CHECK(worst <= (sc > 0 ? sc : 1) * 1e-5 + 1e-6,
              "f32 %s worst %.3e too large", cases[c].name, worst);
        free(x); free(og); free(oc);   /* weights not freed (backend cache) */
        rng_state = rng_saved;
    }
}

/* =========================================================================*/
/* 5. BF16 head GEMV */
static void test_head_gemv(void) {
    int64_t O = 4096, K = 1024;
    uint16_t *w = malloc((size_t)O * K * sizeof(uint16_t));
    float *x = malloc((size_t)K * sizeof(float));
    float *og = malloc((size_t)O * sizeof(float));
    float *oc = malloc((size_t)O * sizeof(float));
    for (int64_t i = 0; i < O * K; i++) {
        /* finite BF16 patterns (RNE of a float) — fully random u16 would
         * include inf/NaN codes and make the compare meaningless */
        float f = rng_float();
        uint32_t u;
        memcpy(&u, &f, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        w[i] = (uint16_t)(u >> 16);
    }
    for (int64_t i = 0; i < K; i++) x[i] = rng_float();
    for (int64_t o = 0; o < O; o++) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            uint32_t u = (uint32_t)w[o * K + k] << 16;
            float f;
            memcpy(&f, &u, 4);
            acc += f * x[k];
        }
        oc[o] = acc;
    }
    int rc = apus_metal_head_gemv_bf16(w, x, og, O, K);
    CHECK(rc == 0, "head gemv rc=%d", rc);
    long nbit = 0;
    double worst = 0, sc = 0;
    if (rc == 0)
        for (int64_t i = 0; i < O; i++) {
            if (f32bits(og[i]) != f32bits(oc[i])) nbit++;
            double d = fabs((double)og[i] - (double)oc[i]);
            if (d > worst) worst = d;
            if (fabs((double)oc[i]) > sc) sc = fabs((double)oc[i]);
        }
    printf("  head gemv: bitwise %lld/%lld worst abs %.3e (scale %.1e)\n",
           (long long)(O - nbit), (long long)O, worst, sc);
    CHECK(worst <= sc * 1e-5 + 1e-6, "head gemv worst %.3e", worst);
    free(x); free(og); free(oc);   /* weight not freed (cache) */
}

/* =========================================================================*/
/* 6. RMSNorm */
static void test_rmsnorm(void) {
    int64_t sizes[] = {64, 256, 4096, 512};
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        int64_t n = sizes[s];
        float *x = malloc((size_t)n * sizeof(float));
        float *w = malloc((size_t)n * sizeof(float));
        float *yg = malloc((size_t)n * sizeof(float));
        float *yc = malloc((size_t)n * sizeof(float));
        for (int64_t i = 0; i < n; i++) {
            x[i] = rng_float();
            w[i] = apus_bf16_round(rng_float());
        }
        apus_rms_norm(x, w, 1e-6f, yc, (size_t)n);
        int rc = apus_metal_rmsnorm(x, w, 1e-6f, yg, n);
        CHECK(rc == 0, "rmsnorm rc=%d n=%lld", rc, (long long)n);
        long nbit = 0;
        double worst = 0;
        if (rc == 0)
            for (int64_t i = 0; i < n; i++) {
                if (f32bits(yg[i]) != f32bits(yc[i])) nbit++;
                double d = fabs((double)yg[i] - (double)yc[i]);
                if (d > worst) worst = d;
            }
        printf("  rmsnorm n=%-5lld bitwise %lld/%lld worst abs %.3e\n",
               (long long)n, (long long)(n - nbit), (long long)n, worst);
        CHECK(worst < 1e-3, "rmsnorm n=%lld worst %.3e", (long long)n, worst);
        free(x); free(yg); free(yc);   /* weight not freed (cache) */
    }
}

/* =========================================================================*/
/* 7. fail-soft: budget exhaustion -> "unsupported" -> CPU fallback */
static void test_failsoft(void) {
    /* a weight too big for a 1 MB budget must be rejected (rc=1), and the
     * hooked CPU path must then produce the CPU result unchanged */
    setenv("APUS_METAL_DENSE_MB", "1", 1);
    apus_metal_disable();
    char err[256];
    CHECK(apus_metal_enable(err, sizeof err) == 0, "failsoft: enable: %s", err);
    size_t O = 2048, K = 2048;    /* 4 MB of codes >> 1 MB budget */
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    ApusFp8W w;
    uint8_t *wc = malloc(O * K), *wsc = malloc(nbo * nb);
    memset(wc, 0x08, O * K);
    memset(wsc, 100, nbo * nb);
    w.codes = wc; w.scales = wsc; w.O = (int64_t)O; w.K = (int64_t)K;
    float *x = malloc(K * sizeof(float));
    float *o1 = malloc(O * sizeof(float));
    float *o2 = malloc(O * sizeof(float));
    for (size_t i = 0; i < K; i++) x[i] = rng_float();
    int rc = apus_metal_fp8_linear(&w, x, o1, 1, (int)K, (int)O);
    CHECK(rc == 1, "failsoft: oversized weight must return unsupported "
          "(rc=%d)", rc);
    /* hooked call: hook declines -> CPU result through apus_fp8_linear */
    apus_fp8_linear(&w, x, o1, 1, (int)K, (int)O);
    ApusBackendHooks saved = apus_backend_hooks;
    memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
    apus_fp8_linear(&w, x, o2, 1, (int)K, (int)O);
    CHECK(memcmp(o1, o2, O * sizeof(float)) == 0,
          "failsoft: hooked fallback != pure CPU");
    apus_backend_hooks = saved;
    /* disable(): hooks cleared, engine fully on CPU */
    apus_metal_disable();
    CHECK(apus_backend_hooks.fp8_linear == NULL
          && apus_backend_hooks.f32_linear == NULL
          && apus_backend_hooks.head_gemv == NULL,
          "failsoft: hooks not cleared by disable");
    /* restore a normal budget for anything running after us */
    unsetenv("APUS_METAL_DENSE_MB");
    CHECK(apus_metal_enable(err, sizeof err) == 0,
          "failsoft: re-enable: %s", err);
    free(wc); free(wsc); free(x); free(o1); free(o2);
}

int main(void) {
    printf("test_kernels: M7b Metal backend kernel verification\n");
    char err[256];
    if (apus_metal_enable(err, sizeof err)) {
        printf("  Metal unavailable (%s) — skipping (not a failure on "
               "non-GPU hosts)\n", err);
        return 0;
    }
    printf("  Metal device ready\n");
    test_act_quant();
    test_gemm_battery();
    test_gemm_edges();
    test_full_linear();
    test_f32_linear();
    test_head_gemv();
    test_rmsnorm();
    printf("  before fail-soft test: zero-copy wrapped %.1f MB, "
           "uploaded %.1f MB, %llu dispatches\n",
           (double)apus_metal_bytes_wrapped() / 1048576.0,
           (double)apus_metal_bytes_uploaded() / 1048576.0,
           (unsigned long long)apus_metal_dispatches());
    CHECK(apus_metal_bytes_wrapped() > 0,
          "zero-copy wrapping never engaged (expected on Apple Silicon)");
    test_failsoft();
    apus_metal_disable();
    printf("test_kernels: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
