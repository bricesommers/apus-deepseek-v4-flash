/*
 * tests/m6c/test_m6c.c — hard-gate tests for the M6c threading/scratch work:
 *
 *   1. Pool correctness: apus_pool_run covers [0, n) exactly once.
 *   2. THREAD INDEPENDENCE (the m6a/m6b invariance contract): the threaded
 *      fp8/fp4 GEMMs (apus_fp8_gemm_mt / apus_fp4_gemm_mt) are BITWISE equal
 *      to the single-thread NEON kernels, and apus_bf16_linear /
 *      apus_f32_linear rows are bitwise M-independent (prefill/decode share
 *      per-row accumulation). The process-level digest printed at the end is
 *      diffed across APUS_THREADS=1/4/8 by the Makefile (bitwise identity).
 *   3. New bf16/f32 NEON dot vs the old scalar-order reference: bounded
 *      reorder error (documented scalar-vs-NEON tolerance class), measured
 *      against the per-output error scale esc = sum |a*b|.
 *   4. Scratch arena: nested mark/reset, growth across segments, alignment.
 *   5. Pool dispatch overhead: informational (stderr), assert < 200 us avg
 *      (budget was <50 us; the assert only catches pathological regressions).
 *
 * Run from the repository root.
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
static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-2, 2) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
}

/* ---- FNV-1a digest over f32 bit patterns (thread-independence proof) ---- */
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

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---- 1. pool coverage ---- */
typedef struct { int *counts; } PoolCnt;
static void pool_cnt_rows(void *ctx, size_t i0, size_t i1) {
    PoolCnt *pc = ctx;
    for (size_t i = i0; i < i1; i++) pc->counts[i]++;
}

static void test_pool(void) {
    enum { N = 1000 };
    int *counts = calloc(N, sizeof(int));
    PoolCnt pc = { counts };
    apus_pool_run(N, pool_cnt_rows, &pc);
    int bad = 0;
    for (int i = 0; i < N; i++) if (counts[i] != 1) bad++;
    CHECK(bad == 0, "pool coverage: %d rows not visited exactly once", bad);
    free(counts);
    fprintf(stderr, "pool: threads=%d\n", apus_pool_threads());
}

/* ---- 2. threaded fp8/fp4 GEMM bitwise == single-thread anchor ----
 * Anchor: the NEON kernels on ARM; the normative scalar kernels elsewhere
 * (M12a-1 x86 — the mt threaded rows mirror the scalar kernels step for
 * step, so the bitwise contract is identical). */
#ifdef __ARM_NEON
#define FP8_GEMM_ANCHOR apus_fp8_gemm_neon
#define FP4_GEMM_ANCHOR apus_fp4_gemm_neon
#define ANCHOR_NAME "neon"
#else
#define FP8_GEMM_ANCHOR apus_fp8_gemm_scalar
#define FP4_GEMM_ANCHOR apus_fp4_gemm_scalar
#define ANCHOR_NAME "scalar"
#endif
static void test_fp8_mt(size_t M, size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    size_t nsb = ((O + 127) / 128) * nb;
    size_t nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * K), *ws = malloc(nsb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *o_neon = malloc(M * O * sizeof(float));
    float *o_mt = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = (uint8_t)(rng_u64() >> 56);
    for (size_t i = 0; i < nsb; i++)
        ws[i] = (uint8_t)(120 + (rng_u64() >> 62) % 12);   /* 2^-7..2^7 */
    for (size_t i = 0; i < M * K; i++) x[i] = rng_float();
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[m * K + i] = apus_bf16_round(x[m * K + i]);
        apus_fp4_act_quant_scalar(x + m * K, K, codes + m * K, as + m * nab);
    }
    FP8_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, M, O, K);
    apus_fp8_gemm_mt(w, ws, codes, as, scratch, o_mt, M, O, K);
    CHECK(memcmp(o_neon, o_mt, M * O * sizeof(float)) == 0,
          "fp8 gemm_mt != gemm_" ANCHOR_NAME " (M=%zu O=%zu K=%zu)", M, O, K);
    digest_f32(o_mt, M * O);
    free(w); free(ws); free(codes); free(as); free(x);
    free(scratch); free(o_neon); free(o_mt);
}

static void test_fp4_mt(size_t M, size_t O, size_t K) {
    size_t nb = K / 32, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2)), *ws = malloc(O * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *o_neon = malloc(M * O * sizeof(float));
    float *o_mt = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)(rng_u64() >> 56);
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(120 + (rng_u64() >> 62) % 12);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_float();
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[m * K + i] = apus_bf16_round(x[m * K + i]);
        apus_fp4_act_quant_scalar(x + m * K, K, codes + m * K, as + m * nab);
    }
    FP4_GEMM_ANCHOR(w, ws, codes, as, scratch, o_neon, M, O, K);
    apus_fp4_gemm_mt(w, ws, codes, as, scratch, o_mt, M, O, K);
    CHECK(memcmp(o_neon, o_mt, M * O * sizeof(float)) == 0,
          "fp4 gemm_mt != gemm_" ANCHOR_NAME " (M=%zu O=%zu K=%zu)", M, O, K);
    digest_f32(o_mt, M * O);
    free(w); free(ws); free(codes); free(as); free(x);
    free(scratch); free(o_neon); free(o_mt);
}

/* ---- 3. bf16/f32 linear: reorder bound + bitwise M-independence ---- */
static void test_f32_path(int M, int K, int O, int bf16) {
    float *w = malloc((size_t)O * K * sizeof(float));
    float *x = malloc((size_t)M * K * sizeof(float));
    float *out = malloc((size_t)M * O * sizeof(float));
    float *out1 = malloc((size_t)O * sizeof(float));
    for (size_t i = 0; i < (size_t)O * K; i++) w[i] = rng_float() * 0.05f;
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = rng_float();
    if (bf16)
        apus_bf16_linear(w, x, out, M, K, O);
    else
        apus_f32_linear(w, x, out, M, K, O);
    /* reference: the OLD scalar semantics (sequential-k accumulation,
     * input rounded per element for bf16). The threaded kernels keep the
     * per-row sequential-k order, so they must match the reference
     * BITWISE — there is no reorder budget on these paths. */
    long nbit = 0;
    for (int m = 0; m < M; m++)
        for (int o = 0; o < O; o++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float xv = x[(size_t)m * K + k];
                if (bf16) xv = apus_bf16_round(xv);
                acc += xv * w[(size_t)o * K + k];
            }
            float ref = bf16 ? apus_bf16_round(acc) : acc;
            float got = out[(size_t)m * O + o];
            uint32_t a, b;
            memcpy(&a, &got, 4);
            memcpy(&b, &ref, 4);
            if (a != b) nbit++;
        }
    CHECK(nbit == 0, "%s not bitwise vs old scalar order: %ld diffs "
          "(M=%d K=%d O=%d)", bf16 ? "bf16" : "f32", nbit, M, K, O);
    /* M-independence: row m computed in the M-row call must be bitwise
     * equal to the same row computed as M=1 (prefill/decode consistency) */
    for (int m = 0; m < M; m++) {
        if (bf16)
            apus_bf16_linear(w, x + (size_t)m * K, out1, 1, K, O);
        else
            apus_f32_linear(w, x + (size_t)m * K, out1, 1, K, O);
        CHECK(memcmp(out + (size_t)m * O, out1, (size_t)O * sizeof(float)) == 0,
              "%s row %d differs between M=%d and M=1", bf16 ? "bf16" : "f32",
              m, M);
    }
    digest_f32(out, (size_t)M * O);
    free(w); free(x); free(out); free(out1);
}

/* ---- 3b. wo_a NEON dot: bounded reorder vs the old scalar order ---- */
static uint32_t f32u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void test_woa(int s, int G, int ol, int sub) {
    int hd = G * sub;
    size_t rows = (size_t)s * (size_t)G * (size_t)ol;
    uint16_t *wa = malloc((size_t)G * ol * sub * sizeof(uint16_t));
    float *o = malloc((size_t)s * hd * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    for (long i = 0; i < (long)G * ol * sub; i++)
        wa[i] = apus_bf16_bits(rng_float() * 0.05f);
    for (long i = 0; i < (long)s * hd; i++) o[i] = apus_bf16_round(rng_float());
    ApusWoAJob job = { wa, o, y, s, G, ol, sub, hd };
    apus_pool_run(rows, apus_woa_rows, &job);
    /* Reference: OLD scalar semantics (sequential-k over the exactly widened
     * BF16 weights, BF16-rounded out). The NEON 4-accumulator dot reorders
     * the FP32 sum (~1e-7 rel), so the BF16-rounded output may flip by AT
     * MOST one bf16 ulp on a rounding boundary (the documented SIMD reorder
     * tolerance class). */
    long nflip = 0;
    for (size_t r = 0; r < rows; r++) {
        size_t t = r / ((size_t)G * (size_t)ol), rr = r % ((size_t)G * (size_t)ol);
        size_t g = rr / (size_t)ol, jj = rr % (size_t)ol;
        const float *og = o + t * (size_t)hd + g * (size_t)sub;
        const uint16_t *wr = wa + (g * (size_t)ol + (size_t)jj) * (size_t)sub;
        float acc = 0.0f;
        for (int k = 0; k < sub; k++) acc += og[k] * apus_bf16_f32(wr[k]);
        float ref = apus_bf16_round(acc);
        float d = fabsf(y[r] - ref);
        float ulp = fabsf(ref) * 0.0078125f;   /* 2^-7: one bf16 ulp */
        if (f32u(y[r]) != f32u(ref)) nflip++;
        CHECK(d <= ulp * 1.5f + 1e-30f,
              "wo_a row %zu: diff %.3e exceeds one bf16 ulp of %.3e",
              r, (double)d, (double)ref);
    }
    CHECK(nflip * 100 <= rows, "wo_a: %ld/%zu bf16 flips (>1%%)", nflip, rows);
    fprintf(stderr, "wo_a %dx%dx%dx%d: %ld/%zu bf16 flips\n",
            s, G, ol, sub, nflip, rows);
    digest_f32(y, rows);
    free(wa); free(o); free(y);
}

/* ---- 4. scratch arena ---- */
static void test_scratch(void) {
    ApusScratchMark m0 = apus_scratch_mark();
    float *a = apus_scratch_alloc(1000 * sizeof(float));
    CHECK(a != NULL && ((uintptr_t)a & 63u) == 0, "scratch alignment");
    for (int i = 0; i < 1000; i++) a[i] = (float)i;
    ApusScratchMark m1 = apus_scratch_mark();
    /* force segment growth (bigger than the geometric cap ladder start) */
    float *b = apus_scratch_alloc((size_t)4 << 20);
    CHECK(b != NULL, "scratch growth alloc");
    for (size_t i = 0; i < (size_t)1 << 20; i++) b[i] = 1.0f;
    apus_scratch_reset(m1);
    float *c = apus_scratch_alloc(64 * sizeof(float));
    CHECK(c != NULL, "scratch reuse after reset");
    apus_scratch_reset(m0);
    int ok = 1;
    for (int i = 0; i < 1000; i++) if (a[i] != (float)i) ok = 0;
    CHECK(ok, "scratch: earlier allocation corrupted by later growth");
    /* dead-segment reuse with a LARGER request must grow the segment
     * (regression: the bump offset used to run past the segment end —
     * heap overflow, m5 non-determinism + T=1 segfault) */
    ApusScratchMark m3 = apus_scratch_mark();
    float *d = apus_scratch_alloc((size_t)1 << 20);      /* new 1 MB segment */
    CHECK(d != NULL, "scratch: segment alloc");
    apus_scratch_reset(m3);
    float *e = apus_scratch_alloc((size_t)4 << 20);      /* reuse, too small */
    CHECK(e != NULL, "scratch: dead-segment growth");
    for (size_t i = 0; i < (size_t)1 << 20; i++) e[i] = 3.0f;   /* touch 4 MB */
    int ok2 = 1;
    for (size_t i = 0; i < (size_t)1 << 20; i++) if (e[i] != 3.0f) ok2 = 0;
    CHECK(ok2, "scratch: grown segment contents");
    apus_scratch_reset(m3);
}

/* ---- 5. dispatch overhead (informational) ---- */
static void nop_rows(void *ctx, size_t i0, size_t i1) {
    (void)ctx; (void)i0; (void)i1;
}

static void test_overhead(void) {
    if (apus_pool_threads() <= 1) {
        CHECK(1, "single-threaded: no dispatch (placeholder check)");
        fprintf(stderr, "pool: single-threaded, no dispatch overhead\n");
        return;
    }
    /* warm up */
    for (int i = 0; i < 64; i++) apus_pool_run(64, nop_rows, NULL);
    int reps = 2000;
    double t0 = now_s();
    for (int i = 0; i < reps; i++) apus_pool_run(64, nop_rows, NULL);
    double us = (now_s() - t0) / reps * 1e6;
    fprintf(stderr, "pool: avg dispatch overhead %.1f us over %d reps\n",
            us, reps);
    CHECK(us < 200.0, "pool dispatch overhead pathological: %.1f us", us);
}

int main(void) {
    test_pool();
    test_scratch();
    /* GEMV (decode) + small GEMM (prefill) shapes */
    test_fp8_mt(1, 384, 512);
    test_fp8_mt(3, 256, 1024);
    test_fp8_mt(1, 300, 256);      /* O not a multiple of 128 (scale rows) */
    test_fp4_mt(1, 192, 256);
    test_fp4_mt(3, 128, 512);
    test_fp4_mt(1, 100, 128);
    test_f32_path(1, 512, 384, 1);
    test_f32_path(4, 384, 256, 1);
    test_f32_path(1, 512, 384, 0);
    test_f32_path(4, 384, 256, 0);
    test_f32_path(1, 100, 66, 1);  /* odd tail lengths */
    test_woa(1, 4, 64, 384);       /* 16-block main loop */
    test_woa(3, 2, 32, 390);       /* + vector/scalar tails, M-independence */
    test_woa(1, 2, 16, 7);         /* scalar tail only */
    test_overhead();
    fprintf(stderr, "threads=%d checks=%ld failures=%d\n",
            apus_pool_threads(), checks, failures);
    /* stdout is diffed across APUS_THREADS=1/4/8 by the Makefile: only
     * thread-independent content (the output digest + pass/fail) */
    printf("digest=%016llx\n", (unsigned long long)g_digest);
    if (failures) {
        printf("RESULT: FAIL (%d failures)\n", failures);
        return 1;
    }
    printf("RESULT: ok (%ld checks)\n", checks);
    return 0;
}
