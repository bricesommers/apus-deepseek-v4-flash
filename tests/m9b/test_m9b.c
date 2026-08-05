/*
 * tests/m9b/test_m9b.c — M9b BLAS (Accelerate/AMX) dispatch verification.
 *
 *   1. fp8/fp4 BLAS GEMM vs FP64 truth: scale-relative error (esc) must sit
 *      in the accepted FP32-accumulation-reorder class (< 2e-5, same bound
 *      as the m7b GPU battery) — the scale-application sequence is exact
 *      (UE8M0 power-of-two folds, c/blas.h), only summation order differs.
 *   2. Dispatch boundary: apus_fp{4,8}_gemm_mt at M=255 is BITWISE the M9a
 *      NEON kernel; at M=256 (APUS_BLAS_M_MIN) it is BITWISE the BLAS
 *      kernel. Decode (M=1) path untouched (gemv/gemm_neon not dispatched).
 *   3. MoE prefill batching (c/moe.h): s>1 forward BITWISE == s per-token
 *      forwards (routed expert grouping + shared expert batching +
 *      (t,j)-order accumulation), non-hash and hash routing.
 *   4. Determinism: repeated BLAS GEMM bitwise identical.
 *
 * The process digest (FNV-1a over output bits) is diffed across
 * APUS_THREADS=1/4/8 by the Makefile — the BLAS path pins
 * VECLIB_MAXIMUM_THREADS=1, so its outputs are thread-count independent.
 *
 * Exit 0 iff all checks pass.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_MHC_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moe.h"
#include "blas.h"

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { g_fails++; printf("  FAIL: " __VA_ARGS__); \
                   printf("\n"); } \
} while (0)

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                   - 2.0);
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

/* Large-M GEMM path under test: Accelerate BLAS where it exists; on x86
 * (M12a-1, APUS_BLAS == 0) the normative scalar GEMM — which is exactly
 * what apus_fp{4,8}_gemm_mt dispatches to at EVERY M there, so the
 * dispatch-boundary checks below pin "no dispatch change on this platform"
 * with the same structure and check counts as on macOS. */
#if APUS_BLAS
#define FP8_GEMM_BIG apus_fp8_gemm_blas
#define FP4_GEMM_BIG apus_fp4_gemm_blas
#define BIG_NAME "blas"
#else
#define FP8_GEMM_BIG apus_fp8_gemm_scalar
#define FP4_GEMM_BIG apus_fp4_gemm_scalar
#define BIG_NAME "scalar(no-blas)"
#endif
/* Small-M anchor: NEON on ARM, scalar elsewhere. */
#ifdef __ARM_NEON
#define FP8_GEMM_ANCHOR apus_fp8_gemm_neon
#define FP4_GEMM_ANCHOR apus_fp4_gemm_neon
#define ANCHOR_NAME "neon"
#else
#define FP8_GEMM_ANCHOR apus_fp8_gemm_scalar
#define FP4_GEMM_ANCHOR apus_fp4_gemm_scalar
#define ANCHOR_NAME "scalar"
#endif

/* ================= 1. BLAS vs FP64 truth (esc class) ====================== */

static void test_fp8_blas(size_t M, size_t O, size_t K, double *worst) {
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    size_t nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * K), *ws = malloc(nbo * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *ob = malloc(M * O * sizeof(float));
    float *on = malloc(M * O * sizeof(float));
    double *ot = malloc(M * O * sizeof(double));
    double *esc = malloc(M * O * sizeof(double));
    for (size_t i = 0; i < O * K; i++) {
        uint8_t b = (uint8_t)rng_u64();
        if ((b & 0x7F) == 0x7F) b ^= 1;   /* keep NaN codes out */
        w[i] = b;
    }
    for (size_t i = 0; i < nbo * nb; i++)
        ws[i] = (uint8_t)(96 + rng_u64() % 65);   /* 2^-31..2^33, like m7b */
    for (size_t i = 0; i < M * K; i++) x[i] = rng_float();
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[m * K + i] = apus_bf16_round(x[m * K + i]);
        apus_fp4_act_quant_scalar(x + m * K, K, codes + m * K, as + m * nab);
    }
    FP8_GEMM_BIG(w, ws, codes, as, scratch, ob, M, O, K);
    /* f64 truth in the normative block-fold order + error scale */
    for (size_t m = 0; m < M; m++)
        for (size_t o = 0; o < O; o++) {
            double tot = 0, e = 0;
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * 128, hi = lo + 128 > K ? K : lo + 128;
                double dot = 0, ad = 0;
                for (size_t i = lo; i < hi; i++) {
                    double a = apus_e4m3_dequant_f32(codes[m * K + i]);
                    double ww = apus_e4m3_dequant_f32(w[o * K + i]);
                    dot += a * ww;
                    ad += fabs(a * ww);
                }
                double sc = (double)as[m * nab + kb]
                          * apus_ue8m0_f32(ws[(o / 128) * nb + kb]);
                tot += dot * sc;
                e += ad * fabs(sc);
            }
            ot[m * O + o] = tot;
            esc[m * O + o] = e > 1e-30 ? e : 1e-30;
        }
    double me = 0;
    for (size_t i = 0; i < M * O; i++) {
        double d = fabs((double)ob[i] - ot[i]);
        if (d / esc[i] > me) me = d / esc[i];
    }
    if (me > *worst) *worst = me;
    CHECK(me < 2e-5, "fp8 " BIG_NAME " M=%zu O=%zu K=%zu err/esc %.3e >= 2e-5",
          M, O, K, me);
    printf("  fp8 " BIG_NAME " M=%-3zu O=%-4zu K=%-4zu err/esc=%.2e\n", M, O, K, me);
    digest_f32(ob, M * O);
    /* anchor comparison for the report (reorder class, not gated bitwise) */
    FP8_GEMM_ANCHOR(w, ws, codes, as, scratch, on, M, O, K);
    double mn = 0;
    for (size_t i = 0; i < M * O; i++) {
        double d = fabs((double)ob[i] - (double)on[i]);
        if (d / esc[i] > mn) mn = d / esc[i];
    }
    printf("         vs " ANCHOR_NAME " (reorder class): %.2e\n", mn);
    CHECK(mn < 2e-5, "fp8 " BIG_NAME "-vs-" ANCHOR_NAME " M=%zu O=%zu K=%zu %.3e >= 2e-5",
          M, O, K, mn);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch);
    free(ob); free(on); free(ot); free(esc);
}

static void test_fp4_blas(size_t M, size_t O, size_t K, double *worst) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    uint8_t *w = malloc(O * (K / 2)), *ws = malloc(O * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *ob = malloc(M * O * sizeof(float));
    float *on = malloc(M * O * sizeof(float));
    double *ot = malloc(M * O * sizeof(double));
    double *esc = malloc(M * O * sizeof(double));
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(96 + rng_u64() % 65);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_float();
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[m * K + i] = apus_bf16_round(x[m * K + i]);
        apus_fp4_act_quant_scalar(x + m * K, K, codes + m * K, as + m * nab);
    }
    FP4_GEMM_BIG(w, ws, codes, as, scratch, ob, M, O, K);
    for (size_t m = 0; m < M; m++)
        for (size_t o = 0; o < O; o++) {
            double tot = 0, e = 0;
            for (size_t kb = 0; kb < nb; kb++) {
                double dot = 0, ad = 0;
                const uint8_t *p = w + o * (K / 2) + kb * 16;
                for (size_t i = 0; i < 16; i++) {
                    double a0 = apus_e4m3_dequant_f32(
                        codes[m * K + kb * 32 + 2 * i]);
                    double a1 = apus_e4m3_dequant_f32(
                        codes[m * K + kb * 32 + 2 * i + 1]);
                    double w0 = apus_fp4_lut[p[i] & 0x0F];
                    double w1 = apus_fp4_lut[p[i] >> 4];
                    dot += a0 * w0 + a1 * w1;
                    ad += fabs(a0 * w0) + fabs(a1 * w1);
                }
                double sc = (double)as[m * nab + kb / 4]
                          * apus_ue8m0_f32(ws[o * nb + kb]);
                tot += dot * sc;
                e += ad * fabs(sc);
            }
            ot[m * O + o] = tot;
            esc[m * O + o] = e > 1e-30 ? e : 1e-30;
        }
    double me = 0;
    for (size_t i = 0; i < M * O; i++) {
        double d = fabs((double)ob[i] - ot[i]);
        if (d / esc[i] > me) me = d / esc[i];
    }
    if (me > *worst) *worst = me;
    CHECK(me < 2e-5, "fp4 " BIG_NAME " M=%zu O=%zu K=%zu err/esc %.3e >= 2e-5",
          M, O, K, me);
    printf("  fp4 " BIG_NAME " M=%-3zu O=%-4zu K=%-4zu err/esc=%.2e\n", M, O, K, me);
    digest_f32(ob, M * O);
    FP4_GEMM_ANCHOR(w, ws, codes, as, scratch, on, M, O, K);
    double mn = 0;
    for (size_t i = 0; i < M * O; i++) {
        double d = fabs((double)ob[i] - (double)on[i]);
        if (d / esc[i] > mn) mn = d / esc[i];
    }
    printf("         vs " ANCHOR_NAME " (reorder class): %.2e\n", mn);
    CHECK(mn < 2e-5, "fp4 " BIG_NAME "-vs-" ANCHOR_NAME " M=%zu O=%zu K=%zu %.3e >= 2e-5",
          M, O, K, mn);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch);
    free(ob); free(on); free(ot); free(esc);
}

/* ================= 2. dispatch boundary =================================== */

static void test_dispatch_fp8(size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K), nbo = (O + 127) / 128;
    size_t nab = apus_fp4_act_blocks(K);
    size_t M = APUS_BLAS_M_MIN;
    uint8_t *w = malloc(O * K), *ws = malloc(nbo * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *o_mt = malloc(M * O * sizeof(float));
    float *o_ref = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * K; i++) {
        uint8_t b = (uint8_t)rng_u64();
        if ((b & 0x7F) == 0x7F) b ^= 1;
        w[i] = b;
    }
    for (size_t i = 0; i < nbo * nb; i++)
        ws[i] = (uint8_t)(110 + rng_u64() % 20);
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[i] = apus_bf16_round(rng_float());
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nab);
    }
    /* M = cutoff: mt must be bitwise the large-M path */
    apus_fp8_gemm_mt(w, ws, codes, as, scratch, o_mt, M, O, K);
    FP8_GEMM_BIG(w, ws, codes, as, scratch, o_ref, M, O, K);
    CHECK(memcmp(o_mt, o_ref, M * O * sizeof(float)) == 0,
          "fp8 dispatch: mt(M=%zu) != " BIG_NAME, M);
    /* M = cutoff-1: mt must be bitwise the small-M anchor */
    apus_fp8_gemm_mt(w, ws, codes, as, scratch, o_mt, M - 1, O, K);
    FP8_GEMM_ANCHOR(w, ws, codes, as, scratch, o_ref, M - 1, O, K);
    CHECK(memcmp(o_mt, o_ref, (M - 1) * O * sizeof(float)) == 0,
          "fp8 dispatch: mt(M=%zu) != " ANCHOR_NAME, M - 1);
    digest_f32(o_mt, (M - 1) * O);
    /* determinism: repeated large-M call bitwise */
    FP8_GEMM_BIG(w, ws, codes, as, scratch, o_mt, M, O, K);
    float *o2 = malloc(M * O * sizeof(float));
    FP8_GEMM_BIG(w, ws, codes, as, scratch, o2, M, O, K);
    CHECK(memcmp(o_mt, o2, M * O * sizeof(float)) == 0,
          "fp8 " BIG_NAME " not deterministic (M=%zu)", M);
    printf("  fp8 dispatch boundary M=%zu/%zu + determinism ok (O=%zu K=%zu)\n",
           M - 1, M, O, K);
    free(o2);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch);
    free(o_mt); free(o_ref);
}

static void test_dispatch_fp4(size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP, nab = apus_fp4_act_blocks(K);
    size_t M = APUS_BLAS_M_MIN;
    uint8_t *w = malloc(O * (K / 2)), *ws = malloc(O * nb);
    uint8_t *codes = malloc(M * K);
    float *as = malloc(M * nab * sizeof(float));
    float *x = malloc(K * sizeof(float));
    float *scratch = malloc(M * K * sizeof(float));
    float *o_mt = malloc(M * O * sizeof(float));
    float *o_ref = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * (K / 2); i++) w[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < O * nb; i++)
        ws[i] = (uint8_t)(110 + rng_u64() % 20);
    for (size_t m = 0; m < M; m++) {
        for (size_t i = 0; i < K; i++) x[i] = apus_bf16_round(rng_float());
        apus_fp4_act_quant_scalar(x, K, codes + m * K, as + m * nab);
    }
    apus_fp4_gemm_mt(w, ws, codes, as, scratch, o_mt, M, O, K);
    FP4_GEMM_BIG(w, ws, codes, as, scratch, o_ref, M, O, K);
    CHECK(memcmp(o_mt, o_ref, M * O * sizeof(float)) == 0,
          "fp4 dispatch: mt(M=%zu) != " BIG_NAME, M);
    apus_fp4_gemm_mt(w, ws, codes, as, scratch, o_mt, M - 1, O, K);
    FP4_GEMM_ANCHOR(w, ws, codes, as, scratch, o_ref, M - 1, O, K);
    CHECK(memcmp(o_mt, o_ref, (M - 1) * O * sizeof(float)) == 0,
          "fp4 dispatch: mt(M=%zu) != " ANCHOR_NAME, M - 1);
    digest_f32(o_mt, (M - 1) * O);
    FP4_GEMM_BIG(w, ws, codes, as, scratch, o_mt, M, O, K);
    float *o2 = malloc(M * O * sizeof(float));
    FP4_GEMM_BIG(w, ws, codes, as, scratch, o2, M, O, K);
    CHECK(memcmp(o_mt, o2, M * O * sizeof(float)) == 0,
          "fp4 " BIG_NAME " not deterministic (M=%zu)", M);
    printf("  fp4 dispatch boundary M=%zu/%zu + determinism ok (O=%zu K=%zu)\n",
           M - 1, M, O, K);
    free(o2);
    free(w); free(ws); free(codes); free(as); free(x); free(scratch);
    free(o_mt); free(o_ref);
}

/* ================= 3. MoE prefill batching bitwise ======================== */

typedef struct {
    ApusMoeW w;
    ApusFp4W *e1, *e2, *e3;          /* [E] views into the blobs below */
    uint8_t *p1, *s1, *p2, *s2, *p3, *s3;
    uint8_t *sp1, *ss1, *sp2, *ss2, *sp3, *ss3;
    float *gate_w, *gate_bias;
    int64_t *tid2eid;
} MoeFixture;

static void moe_fixture_init(MoeFixture *f, int E, int topk, int dim,
                             int inter, int hash) {
    memset(f, 0, sizeof *f);
    size_t nb4a = (size_t)dim / 32, nb4b = (size_t)inter / 32;
    size_t nb8a = apus_fp8_blocks((size_t)dim),
           nb8b = apus_fp8_blocks((size_t)inter);
    size_t nbo8a = ((size_t)inter + 127) / 128, nbo8b = ((size_t)dim + 127) / 128;
    f->p1 = malloc((size_t)E * inter * (dim / 2));
    f->s1 = malloc((size_t)E * inter * nb4a);
    f->p3 = malloc((size_t)E * inter * (dim / 2));
    f->s3 = malloc((size_t)E * inter * nb4a);
    f->p2 = malloc((size_t)E * dim * (inter / 2));
    f->s2 = malloc((size_t)E * dim * nb4b);
    f->sp1 = malloc((size_t)inter * dim); f->ss1 = malloc(nbo8a * nb8a);
    f->sp3 = malloc((size_t)inter * dim); f->ss3 = malloc(nbo8a * nb8a);
    f->sp2 = malloc((size_t)dim * inter); f->ss2 = malloc(nbo8b * nb8b);
    f->gate_w = malloc((size_t)E * dim * sizeof(float));
    f->gate_bias = malloc((size_t)E * sizeof(float));
    f->tid2eid = malloc(16 * (size_t)topk * sizeof(int64_t));
    for (size_t i = 0; i < (size_t)E * inter * (dim / 2); i++)
        f->p1[i] = f->p3[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < (size_t)E * dim * (inter / 2); i++)
        f->p2[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < (size_t)E * inter * nb4a; i++)
        f->s1[i] = f->s3[i] = (uint8_t)(118 + rng_u64() % 12);
    for (size_t i = 0; i < (size_t)E * dim * nb4b; i++)
        f->s2[i] = (uint8_t)(118 + rng_u64() % 12);
    for (size_t i = 0; i < (size_t)inter * dim; i++) {
        uint8_t b = (uint8_t)rng_u64();
        f->sp1[i] = f->sp3[i] = (b & 0x7F) == 0x7F ? b ^ 1 : b;
    }
    for (size_t i = 0; i < (size_t)dim * inter; i++) {
        uint8_t b = (uint8_t)rng_u64();
        f->sp2[i] = (b & 0x7F) == 0x7F ? b ^ 1 : b;
    }
    for (size_t i = 0; i < nbo8a * nb8a; i++)
        f->ss1[i] = f->ss3[i] = (uint8_t)(118 + rng_u64() % 12);
    for (size_t i = 0; i < nbo8b * nb8b; i++)
        f->ss2[i] = (uint8_t)(118 + rng_u64() % 12);
    for (int i = 0; i < E * dim; i++)
        f->gate_w[i] = apus_bf16_round(rng_float() * 0.1f);
    for (int i = 0; i < E; i++)
        f->gate_bias[i] = apus_bf16_round(rng_float() * 0.01f);
    for (int i = 0; i < 16 * topk; i++)
        f->tid2eid[i] = (int64_t)(rng_u64() % (uint64_t)E);
    f->e1 = malloc((size_t)E * sizeof(ApusFp4W));
    f->e2 = malloc((size_t)E * sizeof(ApusFp4W));
    f->e3 = malloc((size_t)E * sizeof(ApusFp4W));
    for (int e = 0; e < E; e++) {
        f->e1[e] = (ApusFp4W){ f->p1 + (size_t)e * inter * (dim / 2),
                               f->s1 + (size_t)e * inter * nb4a, inter, dim };
        f->e3[e] = (ApusFp4W){ f->p3 + (size_t)e * inter * (dim / 2),
                               f->s3 + (size_t)e * inter * nb4a, inter, dim };
        f->e2[e] = (ApusFp4W){ f->p2 + (size_t)e * dim * (inter / 2),
                               f->s2 + (size_t)e * dim * nb4b, dim, inter };
    }
    f->w.E = E; f->w.topk = topk; f->w.inter = inter; f->w.dim = dim;
    f->w.hash = hash;
    f->w.route_scale = 1.5f; f->w.limit = 10.0f;
    f->w.gate_w = f->gate_w;
    f->w.gate_bias = hash ? NULL : f->gate_bias;
    f->w.tid2eid = hash ? f->tid2eid : NULL;
    f->w.w1 = f->e1; f->w.w2 = f->e2; f->w.w3 = f->e3;
    f->w.sw1 = (ApusFp8W){ f->sp1, f->ss1, inter, dim };
    f->w.sw3 = (ApusFp8W){ f->sp3, f->ss3, inter, dim };
    f->w.sw2 = (ApusFp8W){ f->sp2, f->ss2, dim, inter };
    f->w.layer_id = 0;
}

static void moe_fixture_free(MoeFixture *f) {
    free(f->p1); free(f->s1); free(f->p2); free(f->s2); free(f->p3); free(f->s3);
    free(f->sp1); free(f->ss1); free(f->sp2); free(f->ss2); free(f->sp3); free(f->ss3);
    free(f->gate_w); free(f->gate_bias); free(f->tid2eid);
    free(f->e1); free(f->e2); free(f->e3);
}

static void test_moe_batch(int hash) {
    const int E = 8, topk = 3, dim = 128, inter = 64, s = 5;
    MoeFixture f;
    moe_fixture_init(&f, E, topk, dim, inter, hash);
    float *x = malloc((size_t)s * dim * sizeof(float));
    int64_t *ids = malloc((size_t)s * sizeof(int64_t));
    float *o_batch = malloc((size_t)s * dim * sizeof(float));
    float *o_seq = malloc((size_t)s * dim * sizeof(float));
    for (int i = 0; i < s * dim; i++) x[i] = apus_bf16_round(rng_float());
    for (int t = 0; t < s; t++) ids[t] = t;
    apus_moe_forward(&f.w, x, ids, s, o_batch, NULL);
    for (int t = 0; t < s; t++)
        apus_moe_forward(&f.w, x + (size_t)t * dim, ids + t, 1,
                         o_seq + (size_t)t * dim, NULL);
    size_t nbit = 0;
    for (size_t i = 0; i < (size_t)s * dim; i++) {
        uint32_t ua, ub;
        memcpy(&ua, o_batch + i, 4);
        memcpy(&ub, o_seq + i, 4);
        if (ua != ub) nbit++;
    }
    CHECK(nbit == 0, "moe batch (hash=%d): %zu/%d outputs not bitwise vs M=1",
          hash, nbit, s * dim);
    printf("  moe batch s=%d (hash=%d): bitwise %d/%d vs per-token M=1\n",
           s, hash, s * dim - (int)nbit, s * dim);
    digest_f32(o_batch, (size_t)s * dim);
    moe_fixture_free(&f);
    free(x); free(ids); free(o_batch); free(o_seq);
}

int main(void) {
    printf("test_m9b: BLAS (Accelerate) dispatch + MoE prefill batching\n");
    printf("  blas available: %d (cutoff M>=%d)\n",
           apus_blas_available(), APUS_BLAS_M_MIN);
#if APUS_BLAS
    CHECK(apus_blas_available(), "BLAS path not available on this host");
#else
    /* M12a-1 x86: no BLAS — mt stays on the scalar path at every M; the
     * checks below pin exactly that (BIG == scalar, ANCHOR == scalar). */
    CHECK(1, "no BLAS on this platform (placeholder)");
#endif
    double worst8 = 0, worst4 = 0;
    /* cutoff-sized and larger shapes, incl. odd O and partial K blocks */
    test_fp8_blas(256, 384, 256, &worst8);
    test_fp8_blas(300, 200, 384, &worst8);   /* odd O (not x128) */
    test_fp8_blas(256, 128, 1024, &worst8);
    test_fp8_blas(257, 512, 640, &worst8);   /* odd M */
    test_fp4_blas(256, 384, 256, &worst4);
    test_fp4_blas(300, 200, 384, &worst4);
    test_fp4_blas(256, 128, 1024, &worst4);
    test_fp4_blas(257, 512, 640, &worst4);
    printf("  worst err/esc: fp8 %.3e, fp4 %.3e (bound 2e-5)\n", worst8, worst4);
    test_dispatch_fp8(384, 256);
    test_dispatch_fp4(384, 256);
    test_moe_batch(0);
    test_moe_batch(1);
    printf("%s: %d checks, %d failures\n",
           g_fails ? "FAIL" : "ok", g_checks, g_fails);
    printf("digest=%016llx\n", (unsigned long long)g_digest);
    return g_fails != 0;
}
