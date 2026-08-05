/*
 * tests/m9d/test_m9d.c — M9d prefill compute-utilization verification.
 *
 * M9d changes (c/blas.h, c/attn.h, c/layer.h, c/moe.h):
 *   1. apus_f32_linear / apus_bf16_linear dispatch to Accelerate sgemm at
 *      M >= APUS_BLAS_M_MIN (compressor wkv/wgate, router gate, idx_wproj,
 *      pilot mixes) — FP32 summation-order reorder only (no scales in
 *      these dots), the project's accepted reorder tolerance class.
 *   2. The grouped wo_a o-proj dispatches to a batched BF16-weight BLAS
 *      GEMM at s >= APUS_BLAS_M_MIN (exact BF16->FP32 widen per tile;
 *      summation-order reorder only; output BF16-rounded as before).
 *   3. The serial per-row loops (indexer q-prep/scores/topk, compressor
 *      pooling + finish, q/kv norm+rope+QAT, hc pre/post, act quant,
 *      SwiGLU, MoE accumulation) are pooled over their independent rows
 *      with the per-row op sequence UNCHANGED — bitwise identical for any
 *      thread count by the pool.h contract (covered by the unmodified
 *      m4c/m5/m6c/m8/m9c suites, whose digests did not move).
 *
 * This test covers what the older suites cannot (they all run at M<=250):
 *   1. f32/bf16/woa BLAS paths vs FP64 truth: scale-relative error (esc)
 *      in the accepted reorder class (< 2e-5, the m7b/m9b bound).
 *   2. Dispatch boundary: M=255 stays bitwise the pinned scalar/NEON rows,
 *      M=256 is bitwise the BLAS path; repeated calls deterministic.
 *   3. Model-level (m6b fixtures) forward at s=300 (> cutoff): engages
 *      every new path; the FNV digest over logits+tokens is diffed across
 *      APUS_THREADS=1/4/8 by the Makefile (thread-count independence).
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
#define APUS_LAYER_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
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

/* ================= 1. f32 BLAS vs FP64 truth + boundary =================== */

/* The pinned scalar-order row path (identical to apus_f32_linear below the
 * cutoff — duplicated here so the boundary check does not depend on the
 * dispatch staying out of the way). */
static void ref_f32_scalar(const float *w, const float *x, float *out,
                           size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        for (size_t o = 0; o < O; o++) {
            float dot = 0.0f;
            for (size_t i = 0; i < K; i++)
                dot += x[m * K + i] * w[o * K + i];
            out[m * O + o] = dot;
        }
}

/* Large-M f32 path under test: Accelerate sgemm where it exists; on x86
 * (M12a-1, APUS_BLAS == 0) apus_f32_linear itself — which is exactly the
 * scalar-row path at every M there, so the boundary checks pin "no dispatch
 * change on this platform" (linear == ref_f32_scalar bitwise) with the same
 * structure and check counts as on macOS. NOTE the argument order differs:
 * blas (M, O, K) vs linear (M, K, O). */
#if APUS_BLAS
#define F32_GEMM_BIG(w, x, out, M, O, K) \
    apus_f32_gemm_blas(w, x, out, M, O, K)
#define F32_REF_BIG   apus_f32_gemm_blas
#define BIG_NAME "blas"
#else
#define F32_GEMM_BIG(w, x, out, M, O, K) \
    apus_f32_linear(w, x, out, (int)(M), (int)(K), (int)(O))
#define F32_REF_BIG   ref_f32_scalar
#define BIG_NAME "linear(scalar,no-blas)"
#endif

static void test_f32_blas(size_t M, size_t O, size_t K, double *worst) {
    float *w = malloc(O * K * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *ob = malloc(M * O * sizeof(float));
    float *os = malloc(M * O * sizeof(float));
    double *ot = malloc(M * O * sizeof(double));
    double *esc = malloc(M * O * sizeof(double));
    for (size_t i = 0; i < O * K; i++) w[i] = apus_bf16_round(rng_float());
    for (size_t i = 0; i < M * K; i++) x[i] = apus_bf16_round(rng_float());
    F32_GEMM_BIG(w, x, ob, M, O, K);
    ref_f32_scalar(w, x, os, M, O, K);
    for (size_t m = 0; m < M; m++)
        for (size_t o = 0; o < O; o++) {
            double dot = 0, ad = 0;
            for (size_t i = 0; i < K; i++) {
                double a = x[m * K + i], b = w[o * K + i];
                dot += a * b;
                ad += fabs(a * b);
            }
            ot[m * O + o] = dot;
            esc[m * O + o] = ad > 1e-30 ? ad : 1e-30;
        }
    double me = 0, ms = 0;
    for (size_t i = 0; i < M * O; i++) {
        double d = fabs((double)ob[i] - ot[i]);
        if (d / esc[i] > me) me = d / esc[i];
        d = fabs((double)ob[i] - (double)os[i]);
        if (d / esc[i] > ms) ms = d / esc[i];
    }
    if (me > *worst) *worst = me;
    CHECK(me < 2e-5, "f32 " BIG_NAME " M=%zu O=%zu K=%zu err/esc %.3e >= 2e-5",
          M, O, K, me);
    CHECK(ms < 2e-5, "f32 " BIG_NAME "-vs-scalar M=%zu O=%zu K=%zu %.3e >= 2e-5",
          M, O, K, ms);
    printf("  f32 " BIG_NAME " M=%-3zu O=%-4zu K=%-4zu err/esc=%.2e (vs scalar %.2e)\n",
           M, O, K, me, ms);
    digest_f32(ob, M * O);
    free(w); free(x); free(ob); free(os); free(ot); free(esc);
}

/* apus_f32_linear / apus_bf16_linear dispatch boundary at the cutoff. */
static void test_f32_dispatch(size_t O, size_t K) {
    size_t M = APUS_BLAS_M_MIN;
    float *w = malloc(O * K * sizeof(float));
    float *x = malloc(M * K * sizeof(float));
    float *o_fn = malloc(M * O * sizeof(float));
    float *o_ref = malloc(M * O * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = apus_bf16_round(rng_float());
    for (size_t i = 0; i < M * K; i++) x[i] = apus_bf16_round(rng_float());
    /* M = cutoff: the linear must be bitwise the large-M path */
    apus_f32_linear(w, x, o_fn, (int)M, (int)K, (int)O);
    F32_REF_BIG(w, x, o_ref, M, O, K);
    CHECK(memcmp(o_fn, o_ref, M * O * sizeof(float)) == 0,
          "f32 dispatch: linear(M=%zu) != " BIG_NAME, M);
    /* M = cutoff-1: bitwise the pinned scalar-order rows */
    apus_f32_linear(w, x, o_fn, (int)M - 1, (int)K, (int)O);
    ref_f32_scalar(w, x, o_ref, M - 1, O, K);
    CHECK(memcmp(o_fn, o_ref, (M - 1) * O * sizeof(float)) == 0,
          "f32 dispatch: linear(M=%zu) != scalar rows", M - 1);
    /* bf16 linear: same boundary (with its input/output rounding) */
    {
        float *xb = malloc(M * K * sizeof(float));
        float *ob = malloc(M * O * sizeof(float));
        float *os = malloc(M * O * sizeof(float));
        for (size_t i = 0; i < M * K; i++)
            xb[i] = apus_bf16_round(x[i]);
        apus_bf16_linear(w, x, ob, (int)M - 1, (int)K, (int)O);
        ref_f32_scalar(w, xb, os, M - 1, O, K);
        for (size_t i = 0; i < (M - 1) * O; i++)
            os[i] = apus_bf16_round(os[i]);
        CHECK(memcmp(ob, os, (M - 1) * O * sizeof(float)) == 0,
              "bf16 dispatch: linear(M=%zu) != scalar rows", M - 1);
        apus_bf16_linear(w, x, ob, (int)M, (int)K, (int)O);
        F32_REF_BIG(w, xb, os, M, O, K);
        for (size_t i = 0; i < M * O; i++)
            os[i] = apus_bf16_round(os[i]);
        CHECK(memcmp(ob, os, M * O * sizeof(float)) == 0,
              "bf16 dispatch: linear(M=%zu) != " BIG_NAME "+round", M);
        free(xb); free(ob); free(os);
    }
    /* determinism: repeated BLAS call bitwise */
    apus_f32_linear(w, x, o_fn, (int)M, (int)K, (int)O);
    apus_f32_linear(w, x, o_ref, (int)M, (int)K, (int)O);
    CHECK(memcmp(o_fn, o_ref, M * O * sizeof(float)) == 0,
          "f32 blas not deterministic (M=%zu)", M);
    printf("  f32/bf16 dispatch boundary M=%zu/%zu + determinism ok"
           " (O=%zu K=%zu)\n", M - 1, M, O, K);
    digest_f32(o_fn, M * O);
    free(w); free(x); free(o_fn); free(o_ref);
}

/* ================= 2. grouped wo_a BLAS vs FP64 + pinned rows ============= */

#if !APUS_BLAS
/* Raw (unrounded) grouped wo_a dots in the pinned scalar sequential-k
 * order — the x86 (M12a-1, APUS_BLAS == 0) stand-in for apus_woa_gemm_blas:
 * the engine's only path there is apus_woa_rows (which BF16-rounds), so the
 * FP64-truth comparison needs the unrounded scalar values; the flip
 * accounting below then compares round(scalar) vs the rows path (bitwise
 * equal — the rows path IS this dot plus the round). */
static void ref_woa_scalar(const uint16_t *wa, const float *o, float *y,
                           size_t M, size_t G, size_t ol, size_t sub,
                           size_t hd) {
    for (size_t m = 0; m < M; m++)
        for (size_t g = 0; g < G; g++)
            for (size_t jj = 0; jj < ol; jj++) {
                const float *og = o + m * hd + g * sub;
                const uint16_t *wr = wa + (g * ol + jj) * sub;
                float dot = 0.0f;
                for (size_t i = 0; i < sub; i++)
                    dot += og[i] * apus_bf16_f32(wr[i]);
                y[(m * G + g) * ol + jj] = dot;
            }
}
#endif /* !APUS_BLAS */

static void test_woa_blas(size_t M, size_t G, size_t ol, size_t sub,
                          double *worst) {
    size_t hd = G * sub;
    uint16_t *wa = malloc(G * ol * sub * sizeof(uint16_t));
    float *o = malloc(M * hd * sizeof(float));
    float *yb = malloc(M * G * ol * sizeof(float));
    float *yn = malloc(M * G * ol * sizeof(float));
    double *yt = malloc(M * G * ol * sizeof(double));
    double *esc = malloc(M * G * ol * sizeof(double));
    for (size_t i = 0; i < G * ol * sub; i++) {
        /* finite BF16 weights (real checkpoints never carry inf/nan) */
        float v = apus_bf16_round(rng_float());
        uint32_t u;
        memcpy(&u, &v, 4);
        wa[i] = (uint16_t)(u >> 16);
    }
    for (size_t i = 0; i < M * hd; i++) o[i] = apus_bf16_round(rng_float());
#if APUS_BLAS
    apus_woa_gemm_blas(wa, o, yb, M, G, ol, sub, hd);
#else
    ref_woa_scalar(wa, o, yb, M, G, ol, sub, hd);   /* M12a-1 x86 (above) */
#endif
    /* the pinned row path (static in this TU, M-independent per row);
     * it BF16-rounds its outputs, so the comparison below rounds the raw
     * large-M values the same way */
    {
        ApusWoAJob job = { wa, o, yn, (int)M, (int)G, (int)ol, (int)sub,
                           (int)hd };
        apus_woa_rows(&job, 0, M * G * ol);
    }
    for (size_t m = 0; m < M; m++)
        for (size_t g = 0; g < G; g++)
            for (size_t jj = 0; jj < ol; jj++) {
                double dot = 0, ad = 0;
                const float *og = o + m * hd + g * sub;
                const uint16_t *wr = wa + (g * ol + jj) * sub;
                for (size_t i = 0; i < sub; i++) {
                    double a = og[i], b = apus_bf16_f32(wr[i]);
                    dot += a * b;
                    ad += fabs(a * b);
                }
                yt[(m * G + g) * ol + jj] = dot;
                esc[(m * G + g) * ol + jj] = ad > 1e-30 ? ad : 1e-30;
            }
    double me = 0;
    size_t nflip = 0;
    int bad_ulp = 0;
    for (size_t i = 0; i < M * G * ol; i++) {
        double d = fabs((double)yb[i] - yt[i]);
        if (d / esc[i] > me) me = d / esc[i];
        /* BLAS-vs-NEON, both BF16-rounded: a reorder delta can flip an
         * output by ONE bf16 ulp when it lands on a rounding boundary
         * (the accepted reorder class, see the M6c wo_a notes) — count
         * the flips and require every one to be a single-ulp step. */
        float rb = apus_bf16_round(yb[i]);
        if (memcmp(&rb, &yn[i], 4) != 0) {
            uint32_t ub, un;
            memcpy(&ub, &rb, 4);
            memcpy(&un, &yn[i], 4);
            int32_t cb = (int32_t)(ub >> 16), cn = (int32_t)(un >> 16);
            int32_t diff = cb - cn;
            if (diff < 0) diff = -diff;
            if (diff != 1) bad_ulp = 1;
            nflip++;
        }
    }
    if (me > *worst) *worst = me;
    CHECK(me < 2e-5, "woa " BIG_NAME " M=%zu err/esc %.3e >= 2e-5", M, me);
    CHECK(!bad_ulp, "woa " BIG_NAME "-vs-rows M=%zu: a flip exceeds 1 bf16 ulp", M);
    CHECK(nflip * 1000 <= M * G * ol,
          "woa " BIG_NAME "-vs-rows M=%zu: %zu/%zu flips (> 1e-3)", M, nflip,
          M * G * ol);
    printf("  woa " BIG_NAME " M=%-3zu G=%zu ol=%-4zu sub=%-4zu err/esc=%.2e"
           " (vs rows: %zu/%zu 1-ulp flips)\n", M, G, ol, sub, me, nflip,
           M * G * ol);
    digest_f32(yb, M * G * ol);
    free(wa); free(o); free(yb); free(yn); free(yt); free(esc);
}

/* ================= 3. model-level s>cutoff forward digest ================= */

#define FIX "tests/m6b/fixtures"

static void test_model_s300(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, FIX, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        g_checks++; g_fails++;
        return;
    }
    int V = m.cfg.vocab_size;
    int S = 300;                    /* > APUS_BLAS_M_MIN: engages M9d */
    int64_t *ids = malloc((size_t)S * sizeof(int64_t));
    srand(4242);
    for (int i = 0; i < S; i++) ids[i] = rand() % V;
    float *logits = malloc((size_t)V * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &m);
    apus_model_forward(&m, &st, ids, S, logits, 0);
    digest_f32(logits, (size_t)V);
    /* a few greedy decode steps after the s=300 prefill (decode stays on
     * the pinned M=1 paths; the digest pins prefill->decode consistency) */
    void *scr = malloc(apus_sample_scratch_size((size_t)V));
    ApusRng rng;
    apus_rng_seed(&rng, 1);
    int gen[8];
    for (int i = 0; i < 8; i++) {
        int t = apus_sample(logits, (size_t)V, 0.0f, 1.0f, &rng, scr);
        gen[i] = t;
        int64_t nx = t;
        apus_model_forward(&m, &st, &nx, 1, logits, 0);
        digest_f32(logits, (size_t)V);
    }
    printf("  model s=%d forward + 8 greedy: %d %d %d %d %d %d %d %d\n",
           S, gen[0], gen[1], gen[2], gen[3], gen[4], gen[5], gen[6],
           gen[7]);
    apus_model_state_free(&st, &m);
    apus_model_free(&m);
    free(ids); free(logits); free(scr);
}

int main(void) {
    printf("test_m9d: dense/woa BLAS dispatch (M>=%d) + pooled row loops\n",
           APUS_BLAS_M_MIN);
#if APUS_BLAS
    CHECK(apus_blas_available(), "BLAS path not available on this host");
#else
    /* M12a-1 x86: no BLAS — every M stays on the pinned scalar rows; the
     * checks below pin exactly that (BIG == linear/ref scalar). */
    CHECK(1, "no BLAS on this platform (placeholder)");
#endif
    double worstf = 0, worstw = 0;
    /* engaged real-model shapes: compressor wkv/wgate, router, idx_wproj */
    test_f32_blas(256, 1024, 4096, &worstf);   /* compressor (ratio 4) */
    test_f32_blas(300, 256, 4096, &worstf);    /* router gate */
    test_f32_blas(257, 64, 4096, &worstf);     /* idx_wproj, odd M/small O */
    test_f32_blas(512, 512, 4096, &worstf);    /* compressor (ratio 128) */
    test_f32_blas(256, 256, 512, &worstf);     /* idx compressor cd */
    printf("  worst f32 err/esc: %.3e (bound 2e-5)\n", worstf);
    test_f32_dispatch(384, 256);
    test_f32_dispatch(64, 512);
    /* grouped wo_a (real: G=8, ol=1024, sub=4096; smaller here for f64) */
    test_woa_blas(256, 8, 128, 512, &worstw);
    test_woa_blas(300, 8, 64, 256, &worstw);
    test_woa_blas(257, 4, 96, 384, &worstw);
    printf("  worst woa err/esc: %.3e (bound 2e-5)\n", worstw);
    test_model_s300();
    printf("%s: %d checks, %d failures\n",
           g_fails ? "FAIL" : "ok", g_checks, g_fails);
    printf("digest=%016llx\n", (unsigned long long)g_digest);
    return g_fails != 0;
}
