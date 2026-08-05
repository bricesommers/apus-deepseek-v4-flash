/*
 * tests/m9e/test_m9e.c — M9e routed-expert dispatch-grouping verification.
 *
 * M9e changes (c/fp4.h, c/moe.h):
 *   1. apus_fp4_gemm_mt_grouped: several independent same-(O,K) fp4 GEMMs
 *      in ONE pool dispatch (entry-major unit space). Scheduling only —
 *      every output row keeps the exact M9a per-row accumulation order.
 *   2. c/moe.h batches routed experts with count <= APUS_MOE_GROUP_ROWS
 *      into groups sharing one act quant (w1/w3 share it — the per-expert
 *      path computed the same codes twice), one grouped w1+w3 GEMM, one
 *      SwiGLU, one grouped w2 GEMM. Larger counts run the pre-M9e solo
 *      path (count >= 256 keeps the M9b BLAS dispatch).
 *
 * This test covers what the older suites cover only at their own shapes:
 *   1. grouped GEMM BITWISE == per-entry apus_fp4_gemm_mt, at the real
 *      expert shapes and odd ones, incl. single-row entries, a single
 *      entry, >64 entries (chunk loop), and counts {1..33}.
 *   2. M=1 PIN: apus_fp4_gemm_mt at M=1 and the grouped variant are
 *      bitwise identical to the pre-M9e (M9a) 4-row-block kernel body
 *      (a verbatim copy lives below) — the decode path must not move.
 *   3. MoE-level: s>1 forward BITWISE == s per-token M=1 forwards, with
 *      expert counts engineered (hash routing) to force multi-group
 *      flushes AND a solo (>GROUP_ROWS) expert in one call.
 * The FNV digest over all outputs is diffed across APUS_THREADS=1/4/8 by
 * the Makefile (thread-count independence of the grouped dispatches).
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

/* ============ pre-M9e (M9a) reference kernel: verbatim HEAD body ==========*/
/* The 4-row-block NEON GEMM rows as pinned by M9a..M9d, single-threaded
 * (full o range) — the anchor every new dispatch must reproduce bitwise.
 * M12a-1 x86: no NEON — the anchor is the normative scalar GEMM, which
 * apus_fp4_gemm_mt dispatches to (and mirrors step for step) at every M. */
static void ref_gemm_m9a(const uint8_t *w, const uint8_t *ws,
                         const uint8_t *acodes, const float *as,
                         float *scratch, float *out,
                         size_t M, size_t O, size_t K) {
#ifdef __ARM_NEON
    float16_t *acts16 = (float16_t *)scratch;
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_dequant_f16_all(acodes + m * K, acts16 + m * K, K);
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        for (size_t o = 0; o < O; o++) {
            const uint8_t *wp = w + o * (K / 2);
            const uint8_t *sp = ws + o * nb;
            float32x4_t total[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                    vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (size_t kb = 0; kb < nb; kb++) {
                float16x8_t k[4];
                apus_fp4_expand32_f16_neon(wp + kb * 16, k);
                float32x4_t sb = vdupq_n_f32(apus_ue8m0_f32(sp[kb]));
                for (size_t r = 0; r < mc; r++) {
                    const float16_t *ap = acts16 + (m0 + r) * K
                                        + kb * APUS_FP4_GROUP;
                    float32x4_t acc = apus_fp4_dot32_f16_neon(ap, k);
                    float32x4_t t = vmulq_f32(acc, vdupq_n_f32(
                        0.5f * as[(m0 + r) * nab + kb / 4]));
                    total[r] = vaddq_f32(total[r], vmulq_f32(t, sb));
                }
            }
            for (size_t r = 0; r < mc; r++)
                out[(m0 + r) * O + o] = apus_fp4_hsum4_neon(total[r]);
        }
    }
#else
    apus_fp4_gemm_scalar(w, ws, acodes, as, scratch, out, M, O, K);
#endif
}

/* ================= 1.+2. grouped GEMM bitwise + M=1 pin ===================*/

static void test_grouped(size_t O, size_t K, const size_t *counts, size_t n) {
    size_t nab = apus_fp4_act_blocks(K), rows = 0;
    for (size_t e = 0; e < n; e++) rows += counts[e];
    uint8_t **w = malloc(n * sizeof *w), **ws = malloc(n * sizeof *ws);
    float **o_ref = malloc(n * sizeof *o_ref);
    uint8_t *acodes = malloc(rows * K);
    float *as = malloc(rows * nab * sizeof(float));
    float *scratch = malloc(rows * K * sizeof(float));
    ApusFp4GemmEnt *ents = malloc(n * sizeof *ents);
    for (size_t i = 0; i < rows * K; i++) acodes[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < rows * nab; i++)
        as[i] = ldexpf(1.0f, (int)(rng_u64() % 7) - 3);
    size_t m0 = 0;
    for (size_t e = 0; e < n; e++) {
        size_t nb = K / APUS_FP4_GROUP;
        w[e] = malloc(O * (K / 2));
        ws[e] = malloc(O * nb);
        for (size_t i = 0; i < O * (K / 2); i++) w[e][i] = (uint8_t)rng_u64();
        for (size_t i = 0; i < O * nb; i++)
            ws[e][i] = (uint8_t)(112 + (rng_u64() % 24));
        o_ref[e] = malloc(counts[e] * O * sizeof(float));
        /* per-entry reference: a standalone apus_fp4_gemm_mt call (M=1
         * entries included — the decode shape) */
        apus_fp4_gemm_mt(w[e], ws[e], acodes + m0 * K, as + m0 * nab,
                         scratch, o_ref[e], counts[e], O, K);
        ents[e].w = w[e];
        ents[e].ws = ws[e];
        ents[e].out = malloc(counts[e] * O * sizeof(float));
        ents[e].m0 = m0;
        ents[e].M = counts[e];
        m0 += counts[e];
    }
    apus_fp4_gemm_mt_grouped(ents, n, acodes, as, scratch, O, K);
    size_t nbad = 0;
    for (size_t e = 0; e < n; e++)
        if (memcmp(ents[e].out, o_ref[e], counts[e] * O * sizeof(float)))
            nbad++;
    CHECK(nbad == 0,
          "grouped O=%zu K=%zu n=%zu: %zu/%zu entries not bitwise vs mt",
          O, K, n, nbad, n);
    /* M=1 pin: mt and grouped must both equal the M9a anchor bitwise */
    if (counts[0] == 1) {
        float *o_m9a = malloc(O * sizeof(float));
        ref_gemm_m9a(w[0], ws[0], acodes, as, scratch, o_m9a, 1, O, K);
        CHECK(memcmp(o_m9a, o_ref[0], O * sizeof(float)) == 0,
              "M=1 pin: apus_fp4_gemm_mt != M9a kernel (O=%zu K=%zu)", O, K);
        CHECK(memcmp(o_m9a, ents[0].out, O * sizeof(float)) == 0,
              "M=1 pin: grouped != M9a kernel (O=%zu K=%zu)", O, K);
        digest_f32(o_m9a, O);
        free(o_m9a);
    }
    printf("  grouped O=%-5zu K=%-5zu n=%-3zu rows=%-4zu: bitwise == mt"
           " %s\n", O, K, n, rows, counts[0] == 1 ? "(+M=1 pin)" : "");
    m0 = 0;
    for (size_t e = 0; e < n; e++) {
        digest_f32(ents[e].out, counts[e] * O);
        free(w[e]); free(ws[e]); free(o_ref[e]); free(ents[e].out);
    }
    free(w); free(ws); free(o_ref);
    free(acodes); free(as); free(scratch); free(ents);
}

/* ================= 3. MoE group/solo mix bitwise vs per-token =============*/

typedef struct {
    ApusMoeW w;
    ApusFp4W *e1, *e2, *e3;
    uint8_t *p1, *s1, *p2, *s2, *p3, *s3;
    uint8_t *sp1, *ss1, *sp2, *ss2, *sp3, *ss3;
    float *gate_w, *gate_bias;
    int64_t *tid2eid;
} MoeFixture;

static void moe_fixture_init(MoeFixture *f, int E, int topk, int dim,
                             int inter, int vocab) {
    memset(f, 0, sizeof *f);
    size_t nb4a = (size_t)dim / 32, nb4b = (size_t)inter / 32;
    size_t nb8a = apus_fp8_blocks((size_t)dim),
           nb8b = apus_fp8_blocks((size_t)inter);
    size_t nbo8a = ((size_t)inter + 127) / 128,
           nbo8b = ((size_t)dim + 127) / 128;
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
    f->tid2eid = malloc((size_t)vocab * topk * sizeof(int64_t));
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
    f->w.hash = 1;                      /* engineered counts via tid2eid */
    f->w.route_scale = 1.5f; f->w.limit = 10.0f;
    f->w.gate_w = f->gate_w;
    f->w.gate_bias = NULL;
    f->w.tid2eid = f->tid2eid;
    f->w.w1 = f->e1; f->w.w2 = f->e2; f->w.w3 = f->e3;
    f->w.sw1 = (ApusFp8W){ f->sp1, f->ss1, inter, dim };
    f->w.sw3 = (ApusFp8W){ f->sp3, f->ss3, inter, dim };
    f->w.sw2 = (ApusFp8W){ f->sp2, f->ss2, dim, inter };
    f->w.layer_id = 0;
}

static void moe_fixture_free(MoeFixture *f) {
    free(f->p1); free(f->s1); free(f->p2); free(f->s2); free(f->p3);
    free(f->s3);
    free(f->sp1); free(f->ss1); free(f->sp2); free(f->ss2); free(f->sp3);
    free(f->ss3);
    free(f->gate_w); free(f->gate_bias); free(f->tid2eid);
    free(f->e1); free(f->e2); free(f->e3);
}

/* Hash-routed MoE with engineered expert counts: tokens are assigned so
 * that expert e gets counts[e] routings — forces multi-group flushes and
 * (with a count > APUS_MOE_GROUP_ROWS) a solo expert in the same forward.
 * counts must satisfy sum(counts) == s * topk and every count <= s (one
 * routing per (t,j) slot; a token may route to the same expert twice —
 * avoid that here by construction: counts[e] <= s and distinct slots). */
static void test_moe_groupmix(const int *counts, int E, int topk, int dim,
                              int inter) {
    int s = 0;
    for (int e = 0; e < E; e++) s += counts[e];
    s /= topk;
    int vocab = s > 0 ? s : 1;
    MoeFixture f;
    moe_fixture_init(&f, E, topk, dim, inter, vocab);
    /* assign token t's topk slots round-robin so expert totals match:
     * walk experts; expert e claims counts[e] distinct (token,slot) cells
     * with t = running index % s — each token used at most topk times. */
    int *tuse = calloc((size_t)s, sizeof(int));
    for (int e = 0; e < E; e++) {
        int need = counts[e];
        for (int t = 0; t < s && need; t++) {
            while (tuse[t] < topk && need) {
                f.tid2eid[(size_t)t * topk + tuse[t]] = e;
                tuse[t]++;
                need--;
            }
        }
        CHECK(need == 0, "groupmix: count assignment infeasible (e=%d)", e);
    }
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
    int solo = 0;
    for (int e = 0; e < E; e++)
        if (counts[e] > APUS_MOE_GROUP_ROWS) solo = 1;
    CHECK(nbit == 0,
          "moe groupmix (solo=%d): %zu/%d outputs not bitwise vs M=1",
          solo, nbit, s * dim);
    printf("  moe groupmix s=%d E=%d topk=%d (solo=%d): bitwise %d/%d vs"
           " per-token M=1\n", s, E, topk, solo, s * dim - (int)nbit,
           s * dim);
    digest_f32(o_batch, (size_t)s * dim);
    moe_fixture_free(&f);
    free(tuse); free(x); free(ids); free(o_batch); free(o_seq);
}

int main(void) {
    printf("test_m9e: grouped fp4 GEMM dispatch + MoE expert grouping\n");
    /* real expert shapes, mixed counts incl. M=1 entries and odd counts */
    {   size_t c[] = { 1, 6, 8, 12, 16, 24, 3, 5 };
        test_grouped(2048, 4096, c, 8); }     /* w1/w3 shape */
    {   size_t c[] = { 1, 6, 8, 12, 16, 24, 3, 5 };
        test_grouped(4096, 2048, c, 8); }     /* w2 shape */
    {   size_t c[] = { 1 };                   /* single entry, decode row */
        test_grouped(2048, 4096, c, 1); }
    {   size_t c[] = { 2, 9, 17, 33, 1, 4, 31, 7, 13, 22 };
        test_grouped(96, 256, c, 10); }       /* odd O, small K */
    {   size_t c[70];                         /* >64 entries: chunk loop */
        for (int i = 0; i < 70; i++) c[i] = (size_t)(1 + i % 5);
        test_grouped(64, 128, c, 70); }
    /* MoE: counts force several group flushes (sum 96 routings, R=128
     * would fit one group of <=16 experts only if rows fit — here E=8
     * keeps gn<=8 but counts push past R on the 3rd expert) and a second
     * case with a solo expert (count 200 > 128) mixed with groups. */
    {   int counts[] = { 40, 60, 50, 30, 20, 10, 5, 1 };
        test_moe_groupmix(counts, 8, 2, 128, 64); }   /* s=108, 2 flushes */
    {   int counts[] = { 200, 24, 16, 8, 4, 2, 1, 1 };
        test_moe_groupmix(counts, 8, 2, 128, 64); }   /* solo + groups */
    {   int counts[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        test_moe_groupmix(counts, 8, 1, 128, 64); }   /* all singletons */
    printf("%s: %d checks, %d failures\n",
           g_fails ? "FAIL" : "ok", g_checks, g_fails);
    printf("digest=%016llx\n", (unsigned long long)g_digest);
    return g_fails != 0;
}
