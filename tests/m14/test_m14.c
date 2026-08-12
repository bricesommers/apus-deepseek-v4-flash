/*
 * tests/m14/test_m14.c — hard gates for M14 (ARM NEON staged-product
 * interleaved dots; the NEON twin of the M12a-2 x86 pattern):
 *
 *   1. apus_dot4_f32_neon vs apus_dot_f32_scalar: BITWISE per dot, over a
 *      length sweep (all tails) and adversarial value classes (mixed
 *      magnitude, denormals, cancellation, signed zeros). [ARM only; the
 *      x86 twin is gated by tests/m12/test_m12a2.c.]
 *   2. sparse_attn kernel level: the M14 apus_sparse_attn vs a reference
 *      copy of the pre-M14 per-head body (scalar q.k dot; the platform
 *      P*V idiom unchanged) — outputs BITWISE identical across s/h/d,
 *      index-set shapes (n = 0, tails mod 4, full 640) and kv_a/kv_b
 *      boundary-straddling id sets. Portable: meaningful on every
 *      platform (on x86 it re-proves dot4_x86 + saxpy vs scalar).
 *   3. f32/bf16 linear rows: apus_f32_linear / apus_bf16_linear (pool
 *      workers, M14 NEON row grouping on ARM) vs a per-row scalar-dot
 *      reference — BITWISE over an M/K/O sweep, both round_out modes,
 *      M kept below the BLAS cutoff so the pinned path is what runs.
 *   4. THREAD INDEPENDENCE: the process digest printed at the end is
 *      diffed across APUS_THREADS=1/4/8 by the Makefile (bitwise).
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

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_f32(void) { /* uniform in [-1, 1) */
    return 2.0f * ((float)(rng_u64() >> 40) / 16777216.0f) - 1.0f;
}

static uint64_t digest = 1469598103934665603ULL;
static void dig(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        digest ^= b[i];
        digest *= 1099511628211ULL;
    }
}

static int bits_equal(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    return ua == ub;
}

/* --- 1. dot4 vs scalar -------------------------------------------------- */

/* value classes: 0 = uniform [-1,1), 1 = mixed magnitude, 2 = denormals,
 * 3 = cancellation (a=1, b = alternating huge/tiny), 4 = signed zeros */
static void fill_class(float *a, float *b, size_t n, int cls) {
    for (size_t i = 0; i < n; i++) {
        switch (cls) {
        case 0: a[i] = rng_f32(); b[i] = rng_f32(); break;
        case 1:
            a[i] = rng_f32() * ldexpf(1.0f, (int)(rng_u64() % 81) - 40);
            b[i] = rng_f32() * ldexpf(1.0f, (int)(rng_u64() % 81) - 40);
            break;
        case 2:
            a[i] = ldexpf(rng_f32(), -130);
            b[i] = ldexpf(rng_f32(), -120);
            break;
        case 3:
            a[i] = 1.0f;
            b[i] = (i & 1) ? 1e30f : -1e30f;
            if (i % 3 == 0) b[i] = rng_f32() * 1e-30f;
            break;
        default:
            a[i] = (i & 1) ? 0.0f : -0.0f;
            b[i] = rng_f32();
            break;
        }
    }
}

static void test_dot4(void) {
#ifdef __ARM_NEON
    static const size_t lens[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17,
                                   127, 128, 129, 511, 512, 513, 1024,
                                   4095, 4096, 4097 };
    enum { NL = sizeof(lens) / sizeof(lens[0]) };
    float *buf = malloc(4 * 2 * 4104 * sizeof(float));
    if (!buf) { CHECK(0, "alloc"); return; }
    long mism = 0;
    for (int cls = 0; cls < 5; cls++) {
        for (int li = 0; li < NL; li++) {
            size_t n = lens[li];
            const float *a[4], *b[4];
            float *p = buf;
            float ref[4], got[4];
            for (int r = 0; r < 4; r++) {
                a[r] = p; p += n;
                b[r] = p; p += n;
                fill_class((float *)a[r], (float *)b[r], n, cls);
                ref[r] = apus_dot_f32_scalar(a[r], b[r], n);
            }
            apus_dot4_f32_neon(a, b, n, got);
            for (int r = 0; r < 4; r++) {
                CHECK(bits_equal(ref[r], got[r]),
                      "dot4 cls=%d n=%zu row=%d ref=%a got=%a",
                      cls, n, r, ref[r], got[r]);
                if (!bits_equal(ref[r], got[r])) mism++;
            }
            dig(got, sizeof got);
        }
    }
    printf("dot4 bitwise: %ld mismatches over %ld dot checks\n",
           mism, checks > 0 ? (long)(5 * NL * 4) : 0);
    free(buf);
#else
    printf("dot4 bitwise: skipped (NEON helper is ARM-only; x86 twin gated "
           "by test_m12a2)\n");
#endif
}

/* --- 2. sparse_attn kernel level ---------------------------------------- */

/* Reference: the pre-M14 per-head body — scalar q.k dot ONLY (no dot4
 * groups), P*V in the platform idiom exactly as c/attn.h has it (NEON
 * vfmaq on ARM, apus_saxpy_x86 under AVX2, scalar mul+add elsewhere). */
static void ref_sparse_attn(const float *q, const float *sink, int h, int d,
                            const int32_t *idxs, int idxw, int s, float scale,
                            const float *kv_a, int64_t na,
                            const float *kv_b, float *o) {
    int32_t *idsf = malloc((size_t)s * idxw * sizeof(int32_t));
    int *ns = malloc((size_t)s * sizeof(int));
    float *sc = malloc((size_t)idxw * sizeof(float));
    float *p = malloc((size_t)idxw * sizeof(float));
    if (!idsf || !ns || !sc || !p) { CHECK(0, "alloc"); return; }
    for (int t = 0; t < s; t++) {
        int n = 0;
        for (int jj = 0; jj < idxw; jj++)
            if (idxs[(size_t)t * idxw + jj] >= 0)
                idsf[(size_t)t * idxw + n++] = idxs[(size_t)t * idxw + jj];
        ns[t] = n;
    }
#if APUS_X86
    const int use_avx2 = apus_x86_have_avx2();
#endif
    for (size_t r = 0; r < (size_t)s * (size_t)h; r++) {
        size_t t = r / (size_t)h, hh = r % (size_t)h;
        int n = ns[t];
        float *ov = o + (t * (size_t)h + hh) * (size_t)d;
        if (n == 0) {
            memset(ov, 0, (size_t)d * sizeof(float));
            continue;
        }
        const int32_t *ids = idsf + t * (size_t)idxw;
        const float *qv = q + (t * (size_t)h + hh) * (size_t)d;
        float mx = -INFINITY;
        for (int jj = 0; jj < n; jj++) {
            const float *kv = ids[jj] < na
                ? kv_a + (size_t)ids[jj] * (size_t)d
                : kv_b + (size_t)(ids[jj] - na) * (size_t)d;
            float dot = apus_dot_f32_scalar(qv, kv, (size_t)d);
            sc[jj] = dot * scale;
            if (sc[jj] > mx) mx = sc[jj];
        }
        float sum = 0.0f;
        for (int jj = 0; jj < n; jj++) {
            p[jj] = apus_bf16_round(expf(sc[jj] - mx));
            sum += p[jj];
        }
        size_t i = 0;
#ifdef __ARM_NEON
        for (; i + 4 <= (size_t)d; i += 4)
            vst1q_f32(ov + i, vdupq_n_f32(0.0f));
#endif
        for (; i < (size_t)d; i++) ov[i] = 0.0f;
        for (int jj = 0; jj < n; jj++) {
            const float *kv = ids[jj] < na
                ? kv_a + (size_t)ids[jj] * (size_t)d
                : kv_b + (size_t)(ids[jj] - na) * (size_t)d;
            size_t k = 0;
#ifdef __ARM_NEON
            float32x4_t pj = vdupq_n_f32(p[jj]);
            for (; k + 4 <= (size_t)d; k += 4)
                vst1q_f32(ov + k, vfmaq_f32(vld1q_f32(ov + k), pj,
                                            vld1q_f32(kv + k)));
#endif
#if APUS_X86
            if (use_avx2) {
                apus_saxpy_x86(ov, p[jj], kv, (size_t)d);
                k = (size_t)d;
            }
#endif
            for (; k < (size_t)d; k++) ov[k] += p[jj] * kv[k];
        }
        float denom = sum + expf(sink[hh] - mx);
        for (i = 0; i < (size_t)d; i++)
            ov[i] = apus_bf16_round(ov[i] / denom);
    }
    free(idsf); free(ns); free(sc); free(p);
}

static void run_sparse_case(int s, int h, int d, int idxw, int na, int nb,
                            int pat, uint64_t *mism_bits) {
    int64_t nkv = (int64_t)na + nb;
    float *q = malloc((size_t)s * h * d * sizeof(float));
    float *sink = malloc((size_t)h * sizeof(float));
    float *kv = malloc((size_t)nkv * d * sizeof(float));
    int32_t *idxs = malloc((size_t)s * idxw * sizeof(int32_t));
    float *o_new = malloc((size_t)s * h * d * sizeof(float));
    float *o_ref = malloc((size_t)s * h * d * sizeof(float));
    if (!q || !sink || !kv || !idxs || !o_new || !o_ref) {
        CHECK(0, "alloc");
        return;
    }
    for (int64_t i = 0; i < (int64_t)s * h * d; i++) q[i] = rng_f32();
    for (int i = 0; i < h; i++) sink[i] = rng_f32();
    for (int64_t i = 0; i < nkv * d; i++) kv[i] = rng_f32();
    const float *kv_a = kv, *kv_b = kv + (size_t)na * d;
    for (int t = 0; t < s; t++) {
        for (int jj = 0; jj < idxw; jj++) {
            int32_t v;
            switch (pat) {
            case 0: v = -1; break;                          /* n = 0 */
            case 1: v = jj < 1 ? 0 : -1; break;             /* n = 1 */
            case 2: v = jj < 3 ? (int32_t)(jj % nkv) : -1; break;
            case 3: v = jj < 5 ? (int32_t)(jj % nkv) : -1; break;
            case 4: v = jj < 6 ? (int32_t)(jj % nkv) : -1; break;
            case 5: v = jj < 7 ? (int32_t)(jj % nkv) : -1; break;
            case 6: /* straddle the kv_a/kv_b boundary, duplicates */
                v = (int32_t)((na - 2 + jj) % nkv);
                if (jj % 5 == 4) v = (int32_t)(na > 0 ? na - 1 : 0);
                break;
            case 7: /* full width, descending */
                v = (int32_t)(nkv - 1 - (jj % nkv));
                break;
            default: /* random ids, some -1 holes */
                v = (rng_u64() % 4 == 0) ? -1 : (int32_t)(rng_u64() % nkv);
                break;
            }
            idxs[(size_t)t * idxw + jj] = v;
        }
    }
    float scale = 1.0f / sqrtf((float)d);
    apus_sparse_attn(q, sink, h, d, idxs, idxw, s, scale, kv_a, na, kv_b,
                     o_new);
    ref_sparse_attn(q, sink, h, d, idxs, idxw, s, scale, kv_a, na, kv_b,
                    o_ref);
    size_t total = (size_t)s * h * (size_t)d;
    size_t mism = 0;
    for (size_t i = 0; i < total; i++)
        if (!bits_equal(o_new[i], o_ref[i])) {
            if (mism < 3)
                CHECK(0, "sparse_attn s=%d h=%d d=%d idxw=%d pat=%d "
                         "elem=%zu new=%a ref=%a",
                      s, h, d, idxw, pat, i, o_new[i], o_ref[i]);
            mism++;
        }
    CHECK(mism == 0, "sparse_attn s=%d h=%d d=%d idxw=%d pat=%d: %zu/%zu "
                     "mismatch", s, h, d, idxw, pat, mism, total);
    *mism_bits += mism;
    dig(o_new, total * sizeof(float));
    free(q); free(sink); free(kv); free(idxs); free(o_new); free(o_ref);
}

static void test_sparse_attn(void) {
    uint64_t mism = 0;
    /* decode shape: s=1, real h/d, full 640-wide index set */
    for (int pat = 0; pat < 9; pat++)
        run_sparse_case(1, 64, 512, 640, 512, 128, pat, &mism);
    /* small prefill shapes, odd d (exercises the dot4 + P*V scalar tails) */
    for (int pat = 0; pat < 9; pat++)
        run_sparse_case(3, 4, 21, 11, 7, 4, pat, &mism);
    for (int pat = 0; pat < 9; pat++)
        run_sparse_case(5, 8, 64, 33, 20, 13, pat, &mism);
    printf("sparse_attn kernel: %llu mismatched output elems\n",
           (unsigned long long)mism);
}

/* --- 3. f32/bf16 linear rows -------------------------------------------- */

static void ref_linear(const float *w, const float *x, float *out,
                       int M, int K, int O, int round_out) {
    float *xb = (float *)x;
    float *tmp = NULL;
    if (round_out) {
        tmp = malloc((size_t)M * K * sizeof(float));
        for (size_t i = 0; i < (size_t)M * K; i++)
            tmp[i] = apus_bf16_round(x[i]);
        xb = tmp;
    }
    for (int m = 0; m < M; m++)
        for (int o = 0; o < O; o++) {
            float acc = apus_dot_f32_scalar(xb + (size_t)m * K,
                                            w + (size_t)o * K, (size_t)K);
            out[(size_t)m * O + o] = round_out ? apus_bf16_round(acc) : acc;
        }
    free(tmp);
}

static void run_linear_case(int M, int K, int O, uint64_t *mism_bits) {
    float *w = malloc((size_t)O * K * sizeof(float));
    float *x = malloc((size_t)M * K * sizeof(float));
    float *o_new = malloc((size_t)M * O * sizeof(float));
    float *o_ref = malloc((size_t)M * O * sizeof(float));
    if (!w || !x || !o_new || !o_ref) { CHECK(0, "alloc"); return; }
    for (int64_t i = 0; i < (int64_t)O * K; i++) w[i] = rng_f32();
    for (int64_t i = 0; i < (int64_t)M * K; i++) x[i] = rng_f32();
    for (int ro = 0; ro < 2; ro++) {
        if (ro) apus_bf16_linear(w, x, o_new, M, K, O);
        else    apus_f32_linear(w, x, o_new, M, K, O);
        ref_linear(w, x, o_ref, M, K, O, ro);
        size_t total = (size_t)M * O;
        size_t mism = 0;
        for (size_t i = 0; i < total; i++)
            if (!bits_equal(o_new[i], o_ref[i])) {
                if (mism < 3)
                    CHECK(0, "linear M=%d K=%d O=%d ro=%d elem=%zu "
                             "new=%a ref=%a", M, K, O, ro, i,
                          o_new[i], o_ref[i]);
                mism++;
            }
        CHECK(mism == 0, "linear M=%d K=%d O=%d ro=%d: %zu/%zu mismatch",
              M, K, O, ro, mism, total);
        *mism_bits += mism;
        dig(o_new, total * sizeof(float));
    }
    free(w); free(x); free(o_new); free(o_ref);
}

static void test_linears(void) {
    uint64_t mism = 0;
    /* M below the BLAS cutoff (256) everywhere: the pinned row path is
     * what runs. O sweeps across the 4-row group boundary; K across the
     * dot4 tail lengths. */
    static const int shapes[][3] = {
        { 1, 512, 2048 },     /* decode compressor-ish */
        { 1, 4096, 256 },     /* decode gate-ish */
        { 1, 4096, 64 },      /* decode weights_proj-ish */
        { 3, 129, 7 },
        { 4, 4, 4 },
        { 5, 7, 5 },
        { 17, 512, 33 },
        { 250, 16, 9 },
        { 2, 4099, 3 },
        { 1, 1, 1 },
        { 2, 3, 1027 },
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        run_linear_case(shapes[i][0], shapes[i][1], shapes[i][2], &mism);
    printf("linear rows: %llu mismatched output elems\n",
           (unsigned long long)mism);
}

int main(void) {
    fprintf(stderr, "threads=%d\n", apus_pool_threads());
    test_dot4();
    test_sparse_attn();
    test_linears();
    printf("digest=%016llx\n", (unsigned long long)digest);
    printf("RESULT: %s (%ld checks)\n", failures ? "FAIL" : "ok", checks);
    return failures ? 1 : 0;
}
