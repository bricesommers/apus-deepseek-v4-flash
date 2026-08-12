/*
 * c/attn.h — DeepSeek-V4-Flash attention sublayer (M4c): low-rank Q path,
 * sliding-window KV ring, CSA/HCA compressor state machine, lightning
 * indexer, index-gathered sparse attention with attention sink, grouped
 * low-rank o-proj. C11, libc + arm_neon.h only. FP32 arithmetic with BF16
 * rounding at the reference .to() boundaries (tests/m4b/README.md).
 *
 * Normative reference: tools/oracle.py (f32 mode), itself a port of
 * reference/inference/model.py:199-543 + kernel.py. Stage semantics,
 * decode call order and ambiguity resolutions A1-A12 follow
 * tests/m4b/README.md exactly:
 *   - act quant per-128 (per-64 for the window/compressor nope QAT) from the
 *     BF16-rounded input, ue8m0 pow2 scales (fp4.h/fp8.h pinned rule).
 *   - FP4-E2M1 act quant per-32 for the indexer (Hadamard first, Sylvester
 *     convention x d^-0.5, A3).
 *   - sparse_attn: serial equivalent of the 64-wide blocked online softmax
 *     (A7), probabilities BF16-rounded before P*V, sink in the DENOMINATOR
 *     only, all-masked row -> 0 (A8), inverse RoPE on the output.
 *   - indexer scores: BF16 round after the q*k einsum, after the
 *     ReLU*weight mul, and after the head sum (A1); top-k stable with
 *     lower-index-first tie-break (A6); window idx ring order from
 *     start_pos >= window-1 (A12).
 *   - decode call order: q -> win kv -> window idx -> INDEXER (incl. its
 *     compressor) -> ring write -> ATTN compressor -> sparse_attn.
 *
 * Prefill (start_pos == 0, s tokens at once) and decode (start_pos > 0,
 * s == 1) share one entry point like the reference. M8 adds the third
 * shape — a causal multi-token forward from a CARRIED state (start_pos >
 * 0, s > 1), used by speculative verify batches: it is processed as a
 * per-token interleave of the exact s=1 decode steps on the precomputed
 * q/kv rows, so state and outputs are bitwise "as if decoded one-by-one"
 * by construction (the prefill masking logic is start_pos==0-only and the
 * old decode logic s==1-only; neither covers this shape).
 *
 * Usage: #define APUS_ATTN_IMPLEMENTATION in exactly one TU, which must
 * also have APUS_FP4_IMPLEMENTATION / APUS_FP8_IMPLEMENTATION / APUS_ST_IMPLEMENTATION
 * linked in. Scalar only except the fp4/fp8 GEMM kernels (NEON when
 * __ARM_NEON, pinned against scalar in tests/m3, tests/m4a).
 *
 * M7b: this TU owns the backend hook table (apus_backend_hooks,
 * c/backend_metal.h, all-NULL = CPU kernels) and the weak apus_metal_*
 * stubs; apus_fp8_linear / apus_bf16_linear / the wo_a group loop try the
 * hooks first and fall back to the CPU kernels.
 */
#ifndef APUS_ATTN_H
#define APUS_ATTN_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"
#include "backend_metal.h"
#include "x86.h"   /* M12a-2: AVX2 runtime dispatch (no-op off x86-64) */

#ifdef __cplusplus
extern "C" {
#endif

/* --- shared numerics helpers (also used by moe.h) -------------------------*/

/* RoPE cos/sin tables [seqlen, dim/2], FP32 like the reference (A11).
 * original_seq_len == 0 disables YaRN (SWA layers). */
void apus_rope_precompute(float *cos, float *sin, int dim, int seqlen,
                          int original_seq_len, double base, double factor,
                          int beta_fast, int beta_slow);

/* Row-range variant: fills rows [t0, t1) only, bit-identical to the
 * corresponding rows of apus_rope_precompute. Used by the lazily-grown
 * shared tables in c/layer.h (real max_pos = 1M must not be precomputed). */
void apus_rope_precompute_range(float *cos, float *sin, int dim,
                                int64_t t0, int64_t t1,
                                int original_seq_len, double base,
                                double factor, int beta_fast, int beta_slow);

/* Interleaved-pairs RoPE on one vector of rd floats; BF16-rounds the
 * rotated values (f32-faithful mode). inverse = conjugate. */
void apus_apply_rope(float *x, const float *cos, const float *sin,
                     int rd, int inverse);

/* Sylvester Hadamard rotation x @ H * n^-0.5, BF16-rounded (A3). n pow2. */
void apus_hadamard(float *x, int n);

/* FP8-E4M3 quant->dequant simulation per `group` along K (ue8m0 pow2
 * scale, amax floor 1e-4, clamp +-448, RNE), in place. */
void apus_fp8_qat_sim(float *x, size_t K, size_t group);

/* FP4-E2M1 quant->dequant simulation per 32 along K (pow2 scale, amax
 * floor 6*2^-126, clamp +-6, RNE), in place. */
void apus_fp4_qat_sim(float *x, size_t K);

/* RMSNorm: y = w * (x * rsqrt(mean(x^2)+eps)), BF16-rounded output. */
void apus_rms_norm(const float *x, const float *w, float eps,
                   float *y, size_t n);

/* Dense linears. All take x [M,K] row-major, produce out [M,O] row-major.
 * fp8/fp4: BF16-round the input, act-quant per-128, blockwise kernel
 * semantics of fp8.h/fp4.h, BF16-rounded output.
 * bf16: BF16-round input, FP32-accumulate matmul, BF16-rounded output.
 * f32: plain FP32 matmul, no rounding (compressor wkv/wgate, gate). */
void apus_fp8_linear(const ApusFp8W *w, const float *x, float *out,
                     int M, int K, int O);
void apus_fp4_linear(const ApusFp4W *w, const float *x, float *out,
                     int M, int K, int O);
void apus_bf16_linear(const float *w, const float *x, float *out,
                      int M, int K, int O);
void apus_f32_linear(const float *w, const float *x, float *out,
                     int M, int K, int O);

/* Stable descending top-k (lower index first on ties, A6). k <= n. */
void apus_topk_stable(const float *row, int n, int k, int32_t *idx);

/* --- attention ------------------------------------------------------------*/

typedef struct {
    int dim, n_heads, head_dim, rope_dim, q_lora, o_groups, o_lora;
    int window, ratio;              /* ratio == 0 => SWA */
    int has_indexer;                /* ratio == 4 */
    int idx_heads, idx_dim, idx_topk;
    float eps;                      /* norm eps */
    int max_pos;                    /* rope table length / state sizing */
} ApusAttnCfg;

/* Compressor weights. wkv/wgate [coff*d, dim] (BF16 values as f32),
 * ape [ratio, coff*d] f32, norm [d]. */
typedef struct {
    const float *wkv, *wgate, *ape, *norm;
    int ratio, overlap, d, rd, rotate;
    float eps;                      /* norm eps for the pooled RMSNorm */
} ApusCompW;

typedef struct {
    ApusFp8W wq_a, wq_b, wkv, wo_b;
    const uint16_t *wo_a;   /* [G*o_lora, h*d/G] BF16 bits (A4; M6c storage) */
    const float *q_norm;    /* [q_lora] */
    const float *kv_norm;   /* [head_dim] */
    const float *sink;      /* [n_heads] f32 */
    ApusCompW comp;         /* ratio > 0 */
    ApusCompW idx_comp;     /* has_indexer */
    ApusFp8W idx_wq_b;      /* has_indexer */
    const float *idx_wproj; /* [idx_heads, dim] */
} ApusAttnW;

/* Serializable compressor state (matches the oracle's comp_kv_state /
 * comp_score_state / comp_kv arrays). */
typedef struct {
    int rows, cols;         /* coff*ratio x coff*d */
    float *kv, *sc;         /* [rows, cols]; sc init -inf */
    float *cache;           /* [cap, d] compressed entries (BF16 values),
                               grown on demand up to max_nb */
    int nb, max_nb, d;
    int cap;                /* allocated entries in cache (<= max_nb) */
} ApusCompS;

/* Serializable attention state: pos, window ring, compressor carries. */
typedef struct {
    int64_t pos;
    float *win;             /* [window, head_dim] ring, BF16 values */
    ApusCompS comp;         /* ratio > 0 */
    ApusCompS idx_comp;     /* has_indexer */
} ApusAttnS;

void apus_attn_state_init(ApusAttnS *st, const ApusAttnCfg *cfg);
void apus_attn_state_free(ApusAttnS *st, const ApusAttnCfg *cfg);

/* Named intermediates (all optional / NULL to skip). Buffers must be large
 * enough: q/attn_out [s,h,d], win_kv [s,d], comp_kv [max_nb,d],
 * idx_comp_kv [max_nb,idx_dim], idx_scores [s,max_nb], idx_topk [s,idx_topk],
 * o_out [s,dim]. */
typedef struct {
    float *q, *win_kv, *comp_kv, *idx_comp_kv, *idx_scores;
    int32_t *idx_topk;
    float *attn_out, *o_out;
    int comp_nb, idx_nb, idx_k;
} ApusAttnInterm;

/* Attention sublayer. x [s, dim] (BF16 values), out [s, dim].
 * cos/sin: rope tables [max_pos, rope_dim/2]. Mutates st. */
void apus_attention(const ApusAttnCfg *cfg, const ApusAttnW *w, ApusAttnS *st,
                    const float *x, int s, int64_t start_pos,
                    const float *cos, const float *sin,
                    float *out, ApusAttnInterm *interm);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_ATTN_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp4.h"
#include "fp8.h"

/* --- M7b backend hook table (c/backend_metal.h) -----------------------------
 * Defined here because the attn.h implementation TU is linked into every
 * engine/test binary that has linears. All-NULL = CPU kernels (default);
 * apus_metal_enable() fills it. The weak stubs let the plain CPU binary
 * link: c/backend_metal.mm (metal=1 build) provides strong definitions. */
ApusBackendHooks apus_backend_hooks;

#if defined(__GNUC__)
__attribute__((weak)) int apus_metal_enable(char *err, size_t errcap) {
    if (err && errcap)
        snprintf(err, errcap, "metal backend not compiled in (make metal=1)");
    return -1;
}
__attribute__((weak)) void apus_metal_disable(void) {}
__attribute__((weak)) int apus_metal_is_enabled(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_bytes_wrapped(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_bytes_uploaded(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_dispatches(void) { return 0; }
__attribute__((weak)) int apus_metal_fp8_linear(const ApusFp8W *w,
        const float *x, float *out, int M, int K, int O) {
    (void)w; (void)x; (void)out; (void)M; (void)K; (void)O; return 1;
}
__attribute__((weak)) int apus_metal_f32_linear(const float *w,
        const float *x, float *out, int M, int K, int O, int flags) {
    (void)w; (void)x; (void)out; (void)M; (void)K; (void)O; (void)flags;
    return 1;
}
__attribute__((weak)) int apus_metal_head_gemv_bf16(const void *w,
        const float *x, float *out, int64_t O, int64_t K) {
    (void)w; (void)x; (void)out; (void)O; (void)K; return 1;
}
__attribute__((weak)) int apus_metal_fp8_gemm(const uint8_t *w,
        const uint8_t *ws, const uint8_t *acodes, const float *as,
        float *out, int M, int O, int K) {
    (void)w; (void)ws; (void)acodes; (void)as; (void)out;
    (void)M; (void)O; (void)K; return 1;
}
__attribute__((weak)) int apus_metal_fp8_act_quant(const float *x, int M,
        int K, uint8_t *codes, float *scales) {
    (void)x; (void)M; (void)K; (void)codes; (void)scales; return 1;
}
__attribute__((weak)) int apus_metal_rmsnorm(const float *x, const float *w,
        float eps, float *y, int64_t n) {
    (void)x; (void)w; (void)eps; (void)y; (void)n; return 1;
}
#endif /* __GNUC__ */

/* --- helpers ---------------------------------------------------------------*/

void apus_rope_precompute_range(float *cos, float *sin, int dim,
                                int64_t t0, int64_t t1,
                                int original_seq_len, double base,
                                double factor, int beta_fast, int beta_slow) {
    int half = dim / 2;
    float *freqs = malloc((size_t)half * sizeof(float));
    for (int j = 0; j < half; j++)
        freqs[j] = 1.0f / powf((float)base, (float)(2 * j) / (float)dim);
    if (original_seq_len > 0) {
        /* YaRN ramp (model.py:212-228), scalar parts in double like numpy */
        double l2pi = 2.0 * M_PI;
        double low = floor(dim * log(original_seq_len / (beta_fast * l2pi))
                           / (2.0 * log(base)));
        double high = ceil(dim * log(original_seq_len / (beta_slow * l2pi))
                           / (2.0 * log(base)));
        if (low < 0) low = 0;
        if (high > dim - 1) high = dim - 1;
        if (low == high) high += 0.001;
        for (int j = 0; j < half; j++) {
            float ramp = (float)(((double)j - low) / (high - low));
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            float smooth = 1.0f - ramp;
            freqs[j] = freqs[j] / (float)factor * (1.0f - smooth)
                     + freqs[j] * smooth;
        }
    }
    for (int64_t t = t0; t < t1; t++)
        for (int j = 0; j < half; j++) {
            float ang = (float)t * freqs[j];
            cos[t * half + j] = cosf(ang);
            sin[t * half + j] = sinf(ang);
        }
    free(freqs);
}

void apus_rope_precompute(float *cos, float *sin, int dim, int seqlen,
                          int original_seq_len, double base, double factor,
                          int beta_fast, int beta_slow) {
    apus_rope_precompute_range(cos, sin, dim, 0, seqlen, original_seq_len,
                               base, factor, beta_fast, beta_slow);
}

void apus_apply_rope(float *x, const float *cos, const float *sin,
                     int rd, int inverse) {
    int half = rd / 2;
    for (int j = 0; j < half; j++) {
        float x1 = x[2 * j], x2 = x[2 * j + 1];
        float c = cos[j], s = inverse ? -sin[j] : sin[j];
        x[2 * j]     = apus_bf16_round(x1 * c - x2 * s);
        x[2 * j + 1] = apus_bf16_round(x1 * s + x2 * c);
    }
}

void apus_hadamard(float *x, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += 2 * len)
            for (int j = i; j < i + len; j++) {
                float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
    float inv = (float)pow((double)n, -0.5);
    for (int i = 0; i < n; i++) x[i] = apus_bf16_round(x[i] * inv);
}

/* pow2 scale = 2^ceil(log2(p)) for positive normal p via the reference bit
 * trick (kernel.py fast_log2_ceil/fast_pow2). */
static float apus_ceil_pow2(float p) {
    uint32_t b;
    memcpy(&b, &p, 4);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x007FFFFFu) != 0);
    uint32_t sb = (uint32_t)(e + 127) << 23;
    float s;
    memcpy(&s, &sb, 4);
    return s;
}

void apus_fp8_qat_sim(float *x, size_t K, size_t group) {
    for (size_t lo = 0; lo < K; lo += group) {
        size_t hi = lo + group;
        if (hi > K) hi = K;
        float amax = 0.0f;
        for (size_t i = lo; i < hi; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        float s = apus_ceil_pow2(amax * (1.0f / 448.0f));
        for (size_t i = lo; i < hi; i++)
            x[i] = apus_e4m3_dequant_f32(apus_e4m3_quant_f32(x[i] / s)) * s;
    }
}

/* E2M1 RNE magnitude code 0..7 for a >= 0 (tests/m3 pinned rule). */
static int apus_e2m1_mag(float a) {
    static const float T[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    int mag = 0;
    for (int j = 0; j < 7; j++)
        mag += (a > T[j]) || (a == T[j] && (j % 2 == 1));
    return mag;
}

void apus_fp4_qat_sim(float *x, size_t K) {
    static const float FLOOR = 7.05484372e-38f; /* 6 * 2^-126 (exact) */
    for (size_t lo = 0; lo < K; lo += 32) {
        float amax = 0.0f;
        for (size_t i = lo; i < lo + 32; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax < FLOOR) amax = FLOOR;
        float s = apus_ceil_pow2(amax * (1.0f / 6.0f));
        for (size_t i = lo; i < lo + 32; i++) {
            float y = x[i] / s;
            if (y > 6.0f) y = 6.0f;
            else if (y < -6.0f) y = -6.0f;
            int mag = apus_e2m1_mag(fabsf(y));
            int code = signbit(y) ? (mag | 8) : mag;
            x[i] = apus_fp4_lut[code] * s;
        }
    }
}

void apus_rms_norm(const float *x, const float *w, float eps,
                   float *y, size_t n) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++)
        y[i] = apus_bf16_round(w[i] * (x[i] * inv));
}

/* M9d: per-row act-quant worker for the fp8/fp4 linears (each row: BF16
 * round + scalar-order act quant — per-row independent, so pooling the
 * rows is bitwise identical to the serial loop; the pool.h partitioning
 * contract keeps it thread-count independent). */
typedef struct {
    const float *x;
    float *xb;
    uint8_t *codes;
    float *as;
    int K;
    size_t nab;
} ApusActQuantJob;

static void apus_act_quant_rows(void *vjob, size_t m0, size_t m1) {
    const ApusActQuantJob *j = vjob;
    size_t K = (size_t)j->K, nab = j->nab;
    for (size_t m = m0; m < m1; m++) {
        for (size_t i = 0; i < K; i++)
            j->xb[m * K + i] = apus_bf16_round(j->x[m * K + i]);
        apus_fp4_act_quant_scalar(j->xb + m * K, K, j->codes + m * K,
                                  j->as + m * nab);
    }
}

/* Pool the rows when there is enough work to amortize a dispatch (per-row
 * cost ~K ops; below the threshold the same row code runs inline). */
#define APUS_ROW_POOL_MIN 8

void apus_fp8_linear(const ApusFp8W *w, const float *x, float *out,
                     int M, int K, int O) {
    /* M7b backend hook (Metal when enabled): identical semantics — BF16
     * input round, per-128 act quant, blockwise kernel, BF16-rounded out. */
    if (apus_backend_hooks.fp8_linear
        && apus_backend_hooks.fp8_linear(w, x, out, M, K, O) == 0)
        return;
    size_t nab = apus_fp4_act_blocks((size_t)K);
    ApusScratchMark mk = apus_scratch_mark();
    float *xb = apus_scratch_alloc((size_t)M * K * sizeof(float));
    uint8_t *codes = apus_scratch_alloc((size_t)M * K);
    float *as = apus_scratch_alloc((size_t)M * nab * sizeof(float));
    float *scratch = apus_scratch_alloc((size_t)M * K * sizeof(float));
    ApusActQuantJob aq = { x, xb, codes, as, K, nab };
    if (M >= APUS_ROW_POOL_MIN)
        apus_pool_run((size_t)M, apus_act_quant_rows, &aq);
    else
        apus_act_quant_rows(&aq, 0, (size_t)M);
    apus_fp8_gemm_mt(w->codes, w->scales, codes, as, scratch, out,
                     (size_t)M, (size_t)O, (size_t)K);
    for (int i = 0; i < M * O; i++) out[i] = apus_bf16_round(out[i]);
    apus_scratch_reset(mk);
}

void apus_fp4_linear(const ApusFp4W *w, const float *x, float *out,
                     int M, int K, int O) {
    size_t nab = apus_fp4_act_blocks((size_t)K);
    ApusScratchMark mk = apus_scratch_mark();
    float *xb = apus_scratch_alloc((size_t)M * K * sizeof(float));
    uint8_t *codes = apus_scratch_alloc((size_t)M * K);
    float *as = apus_scratch_alloc((size_t)M * nab * sizeof(float));
    float *scratch = apus_scratch_alloc((size_t)M * K * sizeof(float));
    ApusActQuantJob aq = { x, xb, codes, as, K, nab };
    if (M >= APUS_ROW_POOL_MIN)
        apus_pool_run((size_t)M, apus_act_quant_rows, &aq);
    else
        apus_act_quant_rows(&aq, 0, (size_t)M);
    apus_fp4_gemm_mt(w->packed, w->scales, codes, as, scratch, out,
                     (size_t)M, (size_t)O, (size_t)K);
    for (int i = 0; i < M * O; i++) out[i] = apus_bf16_round(out[i]);
    apus_scratch_reset(mk);
}

/* M6c: sequential-k scalar dot for the dense f32/bf16 paths — the exact
 * pre-M6c accumulation order (clang compiles it to vectorized fmul +
 * sequential scalar fadds, value-identical to plain scalar). The dot
 * order is fixed and M-independent (one output row = one call), so
 * prefill and decode stay bitwise consistent, and rows partition freely
 * across the pool. The compressor wkv/wgate paths in particular leave no
 * room for reorder noise (their QAT outputs have m4c bitwise-diff
 * bounds), which is why these linears are threaded across rows instead
 * of NEON-reordered within a row. */
static inline float apus_dot_f32_scalar(const float *a, const float *b,
                                        size_t n) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; i++) dot += a[i] * b[i];
    return dot;
}

#ifdef __ARM_NEON
/* M14: FOUR independent sequential-order dots at once — the NEON mirror of
 * apus_dot4_f32_x86 (c/x86.h). The vmulq_f32 products are the exact IEEE
 * multiplies (bitwise identical to the scalar fmul), staged to stack; the
 * adds run as four interleaved scalar chains in STRICT k order, so each dot
 * is bitwise identical to apus_dot_f32_scalar — only the FP-add latency is
 * hidden, no rounding sequence changes. The accumulators and row pointers
 * are NAMED variables, not arrays: indexed acc[]/a[] spills to memory in
 * the inner loop (the Rosetta lesson, tests/m12/README.md). */
static inline void apus_dot4_f32_neon(const float *const a[4],
                                      const float *const b[4],
                                      size_t n, float out[4]) {
    const float *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3];
    const float *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
    float p0[4], p1[4], p2[4], p3[4];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        vst1q_f32(p0, vmulq_f32(vld1q_f32(a0 + k), vld1q_f32(b0 + k)));
        vst1q_f32(p1, vmulq_f32(vld1q_f32(a1 + k), vld1q_f32(b1 + k)));
        vst1q_f32(p2, vmulq_f32(vld1q_f32(a2 + k), vld1q_f32(b2 + k)));
        vst1q_f32(p3, vmulq_f32(vld1q_f32(a3 + k), vld1q_f32(b3 + k)));
        d0 += p0[0]; d1 += p1[0]; d2 += p2[0]; d3 += p3[0];
        d0 += p0[1]; d1 += p1[1]; d2 += p2[1]; d3 += p3[1];
        d0 += p0[2]; d1 += p1[2]; d2 += p2[2]; d3 += p3[2];
        d0 += p0[3]; d1 += p1[3]; d2 += p2[3]; d3 += p3[3];
    }
    for (; k < n; k++) {
        d0 += a0[k] * b0[k]; d1 += a1[k] * b1[k];
        d2 += a2[k] * b2[k]; d3 += a3[k] * b3[k];
    }
    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3;
}
#endif

typedef struct {
    const float *w, *x;     /* x: [M,K] (bf16_linear: pre-rounded) */
    float *out;
    int M, K, O;
    int round_out;          /* bf16_linear: BF16-round each output */
} ApusF32LinearJob;

#if APUS_X86
/* M12a-2: AVX2 row body for both linear variants — four rows per
 * apus_dot4_f32_x86 group (staged exact products, scalar sequential add
 * order per row => bitwise identical to apus_dot_f32_scalar per row,
 * c/x86.h contract); trailing rows take the scalar loop. round_out
 * selects the BF16 output round, matching the two scalar workers. */
APUS_TGT_AVX2
static void apus_f32_linear_rows_avx2(const ApusF32LinearJob *j,
                                      size_t r0, size_t r1) {
    size_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const float *a[4], *b[4];
        for (int q = 0; q < 4; q++) {
            size_t m = (r + (size_t)q) / (size_t)j->O;
            size_t o = (r + (size_t)q) % (size_t)j->O;
            a[q] = j->x + m * (size_t)j->K;
            b[q] = j->w + o * (size_t)j->K;
        }
        float d[4];
        apus_dot4_f32_x86(a, b, (size_t)j->K, d);
        for (int q = 0; q < 4; q++)
            j->out[r + (size_t)q] = j->round_out ? apus_bf16_round(d[q])
                                                 : d[q];
    }
    for (; r < r1; r++) {
        size_t m = r / (size_t)j->O, o = r % (size_t)j->O;
        float acc = apus_dot_f32_scalar(j->x + m * (size_t)j->K,
                                        j->w + o * (size_t)j->K, (size_t)j->K);
        j->out[r] = j->round_out ? apus_bf16_round(acc) : acc;
    }
}
#endif

#ifdef __ARM_NEON
/* M14: NEON row body for both linear variants — the ARM twin of
 * apus_f32_linear_rows_avx2 above: four rows per apus_dot4_f32_neon group
 * (bitwise identical to apus_dot_f32_scalar per row), trailing rows scalar.
 * round_out selects the BF16 output round, matching the two workers. */
static void apus_f32_linear_rows_neon(const ApusF32LinearJob *j,
                                      size_t r0, size_t r1) {
    size_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const float *a[4], *b[4];
        for (int q = 0; q < 4; q++) {
            size_t m = (r + (size_t)q) / (size_t)j->O;
            size_t o = (r + (size_t)q) % (size_t)j->O;
            a[q] = j->x + m * (size_t)j->K;
            b[q] = j->w + o * (size_t)j->K;
        }
        float d[4];
        apus_dot4_f32_neon(a, b, (size_t)j->K, d);
        for (int q = 0; q < 4; q++)
            j->out[r + (size_t)q] = j->round_out ? apus_bf16_round(d[q])
                                                 : d[q];
    }
    for (; r < r1; r++) {
        size_t m = r / (size_t)j->O, o = r % (size_t)j->O;
        float acc = apus_dot_f32_scalar(j->x + m * (size_t)j->K,
                                        j->w + o * (size_t)j->K,
                                        (size_t)j->K);
        j->out[r] = j->round_out ? apus_bf16_round(acc) : acc;
    }
}
#endif

/* Scalar-order row worker (bitwise identical to the pre-M6c scalar
 * apus_f32_linear — see apus_dot_f32_scalar note). */
static void apus_f32_linear_rows_scalar(void *vjob, size_t r0, size_t r1) {
    const ApusF32LinearJob *j = vjob;
#if APUS_X86
    /* M12a-2: bitwise-identical AVX2 rows when the CPU supports them. */
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_f32_linear_rows_avx2(j, r0, r1);
        return;
    }
#endif
#ifdef __ARM_NEON
    /* M14: bitwise-identical NEON rows (same contract as the x86 branch). */
    apus_f32_linear_rows_neon(j, r0, r1);
    return;
#endif
    for (size_t r = r0; r < r1; r++) {
        size_t m = r / (size_t)j->O, o = r % (size_t)j->O;
        j->out[r] = apus_dot_f32_scalar(j->x + m * (size_t)j->K,
                                        j->w + o * (size_t)j->K, (size_t)j->K);
    }
}

/* bf16 variant: same scalar-order dot, BF16-rounded output (bitwise
 * identical to the pre-M6c scalar apus_bf16_linear). */
static void apus_f32_linear_rows_bf16(void *vjob, size_t r0, size_t r1) {
    const ApusF32LinearJob *j = vjob;
#if APUS_X86
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_f32_linear_rows_avx2(j, r0, r1);
        return;
    }
#endif
#ifdef __ARM_NEON
    apus_f32_linear_rows_neon(j, r0, r1);
    return;
#endif
    for (size_t r = r0; r < r1; r++) {
        size_t m = r / (size_t)j->O, o = r % (size_t)j->O;
        float acc = apus_dot_f32_scalar(j->x + m * (size_t)j->K,
                                        j->w + o * (size_t)j->K, (size_t)j->K);
        j->out[r] = apus_bf16_round(acc);
    }
}

void apus_bf16_linear(const float *w, const float *x, float *out,
                      int M, int K, int O) {
    /* M7b backend hook: BF16-rounded input, FP32 accumulate, BF16 out. */
    if (apus_backend_hooks.f32_linear
        && apus_backend_hooks.f32_linear(w, x, out, M, K, O,
                                         APUS_HOOK_R_IN | APUS_HOOK_R_OUT) == 0)
        return;
    ApusScratchMark mk = apus_scratch_mark();
    float *xb = apus_scratch_alloc((size_t)M * K * sizeof(float));
    /* BF16-round the input once (the old code rounded per (m,o,k) — the
     * rounding is per-element, so the values are identical). */
    for (size_t i = 0; i < (size_t)M * K; i++)
        xb[i] = apus_bf16_round(x[i]);
#if APUS_BLAS
    /* M9d: large-batch calls go to Accelerate (AMX) — FP32 summation-order
     * reorder only (accepted class, tests/m3; the dot has no scales), then
     * the same per-element BF16 output round. Smaller M keeps the pinned
     * scalar-order path below (every pre-M9d bitwise gate runs at M<=250 —
     * c/blas.h cutoff rationale). */
    if (M >= APUS_BLAS_M_MIN && apus_blas_available()) {
        apus_f32_gemm_blas(w, xb, out, (size_t)M, (size_t)O, (size_t)K);
        for (int i = 0; i < M * O; i++) out[i] = apus_bf16_round(out[i]);
        apus_scratch_reset(mk);
        return;
    }
#endif
    ApusF32LinearJob job = { w, xb, out, M, K, O, 1 };
    apus_pool_run((size_t)M * O, apus_f32_linear_rows_bf16, &job);
    apus_scratch_reset(mk);
}

void apus_f32_linear(const float *w, const float *x, float *out,
                     int M, int K, int O) {
#if APUS_BLAS
    /* M9d: large-batch calls go to Accelerate (AMX); FP32 summation-order
     * reorder only. The scalar path below stays for M < APUS_BLAS_M_MIN —
     * it is load-bearing for the m4c/m5/m6c/m7b/m8 bitwise gates (all at
     * M <= 250) and for decode. */
    if (M >= APUS_BLAS_M_MIN && apus_blas_available()) {
        apus_f32_gemm_blas(w, x, out, (size_t)M, (size_t)O, (size_t)K);
        return;
    }
#endif
    ApusF32LinearJob job = { w, x, out, M, K, O, 0 };
    apus_pool_run((size_t)M * O, apus_f32_linear_rows_scalar, &job);
}

void apus_topk_stable(const float *row, int n, int k, int32_t *idx) {
    /* stable descending top-k: strict greater-than, so among EXACTLY equal
     * scores (including -inf masked entries) the lowest index wins, and
     * each index is picked at most once (A6). */
    char stack_used[8192];
    char *used = stack_used;
    if ((size_t)n > sizeof stack_used) {
        used = calloc((size_t)n, 1);
    } else {
        memset(used, 0, (size_t)n);
    }
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int i = 0; i < n; i++)
            if (!used[i] && (best < 0 || row[i] > row[best])) best = i;
        idx[j] = best;
        used[best] = 1;
    }
    if (used != stack_used) free(used);
}

/* --- window / compressed index sets ----------------------------------------*/

/* get_window_topk_idxs (model.py:254-265, A12). Returns width. */
static int apus_window_idxs(int32_t *out, int s, int64_t start_pos, int win) {
    if (start_pos >= win - 1) {
        int sp = (int)(start_pos % win);
        for (int t = 0; t < s; t++) {
            int j = 0;
            for (int i = sp + 1; i < win; i++) out[(size_t)t * win + j++] = i;
            for (int i = 0; i <= sp; i++) out[(size_t)t * win + j++] = i;
        }
        return win;
    }
    if (start_pos > 0) {
        for (int t = 0; t < s; t++)
            for (int j = 0; j < win; j++)
                out[(size_t)t * win + j] = (j <= start_pos) ? j : -1;
        return win;
    }
    int width = s < win ? s : win;
    for (int t = 0; t < s; t++) {
        int64_t base = t - (int64_t)win + 1;
        if (base < 0) base = 0;
        for (int j = 0; j < width; j++) {
            int64_t v = base + j;
            out[(size_t)t * width + j] = (v > t) ? -1 : (int32_t)v;
        }
    }
    return width;
}

/* get_compress_topk_idxs (model.py:268-276), HCA dense selection.
 * Returns width. */
static int apus_compress_idxs(int32_t *out, int ratio, int s,
                              int64_t start_pos, int offset) {
    if (start_pos > 0) {
        int width = (int)((start_pos + 1) / ratio);
        for (int t = 0; t < s; t++)
            for (int j = 0; j < width; j++)
                out[(size_t)t * width + j] = j + offset;
        return width;
    }
    int nb = s / ratio;
    for (int t = 0; t < s; t++)
        for (int j = 0; j < nb; j++)
            out[(size_t)t * nb + j] =
                (j >= (int64_t)(t + 1) / ratio) ? -1 : j + offset;
    return nb;
}

/* --- compressor state machine (model.py:279-377) ----------------------------*/

/* Initial compressed-cache allocation: small contexts (fixtures, short
 * generations) behave exactly as before; multi-hundred-MB preallocations at
 * real max_pos = 1M (CSA ratio 4: 262146 x 512 x 4 B ~ 537 MB per layer)
 * are instead grown geometrically as the sequence lengthens. */
#define APUS_COMP_PREALLOC_NB 4096

static void apus_comp_state_init(ApusCompS *st, int ratio, int overlap,
                                 int d, int max_pos) {
    int coff = 1 + overlap;
    st->rows = coff * ratio;
    st->cols = coff * d;
    st->d = d;
    st->kv = calloc((size_t)st->rows * st->cols, sizeof(float));
    st->sc = malloc((size_t)st->rows * st->cols * sizeof(float));
    for (size_t i = 0; i < (size_t)st->rows * st->cols; i++)
        st->sc[i] = -INFINITY;
    st->max_nb = max_pos / ratio + 2;
    st->cap = st->max_nb < APUS_COMP_PREALLOC_NB ? st->max_nb
                                                 : APUS_COMP_PREALLOC_NB;
    st->cache = malloc((size_t)st->cap * d * sizeof(float));
    st->nb = 0;
}

/* Grow the compressed-entry cache to hold at least `need` entries
 * (geometric, capped at max_nb). Contents preserved. Returns 0 on success. */
static int apus_comp_ensure(ApusCompS *st, int need) {
    if (need <= st->cap) return 0;
    int cap = st->cap ? st->cap : 1;
    while (cap < need && cap < st->max_nb)
        cap = cap > st->max_nb / 2 ? st->max_nb : 2 * cap;
    if (cap < need) return -1;
    float *nc = realloc(st->cache, (size_t)cap * st->d * sizeof(float));
    if (!nc) return -1;
    st->cache = nc;
    st->cap = cap;
    return 0;
}

static void apus_comp_state_free(ApusCompS *st) {
    free(st->kv); free(st->sc); free(st->cache);
    st->kv = st->sc = st->cache = NULL;
}

/* Pooled entry -> normalize -> rope -> QAT -> cache append. M9d: rows
 * pooled over b; each entry's per-element op sequence is exactly the
 * serial loop's (row-independent), so this is bitwise identical for any
 * thread count. */
typedef struct {
    const ApusCompW *w;
    ApusCompS *st;
    float *out, *emit;
    const int64_t *rope_pos;
    const float *cos, *sin;
    int base;               /* st->nb at call time */
} ApusCompFinishJob;

static void apus_comp_finish_rows(void *vjob, size_t b0, size_t b1) {
    const ApusCompFinishJob *j = vjob;
    const ApusCompW *w = j->w;
    int d = w->d, rd = w->rd, half = rd / 2;
    for (size_t b = b0; b < b1; b++) {
        float *o = j->out + b * (size_t)d;
        for (int i = 0; i < d; i++) o[i] = apus_bf16_round(o[i]);
        apus_rms_norm(o, w->norm, w->eps, o, (size_t)d);
        apus_apply_rope(o + d - rd, j->cos + j->rope_pos[b] * half,
                        j->sin + j->rope_pos[b] * half, rd, 0);
        if (w->rotate) {
            apus_hadamard(o, d);
            apus_fp4_qat_sim(o, (size_t)d);
        } else {
            apus_fp8_qat_sim(o, (size_t)(d - rd), 64);
        }
        for (int i = 0; i < d; i++) o[i] = apus_bf16_round(o[i]);
        memcpy(j->st->cache + (size_t)(j->base + (int)b) * d, o,
               (size_t)d * sizeof(float));
        if (j->emit)
            memcpy(j->emit + b * (size_t)d, o, (size_t)d * sizeof(float));
    }
}

static void apus_comp_finish(const ApusCompW *w, ApusCompS *st,
                             float *out, int nb, const int64_t *rope_pos,
                             const float *cos, const float *sin,
                             float *emit) {
    if (apus_comp_ensure(st, st->nb + nb)) return;   /* OOM: unrecoverable */
    ApusCompFinishJob j = { w, st, out, emit, rope_pos, cos, sin, st->nb };
    if (nb >= 4)
        apus_pool_run((size_t)nb, apus_comp_finish_rows, &j);
    else
        apus_comp_finish_rows(&j, 0, (size_t)nb);
    st->nb += nb;
}

/* M9d: prefill pooling worker — one compressed entry (block b) per row.
 * Each output element's gather/softmax/weighted-sum sequence over the ne
 * candidates is EXACTLY the serial loop's (entry-independent), so pooling
 * the entries is bitwise identical for any thread count. */
typedef struct {
    const ApusCompW *w;
    const float *kv, *score;
    float *out;
    int64_t *rpos;
    int nb;
} ApusCompPoolJob;

static void apus_comp_pool_rows(void *vjob, size_t b0, size_t b1) {
    const ApusCompPoolJob *j = vjob;
    const ApusCompW *w = j->w;
    int ratio = w->ratio, overlap = w->overlap, d = w->d;
    int coff = 1 + overlap, cd = coff * d;
    int ne = coff * ratio;
    ApusScratchMark mk = apus_scratch_mark();   /* worker-local TLS */
    float *p = apus_scratch_alloc((size_t)ne * sizeof(float));
    float *kve = apus_scratch_alloc((size_t)ne * sizeof(float));
    for (size_t b = b0; b < b1; b++) {
        for (int c = 0; c < d; c++) {
            /* gather entries (overlap channel split, model.py:307-314):
             * first half = previous block's first channel half,
             * second half = current block's second channel half;
             * block 0's overlap half is 0 / -inf */
            float mx = -INFINITY;
            for (int e = 0; e < ne; e++) {
                if (overlap && e < ratio) {
                    if (b == 0) { p[e] = -INFINITY; kve[e] = 0.0f; }
                    else {
                        kve[e] = j->kv[(size_t)((b - 1) * ratio + e) * cd + c];
                        p[e] = j->score[(size_t)((b - 1) * ratio + e) * cd + c]
                             + w->ape[(size_t)e * cd + c];
                    }
                } else {
                    int e2 = overlap ? e - ratio : e;
                    int cc = overlap ? d + c : c;
                    kve[e] = j->kv[(size_t)(b * ratio + e2) * cd + cc];
                    p[e] = j->score[(size_t)(b * ratio + e2) * cd + cc]
                         + w->ape[(size_t)e2 * cd + cc];
                }
                if (p[e] > mx) mx = p[e];
            }
            float sum = 0.0f, acc = 0.0f;
            for (int e = 0; e < ne; e++) { p[e] = expf(p[e] - mx); sum += p[e]; }
            for (int e = 0; e < ne; e++)
                acc += kve[e] * (p[e] / sum);
            j->out[b * (size_t)d + c] = acc;
        }
        j->rpos[b] = (int64_t)b * ratio;
    }
    apus_scratch_reset(mk);
}

/* Compressor forward. x [s, dim] (BF16 values). Appends produced entries to
 * the cache and (if emit) copies them out. Returns entries produced (>=0). */
static int apus_comp_forward(const ApusCompW *w, ApusCompS *st,
                             const float *x, int s, int dim,
                             int64_t start_pos,
                             const float *cos, const float *sin, float *emit) {
    int ratio = w->ratio, overlap = w->overlap, d = w->d;
    int coff = 1 + overlap, cd = coff * d;
    ApusScratchMark mk = apus_scratch_mark();
    float *kv = apus_scratch_alloc((size_t)s * cd * sizeof(float));
    float *score = apus_scratch_alloc((size_t)s * cd * sizeof(float));
    apus_f32_linear(w->wkv, x, kv, s, dim, cd);
    apus_f32_linear(w->wgate, x, score, s, dim, cd);
    int produced = 0;

    if (start_pos == 0) {
        int should = s >= ratio;
        int rem = s % ratio;
        int cutoff = s - rem;
        int off = overlap ? ratio : 0;
        if (overlap && cutoff >= ratio) {
            memcpy(st->kv, kv + (size_t)(cutoff - ratio) * cd,
                   (size_t)ratio * cd * sizeof(float));
            for (int e = 0; e < ratio; e++)
                for (int c = 0; c < cd; c++)
                    st->sc[(size_t)e * cd + c] =
                        score[(size_t)(cutoff - ratio + e) * cd + c]
                        + w->ape[(size_t)e * cd + c];
        }
        if (rem > 0) {
            memcpy(st->kv + (size_t)off * cd, kv + (size_t)cutoff * cd,
                   (size_t)rem * cd * sizeof(float));
            for (int e = 0; e < rem; e++)
                for (int c = 0; c < cd; c++)
                    st->sc[(size_t)(off + e) * cd + c] =
                        score[(size_t)(cutoff + e) * cd + c]
                        + w->ape[(size_t)e * cd + c];
        }
        if (should) {
            int nb = cutoff / ratio;
            float *out = apus_scratch_alloc((size_t)nb * d * sizeof(float));
            int64_t *rpos = apus_scratch_alloc((size_t)nb * sizeof(int64_t));
            /* M9d: pooled over the nb entries (per-entry sequence unchanged) */
            ApusCompPoolJob pj = { w, kv, score, out, rpos, nb };
            if (nb >= 4)
                apus_pool_run((size_t)nb, apus_comp_pool_rows, &pj);
            else
                apus_comp_pool_rows(&pj, 0, (size_t)nb);
            apus_comp_finish(w, st, out, nb, rpos, cos, sin, emit);
            produced = nb;
        }
    } else {
        /* decode: one token */
        int slot = (int)(start_pos % ratio);
        int should = (start_pos + 1) % ratio == 0;
        if (overlap) {
            memcpy(st->kv + (size_t)(ratio + slot) * cd, kv, (size_t)cd * sizeof(float));
            for (int c = 0; c < cd; c++)
                st->sc[(size_t)(ratio + slot) * cd + c] =
                    score[c] + w->ape[(size_t)slot * cd + c];
            if (should) {
                int ne = 2 * ratio;
                float *outp = apus_scratch_alloc((size_t)d * sizeof(float));
                float *p = apus_scratch_alloc((size_t)ne * sizeof(float));
                for (int c = 0; c < d; c++) {
                    float mx = -INFINITY;
                    for (int e = 0; e < ne; e++) {
                        p[e] = (e < ratio)
                            ? st->sc[(size_t)e * cd + c]
                            : st->sc[(size_t)e * cd + d + c];
                        if (p[e] > mx) mx = p[e];
                    }
                    float sum = 0.0f;
                    for (int e = 0; e < ne; e++) { p[e] = expf(p[e] - mx); sum += p[e]; }
                    float acc = 0.0f;
                    for (int e = 0; e < ne; e++) {
                        float kv_v = (e < ratio)
                            ? st->kv[(size_t)e * cd + c]
                            : st->kv[(size_t)e * cd + d + c];
                        acc += kv_v * (p[e] / sum);
                    }
                    outp[c] = acc;
                }
                int64_t rp = start_pos + 1 - ratio;
                apus_comp_finish(w, st, outp, 1, &rp, cos, sin, emit);
                produced = 1;
                /* decode shift (model.py:353-354) */
                memmove(st->kv, st->kv + (size_t)ratio * cd,
                        (size_t)ratio * cd * sizeof(float));
                memmove(st->sc, st->sc + (size_t)ratio * cd,
                        (size_t)ratio * cd * sizeof(float));
            }
        } else {
            memcpy(st->kv + (size_t)slot * cd, kv, (size_t)cd * sizeof(float));
            for (int c = 0; c < cd; c++)
                st->sc[(size_t)slot * cd + c] =
                    score[c] + w->ape[(size_t)slot * cd + c];
            if (should) {
                float *outp = apus_scratch_alloc((size_t)d * sizeof(float));
                float *p = apus_scratch_alloc((size_t)ratio * sizeof(float));
                for (int c = 0; c < d; c++) {
                    float mx = -INFINITY;
                    for (int e = 0; e < ratio; e++) {
                        p[e] = st->sc[(size_t)e * cd + c];
                        if (p[e] > mx) mx = p[e];
                    }
                    float sum = 0.0f;
                    for (int e = 0; e < ratio; e++) { p[e] = expf(p[e] - mx); sum += p[e]; }
                    float acc = 0.0f;
                    for (int e = 0; e < ratio; e++)
                        acc += st->kv[(size_t)e * cd + c] * (p[e] / sum);
                    outp[c] = acc;
                }
                int64_t rp = start_pos + 1 - ratio;
                apus_comp_finish(w, st, outp, 1, &rp, cos, sin, emit);
                produced = 1;
            }
        }
    }
    apus_scratch_reset(mk);
    return produced;
}

/* --- indexer (model.py:380-433) ---------------------------------------------*/

/* M9d: row workers for the indexer's serial loops. Each (t,hh) / (t) row is
 * computed with EXACTLY the serial loop's per-element op sequence (pinned
 * scalar dot order, BF16 rounding points), so pooling the rows is bitwise
 * identical to the serial code for any thread count (pool.h contract). */

typedef struct {
    float *q;               /* [s, ih, idm] */
    const float *cos, *sin;
    int64_t start_pos;
    int ih, idm, rd;
} ApusIdxQPrepJob;

static void apus_indexer_qprep_rows(void *vjob, size_t r0, size_t r1) {
    const ApusIdxQPrepJob *j = vjob;
    int idm = j->idm, rd = j->rd, half = rd / 2;
    for (size_t r = r0; r < r1; r++) {
        size_t t = r / (size_t)j->ih;
        float *qv = j->q + r * (size_t)idm;
        apus_apply_rope(qv + idm - rd,
                        j->cos + (j->start_pos + (int64_t)t) * half,
                        j->sin + (j->start_pos + (int64_t)t) * half, rd, 0);
        apus_hadamard(qv, idm);
        apus_fp4_qat_sim(qv, (size_t)idm);
        for (int i = 0; i < idm; i++) qv[i] = apus_bf16_round(qv[i]);
    }
}

typedef struct {
    const float *q, *cache, *wt;
    float *sc;
    int ih, idm;
    int64_t nb;
} ApusIdxScoreJob;

static void apus_indexer_score_rows(void *vjob, size_t t0, size_t t1) {
    const ApusIdxScoreJob *j = vjob;
    int ih = j->ih, idm = j->idm;
    for (size_t t = t0; t < t1; t++)
        for (int64_t b = 0; b < j->nb; b++) {
            const float *kv = j->cache + (size_t)b * idm;
            float acc = 0.0f;
            for (int hh = 0; hh < ih; hh++) {
                const float *qv = j->q + (t * (size_t)ih + hh) * idm;
                float dot = 0.0f;
                for (int i = 0; i < idm; i++) dot += qv[i] * kv[i];
                dot = apus_bf16_round(dot);
                float v = dot > 0.0f ? dot * j->wt[t * (size_t)ih + hh] : 0.0f;
                v = apus_bf16_round(v);
                acc += v;
            }
            j->sc[t * (size_t)j->nb + b] = apus_bf16_round(acc);
        }
}

typedef struct {
    const float *sc;
    int32_t *idx_out;
    int64_t nb;
    int k, ratio, start_is_zero, offset;
} ApusIdxTopkJob;

static void apus_indexer_topk_rows(void *vjob, size_t t0, size_t t1) {
    const ApusIdxTopkJob *j = vjob;
    int64_t nb = j->nb;
    int k = j->k;
    ApusScratchMark mk = apus_scratch_mark();   /* worker-local TLS */
    int32_t *sel = apus_scratch_alloc((size_t)k * sizeof(int32_t));
    float *row = apus_scratch_alloc((size_t)nb * sizeof(float));
    for (size_t t = t0; t < t1; t++) {
        for (int64_t b = 0; b < nb; b++) {
            float v = j->sc[t * (size_t)nb + b];
            if (j->start_is_zero && b >= (int64_t)(t + 1) / j->ratio)
                v = -INFINITY;
            row[b] = v;
        }
        apus_topk_stable(row, (int)nb, k, sel);
        for (int jj = 0; jj < k; jj++) {
            int32_t v = sel[jj];
            if (j->start_is_zero) {
                if (v >= (int64_t)(t + 1) / j->ratio) v = -1;
                else v += j->offset;
            } else {
                v += j->offset;
            }
            j->idx_out[t * (size_t)k + jj] = v;
        }
    }
    apus_scratch_reset(mk);
}

static void apus_indexer(const ApusAttnCfg *cfg, const ApusAttnW *w,
                         ApusAttnS *st, const float *x, const float *qr,
                         int s, int64_t start_pos,
                         const float *cos, const float *sin,
                         int32_t *idx_out, int *k_out,
                         ApusAttnInterm *interm) {
    int ih = cfg->idx_heads, idm = cfg->idx_dim, rd = cfg->rope_dim;
    int ratio = cfg->ratio;
    ApusScratchMark mk = apus_scratch_mark();
    /* q from qr (q_norm output, NOT wq_b output — model.py:411) */
    float *q = apus_scratch_alloc((size_t)s * ih * idm * sizeof(float));
    apus_fp8_linear(&w->idx_wq_b, qr, q, s, cfg->q_lora, ih * idm);
    {
        /* M9d: pooled over the s*ih rows; per-row op sequence unchanged. */
        ApusIdxQPrepJob qp = { q, cos, sin, start_pos, ih, idm, rd };
        size_t rows = (size_t)s * ih;
        if (rows >= APUS_ROW_POOL_MIN)
            apus_pool_run(rows, apus_indexer_qprep_rows, &qp);
        else
            apus_indexer_qprep_rows(&qp, 0, rows);
    }
    /* indexer compressor (BEFORE the attn compressor — call order) */
    int inb = apus_comp_forward(&w->idx_comp, &st->idx_comp, x, s, cfg->dim,
                                start_pos, cos, sin,
                                interm ? interm->idx_comp_kv : NULL);
    if (interm) interm->idx_nb = inb;
    /* weights_proj x (idm^-0.5 * ih^-0.5) (model.py:418) */
    float *wt = apus_scratch_alloc((size_t)s * ih * sizeof(float));
    apus_bf16_linear(w->idx_wproj, x, wt, s, cfg->dim, ih);
    float wscale = (float)(pow((double)idm, -0.5) * pow((double)ih, -0.5));
    for (int i = 0; i < s * ih; i++) wt[i] = apus_bf16_round(wt[i] * wscale);
    /* scores (A1: bf16 round after einsum, after mul, after head sum) */
    int64_t nb = (start_pos + s) / ratio;
    float *sc = apus_scratch_alloc((size_t)s * nb * sizeof(float));
    {
        /* M9d: pooled over the s token rows; per-(t,b) op sequence
         * unchanged (the pinned scalar dot + A1 rounding points). */
        ApusIdxScoreJob sj = { q, st->idx_comp.cache, wt, sc, ih, idm, nb };
        if ((size_t)s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_indexer_score_rows, &sj);
        else
            apus_indexer_score_rows(&sj, 0, (size_t)s);
    }
    if (interm && interm->idx_scores)
        memcpy(interm->idx_scores, sc, (size_t)s * nb * sizeof(float));
    /* causal block mask + top-k + legality (model.py:424-432) */
    int k = cfg->idx_topk < nb ? cfg->idx_topk : (int)nb;
    int offset = start_pos == 0 ? s : cfg->window;
    {
        /* M9d: pooled over the s token rows; per-row selection unchanged. */
        ApusIdxTopkJob tj = { sc, idx_out, nb, k, ratio, start_pos == 0,
                              offset };
        if ((size_t)s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_indexer_topk_rows, &tj);
        else
            apus_indexer_topk_rows(&tj, 0, (size_t)s);
    }
    *k_out = k;
    if (interm && interm->idx_topk) {
        memcpy(interm->idx_topk, idx_out, (size_t)s * k * sizeof(int32_t));
        interm->idx_k = k;
    }
    apus_scratch_reset(mk);
}

/* --- sparse_attn (kernel.py:276-368 serial equivalent, A7/A8) ---------------*/

typedef struct {
    const float *q, *sink;
    int h, d, idxw, s;
    float scale;
    const float *kv_a;
    int64_t na;
    const float *kv_b;
    const int32_t *idsf;    /* [s, idxw] non-negative ids only */
    const int *ns;          /* [s] count per token */
    float *o;
} ApusSparseJob;

/* One (t, hh) attention head per row: scores -> max -> BF16-rounded
 * softmax numerator -> P*V -> sink-denominator divide (exact A7/A8 order).
 * M6c: threaded over heads; the per-head accumulation keeps the exact
 * pre-M6c rounding sequence (q*k dot: mul + sequential adds; P*V: one
 * FMA per element over sequential j — NEON across the independent i
 * lanes only), so per-head values are bitwise identical to the old
 * scalar code. M14: the q*k dots run four-at-a-time through
 * apus_dot4_f32_neon / apus_dot4_f32_x86 — staged exact products, per-dot
 * sequential adds — still bitwise identical to the scalar dot. */
static void apus_sparse_attn_head(void *vjob, size_t r0, size_t r1) {
    const ApusSparseJob *j = vjob;
    int h = j->h, d = j->d, idxw = j->idxw;
    ApusScratchMark mk = apus_scratch_mark();   /* worker-local TLS */
    float *sc = apus_scratch_alloc((size_t)idxw * sizeof(float));
    float *p = apus_scratch_alloc((size_t)idxw * sizeof(float));
#if APUS_X86
    const int use_avx2 = apus_x86_have_avx2();
    if (use_avx2) atomic_fetch_add(&apus_x86_hits, 1);
#endif
    for (size_t r = r0; r < r1; r++) {
        size_t t = r / (size_t)h, hh = r % (size_t)h;
        int n = j->ns[t];
        float *ov = j->o + (t * (size_t)h + hh) * d;
        if (n == 0) {
            memset(ov, 0, (size_t)d * sizeof(float));
            continue;
        }
        const int32_t *ids = j->idsf + t * (size_t)idxw;
        const float *qv = j->q + (t * (size_t)h + hh) * d;
        float mx = -INFINITY;
        int jj = 0;
#if APUS_X86
        /* M12a-2: four q.k dots per apus_dot4_f32_x86 group — bitwise
         * identical to the scalar apus_dot_f32_scalar per dot (c/x86.h);
         * sc/mx updates stay in jj order. */
        if (use_avx2) {
            for (; jj + 4 <= n; jj += 4) {
                const float *a[4], *b[4];
                for (int q = 0; q < 4; q++) {
                    int32_t id = ids[jj + q];
                    a[q] = qv;
                    b[q] = id < j->na
                        ? j->kv_a + (size_t)id * d
                        : j->kv_b + (size_t)(id - j->na) * d;
                }
                float d4[4];
                apus_dot4_f32_x86(a, b, (size_t)d, d4);
                for (int q = 0; q < 4; q++) {
                    sc[jj + q] = d4[q] * j->scale;
                    if (sc[jj + q] > mx) mx = sc[jj + q];
                }
            }
        }
#endif
#ifdef __ARM_NEON
        /* M14: four q.k dots per apus_dot4_f32_neon group — bitwise
         * identical to apus_dot_f32_scalar per dot (same contract as the
         * M12a-2 x86 branch above); sc/mx updates stay in jj order. */
        for (; jj + 4 <= n; jj += 4) {
            const float *a[4], *b[4];
            for (int q = 0; q < 4; q++) {
                int32_t id = ids[jj + q];
                a[q] = qv;
                b[q] = id < j->na
                    ? j->kv_a + (size_t)id * d
                    : j->kv_b + (size_t)(id - j->na) * d;
            }
            float d4[4];
            apus_dot4_f32_neon(a, b, (size_t)d, d4);
            for (int q = 0; q < 4; q++) {
                sc[jj + q] = d4[q] * j->scale;
                if (sc[jj + q] > mx) mx = sc[jj + q];
            }
        }
#endif
        for (; jj < n; jj++) {
            const float *kv = ids[jj] < j->na
                ? j->kv_a + (size_t)ids[jj] * d
                : j->kv_b + (size_t)(ids[jj] - j->na) * d;
            float dot = apus_dot_f32_scalar(qv, kv, (size_t)d);
            sc[jj] = dot * j->scale;
            if (sc[jj] > mx) mx = sc[jj];
        }
        float sum = 0.0f;
        for (int jj = 0; jj < n; jj++) {
            p[jj] = apus_bf16_round(expf(sc[jj] - mx));   /* kernel.py:340 */
            sum += p[jj];
        }
        size_t i = 0;
#ifdef __ARM_NEON
        for (; i + 4 <= (size_t)d; i += 4)
            vst1q_f32(ov + i, vdupq_n_f32(0.0f));
#endif
        for (; i < (size_t)d; i++) ov[i] = 0.0f;
        for (int jj = 0; jj < n; jj++) {
            const float *kv = ids[jj] < j->na
                ? j->kv_a + (size_t)ids[jj] * d
                : j->kv_b + (size_t)(ids[jj] - j->na) * d;
            size_t k = 0;
#ifdef __ARM_NEON
            float32x4_t pj = vdupq_n_f32(p[jj]);
            for (; k + 4 <= (size_t)d; k += 4)
                vst1q_f32(ov + k, vfmaq_f32(vld1q_f32(ov + k), pj,
                                            vld1q_f32(kv + k)));
#endif
#if APUS_X86
            /* M12a-2: mul+add (TWO roundings) per element — bitwise vs the
             * x86 scalar loop (NOT fused like the NEON vfmaq above; the
             * x86 anchor is the scalar mul+add, c/x86.h). */
            if (use_avx2) {
                apus_saxpy_x86(ov, p[jj], kv, (size_t)d);
                k = (size_t)d;
            }
#endif
            for (; k < (size_t)d; k++) ov[k] += p[jj] * kv[k];
        }
        /* sink in the DENOMINATOR only (kernel.py:345-348) */
        float denom = sum + expf(j->sink[hh] - mx);
        for (i = 0; i < (size_t)d; i++)
            ov[i] = apus_bf16_round(ov[i] / denom);
    }
    apus_scratch_reset(mk);
}

static void apus_sparse_attn(const float *q, const float *sink, int h, int d,
                             const int32_t *idxs, int idxw, int s, float scale,
                             const float *kv_a, int64_t na,
                             const float *kv_b, float *o) {
    ApusScratchMark mk = apus_scratch_mark();
    int32_t *idsf = apus_scratch_alloc((size_t)s * idxw * sizeof(int32_t));
    int *ns = apus_scratch_alloc((size_t)s * sizeof(int));
    for (int t = 0; t < s; t++) {
        int n = 0;
        for (int jj = 0; jj < idxw; jj++)
            if (idxs[(size_t)t * idxw + jj] >= 0)
                idsf[(size_t)t * idxw + n++] = idxs[(size_t)t * idxw + jj];
        ns[t] = n;
    }
    ApusSparseJob job = { q, sink, h, d, idxw, s, scale, kv_a, na, kv_b,
                          idsf, ns, o };
    apus_pool_run((size_t)s * h, apus_sparse_attn_head, &job);
    apus_scratch_reset(mk);
}

/* --- attention forward (model.py:436-543) ------------------------------------*/

/* M9d: row workers for the attention forward's serial per-token/per-head
 * loops. Each row runs EXACTLY the serial loop body's per-element op
 * sequence (row-independent), so pooling is bitwise identical for any
 * thread count (pool.h contract); below APUS_ROW_POOL_MIN rows the same
 * code runs inline. */

typedef struct {
    const float *x, *w;
    float eps;
    float *y;
    size_t n;
} ApusRmsRowsJob;

static void apus_rms_rows(void *vjob, size_t t0, size_t t1) {
    const ApusRmsRowsJob *j = vjob;
    for (size_t t = t0; t < t1; t++)
        apus_rms_norm(j->x + t * j->n, j->w, j->eps, j->y + t * j->n, j->n);
}

typedef struct {
    float *q;               /* [s, h, d] */
    const float *cos, *sin; /* row 0 = start_pos */
    float eps;
    int h, d, rd;
} ApusQNormJob;

static void apus_qnorm_rows(void *vjob, size_t r0, size_t r1) {
    const ApusQNormJob *j = vjob;
    int d = j->d, rd = j->rd, half = rd / 2;
    for (size_t r = r0; r < r1; r++) {
        size_t t = r / (size_t)j->h;
        float *qv = j->q + r * (size_t)d;
        /* per-head weight-free RMSNorm (model.py:498, A2) */
        float ss = 0.0f;
        for (int i = 0; i < d; i++) ss += qv[i] * qv[i];
        float inv = 1.0f / sqrtf(ss / (float)d + j->eps);
        for (int i = 0; i < d; i++) qv[i] = apus_bf16_round(qv[i] * inv);
        apus_apply_rope(qv + d - rd, j->cos + t * half, j->sin + t * half,
                        rd, 0);
    }
}

typedef struct {
    float *kv;              /* [s, d] */
    const float *cos, *sin; /* row 0 = start_pos */
    int d, rd;
} ApusKvQatJob;

static void apus_kvqat_rows(void *vjob, size_t t0, size_t t1) {
    const ApusKvQatJob *j = vjob;
    int d = j->d, rd = j->rd, half = rd / 2;
    for (size_t t = t0; t < t1; t++) {
        float *kvv = j->kv + t * (size_t)d;
        apus_apply_rope(kvv + d - rd, j->cos + t * half, j->sin + t * half,
                        rd, 0);
        apus_fp8_qat_sim(kvv, (size_t)(d - rd), 64);   /* group-64 nope QAT */
        for (int i = 0; i < d; i++) kvv[i] = apus_bf16_round(kvv[i]);
    }
}

/* M6c: batched grouped wo_a GEMV (all G*ol*s rows in one dispatch).
 * Row r: t = r/(G*ol), g = (r%(G*ol))/ol, j = r%ol; out BF16-rounded,
 * no input rounding (og already holds BF16 values, A4). */
typedef struct {
    const uint16_t *wa;     /* [G*ol, sub] BF16 bits */
    const float *o;         /* [s, h*d] */
    float *y;               /* [s, G*ol] */
    int s, G, ol, sub, hd;
} ApusWoAJob;

#if APUS_X86
/* M12a-2: AVX2 wo_a rows — four rows per apus_dot4_bf16_x86 group
 * (exactly-widened BF16, staged exact products, scalar sequential add
 * order per row => bitwise identical to the scalar row loop, c/x86.h
 * contract — note this takes NO reorder budget, unlike the NEON path);
 * trailing rows take the scalar loop. */
APUS_TGT_AVX2
static void apus_woa_rows_avx2(const ApusWoAJob *j, size_t r0, size_t r1) {
    size_t gl = (size_t)j->G * j->ol;
    size_t n = (size_t)j->sub;
    size_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const float *a[4];
        const uint16_t *b[4];
        for (int q = 0; q < 4; q++) {
            size_t rr = r + (size_t)q;
            size_t t = rr / gl, rem = rr % gl;
            size_t g = rem / (size_t)j->ol, jj = rem % (size_t)j->ol;
            a[q] = j->o + t * (size_t)j->hd + g * (size_t)j->sub;
            b[q] = j->wa + (g * (size_t)j->ol + jj) * (size_t)j->sub;
        }
        float d[4];
        apus_dot4_bf16_x86(a, b, n, d);
        for (int q = 0; q < 4; q++)
            j->y[r + (size_t)q] = apus_bf16_round(d[q]);
    }
    for (; r < r1; r++) {
        size_t t = r / gl, rr = r % gl;
        size_t g = rr / (size_t)j->ol, jj = rr % (size_t)j->ol;
        const float *og = j->o + t * (size_t)j->hd + g * (size_t)j->sub;
        const uint16_t *wr = j->wa + (g * (size_t)j->ol + jj) * (size_t)j->sub;
        float dot = 0.0f;
        for (size_t k = 0; k < n; k++)
            dot += og[k] * apus_bf16_f32(wr[k]);
        j->y[r] = apus_bf16_round(dot);
    }
}
#endif

static void apus_woa_rows(void *vjob, size_t r0, size_t r1) {
    const ApusWoAJob *j = vjob;
#if APUS_X86
    /* M12a-2: bitwise-identical AVX2 rows when the CPU supports them. */
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_woa_rows_avx2(j, r0, r1);
        return;
    }
#endif
    size_t gl = (size_t)j->G * j->ol;
    for (size_t r = r0; r < r1; r++) {
        size_t t = r / gl, rr = r % gl;
        size_t g = rr / (size_t)j->ol, jj = rr % (size_t)j->ol;
        const float *og = j->o + t * (size_t)j->hd + g * (size_t)j->sub;
        const uint16_t *wr = j->wa + (g * (size_t)j->ol + jj) * (size_t)j->sub;
        size_t n = (size_t)j->sub, k = 0;
        float dot = 0.0f;
#ifdef __ARM_NEON
        /* M6c: 4-accumulator NEON dot over exactly-widened BF16 weights
         * (widen == the old f32 storage bitwise). REORDER vs the pre-M6c
         * scalar sequential-k sum (accepted SIMD tolerance class,
         * documented in tests/m6c/README.md): FP32 rel error ~1e-7 vs the
         * FP32 error scale, and the output is BF16-rounded (ulp ~4e-3), so
         * the rounded result is almost always bitwise identical to scalar.
         * Order is fixed per row (M- and thread-count-independent). The
         * scalar path was FMA-latency-bound (~1 elem/4 cycles); this path
         * is DRAM-bound on the (now halved) wo_a weight stream. */
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        for (; k + 16 <= n; k += 16) {
            uint16x8_t b0 = vld1q_u16(wr + k);
            uint16x8_t b1 = vld1q_u16(wr + k + 8);
            a0 = vfmaq_f32(a0, vld1q_f32(og + k),
                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b0), 16)));
            a1 = vfmaq_f32(a1, vld1q_f32(og + k + 4),
                vreinterpretq_f32_u32(vshll_high_n_u16(b0, 16)));
            a2 = vfmaq_f32(a2, vld1q_f32(og + k + 8),
                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b1), 16)));
            a3 = vfmaq_f32(a3, vld1q_f32(og + k + 12),
                vreinterpretq_f32_u32(vshll_high_n_u16(b1, 16)));
        }
        for (; k + 4 <= n; k += 4) {
            uint32x4_t b = vshll_n_u16(vld1_u16(wr + k), 16);
            a0 = vfmaq_f32(a0, vld1q_f32(og + k),
                           vreinterpretq_f32_u32(b));
        }
        a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        dot = vaddvq_f32(a0);
#endif
        for (; k < n; k++) dot += og[k] * apus_bf16_f32(wr[k]);
        j->y[t * gl + rr] = apus_bf16_round(dot);
    }
}

void apus_attn_state_init(ApusAttnS *st, const ApusAttnCfg *cfg) {
    st->pos = 0;
    st->win = calloc((size_t)cfg->window * cfg->head_dim, sizeof(float));
    if (cfg->ratio) {
        apus_comp_state_init(&st->comp, cfg->ratio, cfg->ratio == 4,
                             cfg->head_dim, cfg->max_pos);
        if (cfg->has_indexer)
            apus_comp_state_init(&st->idx_comp, cfg->ratio, 1,
                                 cfg->idx_dim, cfg->max_pos);
    }
}

void apus_attn_state_free(ApusAttnS *st, const ApusAttnCfg *cfg) {
    free(st->win);
    st->win = NULL;
    if (cfg->ratio) {
        apus_comp_state_free(&st->comp);
        if (cfg->has_indexer) apus_comp_state_free(&st->idx_comp);
    }
}

void apus_attention(const ApusAttnCfg *cfg, const ApusAttnW *w, ApusAttnS *st,
                    const float *x, int s, int64_t start_pos,
                    const float *cos, const float *sin,
                    float *out, ApusAttnInterm *interm) {
    int h = cfg->n_heads, d = cfg->head_dim, rd = cfg->rope_dim;
    int win = cfg->window, ratio = cfg->ratio, half = rd / 2;
    const float *fc = cos + start_pos * half;
    const float *fs = sin + start_pos * half;
    ApusScratchMark mk = apus_scratch_mark();

    /* q path (model.py:496-499) */
    float *qa = apus_scratch_alloc((size_t)s * cfg->q_lora * sizeof(float));
    apus_fp8_linear(&w->wq_a, x, qa, s, cfg->dim, cfg->q_lora);
    float *qr = apus_scratch_alloc((size_t)s * cfg->q_lora * sizeof(float));
    {
        ApusRmsRowsJob rj = { qa, w->q_norm, cfg->eps, qr,
                              (size_t)cfg->q_lora };
        if (s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_rms_rows, &rj);
        else
            apus_rms_rows(&rj, 0, (size_t)s);
    }
    float *q = apus_scratch_alloc((size_t)s * h * d * sizeof(float));
    apus_fp8_linear(&w->wq_b, qr, q, s, cfg->q_lora, h * d);
    {
        /* M9d: pooled over the s*h (token, head) rows; per-row op sequence
         * unchanged (norm sums + rope per row). */
        ApusQNormJob qj = { q, fc, fs, cfg->eps, h, d, rd };
        size_t rows = (size_t)s * h;
        if (rows >= APUS_ROW_POOL_MIN)
            apus_pool_run(rows, apus_qnorm_rows, &qj);
        else
            apus_qnorm_rows(&qj, 0, rows);
    }
    if (interm && interm->q)
        memcpy(interm->q, q, (size_t)s * h * d * sizeof(float));

    /* window kv path (model.py:502-506) */
    float *kva = apus_scratch_alloc((size_t)s * d * sizeof(float));
    apus_fp8_linear(&w->wkv, x, kva, s, cfg->dim, d);
    float *kv = apus_scratch_alloc((size_t)s * d * sizeof(float));
    {
        ApusRmsRowsJob rj = { kva, w->kv_norm, cfg->eps, kv, (size_t)d };
        if (s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_rms_rows, &rj);
        else
            apus_rms_rows(&rj, 0, (size_t)s);
    }
    {
        ApusKvQatJob kj = { kv, fc, fs, d, rd };
        if (s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_kvqat_rows, &kj);
        else
            apus_kvqat_rows(&kj, 0, (size_t)s);
    }
    if (interm && interm->win_kv)
        memcpy(interm->win_kv, kv, (size_t)s * d * sizeof(float));

    /* window indices (model.py:507) */
    float *o = apus_scratch_alloc((size_t)s * h * d * sizeof(float));
    float scale = (float)pow((double)d, -0.5);
    if (start_pos > 0 && s > 1) {
        /* M8: causal multi-token forward from a CARRIED state (speculative
         * verify batches). The prefill path below is causal only from
         * start_pos == 0 and the decode path only for s == 1; a mid-stream
         * batch is instead processed as a per-token interleave of the
         * exact s=1 decode steps on the precomputed q/kv rows — state
         * (ring, compressors) and outputs are bitwise "as if the tokens
         * were decoded one-by-one" by construction. */
        int dim = cfg->dim;
        for (int t = 0; t < s; t++) {
            int64_t spt = start_pos + t;
            int32_t *win_idx_t = apus_scratch_alloc((size_t)win * sizeof(int32_t));
            int winw = apus_window_idxs(win_idx_t, 1, spt, win);
            int32_t *comp_idx_t = NULL;
            int compw = 0;
            if (ratio) {
                if (cfg->has_indexer) {
                    comp_idx_t = apus_scratch_alloc((size_t)cfg->idx_topk * sizeof(int32_t));
                    apus_indexer(cfg, w, st, x + (size_t)t * dim,
                                 qr + (size_t)t * cfg->q_lora, 1, spt,
                                 cos, sin, comp_idx_t, &compw, NULL);
                } else {
                    int maxw = (int)((spt + 1) / ratio);
                    comp_idx_t = apus_scratch_alloc((size_t)(maxw ? maxw : 1) * sizeof(int32_t));
                    compw = apus_compress_idxs(comp_idx_t, ratio, 1, spt, win);
                }
            }
            /* ring write (model.py:530) */
            memcpy(st->win + (size_t)(spt % win) * d, kv + (size_t)t * d,
                   (size_t)d * sizeof(float));
            /* attn compressor (model.py:532) */
            if (ratio)
                apus_comp_forward(&w->comp, &st->comp, x + (size_t)t * dim,
                                  1, dim, spt, cos, sin, NULL);
            /* sparse attention for this token */
            int idxw = winw + compw;
            int32_t *idxs_t = apus_scratch_alloc((size_t)idxw * sizeof(int32_t));
            memcpy(idxs_t, win_idx_t, (size_t)winw * sizeof(int32_t));
            if (compw)
                memcpy(idxs_t + winw, comp_idx_t, (size_t)compw * sizeof(int32_t));
            apus_sparse_attn(q + (size_t)t * h * d, w->sink, h, d, idxs_t,
                             idxw, 1, scale, st->win, win,
                             ratio ? st->comp.cache : NULL,
                             o + (size_t)t * h * d);
        }
    } else {
    /* compressed indices (model.py:509-515); INDEXER runs BEFORE the
     * ring write and BEFORE the attn compressor (decode call order) */
    int32_t *win_idx = apus_scratch_alloc((size_t)s * win * sizeof(int32_t));
    int winw = apus_window_idxs(win_idx, s, start_pos, win);
    int32_t *comp_idx = NULL;
    int compw = 0;
    if (ratio) {
        int offset = start_pos == 0 ? s : win;
        if (cfg->has_indexer) {
            comp_idx = apus_scratch_alloc((size_t)s * cfg->idx_topk * sizeof(int32_t));
            apus_indexer(cfg, w, st, x, qr, s, start_pos, cos, sin,
                         comp_idx, &compw, interm);
        } else {
            int maxw = start_pos > 0 ? (int)((start_pos + 1) / ratio) : s / ratio;
            comp_idx = apus_scratch_alloc((size_t)s * (maxw ? maxw : 1) * sizeof(int32_t));
            compw = apus_compress_idxs(comp_idx, ratio, s, start_pos, offset);
        }
    }

    /* ring write (model.py:518-523, 530) */
    if (start_pos == 0) {
        if (s <= win) {
            memcpy(st->win, kv, (size_t)s * d * sizeof(float));
        } else {
            int cut = s % win;
            memcpy(st->win + (size_t)cut * d, kv + (size_t)(s - win) * d,
                   (size_t)(win - cut) * d * sizeof(float));
            memcpy(st->win, kv + (size_t)(s - cut) * d,
                   (size_t)cut * d * sizeof(float));
        }
    } else {
        memcpy(st->win + (size_t)(start_pos % win) * d, kv,
               (size_t)d * sizeof(float));
    }

    /* attn compressor (model.py:525/532) */
    if (ratio) {
        int nb = apus_comp_forward(&w->comp, &st->comp, x, s, cfg->dim,
                                   start_pos, cos, sin,
                                   interm ? interm->comp_kv : NULL);
        if (interm) interm->comp_nb = nb;
    }

    /* sparse attention (model.py:528/533) */
    int idxw = winw + compw;
    int32_t *idxs = apus_scratch_alloc((size_t)s * idxw * sizeof(int32_t));
    for (int t = 0; t < s; t++) {
        memcpy(idxs + (size_t)t * idxw, win_idx + (size_t)t * winw,
               (size_t)winw * sizeof(int32_t));
        if (compw)
            memcpy(idxs + (size_t)t * idxw + winw, comp_idx + (size_t)t * compw,
                   (size_t)compw * sizeof(int32_t));
    }
    if (start_pos == 0)
        apus_sparse_attn(q, w->sink, h, d, idxs, idxw, s, scale,
                         kv, s, st->comp.cache, o);
    else
        apus_sparse_attn(q, w->sink, h, d, idxs, idxw, s, scale,
                         st->win, win, ratio ? st->comp.cache : NULL, o);
    }
    /* inverse RoPE on outputs (model.py:534) */
    for (int t = 0; t < s; t++)
        for (int hh = 0; hh < h; hh++)
            apus_apply_rope(o + ((size_t)t * h + hh) * d + d - rd,
                            fc + (size_t)t * half, fs + (size_t)t * half,
                            rd, 1);
    if (interm && interm->attn_out)
        memcpy(interm->attn_out, o, (size_t)s * h * d * sizeof(float));

    /* grouped low-rank o-proj (model.py:537-542); wo_a BF16 (A4) */
    int G = cfg->o_groups, ol = cfg->o_lora, sub = h * d / G;
    float *y = apus_scratch_alloc((size_t)s * G * ol * sizeof(float));
    /* M7b backend hook: FP32 accumulate, BF16-rounded output (no input
     * rounding — og already holds BF16 values). On any hook failure the
     * whole group loop falls back to the CPU path (writes are pure). */
    int woa_done = 0;
    if (apus_backend_hooks.f32_linear) {
        woa_done = 1;
        for (int t = 0; t < s && woa_done; t++)
            for (int g = 0; g < G; g++) {
                const float *og = o + (size_t)t * h * d + (size_t)g * sub;
                const uint16_t *wa = w->wo_a + (size_t)g * ol * sub;
                float *yg = y + (size_t)t * G * ol + (size_t)g * ol;
                if (apus_backend_hooks.f32_linear(
                        (const float *)wa, og, yg, 1, sub, ol,
                        APUS_HOOK_R_OUT | APUS_HOOK_W_BF16) != 0) {
                    woa_done = 0;
                    break;
                }
            }
    }
#if APUS_BLAS
    /* M9d: at prefill batch sizes the grouped wo_a GEMM goes to Accelerate
     * (per group: M=s, K=sub, O=ol; BF16 weights widened EXACTLY to FP32
     * per tile — c/blas.h). FP32 summation-order reorder vs the NEON rows
     * only (accepted class — the M6c wo_a notes already document this dot
     * as reorder-tolerant, and the output is BF16-rounded); engaged only
     * at s >= APUS_BLAS_M_MIN, so every pre-M9d bitwise gate (M<=250)
     * keeps the NEON kernel, and decode is bit-identical. */
    if (!woa_done && s >= APUS_BLAS_M_MIN && apus_blas_available()) {
        apus_woa_gemm_blas(w->wo_a, o, y, (size_t)s, (size_t)G,
                           (size_t)ol, (size_t)sub, (size_t)h * d);
        for (size_t i = 0; i < (size_t)s * (size_t)(G * ol); i++)
            y[i] = apus_bf16_round(y[i]);
        woa_done = 1;
    }
#endif
    if (!woa_done) {
        /* M6c: all G*ol*s rows in one pool dispatch; per-row FP32
         * accumulation stays inside the row (thread-count independent). */
        ApusWoAJob job = { w->wo_a, o, y, s, G, ol, sub, h * d };
        apus_pool_run((size_t)s * G * ol, apus_woa_rows, &job);
    }
    apus_fp8_linear(&w->wo_b, y, out, s, G * ol, cfg->dim);
    if (interm && interm->o_out)
        memcpy(interm->o_out, out, (size_t)s * cfg->dim * sizeof(float));

    apus_scratch_reset(mk);
}

#endif /* APUS_ATTN_IMPLEMENTATION */
#endif /* APUS_ATTN_H */
