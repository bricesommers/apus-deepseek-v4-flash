/*
 * c/fp8.h — FP8-E4M3 dense weights with blockwise 128x128 UE8M0 scales:
 * GEMV/GEMM kernels for DeepSeek-V4-Flash dense linears (wq_a/wq_b, wkv,
 * wo_a/wo_b, indexer.wq_b, shared-expert w1/w2/w3, MTP e_proj/h_proj).
 * C11, libc + arm_neon.h only.
 *
 * Numerics contract (normative reference: reference/inference/kernel.py
 * fp8_gemm_kernel lines 203-254 + act_quant lines 40-125, invoked from
 * model.py:116-118):
 *
 *   Storage:  weight row-major [O, K] FP8-E4M3 codes (one byte per element).
 *             Scales [ceil(O/128), ceil(K/128)] UE8M0 bytes, row-major,
 *             one per 128x128 tile; scale value = 2^(byte-127)
 *             (apus_ue8m0_f32, shared with fp4.h).
 *
 *   NORMATIVE PATH (what the reference computes for every FP8 linear):
 *   the BF16 activation is first quantized to FP8-E4M3 with a per-128-along-K
 *   power-of-2 scale (act_quant, scale_fmt="ue8m0"):
 *       amax_b = max(|x_b|) floored at 1e-4
 *       s_b    = 2^ceil(log2(amax_b / 448))        (bit trick, see fp4.h)
 *       code   = fp8_e4m3(clamp(x / s_b, +-448))   (round-to-nearest-even)
 *   This is exactly the fp4 path's activation quantization, so the helpers
 *   are reused from fp4.h (apus_fp4_act_quant_scalar et al.), not duplicated.
 *   Then fp8_gemm: per 128-wide K block an FP8xFP8 GEMM accumulates the
 *   block dot in FP32, and the block is folded into the FP32 total with the
 *   scale PRODUCT computed first (kernel.py:242-249):
 *       sc     = scale_a[m,kb] * scale_b[o/128,kb]   (one FP32 rounding)
 *       total += dot_kb * sc                          (one FP32 rounding)
 *   NOTE the different scale order vs the FP4 path ((dot*sa)*sb): here the
 *   reference multiplies the two scales together first. Both kernels below
 *   mirror this (plain multiplies, no FMA at the scale steps).
 *   We reproduce the block dot in FP32: E4M3 codes are exactly representable
 *   in FP32, so the FP32 dot is the same computation the tensor core does,
 *   up to summation order (unspecified in the reference).
 *
 *   Output: the reference casts the accumulator to BF16 (out_dtype=BF16).
 *   These kernels output FP32; the caller must BF16-round (apus_bf16_round)
 *   at the same points the reference does to stay bit-faithful.
 *
 *   E4M3 NaN codes 0x7F/0xFF decode as +-480 here (matching fp4.h's
 *   apus_e4m3_dequant_f32); they never occur in the checkpoint because the
 *   quant rule clamps to +-448.
 *
 * M4 integration notes:
 *   - Round activations to BF16 (apus_bf16_round) before act quant: the
 *     reference act_quant input dtype is BF16 (kernel.py:41).
 *   - The reference asserts K % 128 == 0 for act quant and uses
 *     ceil(N/128) x ceil(K/128) weight scales. These kernels also accept a
 *     partial trailing K block (amax over the remainder) and any O (scale
 *     row o/128); partial-K shapes cannot be validated against the
 *     reference. All real shapes have K, O multiples of 128.
 *   - UE8M0 byte 0 = 2^-127 (FP32 subnormal, exact); byte 255 = +inf
 *     (2^128 overflows FP32), matching the M1-pinned numpy semantics.
 *
 * Usage: #define APUS_FP8_IMPLEMENTATION in exactly one TU. The fp8
 * implementation CALLS fp4.h helpers (act quant, ue8m0/e4m3 conversion), so
 * some TU in the program must also define APUS_FP4_IMPLEMENTATION. */

#ifndef APUS_FP8_H
#define APUS_FP8_H

#include "fp4.h"
#include "pool.h"
#include "x86.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_FP8_GROUP 128u   /* scale tile edge along both O and K */

/* Number of per-128 scale blocks for K (ceil(K/128)). */
size_t apus_fp8_blocks(size_t K);

/* Per-128-along-K FP8-E4M3 activation quantization, ue8m0 scales —
 * thin wrapper over the fp4.h port of the same reference kernel. */
void apus_fp8_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales);

/* NORMATIVE GEMV (decode, M=1):
 *   out[o] = sum_kb dot_kb * (as[kb] * ws[(o/128)*nb + kb])
 * w: [O, K] E4M3 codes, ws: [ceil(O/128), nb] UE8M0, nb = apus_fp8_blocks(K),
 * acodes: [K] FP8 codes, as: [nb] FP32 act scales,
 * scratch: K floats (dequantized act codes), out: [O]. */
void apus_fp8_gemv_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out, size_t O, size_t K);

/* NORMATIVE GEMM (prefill): out[M,O] row-major. acodes: [M,K],
 * as: [M, nb], scratch: M*K floats. */
void apus_fp8_gemm_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out,
                          size_t M, size_t O, size_t K);

#ifdef __ARM_NEON
void apus_fp8_gemv_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K);
void apus_fp8_gemm_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K);
#endif

#if APUS_X86
/* M12a-2: AVX2 kernels — BITWISE identical to the scalar kernels (c/x86.h
 * contract: exact E4M3 expand, staged exact products, scalar sequential
 * summation order). Runtime-dispatched; the scalar kernels remain the
 * fallback on non-AVX2 x86-64. */
void apus_fp8_gemv_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K);
void apus_fp8_gemm_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K);
#endif

/* M6c: THREADED GEMM (decode hot path). Same numerics as the single-thread
 * kernels below M9b's BLAS cutoff: output rows partitioned over the c/pool.h
 * lanes, each out[m,o] computed by exactly one thread with the identical
 * per-output accumulation order, so results are bitwise identical for any
 * thread count (APUS_THREADS=1 included) and bitwise identical to calling
 * the corresponding non-mt kernel. scratch: M*K floats.
 * M9b: at M >= APUS_BLAS_M_MIN this dispatches to the Accelerate BLAS path
 * (c/blas.h) — FP32-accumulation-reorder class, NOT bitwise vs the NEON
 * kernels; deterministic and thread-count independent. */
void apus_fp8_gemm_mt(const uint8_t *w, const uint8_t *ws,
                      const uint8_t *acodes, const float *as,
                      float *scratch, float *out,
                      size_t M, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_FP8_IMPLEMENTATION

#include <math.h>
#include <string.h>

size_t apus_fp8_blocks(size_t K) {
    return (K + APUS_FP8_GROUP - 1) / APUS_FP8_GROUP;
}

void apus_fp8_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales) {
    apus_fp4_act_quant_scalar(x, K, codes, scales);
}

static void apus_fp8_act_dequant_all(const uint8_t *acodes, float *adeq,
                                     size_t K) {
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(acodes[i]);
}

void apus_fp8_gemv_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out, size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    float *adeq = scratch;
    apus_fp8_act_dequant_all(acodes, adeq, K);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * K;
        const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
        float total = 0.0f;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * APUS_FP8_GROUP;
            size_t hi = lo + APUS_FP8_GROUP;
            if (hi > K) hi = K;
            float dot = 0.0f;
            for (size_t i = lo; i < hi; i++)
                dot += adeq[i] * apus_e4m3_dequant_f32(wp[i]);
            /* reference order: scale product first, then dot * sc */
            float sc = as[kb] * apus_ue8m0_f32(sp[kb]);
            total += dot * sc;
        }
        out[o] = total;
    }
}

void apus_fp8_gemm_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out,
                          size_t M, size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    for (size_t m = 0; m < M; m++)
        apus_fp8_act_dequant_all(acodes + m * K, scratch + m * K, K);
    for (size_t m = 0; m < M; m++) {
        const float *adeq = scratch + m * K;
        const float *am = as + m * nb;
        for (size_t o = 0; o < O; o++) {
            const uint8_t *wp = w + o * K;
            const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
            float total = 0.0f;
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * APUS_FP8_GROUP;
                size_t hi = lo + APUS_FP8_GROUP;
                if (hi > K) hi = K;
                float dot = 0.0f;
                for (size_t i = lo; i < hi; i++)
                    dot += adeq[i] * apus_e4m3_dequant_f32(wp[i]);
                float sc = am[kb] * apus_ue8m0_f32(sp[kb]);
                total += dot * sc;
            }
            out[m * O + o] = total;
        }
    }
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

/* Convert 8 E4M3 codes (widened to u16 lanes) to 8 FP32, exactly, via an
 * FP16 bit placement (M9a — replaces the two-path integer construction,
 * ~2x fewer ops, same exact values for all 256 codes):
 *   h16 = (c&0x80)<<8 | (c&0x7F)<<7  ->  f16 = s * (1+m/8) * 2^(e-15)
 * for e>=1, and the f16 subnormal m*2^-17 for e==0. Every E4M3 value is
 * exactly representable in FP16, and BOTH cases are exactly 2^-8 times the
 * E4M3 value, so value = f32(h16) * 256 is EXACT (power-of-2 multiplier):
 * e>=1 gives (1+m/8)*2^(e-7), e==0 gives m*2^-9. NaN codes 0x7F/0xFF decode
 * as +-480 like the old path (they never occur in the checkpoint). */
static inline void apus_e4m3_cvt8_neon(uint16x8_t c, float32x4_t v[2]) {
    uint16x8_t h = vorrq_u16(vshlq_n_u16(vandq_u16(c, vdupq_n_u16(0x80u)), 8),
                             vshlq_n_u16(vandq_u16(c, vdupq_n_u16(0x7Fu)), 7));
    v[0] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))),
                       256.0f);
    v[1] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))),
                       256.0f);
}

/* Expand 16 E4M3 bytes -> 16 FP32 (4 vectors), exact. */
static inline void apus_e4m3_expand16_neon(const uint8_t *p, float32x4_t v[4]) {
    uint8x16_t b = vld1q_u8(p);
    float32x4_t lo[2], hi[2];
    apus_e4m3_cvt8_neon(vmovl_u8(vget_low_u8(b)), lo);
    apus_e4m3_cvt8_neon(vmovl_u8(vget_high_u8(b)), hi);
    v[0] = lo[0]; v[1] = lo[1]; v[2] = hi[0]; v[3] = hi[1];
}

/* M9a: canonical FP32 summation order for the NEON block dots (ILP reorder).
 * Four independent vector accumulators per row kill the FMA-latency chain of
 * the pre-M9a single-accumulator loop (~1 elem/cycle -> FMA-throughput-bound).
 * The order is FIXED and deterministic (thread-count- and M-independent):
 *   acc[j] lane l accumulates elements 4j+l of every 16-element chunk, in
 *   chunk order; the four accumulators combine lane-wise as
 *   ((a0+a1)+(a2+a3)), then horizontally as ((s0+s1)+(s2+s3)); a partial
 *   trailing chunk (< 16) is added scalar AFTER the reduction.
 * The Metal fp8_dot shader (c/backend_metal.mm) mirrors this exact structure
 * so GPU and CPU block dots stay bit-identical. Only the summation order
 * within dots changed vs pre-M9a (the accepted scalar-vs-NEON tolerance
 * class, tests/m3/m4a); the scale-application sequence is untouched. */
static inline float apus_fp8_reduce4_neon(float32x4_t a0, float32x4_t a1,
                                          float32x4_t a2, float32x4_t a3) {
    float32x4_t s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    return (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
         + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
}

/* FP32 dot of one 128-block in the M9a canonical order above. */
static inline float apus_fp8_dot128_neon(const float *ap, const uint8_t *wp,
                                         size_t len) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        float32x4_t wv[4];
        apus_e4m3_expand16_neon(wp + i, wv);
        a0 = vfmaq_f32(a0, wv[0], vld1q_f32(ap + i));
        a1 = vfmaq_f32(a1, wv[1], vld1q_f32(ap + i + 4));
        a2 = vfmaq_f32(a2, wv[2], vld1q_f32(ap + i + 8));
        a3 = vfmaq_f32(a3, wv[3], vld1q_f32(ap + i + 12));
    }
    float dot = apus_fp8_reduce4_neon(a0, a1, a2, a3);
    for (; i < len; i++)   /* partial trailing block tail (< 16) */
        dot += ap[i] * apus_e4m3_dequant_f32(wp[i]);
    return dot;
}

void apus_fp8_gemv_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K) {
    size_t nb = apus_fp8_blocks(K);
    float *adeq = scratch;
    apus_fp8_act_dequant_all(acodes, adeq, K);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * K;
        const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
        float total = 0.0f;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * APUS_FP8_GROUP;
            size_t hi = lo + APUS_FP8_GROUP;
            if (hi > K) hi = K;
            float dot = apus_fp8_dot128_neon(adeq + lo, wp + lo, hi - lo);
            float sc = as[kb] * apus_ue8m0_f32(sp[kb]);
            total += dot * sc;
        }
        out[o] = total;
    }
}

/* Rows [o0, o1) of the NEON GEMM — the one and only row body, shared by
 * apus_fp8_gemm_neon (full range) and the M6c threaded variant, so the mt
 * kernel is bitwise identical to the single-thread one by construction.
 * 4 activation rows per weight pass: each 16-code weight chunk is expanded
 * once and FMA'd against 4 rows. Per row the accumulation is EXACTLY the
 * apus_fp8_dot128_neon M9a order (4 independent accumulators combined
 * ((a0+a1)+(a2+a3)), scalar tail after the reduction), so a row's value is
 * independent of M (mc==4 fast path vs the per-row fallback are bitwise
 * equal). The per-128-block scale correction (scale product first, then
 * dot*sc) matches the scalar kernel step for step. */
static inline void apus_fp8_gemm_rows_neon(const uint8_t *w, const uint8_t *ws,
                                           const float *as,
                                           const float *scratch, float *out,
                                           size_t M, size_t O, size_t K,
                                           size_t o0, size_t o1) {
    size_t nb = apus_fp8_blocks(K);
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = w + o * K;
            const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
            float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * APUS_FP8_GROUP;
                size_t hi = lo + APUS_FP8_GROUP;
                if (hi > K) hi = K;
                size_t len = hi - lo, i = 0;
                float sc_b = apus_ue8m0_f32(sp[kb]);
                if (mc == 4) {
                    /* acc[r][j]: row r, chunk-lane j (M9a canonical order) */
                    float32x4_t acc[4][4];
                    for (int r = 0; r < 4; r++)
                        for (int j = 0; j < 4; j++)
                            acc[r][j] = vdupq_n_f32(0.0f);
                    for (; i + 16 <= len; i += 16) {
                        float32x4_t wv[4];
                        apus_e4m3_expand16_neon(wp + lo + i, wv);
                        for (int r = 0; r < 4; r++) {
                            const float *ap = scratch + (m0 + r) * K + lo + i;
                            acc[r][0] = vfmaq_f32(acc[r][0], wv[0],
                                                  vld1q_f32(ap));
                            acc[r][1] = vfmaq_f32(acc[r][1], wv[1],
                                                  vld1q_f32(ap + 4));
                            acc[r][2] = vfmaq_f32(acc[r][2], wv[2],
                                                  vld1q_f32(ap + 8));
                            acc[r][3] = vfmaq_f32(acc[r][3], wv[3],
                                                  vld1q_f32(ap + 12));
                        }
                    }
                    for (size_t r = 0; r < 4; r++) {
                        const float *ap = scratch + (m0 + r) * K + lo;
                        float dot = apus_fp8_reduce4_neon(acc[r][0], acc[r][1],
                                                          acc[r][2], acc[r][3]);
                        for (size_t t = i; t < len; t++)
                            dot += ap[t] * apus_e4m3_dequant_f32(wp[lo + t]);
                        float sc = as[(m0 + r) * nb + kb] * sc_b;
                        total[r] += dot * sc;
                    }
                } else {
                    for (size_t r = 0; r < mc; r++) {
                        const float *ap = scratch + (m0 + r) * K + lo;
                        float dot = apus_fp8_dot128_neon(ap, wp + lo, len);
                        float sc = as[(m0 + r) * nb + kb] * sc_b;
                        total[r] += dot * sc;
                    }
                }
            }
            for (size_t r = 0; r < mc; r++)
                out[(m0 + r) * O + o] = total[r];
        }
    }
}

void apus_fp8_gemm_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_fp8_act_dequant_all(acodes + m * K, scratch + m * K, K);
    apus_fp8_gemm_rows_neon(w, ws, as, scratch, out, M, O, K, 0, O);
}

#endif /* __ARM_NEON */

/* -------------------------------------------------------------------------*/
#if APUS_X86

/* M12a-2: rows [o0, o1) of the AVX2 GEMM — BITWISE identical to
 * apus_fp8_gemm_scalar_rows (the normative scalar contract, c/x86.h):
 * exact E4M3 expansion, per-element products computed 8-wide (same single
 * rounding as the scalar mul, staged in place), and every sum in the
 * scalar sequential order — per 128-block the dot adds its elements
 * strictly in order, blocks fold into `total` in kb order with the scale
 * product sc = as*ws rounded first. Eight full blocks are processed per
 * chunk with eight independent accumulator chains to hide the FP-add
 * latency; each chain IS the scalar block dot, so no reassociation. The
 * trailing blocks (< 8 left, or the partial last block) run the same
 * order with one chain per block. Per-output values are M- and
 * thread-count-independent by construction. */
APUS_TGT_AVX2
static void apus_fp8_rows_avx2(const uint8_t *w, const uint8_t *ws,
                               const float *as, const float *scratch,
                               float *out, size_t M, size_t O, size_t K,
                               size_t o0, size_t o1) {
    const size_t nb = apus_fp8_blocks(K);
    const size_t nfull = K / APUS_FP8_GROUP;   /* full 128-blocks */
    float wdeq[8 * APUS_FP8_GROUP];            /* 8 staged blocks */
    for (size_t m = 0; m < M; m++) {
        const float *adeq = scratch + m * K;
        const float *am = as + m * nb;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = w + o * K;
            const uint8_t *sp = ws + (o / APUS_FP8_GROUP) * nb;
            float total = 0.0f;
            size_t kb = 0;
            for (; kb + 8 <= nfull; kb += 8) {
                for (int b = 0; b < 8; b++) {
                    float *ob = wdeq + b * APUS_FP8_GROUP;
                    const uint8_t *pb = wp + (kb + (size_t)b) * APUS_FP8_GROUP;
                    for (int i = 0; i < 8; i++)
                        apus_e4m3_expand16_x86(pb + 16 * i, ob + 16 * i);
                    /* in-place exact products with this block's acts */
                    const float *ab = adeq + (kb + (size_t)b) * APUS_FP8_GROUP;
                    for (int i = 0; i < (int)APUS_FP8_GROUP; i += 8)
                        _mm256_storeu_ps(ob + i, _mm256_mul_ps(
                            _mm256_loadu_ps(ob + i), _mm256_loadu_ps(ab + i)));
                }
                /* eight named chains (array accumulators spill — c/x86.h
                 * dot4 note); adds stay in strict per-block order */
                const float *w0 = wdeq,               *w1 = wdeq + 128;
                const float *w2 = wdeq + 2 * 128,     *w3 = wdeq + 3 * 128;
                const float *w4 = wdeq + 4 * 128,     *w5 = wdeq + 5 * 128;
                const float *w6 = wdeq + 6 * 128,     *w7 = wdeq + 7 * 128;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
                for (int i = 0; i < (int)APUS_FP8_GROUP; i++) {
                    d0 += w0[i]; d1 += w1[i]; d2 += w2[i]; d3 += w3[i];
                    d4 += w4[i]; d5 += w5[i]; d6 += w6[i]; d7 += w7[i];
                }
                float sc0 = am[kb + 0] * apus_ue8m0_f32(sp[kb + 0]);
                float sc1 = am[kb + 1] * apus_ue8m0_f32(sp[kb + 1]);
                float sc2 = am[kb + 2] * apus_ue8m0_f32(sp[kb + 2]);
                float sc3 = am[kb + 3] * apus_ue8m0_f32(sp[kb + 3]);
                float sc4 = am[kb + 4] * apus_ue8m0_f32(sp[kb + 4]);
                float sc5 = am[kb + 5] * apus_ue8m0_f32(sp[kb + 5]);
                float sc6 = am[kb + 6] * apus_ue8m0_f32(sp[kb + 6]);
                float sc7 = am[kb + 7] * apus_ue8m0_f32(sp[kb + 7]);
                total += d0 * sc0; total += d1 * sc1;
                total += d2 * sc2; total += d3 * sc3;
                total += d4 * sc4; total += d5 * sc5;
                total += d6 * sc6; total += d7 * sc7;
            }
            for (; kb < nb; kb++) {   /* < 8 left, or the partial block */
                size_t lo = kb * APUS_FP8_GROUP;
                size_t hi = lo + APUS_FP8_GROUP;
                if (hi > K) hi = K;
                size_t len = hi - lo, i = 0;
                for (; i + 16 <= len; i += 16)
                    apus_e4m3_expand16_x86(wp + lo + i, wdeq + i);
                for (; i < len; i++)
                    wdeq[i] = apus_e4m3_dequant_f32(wp[lo + i]);
                float dot = 0.0f;
                for (i = 0; i < len; i++)
                    dot += adeq[lo + i] * wdeq[i];
                float sc = am[kb] * apus_ue8m0_f32(sp[kb]);
                total += dot * sc;
            }
            out[m * O + o] = total;
        }
    }
}

void apus_fp8_gemv_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K) {
    apus_fp8_act_dequant_all(acodes, scratch, K);
    apus_fp8_rows_avx2(w, ws, as, scratch, out, 1, O, K, 0, O);
}

void apus_fp8_gemm_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_fp8_act_dequant_all(acodes + m * K, scratch + m * K, K);
    apus_fp8_rows_avx2(w, ws, as, scratch, out, M, O, K, 0, O);
}

#endif /* APUS_X86 */

/* --- M6c threaded variant (c/pool.h row partition) ------------------------*/

typedef struct {
    const uint8_t *w, *ws, *acodes;
    const float *as;
    float *scratch, *out;
    size_t M, O, K;
} ApusFp8GemmJob;

#ifdef __ARM_NEON
/* Rows [o0, o1) of the NEON GEMM — delegates to the shared row body
 * (apus_fp8_gemm_rows_neon), so per-output values match apus_fp8_gemm_neon
 * bitwise for any row partition (any thread count). */
static void apus_fp8_gemm_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp8GemmJob *j = vjob;
    apus_fp8_gemm_rows_neon(j->w, j->ws, j->as, j->scratch, j->out,
                            j->M, j->O, j->K, o0, o1);
}
#else
#if APUS_X86
/* Rows [o0, o1) of the AVX2 GEMM — delegates to the shared row body
 * (apus_fp8_rows_avx2), bitwise identical to apus_fp8_gemm_scalar_rows for
 * any row partition (c/x86.h contract). */
static void apus_fp8_gemm_avx2_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp8GemmJob *j = vjob;
    apus_fp8_rows_avx2(j->w, j->ws, j->as, j->scratch, j->out,
                       j->M, j->O, j->K, o0, o1);
}
#endif
static void apus_fp8_gemm_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp8GemmJob *j = vjob;
    size_t M = j->M, O = j->O, K = j->K;
    size_t nb = apus_fp8_blocks(K);
    for (size_t m = 0; m < M; m++) {
        const float *adeq = j->scratch + m * K;
        const float *am = j->as + m * nb;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = j->w + o * K;
            const uint8_t *sp = j->ws + (o / APUS_FP8_GROUP) * nb;
            float total = 0.0f;
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * APUS_FP8_GROUP;
                size_t hi = lo + APUS_FP8_GROUP;
                if (hi > K) hi = K;
                float dot = 0.0f;
                for (size_t i = lo; i < hi; i++)
                    dot += adeq[i] * apus_e4m3_dequant_f32(wp[i]);
                float sc = am[kb] * apus_ue8m0_f32(sp[kb]);
                total += dot * sc;
            }
            j->out[m * O + o] = total;
        }
    }
}
#endif

void apus_fp8_gemm_mt(const uint8_t *w, const uint8_t *ws,
                      const uint8_t *acodes, const float *as,
                      float *scratch, float *out,
                      size_t M, size_t O, size_t K) {
#if APUS_BLAS
    /* M9b: large-batch prefill GEMMs go to Accelerate (AMX); single vecLib
     * thread, deterministic and APUS_THREADS-independent. Smaller M stays
     * on the M9a NEON kernels (see c/blas.h for the cutoff rationale). */
    if (M >= APUS_BLAS_M_MIN && apus_blas_available()) {
        apus_fp8_gemm_blas(w, ws, acodes, as, scratch, out, M, O, K);
        return;
    }
#endif
    ApusFp8GemmJob job = { w, ws, acodes, as, scratch, out, M, O, K };
    for (size_t m = 0; m < M; m++)
        apus_fp8_act_dequant_all(acodes + m * K, scratch + m * K, K);
#ifdef __ARM_NEON
    apus_pool_run(O, apus_fp8_gemm_neon_rows, &job);
#else
#if APUS_X86
    /* M12a-2: AVX2 rows when the CPU supports them — bitwise identical to
     * the scalar rows, so dispatch is numerics-neutral. */
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_pool_run(O, apus_fp8_gemm_avx2_rows, &job);
    } else
#endif
    apus_pool_run(O, apus_fp8_gemm_scalar_rows, &job);
#endif
}

#endif /* APUS_FP8_IMPLEMENTATION */
#endif /* APUS_FP8_H */
