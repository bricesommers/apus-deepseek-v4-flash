/*
 * c/fp4.h — MXFP4 (E2M1 weights + UE8M0 per-32 scales) dequant and
 * GEMV/GEMM kernels for DeepSeek-V4-Flash experts. C11, libc + arm_neon.h only.
 *
 * Numerics contract (normative reference: reference/inference/kernel.py,
 * model.py:109-115; dequant storage semantics pinned by tests/m1/test_3_dequant.py):
 *
 *   Storage:  weight row-major [O, K/2] packed bytes, 2 FP4-E2M1 per byte along
 *             K, LOW nibble = even K index. Scales [O, K/32] UE8M0 bytes,
 *             one per 32 elements along K; scale value = 2^(byte-127).
 *   E2M1 LUT: [0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6] (bit3 sign).
 *
 *   NUMERICS-NORMATIVE PATH (what DeepSeek's reference actually computes for
 *   every FP4 linear, model.py `linear()`): the FP32/BF16 activation is first
 *   quantized to FP8-E4M3 with a per-128-along-K power-of-2 scale
 *   (act_quant with scale_fmt="ue8m0", generate.py defaults):
 *       amax_b = max(|x_b|) floored at 1e-4
 *       s_b    = 2^ceil(log2(amax_b / 448))        (bit-trick, see below)
 *       code   = fp8_e4m3(clamp(x / s_b, +-448))   (round-to-nearest-even)
 *   Then fp4_gemm: FP4 codes are upcast FP4->FP32->FP8 (lossless), an FP8xFP8
 *   GEMM with FP32 accumulate runs per 32-wide K block, and each block's dot
 *   is folded into the FP32 total with the scale product:
 *       total += (dot32(codes_a, codes_b) * scale_a[kb/4]) * scale_b[kb]
 *   We reproduce this exactly in FP32 arithmetic: FP8 codes and FP4 LUT values
 *   are exactly representable in FP32, so the per-block dot in FP32 is the
 *   same computation the tensor core does (up to summation order, which the
 *   reference itself leaves unspecified). No requantization anywhere.
 *
 *   DIAGNOSTIC PATH (apus_fp4_gemv_f32_*): FP32 activation dotted against the
 *   fully dequantized FP32 weight. This is NOT what the reference model does
 *   (it skips the FP8 activation quantization); kept for comparison and as a
 *   fallback, never for normative output.
 *
 *   M4 integration notes:
 *   - reference act_quant input dtype is BF16; to be bit-faithful, round
 *     activations to BF16 (apus_bf16_round) before apus_fp4_act_quant_*.
 *   - reference asserts K % 128 == 0 for act quant. This kernel also accepts
 *     K % 32 == 0 with a partial trailing 128-block (amax over the remainder);
 *     such shapes cannot be validated against the reference.
 *   - UE8M0 byte 255 denotes 2^128, which overflows FP32 -> inf (matching the
 *     numpy/pinned M1 semantics, np.exp2(128).astype(f32) == inf). Real
 *     checkpoint scales are far from this; byte 0 (2^-127, FP32 subnormal)
 *     is handled exactly.
 *
 * Usage: #define APUS_FP4_IMPLEMENTATION in exactly one TU. Scalar paths are
 * always compiled; NEON paths are compiled when __ARM_NEON is defined and are
 * bitwise/ulp-checked against the scalar paths in tests/m3.
 */
#ifndef APUS_FP4_H
#define APUS_FP4_H

#include <stddef.h>
#include <stdint.h>

#include "pool.h"
#include "x86.h"   /* M12a-2: AVX2 runtime dispatch (no-op off x86-64) */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "blas.h"   /* M9b: BLAS dispatch decls (self-contained; impl below) */

#define APUS_FP4_GROUP      32u   /* weight scale group along K */
#define APUS_FP4_ACT_GROUP  128u  /* activation scale group along K */

extern const float apus_fp4_lut[16];

/* --- scalar helpers ------------------------------------------------------*/

/* UE8M0 byte -> FP32 scale 2^(b-127). Exact for all b: b==0 gives the FP32
 * subnormal 2^-127; b==255 gives +inf (2^128 overflows FP32), matching the
 * M1-pinned numpy semantics. */
float apus_ue8m0_f32(uint8_t b);

/* Quantize one FP32 value to FP8-E4M3 (round-to-nearest-even, clamped to the
 * finite range +-448). Input must be finite. */
uint8_t apus_e4m3_quant_f32(float x);

/* FP8-E4M3 code -> FP32 (exact). */
float apus_e4m3_dequant_f32(uint8_t c);

/* Round FP32 to BF16 (round-to-nearest-even) and back to FP32. Provided so
 * callers can mirror the reference's BF16 act_quant input. */
float apus_bf16_round(float x);

/* BF16 bit pack/unpack (M6c BF16 weight storage). apus_bf16_bits(x) ==
 * bits(apus_bf16_round(x)) >> 16; apus_bf16_f32 widening is exact. */
uint16_t apus_bf16_bits(float x);
float apus_bf16_f32(uint16_t b);

/* Number of per-128 activation scale blocks for K (ceil(K/128)). */
size_t apus_fp4_act_blocks(size_t K);

/* Per-128-along-K FP8-E4M3 activation quantization, ue8m0 (power-of-2) scale
 * variant — the normative act_quant(scale_fmt="ue8m0") port.
 * x: K floats; codes: K bytes out; scales: apus_fp4_act_blocks(K) floats out.
 * Requires K % 32 == 0. */
void apus_fp4_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales);

/* Dequantize one packed row: packed[K/2], scales[K/32] -> out[K].
 * Exact: every product LUT*n 2^e is exactly representable in FP32. */
void apus_fp4_dequant_row_scalar(const uint8_t *packed, const uint8_t *scales,
                                 float *out, size_t K);

/* NORMATIVE GEMV (decode, M=1): out[o] = sum_kb (dot32 * sa[kb/4]) * sb[o,kb]
 * w: [O, K/2] packed, ws: [O, K/32] UE8M0, acodes: [K] FP8 codes,
 * as: [apus_fp4_act_blocks(K)] FP32 act scales,
 * scratch: K floats of work area (act dequant; the NEON path stores the
 * exact FP16 acts in the same bytes), out: [O]. */
void apus_fp4_gemv_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out, size_t O, size_t K);

/* NORMATIVE GEMM (prefill): out[M,O] row-major. acodes: [M,K],
 * as: [M, apus_fp4_act_blocks(K)], scratch: M*K floats. */
void apus_fp4_gemm_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out,
                          size_t M, size_t O, size_t K);

/* DIAGNOSTIC (non-normative): FP32 activation x[K] dotted with the fully
 * dequantized weight, single FP32 accumulator per output. */
void apus_fp4_gemv_f32_scalar(const uint8_t *w, const uint8_t *ws,
                              const float *x, float *out, size_t O, size_t K);

#ifdef __ARM_NEON
void apus_fp4_dequant_row_neon(const uint8_t *packed, const uint8_t *scales,
                               float *out, size_t K);
void apus_fp4_gemv_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K);
void apus_fp4_gemm_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K);
#endif

#if APUS_X86
/* M12a-2: AVX2 kernels — BITWISE identical to the scalar kernels (c/x86.h
 * contract: exact FP4-code expand, staged exact products, scalar
 * sequential summation order, (dot*sa)*sb in two rounded steps). */
void apus_fp4_gemv_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K);
void apus_fp4_gemm_avx2(const uint8_t *w, const uint8_t *ws,
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
void apus_fp4_gemm_mt(const uint8_t *w, const uint8_t *ws,
                      const uint8_t *acodes, const float *as,
                      float *scratch, float *out,
                      size_t M, size_t O, size_t K);

/* M9e: GROUPED GEMM — several independent same-(O,K) GEMMs in ONE pool
 * dispatch (routed-expert batching, c/moe.h). Entry i computes
 * ents[i].out[m, o] for m in [0, ents[i].M), reading activation row
 * (ents[i].m0 + m) of the SHARED acodes/as/scratch buffers (row space =
 * the concatenation of all entries' rows; scratch holds sum(M) * K
 * floats). The dispatch unit space is n_ent * O (entry-major), partitioned
 * contiguously over the pool lanes; every output row is still computed
 * entirely by one lane with the EXACT apus_fp4_gemm_rows_neon per-row
 * accumulation order, so results are bitwise identical to per-entry
 * apus_fp4_gemm_mt calls for any thread count (APUS_THREADS=1 included).
 * Scheduling only: no numerics change. An entry with M >= APUS_BLAS_M_MIN
 * falls back to its own apus_fp4_gemm_mt call (the M9b BLAS dispatch),
 * exactly as a standalone call would. */
typedef struct {
    const uint8_t *w, *ws;   /* this entry's weights [O, K/2] / [O, K/32] */
    float *out;              /* this entry's output [M, O] row-major */
    size_t m0, M;            /* shared act row range [m0, m0 + M) */
} ApusFp4GemmEnt;

void apus_fp4_gemm_mt_grouped(const ApusFp4GemmEnt *ents, size_t n_ent,
                              const uint8_t *acodes, const float *as,
                              float *scratch, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_FP4_IMPLEMENTATION

#include <math.h>
#include <string.h>

const float apus_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

float apus_ue8m0_f32(uint8_t b) {
    /* Bit construction avoids out-of-range double->float conversion (which
     * would be UB at b==255). b==0 -> 2^-127 (subnormal); b in 1..254 ->
     * 2^(b-127); b==255 -> +inf, matching np.exp2(128).astype(f32). */
    uint32_t bits = (b == 0) ? 0x00400000u : ((uint32_t)b << 23);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint32_t apus_rne_shift(uint32_t m, int sh) {
    /* m >> sh with round-to-nearest-even. 1 <= sh <= 24. */
    uint32_t half = 1u << (sh - 1);
    uint32_t mask = (1u << sh) - 1u;
    uint32_t r = m & mask;
    uint32_t v = m >> sh;
    if (r > half || (r == half && (v & 1u))) v++;
    return v;
}

uint8_t apus_e4m3_quant_f32(float x) {
    if (x > 448.0f) x = 448.0f;
    else if (x < -448.0f) x = -448.0f;
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t sign = (u >> 31) << 7;
    uint32_t a = u & 0x7fffffffu;
    if (a == 0) return (uint8_t)sign;
    int e = (int)(a >> 23) - 127;
    uint32_t m23 = (a & 0x007fffffu) | 0x00800000u;
    uint32_t code;
    if (e >= -6) {
        /* normal: keep 3 mantissa bits, RNE; carry bumps the exponent */
        uint32_t v = apus_rne_shift(m23, 20); /* 8..16 */
        if (v == 16) { v = 8; e++; }
        if (e > 8) code = 0x7Eu;              /* saturate to 448 (finite max) */
        else code = ((uint32_t)(e + 7) << 3) | (v & 7u);
    } else if (e >= -10) {
        /* subnormal grid, quantum 2^-9; code 8 is continuous with min normal */
        code = apus_rne_shift(m23, 14 - e);   /* 0..8 */
    } else {
        code = 0;
    }
    return (uint8_t)(sign | code);
}

float apus_e4m3_dequant_f32(uint8_t c) {
    int e = (c >> 3) & 0xF, m = c & 7;
    float v = (e != 0) ? ldexpf((float)(8 + m), e - 10)   /* (1+m/8)*2^(e-7) */
                       : ldexpf((float)m, -9);            /* m*2^-9 subnormal */
    return (c & 0x80) ? -v : v;
}

float apus_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x; /* NaN: leave alone */
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

/* M6c: BF16 bit helpers (wo_a storage is BF16; widening is EXACT, so the
 * widened values are bitwise identical to the old f32-rounded storage). */
uint16_t apus_bf16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16); /* NaN */
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float apus_bf16_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

size_t apus_fp4_act_blocks(size_t K) {
    return (K + APUS_FP4_ACT_GROUP - 1) / APUS_FP4_ACT_GROUP;
}

void apus_fp4_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales) {
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t b = 0; b < nab; b++) {
        size_t lo = b * APUS_FP4_ACT_GROUP;
        size_t hi = lo + APUS_FP4_ACT_GROUP;
        if (hi > K) hi = K;
        float amax = 0.0f;
        for (size_t i = lo; i < hi; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        /* scale = 2^ceil(log2(amax * (1/448))) via the reference bit trick
         * (kernel.py fast_log2_ceil/fast_pow2): exponent of the FP32 product
         * plus 1 when any mantissa bit is set. amax >= 1e-4 keeps the product
         * normal, so no subnormal case to handle. */
        float p = amax * (1.0f / 448.0f);
        uint32_t pb;
        memcpy(&pb, &p, 4);
        int e = (int)((pb >> 23) & 0xFF) - 127 + ((pb & 0x007fffffu) != 0);
        uint32_t sbits = (uint32_t)(e + 127) << 23;
        float s;
        memcpy(&s, &sbits, 4);
        scales[b] = s;
        for (size_t i = lo; i < hi; i++)
            codes[i] = apus_e4m3_quant_f32(x[i] / s);
    }
}

void apus_fp4_dequant_row_scalar(const uint8_t *packed, const uint8_t *scales,
                                 float *out, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    for (size_t b = 0; b < nb; b++) {
        float s = apus_ue8m0_f32(scales[b]);
        const uint8_t *p = packed + b * (APUS_FP4_GROUP / 2);
        float *o = out + b * APUS_FP4_GROUP;
        for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
            o[2 * i]     = apus_fp4_lut[p[i] & 0x0F] * s;
            o[2 * i + 1] = apus_fp4_lut[p[i] >> 4] * s;
        }
    }
}

static void apus_act_dequant_all(const uint8_t *acodes, float *adeq, size_t K) {
    for (size_t i = 0; i < K; i++)
        adeq[i] = apus_e4m3_dequant_f32(acodes[i]);
}

void apus_fp4_gemv_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out, size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    float *adeq = scratch;
    apus_act_dequant_all(acodes, adeq, K);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * (K / 2);
        const uint8_t *sp = ws + o * nb;
        float total = 0.0f;
        for (size_t kb = 0; kb < nb; kb++) {
            float dot = 0.0f;
            const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
            const float *a = adeq + kb * APUS_FP4_GROUP;
            for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                dot += a[2 * i]     * apus_fp4_lut[p[i] & 0x0F];
                dot += a[2 * i + 1] * apus_fp4_lut[p[i] >> 4];
            }
            /* reference order: (dot * scale_a) * scale_b, FP32 each step */
            total += (dot * as[kb / 4]) * apus_ue8m0_f32(sp[kb]);
        }
        out[o] = total;
    }
}

void apus_fp4_gemm_scalar(const uint8_t *w, const uint8_t *ws,
                          const uint8_t *acodes, const float *as,
                          float *scratch, float *out,
                          size_t M, size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t m = 0; m < M; m++)
        apus_act_dequant_all(acodes + m * K, scratch + m * K, K);
    for (size_t m = 0; m < M; m++) {
        const float *adeq = scratch + m * K;
        const float *am = as + m * nab;
        for (size_t o = 0; o < O; o++) {
            const uint8_t *wp = w + o * (K / 2);
            const uint8_t *sp = ws + o * nb;
            float total = 0.0f;
            for (size_t kb = 0; kb < nb; kb++) {
                float dot = 0.0f;
                const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
                const float *a = adeq + kb * APUS_FP4_GROUP;
                for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                    dot += a[2 * i]     * apus_fp4_lut[p[i] & 0x0F];
                    dot += a[2 * i + 1] * apus_fp4_lut[p[i] >> 4];
                }
                total += (dot * am[kb / 4]) * apus_ue8m0_f32(sp[kb]);
            }
            out[m * O + o] = total;
        }
    }
}

void apus_fp4_gemv_f32_scalar(const uint8_t *w, const uint8_t *ws,
                              const float *x, float *out, size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * (K / 2);
        const uint8_t *sp = ws + o * nb;
        float acc = 0.0f;
        for (size_t kb = 0; kb < nb; kb++) {
            float s = apus_ue8m0_f32(sp[kb]);
            const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
            const float *a = x + kb * APUS_FP4_GROUP;
            for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                acc += a[2 * i]     * (apus_fp4_lut[p[i] & 0x0F] * s);
                acc += a[2 * i + 1] * (apus_fp4_lut[p[i] >> 4] * s);
            }
        }
        out[o] = acc;
    }
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

/* FP4 codes as int8 = LUT value * 2 (all exactly representable). */
static const int8_t apus_fp4_codes_s8[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* Expand 16 packed bytes -> 32 FP4 code values (LUT*2) as 8 float32x4.
 * Even K lanes from low nibbles, odd from high nibbles (vzip restores order). */
static inline void apus_fp4_expand32_neon(const uint8_t *p, float32x4_t v[8]) {
    int8x16_t lut = vld1q_s8(apus_fp4_codes_s8);
    uint8x16_t bytes = vld1q_u8(p);
    int8x16_t lo = vqtbl1q_s8(lut, vreinterpretq_s8_u8(
                       vandq_u8(bytes, vdupq_n_u8(0x0F))));
    int8x16_t hi = vqtbl1q_s8(lut, vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)));
    int8x16_t q0 = vzip1q_s8(lo, hi);   /* elements  0..15 */
    int8x16_t q1 = vzip2q_s8(lo, hi);   /* elements 16..31 */
    int16x8_t s0 = vmovl_s8(vget_low_s8(q0));
    int16x8_t s1 = vmovl_s8(vget_high_s8(q0));
    int16x8_t s2 = vmovl_s8(vget_low_s8(q1));
    int16x8_t s3 = vmovl_s8(vget_high_s8(q1));
    v[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0)));
    v[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0)));
    v[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1)));
    v[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1)));
    v[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2)));
    v[5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2)));
    v[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3)));
    v[7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3)));
}

void apus_fp4_dequant_row_neon(const uint8_t *packed, const uint8_t *scales,
                               float *out, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    for (size_t b = 0; b < nb; b++) {
        /* codes are LUT*2; folding *0.5 into the scale is exact (power of 2,
         * incl. the byte-0 subnormal: 0.5*2^-127 = 2^-128, still exact) */
        float32x4_t hs = vdupq_n_f32(0.5f * apus_ue8m0_f32(scales[b]));
        float32x4_t v[8];
        apus_fp4_expand32_neon(packed + b * 16, v);
        float *o = out + b * APUS_FP4_GROUP;
        for (int i = 0; i < 8; i++)
            vst1q_f32(o + 4 * i, vmulq_f32(v[i], hs));
    }
}

/* M9a: canonical FP32 summation order for the NEON 32-block dot (ILP
 * reorder). Four independent vector accumulators per row: acc[j] lane l
 * accumulates elements 4j+l and 16+4j+l of the block (two FMAs, in that
 * order); the accumulators combine lane-wise as ((a0+a1)+(a2+a3)) BEFORE the
 * scale steps. The scale application is unchanged from the pre-M9a kernel
 * ((dot*sa)*sb in two rounded vector steps, the exact 0.5 folded into sa),
 * as is the per-block fold into `total` lane-wise. The final horizontal sum
 * is the fixed order ((t0+t1)+(t2+t3)) below. Only the summation order
 * within dots changed (the accepted scalar-vs-NEON tolerance class). */
static inline float32x4_t apus_fp4_dot32_neon(const float *ap,
                                              const float32x4_t wv[8]) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    a0 = vfmaq_f32(a0, wv[0], vld1q_f32(ap));
    a1 = vfmaq_f32(a1, wv[1], vld1q_f32(ap + 4));
    a2 = vfmaq_f32(a2, wv[2], vld1q_f32(ap + 8));
    a3 = vfmaq_f32(a3, wv[3], vld1q_f32(ap + 12));
    a0 = vfmaq_f32(a0, wv[4], vld1q_f32(ap + 16));
    a1 = vfmaq_f32(a1, wv[5], vld1q_f32(ap + 20));
    a2 = vfmaq_f32(a2, wv[6], vld1q_f32(ap + 24));
    a3 = vfmaq_f32(a3, wv[7], vld1q_f32(ap + 28));
    return vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
}

/* M9a FMLAL path: FP4 codes*2 as FP16 bit patterns (all exactly
 * representable; every pattern's low byte is 0x00, so a single 16-entry
 * table of high bytes suffices). */
static const uint8_t apus_fp4_codes_f16_hi[16] = {
    0x00, 0x3C, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A,
    0x80, 0xBC, 0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA
};

/* Expand 16 packed bytes -> 32 FP4 code values (LUT*2) as 4 float16x8,
 * exact. Same nibble/zip structure as apus_fp4_expand32_neon. */
static inline void apus_fp4_expand32_f16_neon(const uint8_t *p,
                                              float16x8_t k[4]) {
    uint8x16_t tab = vld1q_u8(apus_fp4_codes_f16_hi);
    uint8x16_t bytes = vld1q_u8(p);
    uint8x16_t lo = vqtbl1q_u8(tab, vandq_u8(bytes, vdupq_n_u8(0x0F)));
    uint8x16_t hi = vqtbl1q_u8(tab, vshrq_n_u8(bytes, 4));
    uint8x16_t q0 = vzip1q_u8(lo, hi);   /* f16 high bytes, elements 0..15 */
    uint8x16_t q1 = vzip2q_u8(lo, hi);   /* f16 high bytes, elements 16..31 */
    k[0] = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(q0), 8));
    k[1] = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(q0), 8));
    k[2] = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(q1), 8));
    k[3] = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(q1), 8));
}

/* FP32 dot of one 32-block via FMLAL (fused FP16xFP16 -> FP32 long FMA,
 * Armv8.2 FP16FML — present on all Apple Silicon). BITWISE identical to
 * apus_fp4_dot32_neon: code*2 and the E4M3 act values are exact in FP16
 * (acts dequantize to FP16 normals — no subnormals anywhere in this path),
 * the products are exact (<= 9 significant bits), and each FMLAL adds its
 * exact product into the FP32 lane with the same single rounding as
 * vfmaq_f32. Lane/accumulator assignment mirrors the anchor: cj lane l gets
 * elements 4j+l then 16+4j+l; combined ((c0+c1)+(c2+c3)). */
static inline float32x4_t apus_fp4_dot32_f16_neon(const float16_t *ap,
                                                  const float16x8_t k[4]) {
    float16x8_t a0 = vld1q_f16(ap);
    float16x8_t a1 = vld1q_f16(ap + 8);
    float16x8_t a2 = vld1q_f16(ap + 16);
    float16x8_t a3 = vld1q_f16(ap + 24);
    float32x4_t c0 = vfmlalq_low_f16(vdupq_n_f32(0.0f), k[0], a0);
    float32x4_t c1 = vfmlalq_high_f16(vdupq_n_f32(0.0f), k[0], a0);
    float32x4_t c2 = vfmlalq_low_f16(vdupq_n_f32(0.0f), k[1], a1);
    float32x4_t c3 = vfmlalq_high_f16(vdupq_n_f32(0.0f), k[1], a1);
    c0 = vfmlalq_low_f16(c0, k[2], a2);
    c1 = vfmlalq_high_f16(c1, k[2], a2);
    c2 = vfmlalq_low_f16(c2, k[3], a3);
    c3 = vfmlalq_high_f16(c3, k[3], a3);
    return vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3));
}

/* Dequantize activation codes straight to FP16 (exact — see above), stored
 * in the float scratch re-interpreted as f16 (row stride K halves; only 2K
 * of each row's 4K scratch bytes are used). */
static void apus_fp4_act_dequant_f16_all(const uint8_t *acodes,
                                         float16_t *adeq, size_t K) {
    for (size_t i = 0; i < K; i++)
        adeq[i] = (float16_t)apus_e4m3_dequant_f32(acodes[i]);
}

/* Fixed horizontal sum ((t0+t1)+(t2+t3)) — replaces vaddvq_f32 so the order
 * is part of the kernel contract, not the compiler's lowering. */
static inline float apus_fp4_hsum4_neon(float32x4_t t) {
    return (vgetq_lane_f32(t, 0) + vgetq_lane_f32(t, 1))
         + (vgetq_lane_f32(t, 2) + vgetq_lane_f32(t, 3));
}

void apus_fp4_gemv_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K) {
    size_t nb = K / APUS_FP4_GROUP;
    float16_t *adeq = (float16_t *)scratch;
    apus_fp4_act_dequant_f16_all(acodes, adeq, K);
    for (size_t o = 0; o < O; o++) {
        const uint8_t *wp = w + o * (K / 2);
        const uint8_t *sp = ws + o * nb;
        const float16_t *ap = adeq;
        float32x4_t total = vdupq_n_f32(0.0f);
        for (size_t kb = 0; kb < nb; kb++) {
            float16x8_t k[4];
            apus_fp4_expand32_f16_neon(wp + kb * 16, k);
            float32x4_t acc = apus_fp4_dot32_f16_neon(ap, k);
            /* codes are LUT*2: fold the exact 0.5 into scale_a.
             * (dot*sa)*sb in two rounded steps like the reference — plain
             * vmulq, no FMA, to keep the two roundings. */
            float32x4_t t = vmulq_f32(acc, vdupq_n_f32(0.5f * as[kb / 4]));
            total = vaddq_f32(total,
                              vmulq_f32(t, vdupq_n_f32(apus_ue8m0_f32(sp[kb]))));
            ap += APUS_FP4_GROUP;
        }
        out[o] = apus_fp4_hsum4_neon(total);
    }
}

/* Rows [o0, o1) of the NEON GEMM — the one and only row body, shared by
 * apus_fp4_gemm_neon (full range) and the M6c threaded variant, so the mt
 * kernel is bitwise identical to the single-thread one by construction.
 * Per row the per-32-block accumulation is EXACTLY the apus_fp4_dot32_neon
 * M9a order (the FMLAL path is bitwise equal to it), so a row's value is
 * independent of M. `scratch16` is the float scratch re-interpreted as f16
 * act storage (see apus_fp4_act_dequant_f16_all). */
static inline void apus_fp4_gemm_rows_neon(const uint8_t *w, const uint8_t *ws,
                                           const float *as,
                                           const float *scratch, float *out,
                                           size_t M, size_t O, size_t K,
                                           size_t o0, size_t o1) {
    const float16_t *acts16 = (const float16_t *)scratch;
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        for (size_t o = o0; o < o1; o++) {
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
}

void apus_fp4_gemm_neon(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    float16_t *acts16 = (float16_t *)scratch;
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_dequant_f16_all(acodes + m * K, acts16 + m * K, K);
    apus_fp4_gemm_rows_neon(w, ws, as, scratch, out, M, O, K, 0, O);
}

#endif /* __ARM_NEON */

/* -------------------------------------------------------------------------*/
#if APUS_X86

/* FP4 codes as int8 = LUT value * 2 (all exactly representable) — the x86
 * copy of the NEON path's table. */
static const int8_t apus_fp4_codes_s8_x86[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* M12a-2: expand one packed 32-group (16 bytes) -> 32 FP32 LUT values,
 * EXACT (same construction as apus_fp4_expand32_neon: nibble -> int8
 * LUT*2 via a 16-entry shuffle table, low/high nibbles interleaved back
 * to K order, widened, and the exact 0.5 folded in). */
APUS_TGT_AVX2
static inline void apus_fp4_expand32_x86(const uint8_t *p, float *out) {
    const __m128i lut = _mm_loadu_si128((const __m128i *)apus_fp4_codes_s8_x86);
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(b, _mm_set1_epi8(0x0F)));
    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(b, 4),
                                                     _mm_set1_epi8(0x0F)));
    __m128i q0 = _mm_unpacklo_epi8(lo, hi);   /* elements  0..15 */
    __m128i q1 = _mm_unpackhi_epi8(lo, hi);   /* elements 16..31 */
    const __m256 half = _mm256_set1_ps(0.5f);
    _mm256_storeu_ps(out, _mm256_mul_ps(_mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(q0)), half));
    _mm256_storeu_ps(out + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), half));
    _mm256_storeu_ps(out + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(q1)), half));
    _mm256_storeu_ps(out + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), half));
}

/* M12a-2: rows [o0, o1) of the AVX2 GEMM — BITWISE identical to
 * apus_fp4_gemm_scalar_rows (c/x86.h contract): exact code expansion,
 * per-element products computed 8-wide (same single rounding as the
 * scalar mul, staged in place), per-32-group dots summed strictly in
 * order, folds (dot*sa)*sb in two rounded steps in group order. Eight
 * groups per chunk with eight independent accumulator chains (each chain
 * IS the scalar group dot); trailing groups run one chain per group.
 * Per-output values are M- and thread-count-independent. */
APUS_TGT_AVX2
static void apus_fp4_rows_avx2(const uint8_t *w, const uint8_t *ws,
                               const float *as, const float *scratch,
                               float *out, size_t M, size_t O, size_t K,
                               size_t o0, size_t o1) {
    const size_t nb = K / APUS_FP4_GROUP;
    const size_t nab = apus_fp4_act_blocks(K);
    float wdeq[8 * APUS_FP4_GROUP];            /* 8 staged groups */
    for (size_t m = 0; m < M; m++) {
        const float *adeq = scratch + m * K;
        const float *am = as + m * nab;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = w + o * (K / 2);
            const uint8_t *sp = ws + o * nb;
            float total = 0.0f;
            size_t kb = 0;
            for (; kb + 8 <= nb; kb += 8) {
                for (int b = 0; b < 8; b++) {
                    float *ob = wdeq + b * APUS_FP4_GROUP;
                    apus_fp4_expand32_x86(wp + (kb + (size_t)b) * 16, ob);
                    const float *ab = adeq + (kb + (size_t)b) * APUS_FP4_GROUP;
                    for (int i = 0; i < (int)APUS_FP4_GROUP; i += 8)
                        _mm256_storeu_ps(ob + i, _mm256_mul_ps(
                            _mm256_loadu_ps(ob + i), _mm256_loadu_ps(ab + i)));
                }
                /* eight named chains (array accumulators spill — c/x86.h
                 * dot4 note); adds stay in strict per-group order */
                const float *w0 = wdeq,              *w1 = wdeq + 32;
                const float *w2 = wdeq + 2 * 32,     *w3 = wdeq + 3 * 32;
                const float *w4 = wdeq + 4 * 32,     *w5 = wdeq + 5 * 32;
                const float *w6 = wdeq + 6 * 32,     *w7 = wdeq + 7 * 32;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
                for (int i = 0; i < (int)APUS_FP4_GROUP; i++) {
                    d0 += w0[i]; d1 += w1[i]; d2 += w2[i]; d3 += w3[i];
                    d4 += w4[i]; d5 += w5[i]; d6 += w6[i]; d7 += w7[i];
                }
                /* (dot*sa)*sb in two rounded steps, folds in group order */
                total += (d0 * am[(kb + 0) / 4]) * apus_ue8m0_f32(sp[kb + 0]);
                total += (d1 * am[(kb + 1) / 4]) * apus_ue8m0_f32(sp[kb + 1]);
                total += (d2 * am[(kb + 2) / 4]) * apus_ue8m0_f32(sp[kb + 2]);
                total += (d3 * am[(kb + 3) / 4]) * apus_ue8m0_f32(sp[kb + 3]);
                total += (d4 * am[(kb + 4) / 4]) * apus_ue8m0_f32(sp[kb + 4]);
                total += (d5 * am[(kb + 5) / 4]) * apus_ue8m0_f32(sp[kb + 5]);
                total += (d6 * am[(kb + 6) / 4]) * apus_ue8m0_f32(sp[kb + 6]);
                total += (d7 * am[(kb + 7) / 4]) * apus_ue8m0_f32(sp[kb + 7]);
            }
            for (; kb < nb; kb++) {   /* < 8 groups left */
                float *ob = wdeq;
                apus_fp4_expand32_x86(wp + kb * 16, ob);
                const float *a = adeq + kb * APUS_FP4_GROUP;
                float dot = 0.0f;
                for (int i = 0; i < (int)APUS_FP4_GROUP; i++)
                    dot += a[i] * ob[i];
                total += (dot * am[kb / 4]) * apus_ue8m0_f32(sp[kb]);
            }
            out[m * O + o] = total;
        }
    }
}

void apus_fp4_gemv_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out, size_t O, size_t K) {
    apus_act_dequant_all(acodes, scratch, K);
    apus_fp4_rows_avx2(w, ws, as, scratch, out, 1, O, K, 0, O);
}

void apus_fp4_gemm_avx2(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_act_dequant_all(acodes + m * K, scratch + m * K, K);
    apus_fp4_rows_avx2(w, ws, as, scratch, out, M, O, K, 0, O);
}

#endif /* APUS_X86 */

/* M9b: instantiate the BLAS (Accelerate) kernels in this TU — needs the
 * fp4 helpers above (apus_fp4_expand32_neon, apus_ue8m0_f32, ...). */
#define APUS_BLAS_IMPLEMENTATION
#include "blas.h"

/* --- M6c threaded variant (c/pool.h row partition) ------------------------*/

typedef struct {
    const uint8_t *w, *ws, *acodes;
    const float *as;
    float *scratch, *out;
    size_t M, O, K;
} ApusFp4GemmJob;

#ifdef __ARM_NEON
/* Rows [o0, o1) of the NEON GEMM — delegates to the shared row body
 * (apus_fp4_gemm_rows_neon), so per-output values match apus_fp4_gemm_neon
 * bitwise for any row partition (any thread count). */
static void apus_fp4_gemm_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp4GemmJob *j = vjob;
    apus_fp4_gemm_rows_neon(j->w, j->ws, j->as, j->scratch, j->out,
                            j->M, j->O, j->K, o0, o1);
}
#else
#if APUS_X86
/* Rows [o0, o1) of the AVX2 GEMM — delegates to the shared row body
 * (apus_fp4_rows_avx2), bitwise identical to apus_fp4_gemm_scalar_rows
 * for any row partition (c/x86.h contract). */
static void apus_fp4_gemm_avx2_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp4GemmJob *j = vjob;
    apus_fp4_rows_avx2(j->w, j->ws, j->as, j->scratch, j->out,
                       j->M, j->O, j->K, o0, o1);
}
#endif
static void apus_fp4_gemm_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusFp4GemmJob *j = vjob;
    size_t M = j->M, O = j->O, K = j->K;
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t m = 0; m < M; m++) {
        const float *adeq = j->scratch + m * K;
        const float *am = j->as + m * nab;
        for (size_t o = o0; o < o1; o++) {
            const uint8_t *wp = j->w + o * (K / 2);
            const uint8_t *sp = j->ws + o * nb;
            float total = 0.0f;
            for (size_t kb = 0; kb < nb; kb++) {
                float dot = 0.0f;
                const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
                const float *a = adeq + kb * APUS_FP4_GROUP;
                for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                    dot += a[2 * i]     * apus_fp4_lut[p[i] & 0x0F];
                    dot += a[2 * i + 1] * apus_fp4_lut[p[i] >> 4];
                }
                total += (dot * am[kb / 4]) * apus_ue8m0_f32(sp[kb]);
            }
            j->out[m * O + o] = total;
        }
    }
}
#endif

void apus_fp4_gemm_mt(const uint8_t *w, const uint8_t *ws,
                      const uint8_t *acodes, const float *as,
                      float *scratch, float *out,
                      size_t M, size_t O, size_t K) {
#if APUS_BLAS
    /* M9b: large-batch prefill GEMMs go to Accelerate (AMX); single vecLib
     * thread, so deterministic and APUS_THREADS-independent. Smaller M
     * stays on the M9a NEON kernels (DRAM-bound there; also keeps every
     * pre-M9b bitwise gate on its pinned path — c/blas.h). */
    if (M >= APUS_BLAS_M_MIN && apus_blas_available()) {
        apus_fp4_gemm_blas(w, ws, acodes, as, scratch, out, M, O, K);
        return;
    }
#endif
    ApusFp4GemmJob job = { w, ws, acodes, as, scratch, out, M, O, K };
#ifdef __ARM_NEON
    float16_t *acts16 = (float16_t *)scratch;
    for (size_t m = 0; m < M; m++)
        apus_fp4_act_dequant_f16_all(acodes + m * K, acts16 + m * K, K);
    apus_pool_run(O, apus_fp4_gemm_neon_rows, &job);
#else
    for (size_t m = 0; m < M; m++)
        apus_act_dequant_all(acodes + m * K, scratch + m * K, K);
#if APUS_X86
    /* M12a-2: AVX2 rows when supported — bitwise identical to the scalar
     * rows, so dispatch is numerics-neutral. */
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_pool_run(O, apus_fp4_gemm_avx2_rows, &job);
    } else
#endif
    apus_pool_run(O, apus_fp4_gemm_scalar_rows, &job);
#endif
}

/* --- M9e grouped variant (one dispatch for several GEMMs) -----------------*/

typedef struct {
    const ApusFp4GemmEnt *ents;
    size_t n_ent;
    const float *as;
    float *scratch;
    size_t O, K;
} ApusFp4GemmGroupJob;

/* Scalar per-entry rows of the grouped unit space — the pre-M12a-2 x86
 * fallback (same math as apus_fp4_gemm_scalar_rows), factored out so both
 * the non-AVX2 dispatch and non-x86 builds share it. */
#if !defined(__ARM_NEON)
static void apus_fp4_grouped_rows_scalar(const ApusFp4GemmGroupJob *j,
                                         const ApusFp4GemmEnt *en,
                                         size_t o, size_t o_end) {
    size_t O = j->O, K = j->K;
    size_t nb = K / APUS_FP4_GROUP;
    size_t nab = apus_fp4_act_blocks(K);
    const float *as = j->as;
    for (size_t m = 0; m < en->M; m++) {
        const float *adeq = j->scratch + (en->m0 + m) * K;
        const float *am = as + (en->m0 + m) * nab;
        for (size_t oo = o; oo < o_end; oo++) {
            const uint8_t *wp = en->w + oo * (K / 2);
            const uint8_t *sp = en->ws + oo * nb;
            float total = 0.0f;
            for (size_t kb = 0; kb < nb; kb++) {
                float dot = 0.0f;
                const uint8_t *p = wp + kb * (APUS_FP4_GROUP / 2);
                const float *a = adeq + kb * APUS_FP4_GROUP;
                for (size_t i = 0; i < APUS_FP4_GROUP / 2; i++) {
                    dot += a[2 * i]     * apus_fp4_lut[p[i] & 0x0F];
                    dot += a[2 * i + 1] * apus_fp4_lut[p[i] >> 4];
                }
                total += (dot * am[kb / 4]) * apus_ue8m0_f32(sp[kb]);
            }
            en->out[m * O + oo] = total;
        }
    }
}
#endif

/* Units [u0, u1) of the entry-major unit space n_ent * O. Unit u = e * O + o
 * is output row o of entry e; a contiguous unit range clips to whole output
 * rows within each entry, so every output row is computed entirely by one
 * lane with the shared per-row body — bitwise identical to per-entry
 * apus_fp4_gemm_mt calls for any lane partition. */
static void apus_fp4_gemm_grouped_units(void *vjob, size_t u0, size_t u1) {
    const ApusFp4GemmGroupJob *j = vjob;
    size_t O = j->O, K = j->K;
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t u = u0; u < u1;) {
        size_t e = u / O;
        size_t o = u - e * O;
        size_t o_end = u1 - e * O < O ? u1 - e * O : O;
        const ApusFp4GemmEnt *en = &j->ents[e];
#ifdef __ARM_NEON
        /* the shared row body expects scratch/out addressed at row 0 of the
         * entry; shift the base pointers by m0 rows (as/scratch) and pass a
         * row-shifted out so out[(m)*O + o] lands at en->out[(m)*O + o] */
        apus_fp4_gemm_rows_neon(en->w, en->ws,
                                j->as + en->m0 * nab,
                                (const float *)((const float16_t *)j->scratch
                                                + en->m0 * K),
                                en->out, en->M, O, K, o, o_end);
#elif APUS_X86
        /* M12a-2: AVX2 rows (bitwise == the scalar rows) when supported.
         * Same base-pointer shift as the NEON branch. */
        if (apus_x86_have_avx2()) {
            atomic_fetch_add(&apus_x86_hits, 1);
            apus_fp4_rows_avx2(en->w, en->ws,
                               j->as + en->m0 * nab,
                               j->scratch + en->m0 * K,
                               en->out, en->M, O, K, o, o_end);
        } else {
            apus_fp4_grouped_rows_scalar(j, en, o, o_end);
        }
#else
        apus_fp4_grouped_rows_scalar(j, en, o, o_end);
#endif
        u = (e + 1) * O;
    }
}

void apus_fp4_gemm_mt_grouped(const ApusFp4GemmEnt *ents, size_t n_ent,
                              const uint8_t *acodes, const float *as,
                              float *scratch, size_t O, size_t K) {
#if APUS_BLAS
    size_t nab = apus_fp4_act_blocks(K);
#endif
    /* Chunks of <=64 entries: fixed storage, any n_ent. Within a chunk,
     * BLAS-sized entries take the standalone fallback; the rest share one
     * pool dispatch over the entry-major unit space. */
    while (n_ent) {
        size_t nc = n_ent < 64 ? n_ent : 64;
        ApusFp4GemmEnt small[64];
        size_t n_small = 0;
        for (size_t e = 0; e < nc; e++) {
#if APUS_BLAS
            /* per-entry fallback for a BLAS-sized batch: identical to a
             * standalone apus_fp4_gemm_mt call for that entry (c/blas.h
             * dispatch contract). Callers (c/moe.h) keep such experts out
             * of groups; this is defensive. */
            if (ents[e].M >= APUS_BLAS_M_MIN && apus_blas_available()) {
                apus_fp4_gemm_mt(ents[e].w, ents[e].ws,
                                 acodes + ents[e].m0 * K,
                                 as + ents[e].m0 * nab,
                                 scratch + ents[e].m0 * K,
                                 ents[e].out, ents[e].M, O, K);
                continue;
            }
#endif
            small[n_small++] = ents[e];
        }
        if (n_small) {
#ifdef __ARM_NEON
            float16_t *acts16 = (float16_t *)scratch;
            for (size_t e = 0; e < n_small; e++)
                for (size_t m = 0; m < small[e].M; m++)
                    apus_fp4_act_dequant_f16_all(
                        acodes + (small[e].m0 + m) * K,
                        acts16 + (small[e].m0 + m) * K, K);
#else
            for (size_t e = 0; e < n_small; e++)
                for (size_t m = 0; m < small[e].M; m++)
                    apus_act_dequant_all(acodes + (small[e].m0 + m) * K,
                                         scratch + (small[e].m0 + m) * K, K);
#endif
            ApusFp4GemmGroupJob job = { small, n_small, as, scratch, O, K };
            apus_pool_run(n_small * O, apus_fp4_gemm_grouped_units, &job);
        }
        ents += nc;
        n_ent -= nc;
    }
}

#endif /* APUS_FP4_IMPLEMENTATION */
#endif /* APUS_FP4_H */
