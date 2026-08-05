/*
 * c/x86.h — M12a-2: AVX2 kernels for the x86-64 hot paths, runtime-dispatched.
 *
 * Design contract (see tests/m12/README.md, M12a-2 section):
 *
 *   Every AVX2 kernel in this tree is BITWISE IDENTICAL to the normative
 *   scalar kernel it replaces — the scalar kernels (c/fp4.h, c/fp8.h,
 *   c/attn.h, c/model.h) are the semantic anchor on x86, and the AVX2 paths
 *   keep their exact rounding sequence:
 *     - expansion/dequantization is per-element EXACT (E4M3 expand proven
 *       bitwise on all 256 codes in tests/m12, both the F16C and the
 *       integer variant; FP4 codes via the same LUT*2 nibble table as NEON;
 *       BF16 widening is a pure 16-bit shift);
 *     - per-element products are computed 8-wide (one IEEE mul per element,
 *       identical to the scalar mul's single rounding — no FMA anywhere:
 *       the scalar anchors use separate mul + add, two roundings);
 *     - every reduction keeps the scalar SEQUENTIAL order: independent
 *       output rows/blocks are interleaved as separate accumulator chains
 *       (each chain is the scalar loop, so no reassociation), but no chain
 *       ever changes the order of adds within one dot.
 *   Consequence: the existing x86 gates (scalar anchors, within-platform
 *   thread-count digests, tolerance classes) hold unchanged, and dispatch
 *   is numerics-neutral.
 *
 * Dispatch: per-function __attribute__((target("avx2"[, "f16c"]))) — no
 * global -mavx2, the binary still runs on baseline x86-64. Callers check
 * apus_x86_have_avx2() once per kernel call and fall back to the scalar
 * path. APUS_X86_DISABLE=1 forces the scalar paths (bench/debug).
 *
 * FMA is intentionally NOT used and NOT required: fusing mul+add would
 * change the two-rounding scalar contract to one rounding (that is what
 * the ARM NEON paths do for the sparse-attn KV sum — on x86 the scalar
 * anchor is mul+add, so the AVX2 kernel mirrors THAT, not ARM).
 *
 * Everything is static/static inline (single-TU header pattern like the
 * rest of c/). APUS_X86 == 0 off x86-64: the header compiles to nothing.
 */

#ifndef APUS_X86_H
#define APUS_X86_H

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define APUS_X86 1
#else
#define APUS_X86 0
#endif

#if APUS_X86

#include <immintrin.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>   /* getenv */
#include <string.h>   /* memcpy */

#define APUS_TGT_AVX2      __attribute__((target("avx2")))
#define APUS_TGT_AVX2_F16C __attribute__((target("avx2,f16c")))

/* Runtime AVX2 gate (cached; the write is idempotent, so the race is
 * benign). AVX2 alone suffices — no FMA is used (see header note). */
static inline int apus_x86_have_avx2(void) {
    static int cached = -1;
    if (cached < 0)
        cached = __builtin_cpu_supports("avx2")
              && !getenv("APUS_X86_DISABLE");
    return cached;
}

static inline int apus_x86_have_f16c(void) {
    static int cached = -1;
    if (cached < 0)
        cached = __builtin_cpu_supports("f16c");
    return cached;
}

/* AVX2-path activity counter (tests/m12 "the AVX2 path was taken HERE"
 * probe). Incremented once per dispatched worker invocation. */
static _Atomic unsigned long apus_x86_hits;
static inline unsigned long apus_x86_avx2_hits(void) {
    return atomic_load(&apus_x86_hits);
}

/* --- E4M3 -> FP32 expand (exact, shared by fp8 weights and fp4 acts) -----*/

/* F16C variant — the M9a NEON trick in SSE/AVX2 form: the bit placement
 * h16 = (c&0x80)<<8 | (c&0x7F)<<7 read as FP16 is exactly 2^-8 x the E4M3
 * value for BOTH normals and subnormals, so f32(h16) * 256 is EXACT for
 * all 256 codes (NaN codes 0x7F/0xFF decode as +-480, matching
 * apus_e4m3_dequant_f32). Proven bitwise on all 256 codes in tests/m12. */
APUS_TGT_AVX2_F16C
static inline void apus_e4m3_expand16_f16c_x86(const uint8_t *p, float *out) {
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    __m256i w = _mm256_cvtepu8_epi16(b);   /* 16 x u16 */
    __m256i h = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x80)), 8),
        _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x7F)), 7));
    __m256 s = _mm256_set1_ps(256.0f);
    _mm256_storeu_ps(out, _mm256_mul_ps(
        _mm256_cvtph_ps(_mm256_castsi256_si128(h)), s));
    _mm256_storeu_ps(out + 8, _mm256_mul_ps(
        _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1)), s));
}

/* Integer variant (AVX2 without F16C): normal codes (e >= 1) construct the
 * FP32 bits directly — value (1+m/8)*2^(e-7) has exponent field e+120 and
 * mantissa m<<20; subnormal codes (e == 0) are m*2^-9 = float(m) * 2^-9,
 * both steps exact. Proven bitwise on all 256 codes in tests/m12. */
APUS_TGT_AVX2
static inline void apus_e4m3_expand16_int_x86(const uint8_t *p, float *out) {
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    for (int half = 0; half < 2; half++) {
        __m256i v = _mm256_cvtepu8_epi32(half ? _mm_srli_si128(b, 8) : b);
        __m256i sign = _mm256_slli_epi32(
            _mm256_and_si256(v, _mm256_set1_epi32(0x80)), 24);
        __m256i e = _mm256_and_si256(_mm256_srli_epi32(v, 3),
                                     _mm256_set1_epi32(0xF));
        __m256i m = _mm256_and_si256(v, _mm256_set1_epi32(7));
        __m256i norm = _mm256_or_si256(
            _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(120)), 23),
            _mm256_slli_epi32(m, 20));
        __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(m),
                                   _mm256_set1_ps(0x1p-9f));
        __m256i isub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
        __m256 val = _mm256_blendv_ps(_mm256_castsi256_ps(norm), sub,
                                      _mm256_castsi256_ps(isub));
        val = _mm256_castsi256_ps(_mm256_xor_si256(
            _mm256_castps_si256(val), sign));
        _mm256_storeu_ps(out + 8 * half, val);
    }
}

static inline void apus_e4m3_expand16_x86(const uint8_t *p, float *out) {
    if (apus_x86_have_f16c())
        apus_e4m3_expand16_f16c_x86(p, out);
    else
        apus_e4m3_expand16_int_x86(p, out);
}

/* Widen 8 BF16 bit patterns -> 8 FP32 (exact: upper 16 bits). */
APUS_TGT_AVX2
static inline __m256 apus_bf16_widen8_x86(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}

/* --- staged-product sequential dots (the bitwise-scalar SIMD pattern) ----*/

/* FOUR independent sequential-order dots at once (four unrelated output
 * rows): out[r] = sum_k a_r[k] * b_r[k], each accumulated STRICTLY in k
 * order — bitwise identical to the scalar loop per row. The products are
 * staged 8-wide (one exact IEEE mul per element) and the adds run as four
 * interleaved scalar chains to hide the FP-add latency; no chain's add
 * order changes. The accumulators and row pointers are NAMED variables,
 * not arrays: indexed acc[]/a[] spills to memory in the inner loop
 * (measured 3x slower under Rosetta translation, tests/m12 README). */
APUS_TGT_AVX2
static inline void apus_dot4_f32_x86(const float *const a[4],
                                     const float *const b[4],
                                     size_t n, float out[4]) {
    const float *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3];
    const float *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
    float p0[8], p1[8], p2[8], p3[8];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(a0 + k),
                                           _mm256_loadu_ps(b0 + k)));
        _mm256_storeu_ps(p1, _mm256_mul_ps(_mm256_loadu_ps(a1 + k),
                                           _mm256_loadu_ps(b1 + k)));
        _mm256_storeu_ps(p2, _mm256_mul_ps(_mm256_loadu_ps(a2 + k),
                                           _mm256_loadu_ps(b2 + k)));
        _mm256_storeu_ps(p3, _mm256_mul_ps(_mm256_loadu_ps(a3 + k),
                                           _mm256_loadu_ps(b3 + k)));
        for (int i = 0; i < 8; i++) {   /* fully unrolled (constant 8) */
            d0 += p0[i]; d1 += p1[i]; d2 += p2[i]; d3 += p3[i];
        }
    }
    for (; k < n; k++) {
        d0 += a0[k] * b0[k]; d1 += a1[k] * b1[k];
        d2 += a2[k] * b2[k]; d3 += a3[k] * b3[k];
    }
    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3;
}

/* Same, with BF16 weight rows (exactly widened). */
APUS_TGT_AVX2
static inline void apus_dot4_bf16_x86(const float *const a[4],
                                      const uint16_t *const b[4],
                                      size_t n, float out[4]) {
    const float *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3];
    const uint16_t *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
    float p0[8], p1[8], p2[8], p3[8];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(a0 + k),
                                           apus_bf16_widen8_x86(b0 + k)));
        _mm256_storeu_ps(p1, _mm256_mul_ps(_mm256_loadu_ps(a1 + k),
                                           apus_bf16_widen8_x86(b1 + k)));
        _mm256_storeu_ps(p2, _mm256_mul_ps(_mm256_loadu_ps(a2 + k),
                                           apus_bf16_widen8_x86(b2 + k)));
        _mm256_storeu_ps(p3, _mm256_mul_ps(_mm256_loadu_ps(a3 + k),
                                           apus_bf16_widen8_x86(b3 + k)));
        for (int i = 0; i < 8; i++) {   /* fully unrolled (constant 8) */
            d0 += p0[i]; d1 += p1[i]; d2 += p2[i]; d3 += p3[i];
        }
    }
    for (; k < n; k++) {
        uint32_t u0 = (uint32_t)b0[k] << 16, u1 = (uint32_t)b1[k] << 16;
        uint32_t u2 = (uint32_t)b2[k] << 16, u3 = (uint32_t)b3[k] << 16;
        float f0, f1, f2, f3;
        memcpy(&f0, &u0, 4); memcpy(&f1, &u1, 4);
        memcpy(&f2, &u2, 4); memcpy(&f3, &u3, 4);
        d0 += a0[k] * f0; d1 += a1[k] * f1;
        d2 += a2[k] * f2; d3 += a3[k] * f3;
    }
    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3;
}

/* ov[k] += p * kv[k] over k in [0, n): per-element mul + add, TWO roundings
 * — the x86 scalar contract for the sparse-attn weighted KV sum (ARM fuses
 * this with vfmaq; on x86 the scalar anchor is mul+add, so the AVX2 kernel
 * mirrors THAT and stays bitwise vs the x86 scalar loop). */
APUS_TGT_AVX2
static inline void apus_saxpy_x86(float *ov, float p, const float *kv,
                                  size_t n) {
    __m256 pj = _mm256_set1_ps(p);
    size_t k = 0;
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(ov + k, _mm256_add_ps(_mm256_loadu_ps(ov + k),
            _mm256_mul_ps(pj, _mm256_loadu_ps(kv + k))));
    for (; k < n; k++) ov[k] += p * kv[k];
}

#endif /* APUS_X86 */
#endif /* APUS_X86_H */
