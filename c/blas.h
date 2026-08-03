/*
 * c/blas.h — Accelerate.framework (vecLib/AMX) FP32 GEMM path for the
 * batch-M prefill shapes (M9b). C11, macOS only (Accelerate is a system
 * framework, linked like libSystem; see tests/m9b/README.md build notes).
 *
 * Dispatch contract:
 *   - M <  APUS_BLAS_M_MIN: caller keeps the M9a NEON kernels (decode GEMV
 *     and small-M work are DRAM-bound / call-overhead-bound; BLAS does not
 *     help and the NEON path's M-independence is load-bearing for the
 *     chunk-invariance bitwise gates in tests/m4c+m5 and the m7b GPU==CPU
 *     bitwise gates, whose largest batched M is 250).
 *   - M >= APUS_BLAS_M_MIN: cblas_sgemm from Accelerate, one vecLib thread
 *     per call (VECLIB_MAXIMUM_THREADS=1 pinned on first use), over a
 *     FIXED grid of <=8 MB weight row tiles distributed across the
 *     c/pool.h lanes (AMX is per-core). The tile grid depends only on
 *     (O, K), so outputs are deterministic and APUS_THREADS-independent.
 *
 * Numerics contract (EXACTLY the fp8.h/fp4.h normative paths — all scale
 * factors are UE8M0 powers of two, so per-element scale folding is exact
 * and the block scale-application sequence is preserved up to FP32
 * summation order, the project's accepted reorder class, tests/m3):
 *   fp8: a'[m,i] = e4m3(acode)·sa[m,i/128]  (exact pow2 product)
 *        w'[o,i] = e4m3(w[o,i])·sb[o/128,i/128] (exact pow2 product)
 *        out[m,o] = Σ_i a'·w'  ==  Σ_kb dot_kb·(sa·sb)  — identical value
 *        whenever no intermediate over/underflows (sa·sb and every fold
 *        are exact scalings by powers of two, which distribute exactly
 *        over the dot's partial sums; only the summation ORDER is BLAS's).
 *   fp4: a'[m,i] = e4m3(acode)·sa[m,i/128]; w'[o,i] = LUT·sb[o,i/32]
 *        (LUT = codes×2×0.5; the 0.5 is an exact pow2 fold into sb,
 *        incl. the byte-0 subnormal) → out == Σ_kb (dot32·sa)·sb likewise.
 *   Degenerate scale bytes (255 = inf) can diverge from the NEON path
 *   (inf folds into weights before the dot); they never occur in the
 *   checkpoint and the BLAS path is only exercised on real-scale data.
 *
 * Include AFTER fp4.h (needs its declarations); the implementation is
 * instantiated once from the end of fp4.h's implementation section.
 */
#ifndef APUS_BLAS_H
#define APUS_BLAS_H

#include <stddef.h>
#include <stdint.h>

#include "pool.h"

#if defined(__APPLE__) && defined(__ARM_NEON)
#define APUS_BLAS 1
#else
#define APUS_BLAS 0
#endif

/* Batch-M dispatch cutoff. 256 = above every pre-M9b suite's largest
 * batched M (m4c 250, m5/m7b 200, m6a/m6b 8, m8 k+1) so all existing
 * bitwise gates keep the NEON path, and far above the measured NEON/BLAS
 * crossover (~M=16..32, tests/m9b). */
#define APUS_BLAS_M_MIN 256

/* 1 when the BLAS path is compiled in and not disabled via APUS_NO_BLAS=1.
 * First call pins VECLIB_MAXIMUM_THREADS=1 (deterministic partitioning). */
int apus_blas_available(void);

/* Same signatures/semantics as apus_fp8_gemm_mt / apus_fp4_gemm_mt
 * (scratch: M*K floats), for M >= APUS_BLAS_M_MIN. */
void apus_fp8_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K);
void apus_fp4_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K);

/* M9d: dense-weight variants for the pinned scalar f32/bf16 linears
 * (compressor wkv/wgate, router gate, idx_wproj, pilot mixes) and the
 * grouped wo_a o-proj. Same dispatch contract: caller engages only at
 * M >= APUS_BLAS_M_MIN, fixed (O,K)-only tile grid, one vecLib thread per
 * sgemm => deterministic and APUS_THREADS-independent. FP32 summation order
 * inside the dots is Accelerate's (the accepted reorder class, tests/m3);
 * there are no scales here, so nothing else changes.
 * apus_f32_gemm_blas:  w [O,K] FP32, x [M,K], out [M,O], all row-major,
 *                      packed rows (ld == K / O); weights stream straight
 *                      into sgemm (no dequant).
 * apus_bf16_gemm_blas: wb [O,K] BF16 bits (uint16), widened EXACTLY to FP32
 *                      per tile (same widening as the NEON wo_a kernel);
 *                      x rows at stride ldx, out rows at stride ldo (the
 *                      grouped wo_a call passes group-slice offsets). */
void apus_f32_gemm_blas(const float *w, const float *x, float *out,
                        size_t M, size_t O, size_t K);
void apus_bf16_gemm_blas(const uint16_t *wb, const float *x, size_t ldx,
                         float *out, size_t ldo,
                         size_t M, size_t O, size_t K);

/* M9d: batched grouped wo_a o-proj (c/attn.h): G independent BF16-weight
 * GEMMs — group g reads the x slice x + g*sub (row stride hd), the weight
 * block wa + g*ol*sub, and writes the out slice y + g*ol (row stride
 * G*ol) — dispatched as ONE pooled grid of G x the fixed (ol,sub) tile
 * grid so every pool lane stays busy across all groups. Same numerics as
 * apus_bf16_gemm_blas per group (exact BF16->FP32 widen, sgemm per tile);
 * the grid depends only on (G, ol, sub). */
void apus_woa_gemm_blas(const uint16_t *wa, const float *x, float *y,
                        size_t M, size_t G, size_t ol, size_t sub,
                        size_t hd);

#endif /* APUS_BLAS_H */

/* =========================================================================*/
#if defined(APUS_BLAS_IMPLEMENTATION) && !defined(APUS_BLAS_IMPL_INCLUDED)
#define APUS_BLAS_IMPL_INCLUDED

#if APUS_BLAS

/* The LP64 cblas interface is marked deprecated in the macOS 13.3 SDK in
 * favor of the ILP64-capable headers; the symbols remain the stable system
 * ABI (Accelerate ships with the OS). Silence the deprecation noise (the
 * warning fires at the call sites, so the pragma wraps the whole impl). */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

int apus_blas_available(void) {
    static int state = -1;
    if (state < 0) {
        const char *no = getenv("APUS_NO_BLAS");
        state = (no && no[0] == '1') ? 0 : 1;
        if (state)
            setenv("VECLIB_MAXIMUM_THREADS", "1", 0); /* before first sgemm */
    }
    return state;
}

/* E4M3 -> FP32 expand, 16 codes -> 4 vectors (exact; copy of fp8.h's
 * apus_e4m3_expand16_neon — fp8.h's static inlines are not visible at the
 * point this implementation is instantiated from fp4.h). */
static inline void apus_blas_e4m3_expand16(const uint8_t *p, float32x4_t v[4]) {
    uint8x16_t b = vld1q_u8(p);
    uint16x8_t cl = vmovl_u8(vget_low_u8(b)), ch = vmovl_u8(vget_high_u8(b));
    uint16x8_t hl = vorrq_u16(vshlq_n_u16(vandq_u16(cl, vdupq_n_u16(0x80u)), 8),
                              vshlq_n_u16(vandq_u16(cl, vdupq_n_u16(0x7Fu)), 7));
    uint16x8_t hh = vorrq_u16(vshlq_n_u16(vandq_u16(ch, vdupq_n_u16(0x80u)), 8),
                              vshlq_n_u16(vandq_u16(ch, vdupq_n_u16(0x7Fu)), 7));
    v[0] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hl))),
                       256.0f);
    v[1] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hl))),
                       256.0f);
    v[2] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hh))),
                       256.0f);
    v[3] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hh))),
                       256.0f);
}

/* Dequant + act-scale fold: scratch[m*K+i] = e4m3(acodes)·sa[m, i/128].
 * sa is a power of two -> every product is EXACT (no rounding). */
static void apus_blas_act_fold(const uint8_t *acodes, const float *as,
                               float *scratch, size_t M, size_t K) {
    size_t nb = (K + 127) / 128;
    for (size_t m = 0; m < M; m++) {
        const uint8_t *cp = acodes + m * K;
        float *d = scratch + m * K;
        for (size_t kb = 0; kb < nb; kb++) {
            size_t lo = kb * 128, hi = lo + 128 > K ? K : lo + 128;
            float32x4_t sa = vdupq_n_f32(as[m * nb + kb]);
            size_t i = lo;
            for (; i + 16 <= hi; i += 16) {
                float32x4_t v[4];
                apus_blas_e4m3_expand16(cp + i, v);
                for (int j = 0; j < 4; j++)
                    vst1q_f32(d + i + 4 * (size_t)j, vmulq_f32(v[j], sa));
            }
            for (; i < hi; i++)
                d[i] = apus_e4m3_dequant_f32(cp[i]) * as[m * nb + kb];
        }
    }
}

/* Weight-tile scratch rows: largest multiple of 128 with rows*K*4 <= cap. */
static size_t apus_blas_tile_rows(size_t O, size_t K, size_t cap_bytes) {
    size_t rows = cap_bytes / (K * sizeof(float));
    rows &= ~(size_t)127;
    if (rows < 128) rows = 128;
    return rows < O ? rows : O;
}

typedef struct {
    const uint8_t *w, *ws;
    const float *a;         /* folded activations [M,K] (the caller scratch) */
    float *out;
    size_t M, O, K, OT;
    int fp4;
} ApusBlasJob;

/* One weight tile [o0, o0+on): dequant + exact scale fold into a
 * thread-local scratch tile, then a single-thread sgemm. Called per tile
 * from the pool lanes (each lane has its own TLS scratch arena). The tile
 * grid is fixed by (O, K) only, so every output element is produced by an
 * sgemm of the same shape regardless of the lane count — bitwise
 * thread-count independent by construction (asserted in tests/m9b). */
static void apus_blas_tile(const ApusBlasJob *j, size_t o0, size_t on) {
    size_t K = j->K;
    ApusScratchMark mk = apus_scratch_mark();
    float *wd = apus_scratch_alloc(on * K * sizeof(float));
    int heap = 0;
    if (!wd) {
        wd = malloc(on * K * sizeof(float));
        heap = 1;
        if (!wd) { apus_scratch_reset(mk); return; }
    }
    if (!j->fp4) {
        size_t nb = (K + 127) / 128;
        for (size_t oo = 0; oo < on; oo++) {
            const uint8_t *wp = j->w + (o0 + oo) * K;
            const uint8_t *sp = j->ws + ((o0 + oo) / 128) * nb;
            float *d = wd + oo * K;
            for (size_t kb = 0; kb < nb; kb++) {
                size_t lo = kb * 128, hi = lo + 128 > K ? K : lo + 128;
                float32x4_t sb = vdupq_n_f32(apus_ue8m0_f32(sp[kb]));
                size_t i = lo;
                for (; i + 16 <= hi; i += 16) {
                    float32x4_t v[4];
                    apus_blas_e4m3_expand16(wp + i, v);
                    for (int q = 0; q < 4; q++)
                        vst1q_f32(d + i + 4 * (size_t)q, vmulq_f32(v[q], sb));
                }
                for (; i < hi; i++)
                    d[i] = apus_e4m3_dequant_f32(wp[i])
                         * apus_ue8m0_f32(sp[kb]);
            }
        }
    } else {
        size_t nb = K / APUS_FP4_GROUP;
        for (size_t oo = 0; oo < on; oo++) {
            const uint8_t *wp = j->w + (o0 + oo) * (K / 2);
            const uint8_t *sp = j->ws + (o0 + oo) * nb;
            float *d = wd + oo * K;
            for (size_t b = 0; b < nb; b++) {
                /* codes expand as LUT*2; the 0.5 fold is an exact pow2
                 * (byte-0 subnormal included: 0.5*2^-127 = 2^-128) */
                float32x4_t hs = vdupq_n_f32(0.5f * apus_ue8m0_f32(sp[b]));
                float32x4_t v[8];
                apus_fp4_expand32_neon(wp + b * 16, v);
                float *o = d + b * APUS_FP4_GROUP;
                for (int i = 0; i < 8; i++)
                    vst1q_f32(o + 4 * (size_t)i, vmulq_f32(v[i], hs));
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)j->M, (int)on, (int)K, 1.0f,
                j->a, (int)K, wd, (int)K, 0.0f, j->out + o0, (int)j->O);
    if (heap) free(wd);
    apus_scratch_reset(mk);
}

static void apus_blas_tiles(void *vjob, size_t t0, size_t t1) {
    const ApusBlasJob *j = vjob;
    for (size_t t = t0; t < t1; t++) {
        size_t o0 = t * j->OT, on = j->O - o0 < j->OT ? j->O - o0 : j->OT;
        apus_blas_tile(j, o0, on);
    }
}

/* 8 MB per-lane dequant tiles: big enough for efficient sgemm shapes,
 * small enough that 8 pool lanes cost 64 MB of TLS scratch. */
#define APUS_BLAS_TILE_CAP ((size_t)8 << 20)

void apus_fp8_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    apus_blas_act_fold(acodes, as, scratch, M, K);
    ApusBlasJob job = { w, ws, scratch, out, M, O, K,
                        apus_blas_tile_rows(O, K, APUS_BLAS_TILE_CAP), 0 };
    apus_pool_run((O + job.OT - 1) / job.OT, apus_blas_tiles, &job);
}

void apus_fp4_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    apus_blas_act_fold(acodes, as, scratch, M, K);
    ApusBlasJob job = { w, ws, scratch, out, M, O, K,
                        apus_blas_tile_rows(O, K, APUS_BLAS_TILE_CAP), 1 };
    apus_pool_run((O + job.OT - 1) / job.OT, apus_blas_tiles, &job);
}

/* --- M9d dense-weight paths -------------------------------------------------*/

typedef struct {
    const float *w;         /* [O,K] FP32, packed rows */
    const uint16_t *wb;     /* [O,K] BF16 bits (when set) */
    const float *x;
    size_t ldx;
    float *out;
    size_t ldo;
    size_t M, O, K, OT;
} ApusBlasDenseJob;

/* One weight tile [o0, o0+on) of the dense paths: FP32 weights stream
 * directly (no scratch); BF16 weights are widened exactly into a
 * thread-local scratch tile first. Single-thread sgemm per tile; the tile
 * grid depends only on (O, K) — bitwise thread-count independent. */
static void apus_blas_dense_tile(const ApusBlasDenseJob *j, size_t o0,
                                 size_t on) {
    size_t K = j->K;
    const float *w = j->w;
    float *wd = NULL;
    ApusScratchMark mk = apus_scratch_mark();
    int heap = 0;
    if (j->wb) {
        wd = apus_scratch_alloc(on * K * sizeof(float));
        if (!wd) {
            wd = malloc(on * K * sizeof(float));
            heap = 1;
            if (!wd) { apus_scratch_reset(mk); return; }
        }
        for (size_t oo = 0; oo < on; oo++) {
            const uint16_t *wp = j->wb + (o0 + oo) * K;
            float *d = wd + oo * K;
            size_t i = 0;
            /* exact BF16 -> FP32 widen (bit shift; same values as the NEON
             * wo_a kernel's vshll widening) */
            for (; i + 8 <= K; i += 8) {
                uint16x8_t b = vld1q_u16(wp + i);
                vst1q_f32(d + i,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16)));
                vst1q_f32(d + i + 4,
                    vreinterpretq_f32_u32(vshll_high_n_u16(b, 16)));
            }
            for (; i < K; i++) {
                uint32_t b = (uint32_t)wp[i] << 16;
                memcpy(&d[i], &b, 4);
            }
        }
        w = wd;
    } else {
        w = j->w + o0 * K;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)j->M, (int)on, (int)K, 1.0f,
                j->x, (int)j->ldx, w, (int)K, 0.0f,
                j->out + o0, (int)j->ldo);
    if (heap) free(wd);
    apus_scratch_reset(mk);
}

static void apus_blas_dense_tiles(void *vjob, size_t t0, size_t t1) {
    const ApusBlasDenseJob *j = vjob;
    for (size_t t = t0; t < t1; t++) {
        size_t o0 = t * j->OT, on = j->O - o0 < j->OT ? j->O - o0 : j->OT;
        apus_blas_dense_tile(j, o0, on);
    }
}

void apus_f32_gemm_blas(const float *w, const float *x, float *out,
                        size_t M, size_t O, size_t K) {
    ApusBlasDenseJob job = { w, NULL, x, K, out, O, M, O, K,
                             apus_blas_tile_rows(O, K, APUS_BLAS_TILE_CAP) };
    apus_pool_run((O + job.OT - 1) / job.OT, apus_blas_dense_tiles, &job);
}

void apus_bf16_gemm_blas(const uint16_t *wb, const float *x, size_t ldx,
                         float *out, size_t ldo,
                         size_t M, size_t O, size_t K) {
    ApusBlasDenseJob job = { NULL, wb, x, ldx, out, ldo, M, O, K,
                             apus_blas_tile_rows(O, K, APUS_BLAS_TILE_CAP) };
    apus_pool_run((O + job.OT - 1) / job.OT, apus_blas_dense_tiles, &job);
}

/* Batched grouped wo_a (M9d): one pool dispatch over G x tiles-per-group.
 * Tile (g, tt) is exactly an apus_blas_dense_tile call on group g's slice,
 * so per-group outputs equal apus_bf16_gemm_blas bitwise; the grid depends
 * only on (G, ol, sub). */
typedef struct {
    const uint16_t *wa;
    const float *x;
    float *y;
    size_t M, G, ol, sub, hd, OT, ntpg;
} ApusBlasWoaJob;

static void apus_blas_woa_tiles(void *vjob, size_t t0, size_t t1) {
    const ApusBlasWoaJob *j = vjob;
    for (size_t t = t0; t < t1; t++) {
        size_t g = t / j->ntpg, tt = t % j->ntpg;
        size_t o0 = tt * j->OT,
               on = j->ol - o0 < j->OT ? j->ol - o0 : j->OT;
        ApusBlasDenseJob dj = {
            NULL, j->wa + g * j->ol * j->sub,
            j->x + g * j->sub, j->hd,
            j->y + g * j->ol, j->G * j->ol,
            j->M, j->ol, j->sub, j->OT
        };
        apus_blas_dense_tile(&dj, o0, on);
    }
}

void apus_woa_gemm_blas(const uint16_t *wa, const float *x, float *y,
                        size_t M, size_t G, size_t ol, size_t sub,
                        size_t hd) {
    ApusBlasWoaJob job = { wa, x, y, M, G, ol, sub, hd, 0, 0 };
    job.OT = apus_blas_tile_rows(ol, sub, APUS_BLAS_TILE_CAP);
    job.ntpg = (ol + job.OT - 1) / job.OT;
    apus_pool_run(G * job.ntpg, apus_blas_woa_tiles, &job);
}

#pragma clang diagnostic pop

#else  /* !APUS_BLAS */

int apus_blas_available(void) { return 0; }

void apus_fp8_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    (void)w; (void)ws; (void)acodes; (void)as; (void)scratch;
    (void)out; (void)M; (void)O; (void)K;
}
void apus_fp8_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    (void)w; (void)ws; (void)acodes; (void)as; (void)scratch;
    (void)out; (void)M; (void)O; (void)K;
}
void apus_fp4_gemm_blas(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as,
                        float *scratch, float *out,
                        size_t M, size_t O, size_t K) {
    (void)w; (void)ws; (void)acodes; (void)as; (void)scratch;
    (void)out; (void)M; (void)O; (void)K;
}
void apus_f32_gemm_blas(const float *w, const float *x, float *out,
                        size_t M, size_t O, size_t K) {
    (void)w; (void)x; (void)out; (void)M; (void)O; (void)K;
}
void apus_bf16_gemm_blas(const uint16_t *wb, const float *x, size_t ldx,
                         float *out, size_t ldo,
                         size_t M, size_t O, size_t K) {
    (void)wb; (void)x; (void)ldx; (void)out; (void)ldo;
    (void)M; (void)O; (void)K;
}
void apus_woa_gemm_blas(const uint16_t *wa, const float *x, float *y,
                        size_t M, size_t G, size_t ol, size_t sub,
                        size_t hd) {
    (void)wa; (void)x; (void)y; (void)M; (void)G; (void)ol; (void)sub;
    (void)hd;
}

#endif /* APUS_BLAS */
#endif /* APUS_BLAS_IMPLEMENTATION */
