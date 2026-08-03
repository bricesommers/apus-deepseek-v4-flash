/*
 * c/mhc.h — mHC (Manifold-Constrained Hyper-Connections) building blocks
 * for DeepSeek-V4-Flash: per-token pre/post/comb map generation from the
 * 4x hidden state, Sinkhorn-20 on the comb matrix, the apply step, and the
 * hc_head 4->1 collapse. C11, libc + arm_neon.h only. FP32 throughout
 * (all hc_* params are FP32 in the checkpoint).
 *
 * Numerics contract (normative reference: reference/inference/model.py
 * Block.hc_pre/hc_post lines 673-686, ParallelHead.hc_head lines 728-735,
 * and kernel.py hc_split_sinkhorn_kernel lines 371-438):
 *
 *   Per token, state X [n, d] (n = hc_mult = 4) flattened to x [n*d] FP32:
 *       rsqrt  = rsqrt(mean(x^2) + norm_eps)          norm_eps = 1e-6
 *       mixes  = (hc_fn @ x) * rsqrt                  hc_fn [24, n*d] FP32
 *   NOTE the reference multiplies by rsqrt AFTER the matmul
 *   (F.linear(x, hc_fn) * rsqrt, model.py:678) — not x*rsqrt before it.
 *   Linear scaling makes these mathematically equal but FP32-rounding
 *   different; these kernels follow the reference order.
 *
 *   Split/gates (kernel.py:391-396), mix_hc = (2+n)*n = 24:
 *       pre[j]    = sigmoid(mixes[j]         * scale[0] + base[j])         + hc_eps
 *       post[j]   = 2 * sigmoid(mixes[n+j]   * scale[1] + base[n+j])
 *       comb[j,k] = mixes[2n + j*n+k] * scale[2] + base[2n + j*n+k]
 *
 *   Sinkhorn (kernel.py:398-423), hc_eps = 1e-6, iters = 20 — EXACT order:
 *       1. row softmax with max subtraction, then PER-ELEMENT + eps:
 *          c[j,k] = exp(c[j,k] - rowmax[j]) / rowsum[j] + eps
 *       2. column normalize with eps IN THE DENOMINATOR:
 *          c[j,k] /= colsum[k] + eps
 *       3. repeat (iters-1 = 19) times: row normalize c[j,k] /= rowsum[j]+eps,
 *          then column normalize c[j,k] /= colsum[k]+eps.
 *   Total: 1 row softmax, 20 column normalizations, 19 row normalizations;
 *   the last op is a column normalization. The eps placements differ
 *   between step 1 (per element, after division) and steps 2-3 (added to
 *   the sum, before division) — do not "unify" them.
 *
 *   Apply (model.py:680-686), all FP32 (reference casts to BF16 at .to() /
 *   .type_as() boundaries — the caller owns those roundings):
 *       collapse: y[i]    = sum_j pre[j] * X[j,i]              (hc_pre output)
 *       expand:   Y[j,i]  = post[j] * f[i] + sum_k comb[k,j] * R[k,i]
 *                 where f is the sublayer output and R the pre-sublayer
 *                 residual state (hc_post). NOTE the comb factor indexes
 *                 comb[residual k][output j] (model.py:685 sums over the
 *                 FIRST comb axis: sum(comb.unsqueeze(-1) *
 *                 residual.unsqueeze(-2), dim=2)). The comb STORAGE layout
 *                 is unchanged from the kernel split (comb[j,k] <-
 *                 mixes[2n + j*n+k]); only the apply indexing matters.
 *
 *   hc_head (model.py:728-735): own fn [n, n*d], scalar scale, base [n];
 *       pre[j] = sigmoid(mixes[j] * scale + base[j]) + hc_eps
 *       y[i]   = sum_j pre[j] * X[j,i]        (no post/comb, no Sinkhorn)
 *
 * The sigmoid is 1/(1+expf(-x)) in FP32; the row softmax subtracts the row
 * max exactly like the reference, so overflow needs no extra guarding.
 *
 * Usage: #define APUS_MHC_IMPLEMENTATION in exactly one TU. Scalar paths
 * always compiled; NEON paths under __ARM_NEON, checked against scalar in
 * tests/m4a. This header is self-contained (does not need fp4.h/fp8.h).
 */
#ifndef APUS_MHC_H
#define APUS_MHC_H

#include <stddef.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_MHC_MULT 4u   /* hc_mult in DeepSeek-V4-Flash (kernels are n-generic) */

/* rsqrt(mean(x^2) + eps) over n floats. */
float apus_mhc_rsqrt_scalar(const float *x, size_t n, float eps);

/* mixes[j] = (dot(fn + j*nx, x, nx)) * rsqrt, j < nmix.
 * FP32 dot; rsqrt applied AFTER the dot (reference order). */
void apus_mhc_mixes_scalar(const float *x, size_t nx,
                           const float *fn, float rsqrt,
                           float *mixes, size_t nmix);

/* sigmoid(x) = 1/(1+expf(-x)), FP32. */
float apus_mhc_sigmoid(float x);

/* Sinkhorn steps on a row-major n x n matrix (exposed individually so tests
 * can verify the reference iteration-for-iteration):
 * row softmax (max-subtracted) then per-element +eps. */
void apus_mhc_row_softmax_eps(float *c, size_t n, float eps);
/* c[j,k] /= rowsum[j] + eps (eps added to the sum). */
void apus_mhc_norm_rows_eps(float *c, size_t n, float eps);
/* c[j,k] /= colsum[k] + eps (eps added to the sum). */
void apus_mhc_norm_cols_eps(float *c, size_t n, float eps);
/* Full driver: row softmax+eps; col norm; (iters-1) x (row norm; col norm). */
void apus_mhc_sinkhorn(float *c, size_t n, int iters, float eps);

/* Per-token pre/post/comb generation from state x4 [n*d].
 * fn: [(2+n)*n, n*d] row-major FP32; scale: [3]; base: [(2+n)*n].
 * Outputs: pre [n], post [n], comb [n*n] (post-Sinkhorn).
 * mixes: scratch [(2+n)*n], may be NULL (then a stack buffer is used for
 * n == APUS_MHC_MULT; pass a buffer for other n). */
void apus_mhc_prepost_scalar(const float *x4, size_t d, size_t n,
                             const float *fn, const float *scale,
                             const float *base, float norm_eps, float hc_eps,
                             int iters, float *pre, float *post, float *comb,
                             float *mixes);

/* Collapse: y[i] = sum_j pre[j] * x4[j*d+i], i < d. */
void apus_mhc_collapse_scalar(const float *x4, const float *pre,
                              float *y, size_t d, size_t n);

/* Apply/expand: y4[j*d+i] = post[j]*x[i] + sum_k comb[k*n+j]*res[k*d+i].
 * comb is indexed [residual k][output j] (reference convention,
 * model.py:685). */
void apus_mhc_apply_scalar(const float *x, const float *res,
                           const float *post, const float *comb,
                           float *y4, size_t d, size_t n);

/* hc_head collapse (no Sinkhorn): fn [n, n*d], scalar scale, base [n].
 * y[i] = sum_j (sigmoid(mixes[j]*scale + base[j]) + hc_eps) * x4[j*d+i].
 * mixes: scratch [n], may be NULL for n == APUS_MHC_MULT. */
void apus_mhc_head_scalar(const float *x4, size_t d, size_t n,
                          const float *fn, float scale, const float *base,
                          float norm_eps, float hc_eps, float *y,
                          float *mixes);

#ifdef __ARM_NEON
float apus_mhc_rsqrt_neon(const float *x, size_t n, float eps);
void apus_mhc_mixes_neon(const float *x, size_t nx,
                         const float *fn, float rsqrt,
                         float *mixes, size_t nmix);
void apus_mhc_prepost_neon(const float *x4, size_t d, size_t n,
                           const float *fn, const float *scale,
                           const float *base, float norm_eps, float hc_eps,
                           int iters, float *pre, float *post, float *comb,
                           float *mixes);
void apus_mhc_collapse_neon(const float *x4, const float *pre,
                            float *y, size_t d, size_t n);
void apus_mhc_apply_neon(const float *x, const float *res,
                         const float *post, const float *comb,
                         float *y4, size_t d, size_t n);
void apus_mhc_head_neon(const float *x4, size_t d, size_t n,
                        const float *fn, float scale, const float *base,
                        float norm_eps, float hc_eps, float *y,
                        float *mixes);
#endif

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MHC_IMPLEMENTATION

#include <math.h>
#include <string.h>

float apus_mhc_rsqrt_scalar(const float *x, size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float mean = ss / (float)n;
    return 1.0f / sqrtf(mean + eps);
}

void apus_mhc_mixes_scalar(const float *x, size_t nx,
                           const float *fn, float rsqrt,
                           float *mixes, size_t nmix) {
    for (size_t j = 0; j < nmix; j++) {
        const float *f = fn + j * nx;
        float dot = 0.0f;
        for (size_t i = 0; i < nx; i++) dot += f[i] * x[i];
        mixes[j] = dot * rsqrt;   /* rsqrt AFTER the dot (reference order) */
    }
}

float apus_mhc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void apus_mhc_row_softmax_eps(float *c, size_t n, float eps) {
    for (size_t j = 0; j < n; j++) {
        float *r = c + j * n;
        float mx = r[0];
        for (size_t k = 1; k < n; k++) if (r[k] > mx) mx = r[k];
        float sum = 0.0f;
        for (size_t k = 0; k < n; k++) { r[k] = expf(r[k] - mx); sum += r[k]; }
        for (size_t k = 0; k < n; k++) r[k] = r[k] / sum + eps;   /* per-element +eps */
    }
}

void apus_mhc_norm_rows_eps(float *c, size_t n, float eps) {
    for (size_t j = 0; j < n; j++) {
        float *r = c + j * n;
        float sum = 0.0f;
        for (size_t k = 0; k < n; k++) sum += r[k];
        float den = sum + eps;   /* eps in the denominator */
        for (size_t k = 0; k < n; k++) r[k] /= den;
    }
}

void apus_mhc_norm_cols_eps(float *c, size_t n, float eps) {
    for (size_t k = 0; k < n; k++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) sum += c[j * n + k];
        float den = sum + eps;   /* eps in the denominator */
        for (size_t j = 0; j < n; j++) c[j * n + k] /= den;
    }
}

void apus_mhc_sinkhorn(float *c, size_t n, int iters, float eps) {
    apus_mhc_row_softmax_eps(c, n, eps);
    apus_mhc_norm_cols_eps(c, n, eps);
    for (int it = 1; it < iters; it++) {
        apus_mhc_norm_rows_eps(c, n, eps);
        apus_mhc_norm_cols_eps(c, n, eps);
    }
}

/* Split the mixes into pre/post/comb with gates (shared by scalar/NEON). */
static void apus_mhc_split_gates(const float *mixes, size_t n,
                                 const float *scale, const float *base,
                                 float hc_eps,
                                 float *pre, float *post, float *comb) {
    for (size_t j = 0; j < n; j++)
        pre[j] = apus_mhc_sigmoid(mixes[j] * scale[0] + base[j]) + hc_eps;
    for (size_t j = 0; j < n; j++)
        post[j] = 2.0f * apus_mhc_sigmoid(mixes[n + j] * scale[1] + base[n + j]);
    for (size_t j = 0; j < n; j++)
        for (size_t k = 0; k < n; k++) {
            size_t idx = 2 * n + j * n + k;
            comb[j * n + k] = mixes[idx] * scale[2] + base[idx];
        }
}

void apus_mhc_prepost_scalar(const float *x4, size_t d, size_t n,
                             const float *fn, const float *scale,
                             const float *base, float norm_eps, float hc_eps,
                             int iters, float *pre, float *post, float *comb,
                             float *mixes) {
    size_t nmix = (2 + n) * n, nx = n * d;
    float stack_mixes[(2 + APUS_MHC_MULT) * APUS_MHC_MULT];
    if (!mixes) mixes = stack_mixes;   /* caller passes a buffer for n != 4 */
    float rsqrt = apus_mhc_rsqrt_scalar(x4, nx, norm_eps);
    apus_mhc_mixes_scalar(x4, nx, fn, rsqrt, mixes, nmix);
    apus_mhc_split_gates(mixes, n, scale, base, hc_eps, pre, post, comb);
    apus_mhc_sinkhorn(comb, n, iters, hc_eps);
}

void apus_mhc_collapse_scalar(const float *x4, const float *pre,
                              float *y, size_t d, size_t n) {
    for (size_t i = 0; i < d; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; j++) acc += pre[j] * x4[j * d + i];
        y[i] = acc;
    }
}

void apus_mhc_apply_scalar(const float *x, const float *res,
                           const float *post, const float *comb,
                           float *y4, size_t d, size_t n) {
    for (size_t j = 0; j < n; j++) {
        float *y = y4 + j * d;
        for (size_t i = 0; i < d; i++) {
            float acc = post[j] * x[i];
            for (size_t k = 0; k < n; k++) acc += comb[k * n + j] * res[k * d + i];
            y[i] = acc;
        }
    }
}

void apus_mhc_head_scalar(const float *x4, size_t d, size_t n,
                          const float *fn, float scale, const float *base,
                          float norm_eps, float hc_eps, float *y,
                          float *mixes) {
    size_t nx = n * d;
    float stack_mixes[APUS_MHC_MULT];
    if (!mixes) mixes = stack_mixes;
    float rsqrt = apus_mhc_rsqrt_scalar(x4, nx, norm_eps);
    apus_mhc_mixes_scalar(x4, nx, fn, rsqrt, mixes, n);
    /* reuse the mixes buffer for pre (mixes not needed afterwards) */
    for (size_t j = 0; j < n; j++)
        mixes[j] = apus_mhc_sigmoid(mixes[j] * scale + base[j]) + hc_eps;
    apus_mhc_collapse_scalar(x4, mixes, y, d, n);
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

float apus_mhc_rsqrt_neon(const float *x, size_t n, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float ss = vaddvq_f32(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    float mean = ss / (float)n;
    return 1.0f / sqrtf(mean + eps);
}

void apus_mhc_mixes_neon(const float *x, size_t nx,
                         const float *fn, float rsqrt,
                         float *mixes, size_t nmix) {
    for (size_t j = 0; j < nmix; j++) {
        const float *f = fn + j * nx;
        float32x4_t acc = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= nx; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(f + i), vld1q_f32(x + i));
        float dot = vaddvq_f32(acc);
        for (; i < nx; i++) dot += f[i] * x[i];
        mixes[j] = dot * rsqrt;
    }
}

void apus_mhc_prepost_neon(const float *x4, size_t d, size_t n,
                           const float *fn, const float *scale,
                           const float *base, float norm_eps, float hc_eps,
                           int iters, float *pre, float *post, float *comb,
                           float *mixes) {
    size_t nmix = (2 + n) * n, nx = n * d;
    float stack_mixes[(2 + APUS_MHC_MULT) * APUS_MHC_MULT];
    if (!mixes) mixes = stack_mixes;
    float rsqrt = apus_mhc_rsqrt_neon(x4, nx, norm_eps);
    apus_mhc_mixes_neon(x4, nx, fn, rsqrt, mixes, nmix);
    /* gates + Sinkhorn are 4x4 scalar work: identical code to the scalar
     * path, so the only FP32 summation-order difference is in mixes */
    apus_mhc_split_gates(mixes, n, scale, base, hc_eps, pre, post, comb);
    apus_mhc_sinkhorn(comb, n, iters, hc_eps);
}

void apus_mhc_collapse_neon(const float *x4, const float *pre,
                            float *y, size_t d, size_t n) {
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t j = 0; j < n; j++)
            acc = vfmaq_f32(acc, vdupq_n_f32(pre[j]), vld1q_f32(x4 + j * d + i));
        vst1q_f32(y + i, acc);
    }
    for (; i < d; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; j++) acc += pre[j] * x4[j * d + i];
        y[i] = acc;
    }
}

void apus_mhc_apply_neon(const float *x, const float *res,
                         const float *post, const float *comb,
                         float *y4, size_t d, size_t n) {
    for (size_t j = 0; j < n; j++) {
        float *y = y4 + j * d;
        float32x4_t pj = vdupq_n_f32(post[j]);
        size_t i = 0;
        for (; i + 4 <= d; i += 4) {
            float32x4_t acc = vmulq_f32(pj, vld1q_f32(x + i));
            for (size_t k = 0; k < n; k++)
                acc = vfmaq_f32(acc, vdupq_n_f32(comb[k * n + j]),
                                vld1q_f32(res + k * d + i));
            vst1q_f32(y + i, acc);
        }
        for (; i < d; i++) {
            float acc = post[j] * x[i];
            for (size_t k = 0; k < n; k++) acc += comb[k * n + j] * res[k * d + i];
            y[i] = acc;
        }
    }
}

void apus_mhc_head_neon(const float *x4, size_t d, size_t n,
                        const float *fn, float scale, const float *base,
                        float norm_eps, float hc_eps, float *y,
                        float *mixes) {
    size_t nx = n * d;
    float stack_mixes[APUS_MHC_MULT];
    if (!mixes) mixes = stack_mixes;
    float rsqrt = apus_mhc_rsqrt_neon(x4, nx, norm_eps);
    apus_mhc_mixes_neon(x4, nx, fn, rsqrt, mixes, n);
    for (size_t j = 0; j < n; j++)
        mixes[j] = apus_mhc_sigmoid(mixes[j] * scale + base[j]) + hc_eps;
    apus_mhc_collapse_neon(x4, mixes, y, d, n);
}

#endif /* __ARM_NEON */
#endif /* APUS_MHC_IMPLEMENTATION */
#endif /* APUS_MHC_H */
