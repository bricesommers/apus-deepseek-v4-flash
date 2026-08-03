/*
 * c/moe.h — DeepSeek-V4-Flash MoE sublayer (M4c): sqrtsoftplus router with
 * selection-only bias (or hash routing via tid2eid), MXFP4 routed-expert
 * dispatch with FP32 SwiGLU, FP8 shared expert, FP32 weighted accumulation.
 * C11, libc + arm_neon.h only.
 *
 * Normative reference: tools/oracle.py (f32 mode) porting
 * reference/inference/model.py:546-644. Semantics pins
 * (tests/m4b/README.md stage 13-14):
 *   - scores = x @ gate_w^T in FP32 (no rounding); sqrtsoftplus =
 *     sqrt(F.softplus(x, threshold=20)).
 *   - gate.bias added for top-k SELECTION only (A6 stable tie-break);
 *     weights gathered from the UNBIASED scores, normalized to sum 1,
 *     x route_scale (1.5). Hash-routed layers (swa) take the expert indices
 *     from tid2eid[input_id] but the weights still come from the scores.
 *   - routed experts: MXFP4 GEMMs (fp4.h normative path), FP32 SwiGLU with
 *     up clamped to +-limit and gate to max +limit, BF16 round before w2;
 *     shared expert: FP8 GEMMs (fp8.h), same SwiGLU.
 *   - FP32 accumulation across routed experts; shared added; BF16 out.
 *
 * Usage: #define APUS_MOE_IMPLEMENTATION in exactly one TU (needs the
 * fp4/fp8/attn implementations linked for the GEMM + linear helpers).
 */
#ifndef APUS_MOE_H
#define APUS_MOE_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"
#include "attn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int E, topk, inter, dim, hash;
    float route_scale, limit;
    const float *gate_w;     /* [E, dim] BF16-valued f32 */
    const float *gate_bias;  /* [E] f32, NULL when hash */
    const int64_t *tid2eid;  /* [vocab, topk], NULL unless hash */
    ApusFp4W *w1, *w2, *w3;  /* [E] routed experts (MXFP4) */
    ApusFp8W sw1, sw2, sw3;  /* shared expert (FP8) */
    /* Optional expert-resolve hook (M6a tiering, c/cache.h). When
     * hook_resolve is set, expert weights come from the hook instead of the
     * w1/w2/w3 arrays (which may be empty); hook_hint is called for each
     * token's routed experts BEFORE the compute loop (miss overlap /
     * batch-union), hook_layer_end after the sublayer (end-of-block
     * promotion). Numerics are unchanged: the hook only supplies the same
     * slab bytes the arrays would have pointed at. */
    void *hook_ctx;
    int  (*hook_resolve)(void *ctx, int layer, int eid,
                         ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3);
    void (*hook_hint)(void *ctx, int layer, int eid);
    void (*hook_layer_end)(void *ctx, int layer);
    /* Optional separate contexts for hint/layer_end (M6b pilot wraps those
     * two hooks while leaving resolve on the store's ctx). NULL = use
     * hook_ctx. */
    void *hook_hint_ctx;
    void *hook_layer_end_ctx;
    int layer_id;
} ApusMoeW;

/* Named intermediates (NULL to skip). Buffers: scores/biased [s,E],
 * router_w [s,topk], router_idx [s,topk], moe_routed/moe_shared [s,dim]. */
typedef struct {
    float *router_scores, *router_scores_biased, *router_w;
    int32_t *router_idx;
    float *moe_routed, *moe_shared;
} ApusMoeInterm;

/* Router scores only (M6b pilot reuses this — the SAME code path as the
 * MoE gate, do not duplicate the math): sp = sqrtsoftplus(x @ gate_w^T)
 * [s,E] FP32; biased = sp + gate_bias for top-k SELECTION (non-hash only;
 * pass NULL for hash layers / when not needed). Buffers [s,E]. */
void apus_router_score(const ApusMoeW *w, const float *x, int s,
                       float *sp, float *biased);

/* MoE sublayer. x [s, dim] (BF16 values), ids [s] input token ids (hash
 * routing), out [s, dim] BF16-rounded. */
void apus_moe_forward(const ApusMoeW *w, const float *x, const int64_t *ids,
                      int s, float *out, ApusMoeInterm *interm);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MOE_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* torch F.softplus (beta=1, threshold=20). */
static float apus_softplus(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* Expert SwiGLU (model.py:596-606) over n rows' worth of gate/up buffers
 * (elementwise — n = rows*inter; the per-element op sequence is identical
 * for any n, so the batched prefill path is bitwise equal to M=1 calls).
 * M9d: element ranges pooled at larger n (elementwise, per-element order
 * fixed — bitwise identical for any thread count). */
typedef struct {
    const float *g, *u;
    float *h;
    float limit;
} ApusSwigluJob;

static void apus_expert_swiglu_range(void *vjob, size_t i0, size_t i1) {
    const ApusSwigluJob *j = vjob;
    float limit = j->limit;
    for (size_t i = i0; i < i1; i++) {
        float gv = j->g[i], uv = j->u[i];
        if (limit > 0.0f) {
            if (uv > limit) uv = limit;
            else if (uv < -limit) uv = -limit;
            if (gv > limit) gv = limit;
        }
        float hv = (gv * (1.0f / (1.0f + expf(-gv)))) * uv;
        j->h[i] = apus_bf16_round(hv);
    }
}

static void apus_expert_swiglu(const float *g, const float *u, float *h,
                               size_t n, float limit) {
    ApusSwigluJob j = { g, u, h, limit };
    if (n >= 16384)
        apus_pool_run(n, apus_expert_swiglu_range, &j);
    else
        apus_expert_swiglu_range(&j, 0, n);
}

/* M9d: per-token weighted expert accumulation (prefill batch path): for
 * each token t, y[t] += wgt[t,j] * eo_all[t,j] over j in the exact (t,j)
 * order of the M=1 path. Rows are token-independent and the per-element
 * accumulation keeps its j order, so pooling over tokens is bitwise
 * identical for any thread count. */
typedef struct {
    const float *wgt, *eo_all;
    float *y;
    int topk, dim;
} ApusMoeAccumJob;

static void apus_moe_accum_rows(void *vjob, size_t t0, size_t t1) {
    const ApusMoeAccumJob *j = vjob;
    int topk = j->topk, dim = j->dim;
    for (size_t t = t0; t < t1; t++)
        for (int jj = 0; jj < topk; jj++) {
            float wt = j->wgt[t * (size_t)topk + jj];
            const float *eov = j->eo_all + (t * (size_t)topk + jj) * dim;
            for (int i = 0; i < dim; i++)
                j->y[t * (size_t)dim + i] += wt * eov[i];
        }
}

/* M9e: routed-expert dispatch grouping. The M9b batched path ran each
 * expert's three linears as separate apus_fp4_linear calls — ~5-7 pool
 * dispatches per expert (act quant x2, three GEMMs, SwiGLU when pooled),
 * ~108 experts per layer at real-prompt batch sizes. Grouping runs a
 * SMALL batch of experts (all with count < APUS_BLAS_M_MIN) through ONE
 * act quant, ONE grouped GEMM dispatch per matrix (w1, w3, w2), ONE
 * SwiGLU, and pooled BF16 rounds (c/fp4.h apus_fp4_gemm_mt_grouped).
 * Scheduling only: every per-row op sequence is EXACTLY the per-expert
 * loop's (the w1/w3 act quant of the same xg rows was already computed
 * twice identically; computing it once is bitwise the same value), so
 * outputs are bitwise identical for any thread count and bitwise
 * identical to the per-expert path. Experts with count >
 * APUS_MOE_GROUP_ROWS run solo (the pre-M9e path; count >= 256 still
 * takes the M9b BLAS dispatch there).
 *
 * Group size is cache-bound, measured (tests/m9e/README.md): the m-block
 * passes of the M9a kernel re-read each weight slab ~count/4 times, so a
 * dispatch is fast only while the slabs it streams concurrently stay
 * L2-resident. ONE slab per dispatch (the M9d granularity) is optimal;
 * ANY multi-expert GEMM dispatch is a net LOSS here (512-tok prefill,
 * fixed code: MAX=2 matrix-split 57.9s, MAX=1 merged w1+w3 55.6s, vs
 * M9d 55.0-56.0s; an earlier sweep that suggested MAX=2 won carried an
 * entry-copy bug that re-read the same slab L2-hot). APUS_MOE_GROUP_MAX
 * is therefore 1: GEMM dispatches stay one slab per matrix per expert,
 * and the retained wins are the scheduling-only work around them — the
 * act quant of xg computed ONCE for both w1 and w3 (was twice, the same
 * value), and the BF16 output rounds POOLED instead of serial per-linear
 * loops. */
#define APUS_MOE_GROUP_ROWS 128
#define APUS_MOE_GROUP_MAX  1
/* 1 = w1+w3 share one grouped dispatch; 0 = one dispatch per matrix
 * (fewer slabs concurrently streamed; see tests/m9e/README.md). */
#define APUS_MOE_MERGE_W13  0

/* Elementwise BF16 round, pooled at larger n (same rounding as the
 * per-element loops inside apus_fp4_linear — bitwise identical). */
typedef struct { float *v; } ApusMoeRoundJob;

static void apus_moe_round_range(void *vjob, size_t i0, size_t i1) {
    float *v = ((ApusMoeRoundJob *)vjob)->v;
    for (size_t i = i0; i < i1; i++) v[i] = apus_bf16_round(v[i]);
}

static void apus_moe_round(float *v, size_t n) {
    ApusMoeRoundJob j = { v };
    if (n >= 16384)
        apus_pool_run(n, apus_moe_round_range, &j);
    else
        apus_moe_round_range(&j, 0, n);
}

/* Run one routed-expert group: resolve + gather, ONE act quant of the
 * gathered rows, ONE grouped w1+w3 GEMM dispatch, BF16 rounds, ONE
 * SwiGLU, ONE act quant of h, ONE grouped w2 GEMM, scatter to eo_all.
 * ge[i] = expert id, rc[i] = its row count, roff[i] = its first row in
 * the group's row space (grows = total rows). Buffers are caller-owned
 * (sized APUS_MOE_GROUP_ROWS rows). */
static void apus_moe_expert_group(const ApusMoeW *w, const int *ge,
                                  const int *rc, const size_t *roff, int gn,
                                  size_t grows, const int *cnt,
                                  const int32_t *slot, const float *x,
                                  float *eo_all, float *xg, float *xb,
                                  uint8_t *codes, float *asx, float *scr,
                                  float *g, float *u, float *h, float *hb,
                                  uint8_t *codes2, float *as2, float *eo) {
    int dim = w->dim, inter = w->inter, topk = w->topk;
    size_t nab_d = apus_fp4_act_blocks((size_t)dim);
    size_t nab_i = apus_fp4_act_blocks((size_t)inter);
    ApusFp4W r1[APUS_MOE_GROUP_MAX], r2[APUS_MOE_GROUP_MAX],
             r3[APUS_MOE_GROUP_MAX];
    for (int i = 0; i < gn; i++) {
        int e = ge[i];
        if (w->hook_resolve) {
            w->hook_resolve(w->hook_ctx, w->layer_id, e,
                            &r1[i], &r2[i], &r3[i]);
        } else {
            r1[i] = w->w1[e];
            r2[i] = w->w2[e];
            r3[i] = w->w3[e];
        }
        int c0 = cnt[e];
        for (int q = 0; q < rc[i]; q++)
            memcpy(xg + (roff[i] + (size_t)q) * (size_t)dim,
                   x + (size_t)(slot[c0 + q] / topk) * (size_t)dim,
                   (size_t)dim * sizeof(float));
    }
    /* one act quant for both w1 and w3 (the per-expert path computed the
     * same codes twice; per-row quant math is unchanged) */
    {
        ApusActQuantJob aq = { xg, xb, codes, asx, dim, nab_d };
        if (grows >= APUS_ROW_POOL_MIN)
            apus_pool_run(grows, apus_act_quant_rows, &aq);
        else
            apus_act_quant_rows(&aq, 0, grows);
    }
    /* Work-balance the dispatch unit space: the pool partitions it into
     * contiguous spans, and a span's cost is proportional to the SUM of
     * the row counts of the entries it covers — with experts in ascending
     * eid order that sum varies by >10x between lanes (the whole group
     * then runs at the slowest lane's pace, which is what cvwait showed).
     * Ordering the entries high-low by row count (heaviest, lightest,
     * 2nd-heaviest, 2nd-lightest, ...) gives every contiguous span a mix.
     * Each output row is still computed entirely by one lane with the
     * identical per-row math, so results are bitwise identical for any
     * thread count; only the lane assignment of rows changes. */
    int ord[APUS_MOE_GROUP_MAX];
    {
        int idx[APUS_MOE_GROUP_MAX];
        for (int i = 0; i < gn; i++) idx[i] = i;
        for (int i = 1; i < gn; i++) {          /* insertion sort, rc desc */
            int v = idx[i], j = i - 1;
            while (j >= 0 && rc[idx[j]] < rc[v]) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = v;
        }
        int lo = 0, hi = gn - 1, p = 0;
        while (lo <= hi) {
            ord[p++] = idx[lo++];
            if (lo <= hi) ord[p++] = idx[hi--];
        }
    }
    ApusFp4GemmEnt ents[2 * APUS_MOE_GROUP_MAX];
    for (int p = 0; p < gn; p++) {
        int i = ord[p];
        ents[2 * p]     = (ApusFp4GemmEnt){ r1[i].packed, r1[i].scales,
                              g + roff[i] * (size_t)inter,
                              roff[i], (size_t)rc[i] };
        ents[2 * p + 1] = (ApusFp4GemmEnt){ r3[i].packed, r3[i].scales,
                              u + roff[i] * (size_t)inter,
                              roff[i], (size_t)rc[i] };
    }
#if APUS_MOE_MERGE_W13
    apus_fp4_gemm_mt_grouped(ents, (size_t)(2 * gn), codes, asx, scr,
                             (size_t)inter, (size_t)dim);
#else
    /* one dispatch per matrix (fewer slabs concurrently streamed): split
     * the interleaved array into per-matrix arrays */
    {
        ApusFp4GemmEnt e1[APUS_MOE_GROUP_MAX], e3[APUS_MOE_GROUP_MAX];
        for (int p = 0; p < gn; p++) { e1[p] = ents[2 * p]; e3[p] = ents[2 * p + 1]; }
        apus_fp4_gemm_mt_grouped(e1, (size_t)gn, codes, asx, scr,
                                 (size_t)inter, (size_t)dim);
        apus_fp4_gemm_mt_grouped(e3, (size_t)gn, codes, asx, scr,
                                 (size_t)inter, (size_t)dim);
    }
#endif
    apus_moe_round(g, grows * (size_t)inter);
    apus_moe_round(u, grows * (size_t)inter);
    apus_expert_swiglu(g, u, h, grows * (size_t)inter, w->limit);
    {
        ApusActQuantJob aq = { h, hb, codes2, as2, inter, nab_i };
        if (grows >= APUS_ROW_POOL_MIN)
            apus_pool_run(grows, apus_act_quant_rows, &aq);
        else
            apus_act_quant_rows(&aq, 0, grows);
    }
    for (int p = 0; p < gn; p++) {
        int i = ord[p];
        ents[p] = (ApusFp4GemmEnt){ r2[i].packed, r2[i].scales,
                      eo + roff[i] * (size_t)dim, roff[i], (size_t)rc[i] };
    }
    apus_fp4_gemm_mt_grouped(ents, (size_t)gn, codes2, as2, scr,
                             (size_t)dim, (size_t)inter);
    apus_moe_round(eo, grows * (size_t)dim);
    for (int i = 0; i < gn; i++) {
        int c0 = cnt[ge[i]];
        for (int q = 0; q < rc[i]; q++)
            memcpy(eo_all + (size_t)slot[c0 + q] * (size_t)dim,
                   eo + (roff[i] + (size_t)q) * (size_t)dim,
                   (size_t)dim * sizeof(float));
    }
}

/* One expert SwiGLU (model.py:596-606). kind: 0 = fp4, 1 = fp8. */
static void apus_expert(const void *w1, const void *w2, const void *w3,
                        int kind, float limit, const float *x, int dim,
                        int inter, float *out) {    ApusScratchMark mk = apus_scratch_mark();
    float *g = apus_scratch_alloc((size_t)inter * sizeof(float));
    float *u = apus_scratch_alloc((size_t)inter * sizeof(float));
    float *h = apus_scratch_alloc((size_t)inter * sizeof(float));
    if (kind == 0) {
        apus_fp4_linear(w1, x, g, 1, dim, inter);
        apus_fp4_linear(w3, x, u, 1, dim, inter);
    } else {
        apus_fp8_linear(w1, x, g, 1, dim, inter);
        apus_fp8_linear(w3, x, u, 1, dim, inter);
    }
    apus_expert_swiglu(g, u, h, (size_t)inter, limit);
    if (kind == 0) apus_fp4_linear(w2, h, out, 1, inter, dim);
    else           apus_fp8_linear(w2, h, out, 1, inter, dim);
    apus_scratch_reset(mk);
}

void apus_router_score(const ApusMoeW *w, const float *x, int s,
                       float *sp, float *biased) {
    int E = w->E, dim = w->dim;
    ApusScratchMark mk = apus_scratch_mark();
    float *scores = apus_scratch_alloc((size_t)s * E * sizeof(float));
    /* M7b backend hook: plain FP32 matmul, no rounding (c/backend_metal.h) */
    if (!(apus_backend_hooks.f32_linear
          && apus_backend_hooks.f32_linear(w->gate_w, x, scores, s, dim, E,
                                           0) == 0))
        apus_f32_linear(w->gate_w, x, scores, s, dim, E);
    for (int i = 0; i < s * E; i++)
        sp[i] = sqrtf(apus_softplus(scores[i]));
    if (biased)
        for (int i = 0; i < s * E; i++) biased[i] = sp[i] + w->gate_bias[i % E];
    apus_scratch_reset(mk);
}

void apus_moe_forward(const ApusMoeW *w, const float *x, const int64_t *ids,
                      int s, float *out, ApusMoeInterm *interm) {
    int E = w->E, topk = w->topk, dim = w->dim, inter = w->inter;
    ApusScratchMark mk = apus_scratch_mark();
    float *sp = apus_scratch_alloc((size_t)s * E * sizeof(float));
    float *biased = w->hash ? NULL
        : apus_scratch_alloc((size_t)s * E * sizeof(float));
    int32_t *idx = apus_scratch_alloc((size_t)s * topk * sizeof(int32_t));
    float *wgt = apus_scratch_alloc((size_t)s * topk * sizeof(float));

    /* gate (model.py:564-584) */
    apus_router_score(w, x, s, sp, biased);
    if (interm && interm->router_scores)
        memcpy(interm->router_scores, sp, (size_t)s * E * sizeof(float));
    if (!w->hash) {
        if (interm && interm->router_scores_biased)
            memcpy(interm->router_scores_biased, biased,
                   (size_t)s * E * sizeof(float));
        for (int t = 0; t < s; t++)
            apus_topk_stable(biased + (size_t)t * E, E, topk,
                             idx + (size_t)t * topk);
    } else {
        for (int t = 0; t < s; t++)
            for (int j = 0; j < topk; j++)
                idx[(size_t)t * topk + j] =
                    (int32_t)w->tid2eid[ids[t] * topk + j];
    }
    /* weights from UNBIASED scores, normalized, x route_scale */
    for (int t = 0; t < s; t++) {
        float sum = 0.0f;
        for (int j = 0; j < topk; j++) {
            wgt[(size_t)t * topk + j] = sp[(size_t)t * E + idx[(size_t)t * topk + j]];
            sum += wgt[(size_t)t * topk + j];
        }
        for (int j = 0; j < topk; j++)
            wgt[(size_t)t * topk + j] =
                wgt[(size_t)t * topk + j] / sum * w->route_scale;
    }
    if (interm) {
        if (interm->router_idx)
            memcpy(interm->router_idx, idx, (size_t)s * topk * sizeof(int32_t));
        if (interm->router_w)
            memcpy(interm->router_w, wgt, (size_t)s * topk * sizeof(float));
    }

    /* routed experts, FP32 accumulation (model.py:633-640).
     * With a resolve hook (M6a tiering): submit all unique routed experts
     * of this block first (batch-union miss overlap), then resolve
     * just-in-time per expert — same slab bytes, same kernels, same
     * accumulation order as the eager path. */
    if (w->hook_resolve && w->hook_hint) {
        uint64_t *seen = apus_scratch_alloc(((size_t)E + 63) / 64 * sizeof(uint64_t));
        memset(seen, 0, ((size_t)E + 63) / 64 * sizeof(uint64_t));
        for (int t = 0; t < s; t++)
            for (int j = 0; j < topk; j++) {
                int e = idx[(size_t)t * topk + j];
                if (!(seen[e >> 6] & (1ull << (e & 63)))) {
                    seen[e >> 6] |= 1ull << (e & 63);
                    w->hook_hint(w->hook_hint_ctx ? w->hook_hint_ctx
                                                  : w->hook_ctx,
                               w->layer_id, e);
                }
            }
    }
    float *y = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    memset(y, 0, (size_t)s * dim * sizeof(float));
    if (s == 1) {
        float *eo = apus_scratch_alloc((size_t)dim * sizeof(float));
        for (int j = 0; j < topk; j++) {
            int e = idx[j];
            if (w->hook_resolve) {
                ApusFp4W e1, e2, e3;
                w->hook_resolve(w->hook_ctx, w->layer_id, e, &e1, &e2, &e3);
                apus_expert(&e1, &e2, &e3, 0, w->limit, x, dim, inter, eo);
            } else {
                apus_expert(&w->w1[e], &w->w2[e], &w->w3[e], 0, w->limit,
                            x, dim, inter, eo);
            }
            float wt = wgt[j];
            for (int i = 0; i < dim; i++)
                y[i] += wt * eo[i];
        }
    } else {
        /* M9b batched prefill: group the (t,j) routings by expert, run each
         * expert's three linears ONCE at M=count (weight slab streamed once
         * per layer instead of once per token), then accumulate per token in
         * the EXACT (t,j) order of the M=1 path. Bitwise identical to the
         * per-token loop: the fp4 GEMM rows are M-independent bitwise (M9a,
         * apus_fp4_gemm_rows_neon), act quant/SwiGLU are per-row elementwise,
         * and the y accumulation below is the same per-element expression in
         * the same order. Resolve order (ascending eid, once per unique
         * expert) only changes store-internal accounting, not slab bytes.
         * M9e: experts with count <= APUS_MOE_GROUP_ROWS are batched into
         * groups sharing one quant/GEMM/SwiGLU dispatch set per group
         * (apus_moe_expert_group; scheduling only, bitwise by the same
         * argument); larger counts run the solo per-expert path below. */
        size_t nj = (size_t)s * topk;
        int *cnt = apus_scratch_alloc(((size_t)E + 1) * sizeof(int));
        int *cur = apus_scratch_alloc((size_t)E * sizeof(int));
        int32_t *slot = apus_scratch_alloc(nj * sizeof(int32_t));
        float *eo_all = apus_scratch_alloc(nj * (size_t)dim * sizeof(float));
        memset(cnt, 0, ((size_t)E + 1) * sizeof(int));
        for (size_t q = 0; q < nj; q++) cnt[idx[q] + 1]++;
        for (int e = 0; e < E; e++) cnt[e + 1] += cnt[e];
        memcpy(cur, cnt, (size_t)E * sizeof(int));
        for (size_t q = 0; q < nj; q++) slot[cur[idx[q]]++] = (int32_t)q;
        size_t maxc = 1;
        for (int e = 0; e < E; e++)
            if ((size_t)(cnt[e + 1] - cnt[e]) > maxc)
                maxc = (size_t)(cnt[e + 1] - cnt[e]);
        float *xg = apus_scratch_alloc(maxc * (size_t)dim * sizeof(float));
        float *g = apus_scratch_alloc(maxc * (size_t)inter * sizeof(float));
        float *u = apus_scratch_alloc(maxc * (size_t)inter * sizeof(float));
        float *h = apus_scratch_alloc(maxc * (size_t)inter * sizeof(float));
        float *eo = apus_scratch_alloc(maxc * (size_t)dim * sizeof(float));
        /* M9e grouped-path buffers (APUS_MOE_GROUP_ROWS rows; the solo
         * buffers above serve experts with count > GROUP_ROWS). */
        const size_t R = APUS_MOE_GROUP_ROWS;
        size_t nab_d = apus_fp4_act_blocks((size_t)dim);
        size_t nab_i = apus_fp4_act_blocks((size_t)inter);
        size_t kmax = (size_t)dim > (size_t)inter ? (size_t)dim
                                                  : (size_t)inter;
        float *gx = apus_scratch_alloc(R * (size_t)dim * sizeof(float));
        float *gxb = apus_scratch_alloc(R * (size_t)dim * sizeof(float));
        uint8_t *gcodes = apus_scratch_alloc(R * (size_t)dim);
        float *gasx = apus_scratch_alloc(R * nab_d * sizeof(float));
        float *gscr = apus_scratch_alloc(R * kmax * sizeof(float));
        float *gg = apus_scratch_alloc(R * (size_t)inter * sizeof(float));
        float *gu = apus_scratch_alloc(R * (size_t)inter * sizeof(float));
        float *gh = apus_scratch_alloc(R * (size_t)inter * sizeof(float));
        float *ghb = apus_scratch_alloc(R * (size_t)inter * sizeof(float));
        uint8_t *gcodes2 = apus_scratch_alloc(R * (size_t)inter);
        float *gas2 = apus_scratch_alloc(R * nab_i * sizeof(float));
        float *geo = apus_scratch_alloc(R * (size_t)dim * sizeof(float));
        int ge[APUS_MOE_GROUP_MAX], grc[APUS_MOE_GROUP_MAX];
        size_t groff[APUS_MOE_GROUP_MAX];
        int gn = 0;
        size_t grows = 0;
        for (int e = 0; e < E; e++) {
            int c0 = cnt[e], c = cnt[e + 1] - c0;
            if (!c) continue;
            if ((size_t)c <= R &&
                (gn == APUS_MOE_GROUP_MAX || grows + (size_t)c > R)) {
                /* group full: flush */
                apus_moe_expert_group(w, ge, grc, groff, gn, grows, cnt,
                                      slot, x, eo_all, gx, gxb, gcodes,
                                      gasx, gscr, gg, gu, gh, ghb,
                                      gcodes2, gas2, geo);
                gn = 0;
                grows = 0;
            }
            if ((size_t)c <= R) {
                ge[gn] = e;
                grc[gn] = c;
                groff[gn] = grows;
                gn++;
                grows += (size_t)c;
                continue;
            }
            /* solo expert (count > GROUP_ROWS): the pre-M9e per-expert
             * path (count >= APUS_BLAS_M_MIN still dispatches to BLAS
             * inside apus_fp4_linear, exactly as before M9e). */
            const ApusFp4W *e1, *e2, *e3;
            ApusFp4W r1, r2, r3;
            if (w->hook_resolve) {
                w->hook_resolve(w->hook_ctx, w->layer_id, e, &r1, &r2, &r3);
                e1 = &r1; e2 = &r2; e3 = &r3;
            } else {
                e1 = &w->w1[e]; e2 = &w->w2[e]; e3 = &w->w3[e];
            }
            for (int q = 0; q < c; q++)
                memcpy(xg + (size_t)q * dim,
                       x + (size_t)(slot[c0 + q] / topk) * dim,
                       (size_t)dim * sizeof(float));
            apus_fp4_linear(e1, xg, g, c, dim, inter);
            apus_fp4_linear(e3, xg, u, c, dim, inter);
            apus_expert_swiglu(g, u, h, (size_t)c * (size_t)inter, w->limit);
            apus_fp4_linear(e2, h, eo, c, inter, dim);
            for (int q = 0; q < c; q++)
                memcpy(eo_all + (size_t)slot[c0 + q] * dim,
                       eo + (size_t)q * dim, (size_t)dim * sizeof(float));
        }
        if (gn)
            apus_moe_expert_group(w, ge, grc, groff, gn, grows, cnt,
                                  slot, x, eo_all, gx, gxb, gcodes,
                                  gasx, gscr, gg, gu, gh, ghb,
                                  gcodes2, gas2, geo);
        /* M9d: pooled over tokens; per-element (t,j) accumulation order
         * unchanged (see ApusMoeAccumJob). */
        ApusMoeAccumJob aj = { wgt, eo_all, y, topk, dim };
        if (s >= 8)
            apus_pool_run((size_t)s, apus_moe_accum_rows, &aj);
        else
            apus_moe_accum_rows(&aj, 0, (size_t)s);
    }
    if (interm && interm->moe_routed)
        memcpy(interm->moe_routed, y, (size_t)s * dim * sizeof(float));

    /* shared expert (model.py:643); s>1 batched like the routed path
     * (fp8 GEMM rows are M-independent bitwise — one weight stream for the
     * whole block instead of one per token) */
    float *sh = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    if (s == 1) {
        apus_expert(&w->sw1, &w->sw2, &w->sw3, 1, w->limit, x, dim, inter, sh);
    } else {
        float *g = apus_scratch_alloc((size_t)s * inter * sizeof(float));
        float *u = apus_scratch_alloc((size_t)s * inter * sizeof(float));
        float *h = apus_scratch_alloc((size_t)s * inter * sizeof(float));
        apus_fp8_linear(&w->sw1, x, g, s, dim, inter);
        apus_fp8_linear(&w->sw3, x, u, s, dim, inter);
        apus_expert_swiglu(g, u, h, (size_t)s * (size_t)inter, w->limit);
        apus_fp8_linear(&w->sw2, h, sh, s, inter, dim);
    }
    if (interm && interm->moe_shared)
        memcpy(interm->moe_shared, sh, (size_t)s * dim * sizeof(float));
    for (int i = 0; i < s * dim; i++)
        out[i] = apus_bf16_round(y[i] + sh[i]);

    /* end-of-block hook (M6a): working-set promotion happens after all
     * slab views of this sublayer are consumed */
    if (w->hook_layer_end)
        w->hook_layer_end(w->hook_layer_end_ctx ? w->hook_layer_end_ctx
                                                : w->hook_ctx,
                          w->layer_id);

    apus_scratch_reset(mk);
}

#endif /* APUS_MOE_IMPLEMENTATION */
#endif /* APUS_MOE_H */
