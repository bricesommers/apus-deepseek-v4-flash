/*
 * c/mtp.h — M8: DeepSeek-V4-Flash MTP head + speculative decoding.
 *
 * MTP block (reference/inference/model.py:738-766, ARCHITECTURE §3.7):
 * depth-1, V3-style. Tensor namespace "mtp.<idx>.*" in the shard set:
 *   e = enorm(embed(ids))            -> e_proj (FP8 [dim,dim])
 *   x = hnorm(prev 4x-dim mHC hidden) -> h_proj (FP8), per hc row
 *   x = bf16(e_proj(e) + h_proj(x))  -> a full Block (SWA ratio 0, full
 *   256-expert MoE WITH gate.bias, FP4 experts) -> own hc_head collapse
 *   (sigmoid, NO Sinkhorn) -> own norm -> SHARED lm head.
 * The block itself is loaded by the standard c/layer.h machinery
 * (apus_layer_load_named with prefix "mtp.<idx>"); this header adds the
 * e/h glue + its own hc_head/norm + the shared-head logits.
 *
 * Speculative decoding (ApusSpec): draft/verify with EXACT output
 * equivalence to non-speculative decoding. The accept rule: a draft token
 * is accepted iff it equals the MAIN MODEL'S OWN pick at that position —
 * argmax for greedy, the model's own apus_sample() draw for sampled. Every
 * emitted token is sampled from the main model's own logits row (produced
 * by a context that is bitwise chunk-invariant to one-by-one decoding,
 * the m4c/m5 property), consuming exactly one RNG uniform per emitted
 * token in position order; drafts consume no RNG. The emitted stream is
 * therefore bitwise identical to non-speculative decoding for the same
 * seed — greedy included, by construction, not by tolerance. (Chosen over
 * classical rejection sampling: it needs no draft-probability evaluation
 * and reproduces the deterministic per-seed stream exactly; distribution-
 * preserving rejection sampling would only guarantee equality in
 * distribution, not the bitwise stream.)
 *
 * Step shape (depth D = drafts chained per step; batch = D tokens):
 *   invariant: main state fed <= q-1; x_q held (already sampled from a
 *   valid main logits row); h_{q-1}, x_{q-1} known; drafts d1..dD chained
 *   from (h_{q-1}, x_{q-1}) (d_i is the candidate for position q+i-1);
 *   MTP state consistent through q-1 (true pairs only).
 *   1. snapshot main + MTP state (rollback = memcpy restore, no recompute
 *      of the snapshot; see below).
 *   2. verify batch: main forward of [x_q, d2, ..., d_D] at q..q+D-1
 *      (ONE batched call) -> rows R[j] = logits for q+j+1, hiddens H[j].
 *   3. walk: emit x_q; for j=1..D-1 sample x_{q+j} from R[j-1]; accept
 *      d_{j+1} iff d_{j+1} == x_{q+j}; stop at the first mismatch. Full
 *      match -> bonus x_{q+D} from R[D-1].
 *   4. state fixup: fed-true run is b_0..b_matched. Full match: keep the
 *      batch state. Partial: restore the snapshot and re-feed the true
 *      prefix b_0..b_matched in one batched call (bitwise chunk-invariant,
 *      so identical to one-by-one; logits/hiddens from the verify batch
 *      remain valid and are reused). MTP: restore its snapshot, replay the
 *      true pairs (H[j], b_j) j<=matched in one batched MTP forward, then
 *      chain the next D drafts from the replay's last hidden.
 *
 * Rollback design: ApusSnap copies pos + SWA window rings + compressor
 * carries (kv/sc) + compressed-entry counts per layer; restore is a
 * truncation (memcpy + nb reset — cache entries beyond nb are dead
 * capacity, rope tables are stateless). The verify batch's pollution of
 * the rejected tail is fully undone; accepted-prefix state is rebuilt by
 * the batched re-feed, which the m4c chunk-invariance property makes
 * bitwise equal to sequential decoding.
 *
 * Usage: #define APUS_MTP_IMPLEMENTATION in exactly one TU (needs the
 * model.h implementation TU — apus_mtp_forward reuses its statics).
 */
#ifndef APUS_MTP_H
#define APUS_MTP_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"
#include "sample.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- MTP block weights --------------------------------------------------*/

typedef struct {
    const ApusModel *m;       /* parent model (embed/head shared) */
    int idx;                  /* MTP block index (0) */
    ApusLayer block;          /* the full block (ratio 0 SWA, bias gate) */
    ApusFp8W e_proj, h_proj;  /* FP8 dense [dim, dim] views */
    float *enorm_w, *hnorm_w, *norm_w;  /* [dim] owned (bf16 widened) */
    float *hc_head_fn;        /* [hc, hc*dim] owned F32 */
    float *hc_head_base;      /* [hc] owned F32 */
    float hc_head_scale;
} ApusMtp;

/* Load MTP block `idx` ("mtp.<idx>.*") from the model's shard set. The
 * block's MoE expert-store layer id is m->n_layers + idx (attach the M6a
 * store to mt->block.mw afterwards in tiered mode). Returns 0 on success;
 * -1 with err when the model has no MTP tensors (m->n_mtp == 0). */
int  apus_mtp_load(ApusMtp *mt, ApusModel *m, int idx, char *err,
                   size_t errcap);
void apus_mtp_free(ApusMtp *mt);

typedef struct {
    ApusLayerState block;
} ApusMtpState;

void apus_mtp_state_init(ApusMtpState *st, const ApusMtp *mt);
void apus_mtp_state_free(ApusMtpState *st, const ApusMtp *mt);

/* MTP forward (model.py:757-766). h_in [s, hc*dim] previous mHC hidden
 * (main-model post-block hidden, or the MTP block's own when chaining),
 * ids [s] tokens at positions start_pos..start_pos+s-1. Mutates the MTP
 * attention state. Outputs, for the LAST position only: logits [V] (FP32,
 * unrounded — the draft distribution for position start_pos+s) and
 * h_last [hc*dim] (post-block hidden, the chaining input). h_last may be
 * NULL. */
void apus_mtp_forward(const ApusMtp *mt, ApusMtpState *mst,
                      const float *h_in, const int64_t *ids, int s,
                      int64_t start_pos, float *logits, float *h_last);

/* ---- state snapshots (rollback = truncation) -----------------------------*/

typedef struct {
    int64_t pos;
    float *win;               /* [window * head_dim] */
    size_t win_bytes;
    int comp_nb, idx_nb;
    float *comp_kv, *comp_sc; /* [comp.rows * comp.cols] each, or NULL */
    float *idx_kv, *idx_sc;
    size_t comp_bytes, idx_bytes;
} ApusSnapLayer;

typedef struct {
    int n;
    int64_t pos;              /* model-level pos (tokens consumed) */
    ApusSnapLayer *L;         /* [n] */
} ApusSnap;

/* Allocate snapshot storage mirroring the given states. layers/states are
 * the main model's (n = m->n_layers) or the MTP block's (n = 1). */
void apus_snap_alloc(ApusSnap *sn, const ApusLayer *layers,
                     const ApusLayerState *states, int n);
void apus_snap_free(ApusSnap *sn);
void apus_snap_take(ApusSnap *sn, ApusLayerState *states, int64_t pos);
void apus_snap_restore(const ApusSnap *sn, ApusLayerState *states,
                       int64_t *pos);

/* ---- speculative decode engine --------------------------------------------*/

typedef struct ApusSpec {
    const ApusModel *m;
    ApusModelState *st;
    const ApusMtp *mt;        /* NULL iff draft_override set */
    ApusMtpState *mst;
    int depth;                /* D: drafts chained per step (>= 1) */
    float temp, top_p;
    ApusRng *rng;
    void *sample_scratch;
    /* Optional test hook replacing the MTP proposer: fill drafts[0..n-1]
     * (d1..dD candidates for positions q..q+n-1). When set, the MTP block
     * is never run. */
    void (*draft_override)(void *ctx, int64_t q, int64_t *drafts, int n);
    void *draft_ctx;
    /* Optional hook invoked with the verify-batch token ids right before
     * the batched forward (apus.c wires the M6b hash-layer prefetch here).
     * Hint-only; must never touch numerics. */
    void (*pre_batch)(void *ctx, const int64_t *ids, int n, int64_t pos);
    void *pre_batch_ctx;
    /* stats */
    uint64_t emitted;         /* tokens emitted across all steps */
    uint64_t batches;         /* verify batches run */
    uint64_t batch_tokens;    /* tokens processed by verify batches */
    uint64_t refeed_tokens;   /* tokens re-fed after partial rejects */
    uint64_t offered;         /* speculative draft tokens checked (d2..dD) */
    uint64_t accepted;        /* ... accepted */
    uint64_t d1_offered, d1_hits;  /* d1 vs the held token (informational) */
    /* internals (opaque to callers) */
    int64_t q;                /* position of the held token */
    int64_t held;             /* x_q: sampled, not yet emitted */
    int64_t tok_prev;         /* x_{q-1} */
    float *h_prev;            /* [hc*dim] main hidden at q-1 */
    int64_t *drafts;          /* [depth] d1..dD */
    int64_t *batch;           /* [depth] */
    float *R;                 /* [depth * V] verify batch logits */
    float *H;                 /* [depth * hc*dim] verify batch hiddens */
    float *mtp_logits;        /* [V] */
    float *hmtp;              /* [hc*dim] chaining hidden */
    float *refeed_logits;     /* [V] */
    ApusSnap snap;            /* main state snapshot */
    ApusSnap mtp_snap;        /* MTP state snapshot */
    int V, hcdim;
} ApusSpec;

void apus_spec_init(ApusSpec *sp, const ApusModel *m, ApusModelState *st,
                    const ApusMtp *mt, ApusMtpState *mst, int depth,
                    float temp, float top_p, ApusRng *rng,
                    void *sample_scratch);
void apus_spec_free(ApusSpec *sp);

/* Prefill: main forward over ids[n] + MTP true-pair replay + first draft
 * chain; samples the first held token. Returns 0. */
int  apus_spec_prefill(ApusSpec *sp, const int64_t *ids, int n);

/* One speculative step. Appends emitted tokens to out (cap must be >=
 * depth + 1); returns the count emitted (>= 1, 0 at the position table
 * limit). The caller stops on EOS / max_tokens — the emitted stream is a
 * prefix of the non-speculative stream, so truncation is safe. */
int  apus_spec_step(ApusSpec *sp, int *out, int cap);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MTP_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- MTP weights ----------------------------------------------------------*/

int apus_mtp_load(ApusMtp *mt, ApusModel *m, int idx, char *err,
                  size_t errcap) {
    memset(mt, 0, sizeof *mt);
    mt->m = m;
    mt->idx = idx;
    if (m->n_mtp <= idx) {
        snprintf(err, errcap, "mtp: model has no MTP block %d", idx);
        return -1;
    }
    char prefix[32], n[192];
    snprintf(prefix, sizeof prefix, "mtp.%d", idx);
    if (apus_layer_load_named(&mt->block, m->set, &m->cfg, prefix,
                              m->mtp_ratio, m->mtp_hash,
                              m->n_layers + idx, err, errcap))
        return -1;
    int dim = m->cfg.dim, hc = m->cfg.hc_mult;
    snprintf(n, sizeof n, "%s.e_proj", prefix);
    if (apus_st_fp8w(m->set, n, &mt->e_proj)
        || mt->e_proj.O != dim || mt->e_proj.K != dim) {
        snprintf(err, errcap, "mtp: bad tensor %s", n);
        return -1;
    }
    snprintf(n, sizeof n, "%s.h_proj", prefix);
    if (apus_st_fp8w(m->set, n, &mt->h_proj)
        || mt->h_proj.O != dim || mt->h_proj.K != dim) {
        snprintf(err, errcap, "mtp: bad tensor %s", n);
        return -1;
    }
    snprintf(n, sizeof n, "%s.enorm.weight", prefix);
    mt->enorm_w = apus_load_f32(m->set, n, (size_t)dim, err, errcap);
    if (!mt->enorm_w) return -1;
    snprintf(n, sizeof n, "%s.hnorm.weight", prefix);
    mt->hnorm_w = apus_load_f32(m->set, n, (size_t)dim, err, errcap);
    if (!mt->hnorm_w) return -1;
    snprintf(n, sizeof n, "%s.norm.weight", prefix);
    mt->norm_w = apus_load_f32(m->set, n, (size_t)dim, err, errcap);
    if (!mt->norm_w) return -1;
    /* hc_head_* (F32, own sigmoid-gated collapse — NO Sinkhorn) */
    const ApusStTensor *t;
    snprintf(n, sizeof n, "%s.hc_head_fn", prefix);
    t = apus_st_set_get(m->set, n);
    if (!t || t->dtype != APUS_ST_F32
        || apus_st_nelem(t) != (size_t)hc * hc * dim) {
        snprintf(err, errcap, "mtp: bad tensor %s", n);
        return -1;
    }
    mt->hc_head_fn = malloc((size_t)hc * hc * dim * sizeof(float));
    apus_st_f32(t, mt->hc_head_fn, (size_t)hc * hc * dim);
    snprintf(n, sizeof n, "%s.hc_head_base", prefix);
    t = apus_st_set_get(m->set, n);
    if (!t || apus_st_nelem(t) != (size_t)hc) {
        snprintf(err, errcap, "mtp: bad tensor %s", n);
        return -1;
    }
    mt->hc_head_base = malloc((size_t)hc * sizeof(float));
    apus_st_f32(t, mt->hc_head_base, (size_t)hc);
    snprintf(n, sizeof n, "%s.hc_head_scale", prefix);
    t = apus_st_set_get(m->set, n);
    if (!t || apus_st_nelem(t) != 1) {
        snprintf(err, errcap, "mtp: bad tensor %s", n);
        return -1;
    }
    apus_st_f32(t, &mt->hc_head_scale, 1);
    return 0;
}

void apus_mtp_free(ApusMtp *mt) {
    if (!mt) return;
    apus_layer_free(&mt->block);
    free(mt->enorm_w);
    free(mt->hnorm_w);
    free(mt->norm_w);
    free(mt->hc_head_fn);
    free(mt->hc_head_base);
    memset(mt, 0, sizeof *mt);
}

void apus_mtp_state_init(ApusMtpState *st, const ApusMtp *mt) {
    apus_layer_state_init(&st->block, &mt->block);
}

void apus_mtp_state_free(ApusMtpState *st, const ApusMtp *mt) {
    apus_layer_state_free(&st->block, &mt->block);
}

/* ---- MTP forward (model.py:757-766) ----------------------------------------*/

void apus_mtp_forward(const ApusMtp *mt, ApusMtpState *mst,
                      const float *h_in, const int64_t *ids, int s,
                      int64_t start_pos, float *logits, float *h_last) {
    const ApusModel *m = mt->m;
    int dim = m->cfg.dim, hc = m->cfg.hc_mult, V = m->cfg.vocab_size;
    float eps = m->cfg.norm_eps;
    ApusScratchMark mk = apus_scratch_mark();

    /* e = enorm(embed(ids)) -> e_proj  (model.py:760-761, 763) */
    float *e = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    float *erow = apus_scratch_alloc((size_t)hc * dim * sizeof(float));
    for (int t = 0; t < s; t++) {
        apus_model_embed(m, ids[t], erow);   /* hc-identical rows; take 1st */
        memcpy(e + (size_t)t * dim, erow, (size_t)dim * sizeof(float));
        apus_rms_norm(e + (size_t)t * dim, mt->enorm_w, eps,
                      e + (size_t)t * dim, (size_t)dim);
    }
    float *ep = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    apus_fp8_linear(&mt->e_proj, e, ep, s, dim, dim);

    /* x = hnorm(h_in) -> h_proj (per hc row)  (model.py:762-763) */
    float *hn = apus_scratch_alloc((size_t)s * hc * dim * sizeof(float));
    for (int r = 0; r < s * hc; r++)
        apus_rms_norm(h_in + (size_t)r * dim, mt->hnorm_w, eps,
                      hn + (size_t)r * dim, (size_t)dim);
    float *hp = apus_scratch_alloc((size_t)s * hc * dim * sizeof(float));
    apus_fp8_linear(&mt->h_proj, hn, hp, s * hc, dim, dim);

    /* x = bf16(e_proj(e) + h_proj(x)) -> Block  (model.py:763-764) */
    float *x = apus_scratch_alloc((size_t)s * hc * dim * sizeof(float));
    for (int t = 0; t < s; t++)
        for (int j = 0; j < hc; j++)
            for (int i = 0; i < dim; i++)
                x[((size_t)t * hc + j) * dim + i] =
                    apus_bf16_round(ep[(size_t)t * dim + i]
                                    + hp[((size_t)t * hc + j) * dim + i]);
    apus_block_forward(&mt->block, &mst->block, x, ids, s, start_pos, NULL);

    /* head: own hc_head collapse -> own norm -> SHARED lm head, last
     * position only (model.py:765 -> ParallelHead.forward 718-726) */
    float *hL = x + (size_t)(s - 1) * hc * dim;
    if (h_last) memcpy(h_last, hL, (size_t)hc * dim * sizeof(float));
    float *y = apus_scratch_alloc((size_t)dim * sizeof(float));
    float *yn = apus_scratch_alloc((size_t)dim * sizeof(float));
    float *mixes = apus_scratch_alloc((size_t)hc * sizeof(float));
    apus_mhc_head_scalar(hL, (size_t)dim, (size_t)hc, mt->hc_head_fn,
                         mt->hc_head_scale, mt->hc_head_base,
                         m->cfg.norm_eps, m->cfg.hc_eps, y, mixes);
    for (int i = 0; i < dim; i++) y[i] = apus_bf16_round(y[i]);
    apus_rms_norm(y, mt->norm_w, eps, yn, (size_t)dim);
    apus_head_gemv(m->head, yn, logits, V, dim);
    apus_scratch_reset(mk);
}

/* ---- snapshots --------------------------------------------------------------*/

void apus_snap_alloc(ApusSnap *sn, const ApusLayer *layers,
                     const ApusLayerState *states, int n) {
    sn->n = n;
    sn->pos = 0;
    sn->L = calloc((size_t)n, sizeof(ApusSnapLayer));
    for (int i = 0; i < n; i++) {
        const ApusAttnCfg *a = &layers[i].acfg;
        const ApusAttnS *as = &states[i].attn;
        ApusSnapLayer *sl = &sn->L[i];
        sl->win_bytes = (size_t)a->window * a->head_dim * sizeof(float);
        sl->win = malloc(sl->win_bytes);
        if (as->comp.kv) {
            sl->comp_bytes = (size_t)as->comp.rows * as->comp.cols
                             * sizeof(float);
            sl->comp_kv = malloc(sl->comp_bytes);
            sl->comp_sc = malloc(sl->comp_bytes);
        }
        if (as->idx_comp.kv) {
            sl->idx_bytes = (size_t)as->idx_comp.rows * as->idx_comp.cols
                            * sizeof(float);
            sl->idx_kv = malloc(sl->idx_bytes);
            sl->idx_sc = malloc(sl->idx_bytes);
        }
    }
}

void apus_snap_free(ApusSnap *sn) {
    if (!sn->L) return;
    for (int i = 0; i < sn->n; i++) {
        free(sn->L[i].win);
        free(sn->L[i].comp_kv);
        free(sn->L[i].comp_sc);
        free(sn->L[i].idx_kv);
        free(sn->L[i].idx_sc);
    }
    free(sn->L);
    sn->L = NULL;
}

void apus_snap_take(ApusSnap *sn, ApusLayerState *states, int64_t pos) {
    sn->pos = pos;
    for (int i = 0; i < sn->n; i++) {
        ApusAttnS *as = &states[i].attn;
        ApusSnapLayer *sl = &sn->L[i];
        sl->pos = as->pos;
        memcpy(sl->win, as->win, sl->win_bytes);
        if (sl->comp_kv) {
            sl->comp_nb = as->comp.nb;
            memcpy(sl->comp_kv, as->comp.kv, sl->comp_bytes);
            memcpy(sl->comp_sc, as->comp.sc, sl->comp_bytes);
        }
        if (sl->idx_kv) {
            sl->idx_nb = as->idx_comp.nb;
            memcpy(sl->idx_kv, as->idx_comp.kv, sl->idx_bytes);
            memcpy(sl->idx_sc, as->idx_comp.sc, sl->idx_bytes);
        }
    }
}

/* Rollback = truncation: pos/window/carries restored, comp.nb rewound
 * (cache entries beyond nb are dead capacity and are never read; rope
 * tables are stateless). */
void apus_snap_restore(const ApusSnap *sn, ApusLayerState *states,
                       int64_t *pos) {
    *pos = sn->pos;
    for (int i = 0; i < sn->n; i++) {
        ApusAttnS *as = &states[i].attn;
        const ApusSnapLayer *sl = &sn->L[i];
        as->pos = sl->pos;
        memcpy(as->win, sl->win, sl->win_bytes);
        if (sl->comp_kv) {
            as->comp.nb = sl->comp_nb;
            memcpy(as->comp.kv, sl->comp_kv, sl->comp_bytes);
            memcpy(as->comp.sc, sl->comp_sc, sl->comp_bytes);
        }
        if (sl->idx_kv) {
            as->idx_comp.nb = sl->idx_nb;
            memcpy(as->idx_comp.kv, sl->idx_kv, sl->idx_bytes);
            memcpy(as->idx_comp.sc, sl->idx_sc, sl->idx_bytes);
        }
    }
}

/* ---- speculative decode engine -------------------------------------------*/

void apus_spec_init(ApusSpec *sp, const ApusModel *m, ApusModelState *st,
                    const ApusMtp *mt, ApusMtpState *mst, int depth,
                    float temp, float top_p, ApusRng *rng,
                    void *sample_scratch) {
    memset(sp, 0, sizeof *sp);
    sp->m = m;
    sp->st = st;
    sp->mt = mt;
    sp->mst = mst;
    sp->depth = depth > 0 ? depth : 1;
    sp->temp = temp;
    sp->top_p = top_p;
    sp->rng = rng;
    sp->sample_scratch = sample_scratch;
    sp->V = m->cfg.vocab_size;
    sp->hcdim = m->cfg.hc_mult * m->cfg.dim;
    int D = sp->depth, V = sp->V;
    sp->h_prev = malloc((size_t)sp->hcdim * sizeof(float));
    sp->hmtp = malloc((size_t)sp->hcdim * sizeof(float));
    sp->drafts = malloc((size_t)D * sizeof(int64_t));
    sp->batch = malloc((size_t)D * sizeof(int64_t));
    sp->R = malloc((size_t)D * V * sizeof(float));
    sp->H = malloc((size_t)D * sp->hcdim * sizeof(float));
    sp->mtp_logits = malloc((size_t)V * sizeof(float));
    sp->refeed_logits = malloc((size_t)V * sizeof(float));
    apus_snap_alloc(&sp->snap, m->layers, st->layers, m->n_layers);
    if (mt)
        apus_snap_alloc(&sp->mtp_snap, &mt->block, &mst->block, 1);
}

void apus_spec_free(ApusSpec *sp) {
    free(sp->h_prev);
    free(sp->hmtp);
    free(sp->drafts);
    free(sp->batch);
    free(sp->R);
    free(sp->H);
    free(sp->mtp_logits);
    free(sp->refeed_logits);
    apus_snap_free(&sp->snap);
    if (sp->mt) apus_snap_free(&sp->mtp_snap);
    memset(sp, 0, sizeof *sp);
}

/* Draft chain: drafts[0] = argmax of the current MTP logits; d_{i+1} =
 * MTP(hmtp_i, d_i) at position start_pos+i-1 (predicting start_pos+i).
 * The MTP block's own post-block hidden takes the role of the main hidden
 * when chaining (the depth-1 head autoregresses; acceptance is measured,
 * correctness is guaranteed by the verify walk regardless). Pollutes the
 * MTP attention state — cleaned by the next snapshot restore. */
static void apus_spec_chain(ApusSpec *sp, int64_t start_pos) {
    sp->drafts[0] = apus_sample_argmax(sp->mtp_logits, (size_t)sp->V);
    for (int i = 1; i < sp->depth; i++) {
        apus_mtp_forward(sp->mt, sp->mst, sp->hmtp, &sp->drafts[i - 1], 1,
                         start_pos + i - 1, sp->mtp_logits, sp->hmtp);
        sp->drafts[i] = apus_sample_argmax(sp->mtp_logits, (size_t)sp->V);
    }
}

int apus_spec_prefill(ApusSpec *sp, const int64_t *ids, int n) {
    int V = sp->V;
    size_t hcdim = (size_t)sp->hcdim;
    float *h_all = malloc((size_t)n * hcdim * sizeof(float));
    /* main prefill (mtp_logits doubles as the [V] logits buffer here) */
    apus_model_forward_h(sp->m, sp->st, ids, n, sp->mtp_logits, 0, h_all);
    /* first held token: exactly what non-spec sampling draws first */
    sp->held = apus_sample(sp->mtp_logits, (size_t)V, sp->temp, sp->top_p,
                           sp->rng, sp->sample_scratch);
    sp->q = n;
    sp->tok_prev = ids[n - 1];
    memcpy(sp->h_prev, h_all + (size_t)(n - 1) * hcdim,
           hcdim * sizeof(float));
    if (sp->draft_override) {
        sp->draft_override(sp->draft_ctx, sp->q, sp->drafts, sp->depth);
    } else {
        /* MTP true-pair replay over the whole prompt (batched): builds the
         * draft head's SWA window from the true (h, id) pairs and yields
         * the first draft; then chain the remaining depth-1 drafts. */
        apus_mtp_forward(sp->mt, sp->mst, h_all, ids, n, 0,
                         sp->mtp_logits, sp->hmtp);
        apus_snap_take(&sp->mtp_snap, &sp->mst->block, (int64_t)n);
        apus_spec_chain(sp, n);
    }
    free(h_all);
    return 0;
}

int apus_spec_step(ApusSpec *sp, int *out, int cap) {
    int D = sp->depth, V = sp->V;
    size_t hcdim = (size_t)sp->hcdim;
    if (cap < D + 1) return 0;
    int64_t maxp = sp->m->cfg.max_pos;
    int D_eff = D;
    if (sp->q + D_eff > maxp) D_eff = (int)(maxp - sp->q);
    if (D_eff <= 0) return 0;

    sp->d1_offered++;
    if (sp->drafts[0] == sp->held) sp->d1_hits++;

    /* verify batch: [x_q, d2, ..., d_{D_eff}] at positions q..q+D_eff-1,
     * ONE batched forward (m4c chunk-invariance: row j is bitwise the
     * one-by-one decode of the same context) */
    apus_snap_take(&sp->snap, sp->st->layers, sp->st->pos);
    sp->batch[0] = sp->held;
    for (int j = 1; j < D_eff; j++) sp->batch[j] = sp->drafts[j];
    if (sp->pre_batch)
        sp->pre_batch(sp->pre_batch_ctx, sp->batch, D_eff, sp->q);
    apus_model_forward_h(sp->m, sp->st, sp->batch, D_eff, sp->R, 1, sp->H);
    sp->batches++;
    sp->batch_tokens += (uint64_t)D_eff;

    /* accept walk: every sampled token is the main model's own draw from
     * its own logits row — one RNG uniform per token, in position order.
     * Accepted drafts are emitted at once; the token that ends the walk
     * (the replacement after a mismatch, or the bonus after a full match)
     * becomes the next HELD token: sampled now, emitted at the start of
     * the next step (it is a true token either way — the rows it is drawn
     * from are valid exactly up to that position). */
    int ne = 0, matched = 0, last = -1;
    out[ne++] = (int)sp->held;
    for (int j = 1; j < D_eff; j++) {
        int x = apus_sample(sp->R + (size_t)(j - 1) * V, (size_t)V,
                            sp->temp, sp->top_p, sp->rng,
                            sp->sample_scratch);
        sp->offered++;
        if ((int)sp->drafts[j] == x) {
            matched++;
            sp->accepted++;
            out[ne++] = x;
        } else {
            last = x;
            break;
        }
    }
    if (last < 0)   /* full match: bonus token from R[D_eff-1] (valid —
                       the whole batch proved true) */
        last = apus_sample(sp->R + (size_t)(D_eff - 1) * V, (size_t)V,
                           sp->temp, sp->top_p, sp->rng,
                           sp->sample_scratch);
    sp->emitted += (uint64_t)ne;

    /* state fixup: fed-true run = batch[0..matched] */
    if (matched < D_eff - 1) {
        /* partial reject: rollback = snapshot restore (truncation), then
         * re-feed the true prefix in one batched call (chunk-invariant,
         * hence bitwise the one-by-one state) */
        apus_snap_restore(&sp->snap, sp->st->layers, &sp->st->pos);
        apus_model_forward(sp->m, sp->st, sp->batch, matched + 1,
                           sp->refeed_logits, 0);
        sp->refeed_tokens += (uint64_t)(matched + 1);
    }

    /* next drafts */
    sp->held = last;
    sp->q += matched + 1;
    sp->tok_prev = sp->batch[matched];
    memcpy(sp->h_prev, sp->H + (size_t)matched * hcdim,
           hcdim * sizeof(float));
    if (sp->mt) {
        /* MTP: restore the clean (true-pairs-only) state, replay the newly
         * accepted true pairs (H[j], batch[j]) j<=matched — batched — then
         * snapshot the clean state and chain the next depth drafts. */
        int64_t mtp_pos = 0;
        apus_snap_restore(&sp->mtp_snap, &sp->mst->block, &mtp_pos);
        apus_mtp_forward(sp->mt, sp->mst, sp->H, sp->batch, matched + 1,
                         sp->q - matched - 1, sp->mtp_logits, sp->hmtp);
        apus_snap_take(&sp->mtp_snap, &sp->mst->block, sp->q);
        apus_spec_chain(sp, sp->q);
    } else if (sp->draft_override) {
        sp->draft_override(sp->draft_ctx, sp->q, sp->drafts, D);
    }
    return ne;
}

#endif /* APUS_MTP_IMPLEMENTATION */
#endif /* APUS_MTP_H */
