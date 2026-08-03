/*
 * c/dspark.h — M11b: DSpark speculative decoding for DeepSeek-V4-Flash-0731.
 *
 * DSpark is the 0731-native draft module: n_stages (3) full transformer
 * blocks under the "mtp.{0,1,2}" checkpoint namespace (SWA ratio 0, bias-gate
 * MoE with FP4 experts — loaded by the standard c/layer.h machinery), plus
 * stage glue: stage 0 main_proj (FP8 [dim, n_targets*dim]) + main_norm,
 * last stage norm + own hc_head (sigmoid, NO Sinkhorn) + markov head
 * (markov_w1/w2 [V, rank]) + confidence head (proj [1, dim+rank]).
 *
 * Normative reference: reference-0731/inference/model.py (DSparkAttention
 * 749-792, DSparkMarkovHead 795-804, DSparkConfidenceHead 807-815,
 * DSparkBlock 819-874, get_dspark_topk_idxs 743-747, Transformer.forward /
 * forward_spec 912-936) and the M11a numpy oracle tools/oracle_dspark.py,
 * whose fixtures (tests/m11a) this port is verified against in tests/m11b.
 * Data flow, accept rule and the D1-D14 ambiguity resolutions are documented
 * in tests/m11a/README.md; the stage-by-stage C checklist is its "M11b
 * implementation checklist" section. Key pins:
 *
 *   - Draft round (forward_spec decode, start_pos = f): draft ids
 *     [anchor, noise x (B-1)] -> embed -> hc expand; each stage writes its
 *     ring slot f%win with kv_norm(wkv(main_x)) rope'd at f + FP8 QAT
 *     (model.py:783), main_x = stage-0 main_norm(main_proj(main_hidden[f]))
 *     SHARED by all stages (D7). The B draft rows rope q/kv at
 *     f+1..f+B (D8), attend over ONE SHARED non-causal index row (D6):
 *     window slots 0..min(win,f+1)-1 ++ win+0..win+B-1, plain arange slot
 *     order (D11) over [ring ++ draft KVs] (draft KVs never enter the
 *     ring, D12). Stage rope: base theta, NO YaRN (D9).
 *   - Head (forward_head 860-874): stage-last hc_head collapse -> own norm
 *     -> SHARED lm head; per row i the markov bias w2 @ w1(out_ids[i]) is
 *     added (bigram from the PRECEDING token) before sampling the next
 *     draft; confidence = proj(cat([pre-norm collapse, markov embeds]))
 *     (D4) in fp32 (D14). Confidence is telemetry only — it takes NO part
 *     in acceptance (D1).
 *   - Prefill (763-769): build each stage's ring from main_x over the whole
 *     prompt (same circular write as SWA prefill); draft embeddings
 *     discarded.
 *
 * THE ACCEPT RULE (M8 rule restated for DSpark — the rule that makes the
 * emitted stream BITWISE equal to non-speculative decoding):
 *   Every emitted token is drawn from the MAIN model's own logits row for
 *   its position (argmax greedy / the engine's pinned-uniform draw for
 *   sampled), exactly one RNG uniform per emitted token in position order.
 *   A draft token is accepted iff it equals the main model's own draw at
 *   that position. Drafts consume a SEPARATE uniform stream (drng) and
 *   never influence the emitted stream.
 *
 * Round shape (invariant at round top: main state fed through TRUE
 * position f; held token for f+1 drawn from a valid main row, not yet
 * emitted; main_hidden row of f at hand; stage rings hold true 0..f):
 *   1. Draft at start_pos=f -> d1..dB for f+2..f+B+1 (+ confidence).
 *   2. Snapshot main state. Verify batch [held, d1..dB] at f+1..f+B+1 in
 *      ONE batched forward (m4c/M8 chunk-invariance: bitwise the
 *      one-by-one decode) -> rows R[0..B] (dists for f+2..f+B+2) +
 *      per-position main_hiddens H[0..B].
 *   3. Walk: emit held; for j=1..B draw m_j from R[j-1] (one uniform);
 *      accept d_j iff d_j == m_j; stop at the first mismatch — the drawn
 *      replacement becomes the next held (NOT emitted). Full match (a=B):
 *      held = draw from R[B] (bonus). Emitted per round = 1 + a.
 *   4. Fixup: full match keeps the batch state; partial restores the
 *      snapshot and re-feeds [held, d1..d_a] in one batched call. Next
 *      round: f' = f+1+a, main_hidden = H[a] reused, held from 3.
 *   5. Stage-ring maintenance (D13): write each stage's ring slots for ALL
 *      newly-true fed positions f+1..f' from H[0..a], so every ring always
 *      holds exactly the true positions 0..f'. The next round's
 *      in-attention slot write (model.py:783) rewrites the same value —
 *      idempotent. Stage rings never need rollback (D12).
 *
 * Usage: #define APUS_DSPARK_IMPLEMENTATION in exactly one TU, AFTER the
 * attn/layer/model/mtp implementation includes — the implementation reuses
 * their same-TU statics (apus_sparse_attn, the q/kv/wo_a row jobs,
 * apus_hc_pre/post, apus_rope_tab_ensure, apus_load_f32, the snapshot
 * machinery), the same include-order pattern c/mtp.h documents.
 */
#ifndef APUS_DSPARK_H
#define APUS_DSPARK_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"
#include "mtp.h"
#include "sample.h"

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_DSPARK_MAX_STAGES 8

/* ---- DSpark weights --------------------------------------------------------*/

typedef struct {
    const ApusModel *m;           /* parent model (embed/head shared) */
    ApusDsparkCfg cfg;            /* copied from m->dsc */
    int n_stages;
    ApusLayer stages[APUS_DSPARK_MAX_STAGES];   /* full SWA blocks */
    /* stage 0 glue */
    ApusFp8W main_proj;           /* [dim, n_targets*dim] FP8 view */
    float *main_norm_w;           /* [dim] owned */
    /* last-stage glue */
    float *norm_w;                /* [dim] owned */
    float *hc_head_fn;            /* [hc, hc*dim] owned F32 */
    float *hc_head_base;          /* [hc] owned F32 */
    float hc_head_scale;
    float *markov_w1;             /* [V, rank] owned F32 (BF16 widened, D14) */
    float *markov_w2;             /* [V, rank] owned F32 */
    float *conf_proj;             /* [dim+rank] owned F32 */
} ApusDspark;

/* Load the DSpark stages + glue from the model's shard set. The stages'
 * MoE expert-store layer ids are m->n_layers + k (attach the M6a store to
 * stages[k].mw afterwards in tiered mode — the "mtp.K.ffn.experts.*"
 * tensors already map to those layers in c/cache.h). Returns 0 on success;
 * -1 with err when the model declares no DSpark config or tensors are
 * missing. */
int  apus_dspark_load(ApusDspark *ds, ApusModel *m, char *err, size_t errcap);
void apus_dspark_free(ApusDspark *ds);

typedef struct {
    ApusLayerState stages[APUS_DSPARK_MAX_STAGES];
} ApusDsparkState;

void apus_dspark_state_init(ApusDsparkState *dst, const ApusDspark *ds);
void apus_dspark_state_free(ApusDsparkState *dst, const ApusDspark *ds);

/* main_x = main_norm(main_proj(main_hidden)) (model.py:853).
 * mh [rows, n_targets*dim] -> out [rows, dim] (BF16 values). */
void apus_dspark_stage0_project(const ApusDspark *ds, const float *mh,
                                int rows, float *out);

/* One stage's ring-slot KV row for MAIN position pos from its main_x row
 * (model.py:759-761): kv_norm(wkv(mx)) -> rope @ pos -> FP8 QAT on the
 * non-rope dims -> BF16. Exposed for the catch-up correctness test. */
void apus_dspark_stage_kv(const ApusDspark *ds, int k, const float *mx_row,
                          int64_t pos, float *out /* [head_dim] */);

/* Prefill (model.py:763-769): build every stage's ring from main_x over
 * the whole prompt (circular write, like SWA prefill). */
void apus_dspark_prefill(const ApusDspark *ds, ApusDsparkState *dst,
                         const float *mx_all, int s);

/* D13 catch-up: write each stage's ring slots for the newly-true fed
 * positions pos0..pos0+n-1 from their main_x rows. */
void apus_dspark_catchup(const ApusDspark *ds, ApusDsparkState *dst,
                         const float *mx_rows, int64_t pos0, int n);

/* Optional draft-round golden intermediates (tests/m11b; NULL to skip). */
typedef struct {
    float *logits_base;     /* [B*V] shared-head logits before the bias */
    float *markov_bias;     /* [B*V] */
    float *logits_final;    /* [B*V] after the bias */
    float *markov_embed;    /* [B*rank] */
    float *conf_hidden;     /* [B*dim] pre-norm hc_head collapse (D4) */
    float *confidence;      /* [B] */
    float *stage_h[APUS_DSPARK_MAX_STAGES];  /* [B*hc*dim] block outputs */
} ApusDsparkDbg;

/* One DSpark draft round (model.py:928-936): (re)writes each stage's ring
 * slot start_pos%win from mx_row (idempotent with the catch-up writes),
 * runs the 3-stage block forward over the B draft rows and the head loop.
 * anchor = the held token; mx_row = stage-0 projection of position
 * start_pos's main_hidden [dim]. Draft sampling: temp <= 0 -> argmax; else
 * one uniform per row from drng (the SEPARATE draft stream, D2), top_p
 * fixed 1.0. Outputs drafts [B] (candidates for positions
 * start_pos+2..start_pos+B+1). dbg optional. */
void apus_dspark_draft_round(const ApusDspark *ds, ApusDsparkState *dst,
                             int64_t anchor, const float *mx_row,
                             int64_t start_pos, float temp, ApusRng *drng,
                             void *sample_scratch, int64_t *drafts,
                             ApusDsparkDbg *dbg);

/* ---- speculative decode engine ---------------------------------------------*/

typedef struct ApusDspec {
    const ApusModel *m;
    ApusModelState *st;
    const ApusDspark *ds;
    ApusDsparkState *dst;
    float temp, top_p;
    ApusRng *rng;             /* MAIN stream: one uniform per emitted token */
    ApusRng drng;             /* draft stream (separate; seeded apart, D2) */
    void *sample_scratch;
    /* Optional test hook replacing the DSpark proposer: fill drafts[0..B-1]
     * (candidates for f+2..f+B+1) at round top (f = last true fed position,
     * anchor = held token). The stage slot-f ring write still happens. */
    void (*draft_override)(void *ctx, int64_t f, int64_t anchor,
                           int64_t *drafts, int n);
    void *draft_ctx;
    /* Optional hook invoked with the verify-batch token ids right before
     * the batched forward (apus.c wires the M6b hash-layer prefetch here).
     * Hint-only; must never touch numerics. */
    void (*pre_batch)(void *ctx, const int64_t *ids, int n, int64_t pos);
    void *pre_batch_ctx;
    /* stats */
    uint64_t emitted;         /* tokens emitted across all rounds */
    uint64_t batches;         /* verify batches run */
    uint64_t batch_tokens;    /* tokens processed by verify batches */
    uint64_t refeed_tokens;   /* tokens re-fed after partial rejects */
    uint64_t offered;         /* draft tokens checked */
    uint64_t accepted;        /* ... accepted */
    uint64_t bonus_rounds;    /* full-match rounds (bonus draw from R[B]) */
    /* internals (opaque to callers) */
    int B;                    /* block_size (0731: 5, fixed) */
    int V, mhdim;             /* vocab, n_targets*dim */
    int64_t f;                /* last TRUE fed position */
    int64_t held;             /* token for f+1: drawn, not yet emitted */
    float *mh_last;           /* [mhdim] main_hidden of position f */
    int64_t *drafts;          /* [B] */
    int64_t *batch;           /* [B+1] */
    float *R;                 /* [(B+1)*V] verify batch logits */
    float *H;                 /* [(B+1)*mhdim] verify batch main_hiddens */
    float *mx;                /* [dim] stage-0 projection scratch */
    float *refeed_logits;     /* [V] */
    ApusSnap snap;            /* main-state snapshot (stage rings never
                               rolled back, D12) */
} ApusDspec;

void apus_dspec_init(ApusDspec *sp, const ApusModel *m, ApusModelState *st,
                     const ApusDspark *ds, ApusDsparkState *dst,
                     float temp, float top_p, ApusRng *rng,
                     uint64_t draft_seed, void *sample_scratch);
void apus_dspec_free(ApusDspec *sp);

/* Prefill: main forward over ids[n] with main_hidden collection, stage-ring
 * prefill from the stage-0 projection, samples the first held token (main
 * stream — exactly what non-spec decoding draws first). Returns 0. */
int  apus_dspec_prefill(ApusDspec *sp, const int64_t *ids, int n);

/* One DSpark draft/verify round. Appends emitted tokens to out (cap must
 * be >= B + 1); returns the count emitted (>= 1, 0 at the position table
 * limit). The emitted stream is a prefix of the non-speculative stream, so
 * truncation is safe. */
int  apus_dspec_step(ApusDspec *sp, int *out, int cap);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_DSPARK_IMPLEMENTATION

#ifndef APUS_ATTN_IMPLEMENTATION
#error "c/dspark.h reuses same-TU statics: define APUS_ATTN_IMPLEMENTATION, APUS_LAYER_IMPLEMENTATION, APUS_MODEL_IMPLEMENTATION and APUS_MTP_IMPLEMENTATION in this TU and include dspark.h AFTER attn.h/layer.h/model.h/mtp.h"
#endif
#ifndef APUS_MTP_IMPLEMENTATION
#error "c/dspark.h needs the APUS_MTP_IMPLEMENTATION statics (ApusSnap) in the same TU"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- loader -----------------------------------------------------------------*/

int apus_dspark_load(ApusDspark *ds, ApusModel *m, char *err, size_t errcap) {
    memset(ds, 0, sizeof *ds);
    ds->m = m;
    ds->cfg = m->dsc;
    if (!m->dspark || ds->cfg.n_stages <= 0
        || ds->cfg.n_stages > APUS_DSPARK_MAX_STAGES) {
        snprintf(err, errcap, "dspark: model declares no DSpark stages");
        return -1;
    }
    ds->n_stages = ds->cfg.n_stages;
    int dim = m->cfg.dim, V = m->cfg.vocab_size, hc = m->cfg.hc_mult;
    int rank = ds->cfg.markov_rank;
    if (ds->cfg.block_size < 1 || ds->cfg.n_targets < 1
        || ds->cfg.noise_id < 0 || ds->cfg.noise_id >= V || rank < 1) {
        snprintf(err, errcap, "dspark: bad config (block %d, targets %d, "
                 "noise %d, rank %d)", ds->cfg.block_size,
                 ds->cfg.n_targets, ds->cfg.noise_id, rank);
        return -1;
    }
    for (int tg = 0; tg < ds->cfg.n_targets; tg++)
        if (ds->cfg.targets[tg] < 0 || ds->cfg.targets[tg] >= m->n_layers) {
            snprintf(err, errcap, "dspark: target layer %d out of range",
                     ds->cfg.targets[tg]);
            return -1;
        }
    /* the stages: full SWA blocks (ratio 0, bias gate, never hash — D5) */
    char prefix[32], n[192];
    for (int k = 0; k < ds->n_stages; k++) {
        snprintf(prefix, sizeof prefix, "mtp.%d", k);
        if (apus_layer_load_named(&ds->stages[k], m->set, &m->cfg, prefix,
                                  0, 0, m->n_layers + k, err, errcap))
            return -1;
    }
    /* stage 0 glue: main_proj (FP8 [dim, n_targets*dim]) + main_norm */
    if (apus_st_fp8w(m->set, "mtp.0.main_proj", &ds->main_proj)
        || ds->main_proj.O != dim
        || ds->main_proj.K != ds->cfg.n_targets * dim) {
        snprintf(err, errcap, "dspark: bad tensor mtp.0.main_proj");
        return -1;
    }
    ds->main_norm_w = apus_load_f32(m->set, "mtp.0.main_norm.weight",
                                    (size_t)dim, err, errcap);
    if (!ds->main_norm_w) return -1;
    /* last-stage glue */
    int last = ds->n_stages - 1;
    snprintf(n, sizeof n, "mtp.%d.norm.weight", last);
    ds->norm_w = apus_load_f32(m->set, n, (size_t)dim, err, errcap);
    if (!ds->norm_w) return -1;
    const ApusStTensor *t;
    snprintf(n, sizeof n, "mtp.%d.hc_head_fn", last);
    t = apus_st_set_get(m->set, n);
    if (!t || t->dtype != APUS_ST_F32
        || apus_st_nelem(t) != (size_t)hc * hc * dim) {
        snprintf(err, errcap, "dspark: bad tensor %s", n);
        return -1;
    }
    ds->hc_head_fn = malloc((size_t)hc * hc * dim * sizeof(float));
    apus_st_f32(t, ds->hc_head_fn, (size_t)hc * hc * dim);
    snprintf(n, sizeof n, "mtp.%d.hc_head_base", last);
    t = apus_st_set_get(m->set, n);
    if (!t || apus_st_nelem(t) != (size_t)hc) {
        snprintf(err, errcap, "dspark: bad tensor %s", n);
        return -1;
    }
    ds->hc_head_base = malloc((size_t)hc * sizeof(float));
    apus_st_f32(t, ds->hc_head_base, (size_t)hc);
    snprintf(n, sizeof n, "mtp.%d.hc_head_scale", last);
    t = apus_st_set_get(m->set, n);
    if (!t || apus_st_nelem(t) != 1) {
        snprintf(err, errcap, "dspark: bad tensor %s", n);
        return -1;
    }
    apus_st_f32(t, &ds->hc_head_scale, 1);
    snprintf(n, sizeof n, "mtp.%d.markov_head.markov_w1.weight", last);
    ds->markov_w1 = apus_load_f32(m->set, n, (size_t)V * rank, err, errcap);
    if (!ds->markov_w1) return -1;
    snprintf(n, sizeof n, "mtp.%d.markov_head.markov_w2.weight", last);
    ds->markov_w2 = apus_load_f32(m->set, n, (size_t)V * rank, err, errcap);
    if (!ds->markov_w2) return -1;
    snprintf(n, sizeof n, "mtp.%d.confidence_head.proj.weight", last);
    ds->conf_proj = apus_load_f32(m->set, n, (size_t)(dim + rank),
                                  err, errcap);
    if (!ds->conf_proj) return -1;
    return 0;
}

void apus_dspark_free(ApusDspark *ds) {
    if (!ds) return;
    for (int k = 0; k < ds->n_stages; k++) apus_layer_free(&ds->stages[k]);
    free(ds->main_norm_w);
    free(ds->norm_w);
    free(ds->hc_head_fn);
    free(ds->hc_head_base);
    free(ds->markov_w1);
    free(ds->markov_w2);
    free(ds->conf_proj);
    memset(ds, 0, sizeof *ds);
}

void apus_dspark_state_init(ApusDsparkState *dst, const ApusDspark *ds) {
    for (int k = 0; k < ds->n_stages; k++)
        apus_layer_state_init(&dst->stages[k], &ds->stages[k]);
}

void apus_dspark_state_free(ApusDsparkState *dst, const ApusDspark *ds) {
    for (int k = 0; k < ds->n_stages; k++)
        apus_layer_state_free(&dst->stages[k], &ds->stages[k]);
}

/* ---- stage-0 projection + ring-slot KV rows ----------------------------------*/

void apus_dspark_stage0_project(const ApusDspark *ds, const float *mh,
                                int rows, float *out) {
    int dim = ds->m->cfg.dim;
    int mhd = ds->cfg.n_targets * dim;
    apus_fp8_linear(&ds->main_proj, mh, out, rows, mhd, dim);
    ApusRmsRowsJob rj = { out, ds->main_norm_w, ds->m->cfg.norm_eps, out,
                          (size_t)dim };
    if (rows >= APUS_ROW_POOL_MIN)
        apus_pool_run((size_t)rows, apus_rms_rows, &rj);
    else
        apus_rms_rows(&rj, 0, (size_t)rows);
}

void apus_dspark_stage_kv(const ApusDspark *ds, int k, const float *mx_row,
                          int64_t pos, float *out) {
    const ApusLayer *L = &ds->stages[k];
    int dim = ds->m->cfg.dim, d = L->acfg.head_dim, rd = L->acfg.rope_dim;
    int half = rd / 2;
    apus_rope_tab_ensure(L->rope, pos + 1);
    ApusScratchMark mk = apus_scratch_mark();
    float *kv = apus_scratch_alloc((size_t)d * sizeof(float));
    apus_fp8_linear(&L->aw.wkv, mx_row, kv, 1, dim, d);
    apus_rms_norm(kv, L->aw.kv_norm, L->acfg.eps, kv, (size_t)d);
    /* rope @ pos -> FP8 QAT on the non-rope dims -> BF16 (759-761) */
    ApusKvQatJob kj = { kv, L->rope->cos + pos * half,
                        L->rope->sin + pos * half, d, rd };
    apus_kvqat_rows(&kj, 0, 1);
    memcpy(out, kv, (size_t)d * sizeof(float));
    apus_scratch_reset(mk);
}

void apus_dspark_prefill(const ApusDspark *ds, ApusDsparkState *dst,
                         const float *mx_all, int s) {
    int dim = ds->m->cfg.dim;
    for (int k = 0; k < ds->n_stages; k++) {
        const ApusLayer *L = &ds->stages[k];
        int d = L->acfg.head_dim, rd = L->acfg.rope_dim, win = L->acfg.window;
        apus_rope_tab_ensure(L->rope, s);
        ApusScratchMark mk = apus_scratch_mark();
        float *kva = apus_scratch_alloc((size_t)s * d * sizeof(float));
        float *kv = apus_scratch_alloc((size_t)s * d * sizeof(float));
        apus_fp8_linear(&L->aw.wkv, mx_all, kva, s, dim, d);
        ApusRmsRowsJob rj = { kva, L->aw.kv_norm, L->acfg.eps, kv,
                              (size_t)d };
        if (s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_rms_rows, &rj);
        else
            apus_rms_rows(&rj, 0, (size_t)s);
        /* rope @ 0..s-1 + QAT + BF16 per row (model.py:764-767) */
        ApusKvQatJob kj = { kv, L->rope->cos, L->rope->sin, d, rd };
        if (s >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)s, apus_kvqat_rows, &kj);
        else
            apus_kvqat_rows(&kj, 0, (size_t)s);
        /* circular ring write (model.py:763-769, same as SWA prefill) */
        float *winp = dst->stages[k].attn.win;
        if (s <= win) {
            memcpy(winp, kv, (size_t)s * d * sizeof(float));
        } else {
            int cut = s % win;
            memcpy(winp + (size_t)cut * d, kv + (size_t)(s - win) * d,
                   (size_t)(win - cut) * d * sizeof(float));
            memcpy(winp, kv + (size_t)(s - cut) * d,
                   (size_t)cut * d * sizeof(float));
        }
        dst->stages[k].attn.pos = s;
        apus_scratch_reset(mk);
    }
}

void apus_dspark_catchup(const ApusDspark *ds, ApusDsparkState *dst,
                         const float *mx_rows, int64_t pos0, int n) {
    int dim = ds->m->cfg.dim;
    for (int k = 0; k < ds->n_stages; k++) {
        const ApusLayer *L = &ds->stages[k];
        int d = L->acfg.head_dim, win = L->acfg.window;
        ApusScratchMark mk = apus_scratch_mark();
        float *kv = apus_scratch_alloc((size_t)d * sizeof(float));
        for (int j = 0; j < n; j++) {
            apus_dspark_stage_kv(ds, k, mx_rows + (size_t)j * dim, pos0 + j,
                                 kv);
            memcpy(dst->stages[k].attn.win
                   + (size_t)((pos0 + j) % win) * d, kv,
                   (size_t)d * sizeof(float));
        }
        apus_scratch_reset(mk);
    }
}

/* ---- DSpark attention (model.py:771-792) -------------------------------------*/

/* x [B, dim] (the stage's attention input for the B draft rows), mx_row
 * [dim] (stage-0 projection of position start_pos's main_hidden — the
 * ring slot write at 783). out [B, dim]. */
static void apus_dspark_attn(const ApusLayer *L, ApusLayerState *st,
                             const float *x, const float *mx_row, int B,
                             int64_t start_pos, float *out) {
    const ApusAttnCfg *cfg = &L->acfg;
    const ApusAttnW *w = &L->aw;
    int h = cfg->n_heads, d = cfg->head_dim, rd = cfg->rope_dim;
    int win = cfg->window, half = rd / 2, dim = cfg->dim;
    /* draft rows rope at start_pos+1..start_pos+B (D8); base-theta table,
     * no YaRN (D9 — ratio 0 layers share the plain SWA table) */
    apus_rope_tab_ensure(L->rope, start_pos + 1 + B);
    const float *fc = L->rope->cos + (start_pos + 1) * half;
    const float *fs = L->rope->sin + (start_pos + 1) * half;
    ApusScratchMark mk = apus_scratch_mark();

    /* q path (774-777) */
    float *qa = apus_scratch_alloc((size_t)B * cfg->q_lora * sizeof(float));
    apus_fp8_linear(&w->wq_a, x, qa, B, dim, cfg->q_lora);
    float *qr = apus_scratch_alloc((size_t)B * cfg->q_lora * sizeof(float));
    {
        ApusRmsRowsJob rj = { qa, w->q_norm, cfg->eps, qr,
                              (size_t)cfg->q_lora };
        if (B >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)B, apus_rms_rows, &rj);
        else
            apus_rms_rows(&rj, 0, (size_t)B);
    }
    float *q = apus_scratch_alloc((size_t)B * h * d * sizeof(float));
    apus_fp8_linear(&w->wq_b, qr, q, B, cfg->q_lora, h * d);
    {
        ApusQNormJob qj = { q, fc, fs, cfg->eps, h, d, rd };
        size_t rows = (size_t)B * h;
        if (rows >= APUS_ROW_POOL_MIN)
            apus_pool_run(rows, apus_qnorm_rows, &qj);
        else
            apus_qnorm_rows(&qj, 0, rows);
    }

    /* draft-row kv (778-780) — concatenated AFTER the ring, never written
     * into it (D12) */
    float *kva = apus_scratch_alloc((size_t)B * d * sizeof(float));
    apus_fp8_linear(&w->wkv, x, kva, B, dim, d);
    float *kv = apus_scratch_alloc((size_t)B * d * sizeof(float));
    {
        ApusRmsRowsJob rj = { kva, w->kv_norm, cfg->eps, kv, (size_t)d };
        if (B >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)B, apus_rms_rows, &rj);
        else
            apus_rms_rows(&rj, 0, (size_t)B);
    }
    {
        ApusKvQatJob kj = { kv, fc, fs, d, rd };
        if (B >= APUS_ROW_POOL_MIN)
            apus_pool_run((size_t)B, apus_kvqat_rows, &kj);
        else
            apus_kvqat_rows(&kj, 0, (size_t)B);
    }

    /* main_kv from main_x at start_pos -> ring slot start_pos%win (783) */
    {
        float *mkv = apus_scratch_alloc((size_t)d * sizeof(float));
        apus_fp8_linear(&w->wkv, mx_row, mkv, 1, dim, d);
        apus_rms_norm(mkv, w->kv_norm, cfg->eps, mkv, (size_t)d);
        ApusKvQatJob kj = { mkv, L->rope->cos + start_pos * half,
                            L->rope->sin + start_pos * half, d, rd };
        apus_kvqat_rows(&kj, 0, 1);
        memcpy(st->attn.win + (size_t)(start_pos % win) * d, mkv,
               (size_t)d * sizeof(float));
        st->attn.pos = start_pos + 1;
    }

    /* ONE SHARED non-causal topk row for all B queries (743-747, D6/D11):
     * window slots 0..min(win,start_pos+1)-1 in plain arange slot order,
     * then win+0..win+B-1 (the B draft KVs after the ring, 784) */
    int nwin = (int)(start_pos + 1 < win ? start_pos + 1 : win);
    int idxw = nwin + B;
    int32_t *idxs = apus_scratch_alloc((size_t)B * idxw * sizeof(int32_t));
    for (int t = 0; t < B; t++) {
        for (int j = 0; j < nwin; j++) idxs[(size_t)t * idxw + j] = j;
        for (int j = 0; j < B; j++)
            idxs[(size_t)t * idxw + nwin + j] = win + j;
    }
    float *o = apus_scratch_alloc((size_t)B * h * d * sizeof(float));
    float scale = (float)pow((double)d, -0.5);
    apus_sparse_attn(q, w->sink, h, d, idxs, idxw, B, scale,
                     st->attn.win, win, kv, o);
    /* inverse rope on outputs (786) at the draft positions */
    for (int t = 0; t < B; t++)
        for (int hh = 0; hh < h; hh++)
            apus_apply_rope(o + ((size_t)t * h + hh) * d + d - rd,
                            fc + (size_t)t * half, fs + (size_t)t * half,
                            rd, 1);

    /* grouped low-rank o-proj (788-791) */
    int G = cfg->o_groups, ol = cfg->o_lora, sub = h * d / G;
    float *y = apus_scratch_alloc((size_t)B * G * ol * sizeof(float));
    ApusWoAJob wj = { w->wo_a, o, y, B, G, ol, sub, h * d };
    apus_pool_run((size_t)B * G * ol, apus_woa_rows, &wj);
    apus_fp8_linear(&w->wo_b, y, out, B, G * ol, dim);
    apus_scratch_reset(mk);
}

/* One stage's block forward over the B draft rows (DSparkBlock.forward ->
 * Block.forward 695-707 with DSparkAttention): hc_pre -> attn_norm ->
 * DSpark attn -> hc_post -> hc_pre -> ffn_norm -> MoE -> hc_post.
 * h [B, hc*dim] in/out. */
static void apus_dspark_stage_forward(const ApusLayer *L, ApusLayerState *st,
                                      float *h, const int64_t *draft_ids,
                                      int B, const float *mx_row,
                                      int64_t start_pos) {
    int dim = L->cfg.dim, hc = L->cfg.hc_mult;
    ApusScratchMark mk = apus_scratch_mark();
    float *x = apus_scratch_alloc((size_t)B * dim * sizeof(float));
    float *post = apus_scratch_alloc((size_t)B * hc * sizeof(float));
    float *comb = apus_scratch_alloc((size_t)B * hc * hc * sizeof(float));
    float *sub = apus_scratch_alloc((size_t)B * dim * sizeof(float));
    float *hn = apus_scratch_alloc((size_t)B * hc * dim * sizeof(float));

    apus_hc_pre(L, h, B, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base,
                x, post, comb, NULL, NULL, NULL);
    apus_layer_norm(L, L->attn_norm_w, x, B);
    apus_dspark_attn(L, st, x, mx_row, B, start_pos, sub);
    apus_hc_post(L, sub, h, post, comb, hn, B);
    memcpy(h, hn, (size_t)B * hc * dim * sizeof(float));

    apus_hc_pre(L, h, B, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base,
                x, post, comb, NULL, NULL, NULL);
    apus_layer_norm(L, L->ffn_norm_w, x, B);
    apus_moe_forward(&L->mw, x, draft_ids, B, sub, NULL);
    apus_hc_post(L, sub, h, post, comb, hn, B);
    memcpy(h, hn, (size_t)B * hc * dim * sizeof(float));
    apus_scratch_reset(mk);
}

/* ---- draft round (forward_spec decode, model.py:928-936) ---------------------*/

void apus_dspark_draft_round(const ApusDspark *ds, ApusDsparkState *dst,
                             int64_t anchor, const float *mx_row,
                             int64_t start_pos, float temp, ApusRng *drng,
                             void *sample_scratch, int64_t *drafts,
                             ApusDsparkDbg *dbg) {
    const ApusModel *m = ds->m;
    int B = ds->cfg.block_size, dim = m->cfg.dim, hc = m->cfg.hc_mult;
    int V = m->cfg.vocab_size, rank = ds->cfg.markov_rank;
    ApusScratchMark mk = apus_scratch_mark();

    /* forward_embed (851-858): [anchor, noise x (B-1)] -> embed -> hc
     * expand (the noise embeddings are real block inputs, D10) */
    int64_t *draft_ids = apus_scratch_alloc((size_t)B * sizeof(int64_t));
    draft_ids[0] = anchor;
    for (int i = 1; i < B; i++) draft_ids[i] = ds->cfg.noise_id;
    float *h = apus_scratch_alloc((size_t)B * hc * dim * sizeof(float));
    for (int t = 0; t < B; t++)
        apus_model_embed(m, draft_ids[t], h + (size_t)t * hc * dim);

    /* the stages (each writes its ring slot start_pos%win, 783) */
    for (int k = 0; k < ds->n_stages; k++) {
        apus_dspark_stage_forward(&ds->stages[k], &dst->stages[k], h,
                                  draft_ids, B, mx_row, start_pos);
        if (dbg && dbg->stage_h[k])
            memcpy(dbg->stage_h[k], h,
                   (size_t)B * hc * dim * sizeof(float));
    }

    /* forward_head (860-874) on the last stage: own hc_head collapse
     * (sigmoid, NO Sinkhorn) -> own norm -> SHARED lm head */
    float *x = apus_scratch_alloc((size_t)B * dim * sizeof(float));
    float *yn = apus_scratch_alloc((size_t)B * dim * sizeof(float));
    for (int t = 0; t < B; t++) {
        apus_mhc_head_scalar(h + (size_t)t * hc * dim, (size_t)dim,
                             (size_t)hc, ds->hc_head_fn, ds->hc_head_scale,
                             ds->hc_head_base, m->cfg.norm_eps,
                             m->cfg.hc_eps, x + (size_t)t * dim, NULL);
        for (int i = 0; i < dim; i++)
            x[(size_t)t * dim + i] = apus_bf16_round(x[(size_t)t * dim + i]);
        apus_rms_norm(x + (size_t)t * dim, ds->norm_w, m->cfg.norm_eps,
                      yn + (size_t)t * dim, (size_t)dim);
    }
    if (dbg && dbg->conf_hidden)
        memcpy(dbg->conf_hidden, x, (size_t)B * dim * sizeof(float));
    float *logits = apus_scratch_alloc((size_t)B * V * sizeof(float));
    for (int t = 0; t < B; t++)
        apus_head_gemv(m->head, yn + (size_t)t * dim,
                       logits + (size_t)t * V, V, dim);
    if (dbg && dbg->logits_base)
        memcpy(dbg->logits_base, logits, (size_t)B * V * sizeof(float));

    /* markov bias loop (864-871): logits[i] += w2 @ w1(out_ids[i]), then
     * sample row i — the bigram bias from the PRECEDING token. markov
     * params BF16-stored, fp32 compute, no output rounding (D14). */
    float *embeds = apus_scratch_alloc((size_t)B * rank * sizeof(float));
    float *bias = apus_scratch_alloc((size_t)V * sizeof(float));
    int64_t cur = anchor;
    for (int i = 0; i < B; i++) {
        const float *e = ds->markov_w1 + (size_t)cur * rank;
        memcpy(embeds + (size_t)i * rank, e, (size_t)rank * sizeof(float));
        apus_f32_linear(ds->markov_w2, e, bias, 1, rank, V);
        if (dbg && dbg->markov_bias)
            memcpy(dbg->markov_bias + (size_t)i * V, bias,
                   (size_t)V * sizeof(float));
        float *row = logits + (size_t)i * V;
        for (int v = 0; v < V; v++) row[v] += bias[v];
        int tok;
        if (temp <= 0.0f) {
            tok = apus_sample_argmax(row, (size_t)V);
        } else {
            /* separate draft stream: one uniform per draft row (D2),
             * top_p 1.0 (the reference has no top-p, D2) */
            tok = apus_sample_logits_u(row, (size_t)V, temp, 1.0f,
                                       apus_rng_uniform(drng),
                                       sample_scratch);
        }
        drafts[i] = tok;
        cur = tok;
    }
    if (dbg && dbg->logits_final)
        memcpy(dbg->logits_final, logits, (size_t)B * V * sizeof(float));
    if (dbg && dbg->markov_embed)
        memcpy(dbg->markov_embed, embeds, (size_t)B * rank * sizeof(float));

    /* confidence (872-873, D4/D14): proj(cat([pre-norm collapse, markov
     * embeds])), fp32, no rounding. Telemetry only — no part in the
     * accept rule (D1). */
    if (dbg && dbg->confidence) {
        float *ci = apus_scratch_alloc((size_t)B * (dim + rank)
                                       * sizeof(float));
        for (int t = 0; t < B; t++) {
            memcpy(ci + (size_t)t * (dim + rank), x + (size_t)t * dim,
                   (size_t)dim * sizeof(float));
            memcpy(ci + (size_t)t * (dim + rank) + dim,
                   embeds + (size_t)t * rank, (size_t)rank * sizeof(float));
        }
        apus_f32_linear(ds->conf_proj, ci, dbg->confidence, B, dim + rank,
                        1);
    }
    apus_scratch_reset(mk);
}

/* ---- speculative decode engine ------------------------------------------------*/

void apus_dspec_init(ApusDspec *sp, const ApusModel *m, ApusModelState *st,
                     const ApusDspark *ds, ApusDsparkState *dst,
                     float temp, float top_p, ApusRng *rng,
                     uint64_t draft_seed, void *sample_scratch) {
    memset(sp, 0, sizeof *sp);
    sp->m = m;
    sp->st = st;
    sp->ds = ds;
    sp->dst = dst;
    sp->temp = temp;
    sp->top_p = top_p;
    sp->rng = rng;
    apus_rng_seed(&sp->drng, draft_seed);
    sp->sample_scratch = sample_scratch;
    sp->B = ds->cfg.block_size;
    sp->V = m->cfg.vocab_size;
    sp->mhdim = ds->cfg.n_targets * m->cfg.dim;
    int B = sp->B, V = sp->V;
    sp->mh_last = malloc((size_t)sp->mhdim * sizeof(float));
    sp->drafts = malloc((size_t)B * sizeof(int64_t));
    sp->batch = malloc((size_t)(B + 1) * sizeof(int64_t));
    sp->R = malloc((size_t)(B + 1) * V * sizeof(float));
    sp->H = malloc((size_t)(B + 1) * sp->mhdim * sizeof(float));
    sp->mx = malloc((size_t)m->cfg.dim * sizeof(float));
    sp->refeed_logits = malloc((size_t)V * sizeof(float));
    apus_snap_alloc(&sp->snap, m->layers, st->layers, m->n_layers);
}

void apus_dspec_free(ApusDspec *sp) {
    free(sp->mh_last);
    free(sp->drafts);
    free(sp->batch);
    free(sp->R);
    free(sp->H);
    free(sp->mx);
    free(sp->refeed_logits);
    apus_snap_free(&sp->snap);
    memset(sp, 0, sizeof *sp);
}

int apus_dspec_prefill(ApusDspec *sp, const int64_t *ids, int n) {
    const ApusModel *m = sp->m;
    int V = sp->V, dim = m->cfg.dim;
    float *logits = malloc((size_t)V * sizeof(float));
    float *mh_all = malloc((size_t)n * sp->mhdim * sizeof(float));
    /* main prefill with main_hidden collection */
    apus_model_forward_mh(m, sp->st, ids, n, logits, 0, mh_all);
    /* first held token: exactly what non-spec sampling draws first */
    sp->held = apus_sample(logits, (size_t)V, sp->temp, sp->top_p,
                           sp->rng, sp->sample_scratch);
    sp->f = n - 1;
    memcpy(sp->mh_last, mh_all + (size_t)(n - 1) * sp->mhdim,
           (size_t)sp->mhdim * sizeof(float));
    /* stage-ring prefill from the SHARED stage-0 projection (D7) */
    float *mx_all = malloc((size_t)n * dim * sizeof(float));
    apus_dspark_stage0_project(sp->ds, mh_all, n, mx_all);
    apus_dspark_prefill(sp->ds, sp->dst, mx_all, n);
    free(mx_all);
    free(mh_all);
    free(logits);
    return 0;
}

int apus_dspec_step(ApusDspec *sp, int *out, int cap) {
    const ApusModel *m = sp->m;
    int B = sp->B, V = sp->V;
    size_t mhdim = (size_t)sp->mhdim;
    if (cap < B + 1) return 0;
    /* the verify batch feeds B_eff+1 tokens at f+1..f+1+B_eff; the last
     * fed position must stay inside the rope tables (M8's guard shape:
     * q + D > maxp clamps D) */
    int64_t maxp = m->cfg.max_pos;
    int B_eff = B;
    if (sp->f + 2 + B_eff > maxp) B_eff = (int)(maxp - sp->f - 2);
    if (B_eff <= 0) return 0;

    /* 1. draft at start_pos = f (stage rings hold true 0..f) */
    apus_dspark_stage0_project(sp->ds, sp->mh_last, 1, sp->mx);
    if (sp->draft_override) {
        /* forced drafts (tests): the slot-f ring write still happens
         * (oracle parity: pos bookkeeping matches the draft round's) */
        apus_dspark_catchup(sp->ds, sp->dst, sp->mx, sp->f, 1);
        for (int k = 0; k < sp->ds->n_stages; k++)
            sp->dst->stages[k].attn.pos = sp->f + 1;
        sp->draft_override(sp->draft_ctx, sp->f, sp->held, sp->drafts, B);
    } else {
        apus_dspark_draft_round(sp->ds, sp->dst, sp->held, sp->mx, sp->f,
                                sp->temp, &sp->drng, sp->sample_scratch,
                                sp->drafts, NULL);
    }

    /* 2. snapshot + verify batch [held, d1..d_{B_eff}] at f+1..f+B_eff+1,
     * ONE batched forward (m4c/M8 chunk-invariance: row j is bitwise the
     * one-by-one decode of the same context) */
    apus_snap_take(&sp->snap, sp->st->layers, sp->st->pos);
    sp->batch[0] = sp->held;
    for (int j = 1; j <= B_eff; j++) sp->batch[j] = sp->drafts[j - 1];
    if (sp->pre_batch)
        sp->pre_batch(sp->pre_batch_ctx, sp->batch, B_eff + 1, sp->f + 1);
    apus_model_forward_mh(m, sp->st, sp->batch, B_eff + 1, sp->R, 1, sp->H);
    sp->batches++;
    sp->batch_tokens += (uint64_t)(B_eff + 1);

    /* 3. the walk (THE ACCEPT RULE): every drawn token is the main model's
     * own draw from its own logits row — one RNG uniform per token, in
     * position order. The token that ends the walk (mismatch replacement
     * or full-match bonus) becomes the next HELD token: drawn, NOT yet
     * emitted. */
    int ne = 0, a = 0;
    int64_t newheld = -1;
    out[ne++] = (int)sp->held;
    for (int j = 1; j <= B_eff; j++) {
        int mj = apus_sample(sp->R + (size_t)(j - 1) * V, (size_t)V,
                             sp->temp, sp->top_p, sp->rng,
                             sp->sample_scratch);
        sp->offered++;
        if ((int)sp->drafts[j - 1] == mj) {
            a = j;
            sp->accepted++;
            out[ne++] = mj;
        } else {
            newheld = mj;
            break;
        }
    }
    if (newheld < 0) {
        /* full match: bonus token from R[B_eff] (valid — the whole batch
         * proved true) */
        newheld = apus_sample(sp->R + (size_t)B_eff * V, (size_t)V,
                              sp->temp, sp->top_p, sp->rng,
                              sp->sample_scratch);
        sp->bonus_rounds++;
    }
    sp->emitted += (uint64_t)ne;

    /* 4. state fixup: fed-true run = batch[0..a]. Full match keeps the
     * batch state; partial = snapshot restore (truncation) + one batched
     * re-feed (bitwise the one-by-one state by chunk invariance). The
     * verify batch's rows/hiddens stay valid and are reused. */
    if (a < B_eff) {
        apus_snap_restore(&sp->snap, sp->st->layers, &sp->st->pos);
        apus_model_forward(m, sp->st, sp->batch, a + 1,
                           sp->refeed_logits, 0);
        sp->refeed_tokens += (uint64_t)(a + 1);
    }

    /* 5. stage-ring maintenance (D13): write the slots for ALL newly-true
     * fed positions f+1..f+1+a from the verify batch's hiddens H[0..a].
     * The next round's in-attention slot write rewrites slot f' with the
     * identical value — idempotent. */
    {
        float *mxw = malloc((size_t)(a + 1) * m->cfg.dim * sizeof(float));
        apus_dspark_stage0_project(sp->ds, sp->H, a + 1, mxw);
        apus_dspark_catchup(sp->ds, sp->dst, mxw, sp->f + 1, a + 1);
        free(mxw);
    }

    sp->held = newheld;
    sp->f = sp->f + 1 + a;
    memcpy(sp->mh_last, sp->H + (size_t)a * mhdim, mhdim * sizeof(float));
    return ne;
}

#endif /* APUS_DSPARK_IMPLEMENTATION */
#endif /* APUS_DSPARK_H */
