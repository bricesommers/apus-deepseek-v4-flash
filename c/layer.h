/*
 * c/layer.h — one DeepSeek-V4-Flash transformer Block (M4c): mHC wiring
 * around the attention (c/attn.h) and MoE (c/moe.h) sublayers, plus the
 * fixture/config-driven weight loader built on c/st.h. C11.
 *
 * Normative reference: tools/oracle.py block_forward (f32 mode) porting
 * reference/inference/model.py:647-700:
 *   h -> hc_pre(attn) -> attn_norm -> attention -> hc_post
 *     -> hc_pre(ffn)  -> ffn_norm  -> moe        -> hc_post -> h
 * mHC per token (c/mhc.h): flatten [4,dim] -> rsqrt over the UNNORMALIZED
 * vector, mixes in FP32 (rsqrt AFTER the matmul), sinkhorn-20 in the exact
 * kernel order, y = sum pre*x BF16-rounded; hc_post y_j = post_j*x +
 * sum_i comb[i,j]*res_i BF16-rounded.
 *
 * Usage: #define APUS_LAYER_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_LAYER_H
#define APUS_LAYER_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"
#include "attn.h"
#include "moe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model config (fixture config.json / real config.json). */
typedef struct {
    int dim, n_heads, head_dim, rope_head_dim, q_lora_rank;
    int o_groups, o_lora_rank, window_size;
    int moe_inter_dim, n_routed_experts, n_activated_experts;
    int index_n_heads, index_head_dim, index_topk;
    int hc_mult, hc_sinkhorn_iters;
    int original_seq_len, beta_fast, beta_slow, vocab_size, max_pos;
    float route_scale, swiglu_limit, hc_eps, norm_eps;
    double rope_theta, compress_rope_theta, rope_factor;
} ApusCfg;

typedef struct {
    ApusCfg cfg;
    int ratio;          /* 0 = swa, 4 = csa, 128 = hca */
    int hash;           /* hash routing (gate.tid2eid) */
    /* owned buffers (f32; BF16 tensors widened, FP8/FP4 stay as views) */
    uint16_t *wo_a;     /* BF16 bits (M6c: was f32; widening at use is exact) */
    float *q_norm, *kv_norm, *attn_sink, *attn_norm_w, *ffn_norm_w;
    float *comp_wkv, *comp_wgate, *comp_ape, *comp_norm;
    float *idx_wkv, *idx_wgate, *idx_ape, *idx_norm, *idx_wproj;
    float *gate_w, *gate_bias;
    int64_t *tid2eid;
    ApusFp4W *ew1, *ew2, *ew3;
    float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    float *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    /* shared lazily-grown rope table (see implementation): one table per
     * (theta class, YaRN params) across all layers, rows built on demand
     * up to the max position actually used — NOT [max_pos] per layer
     * (real max_pos = 1M would cost ~270 MB x 2 x 43 layers). */
    struct ApusRopeTab *rope;
    ApusAttnCfg acfg;
    ApusAttnW aw;
    ApusMoeW mw;
    /* M6b pilot hook (c/pilot.h): invoked with the post-attention hidden
     * state h [s, hc*dim] right after the attention half, before the ffn
     * half. Must never block the compute thread. NULL = no pilot. */
    void *pilot_ctx;
    void (*pilot_post_attn)(void *ctx, int layer, const float *h, int s,
                            int64_t start_pos);
} ApusLayer;

/* Load layer `layer_idx` (real checkpoint naming) from the shard set.
 * Returns 0 on success. */
int  apus_layer_load(ApusLayer *L, ApusStSet *set, const ApusCfg *cfg,
                     int layer_idx, int ratio, int hash,
                     char *err, size_t errcap);
/* Named-prefix variant (M8): prefix is the tensor namespace, e.g. "layers.5"
 * or "mtp.0"; layer_id is the MoE expert-store index (mw.layer_id), which
 * for MTP blocks is n_main_layers + mtp_idx (see c/mtp.h, c/cache.h). */
int  apus_layer_load_named(ApusLayer *L, ApusStSet *set, const ApusCfg *cfg,
                           const char *prefix, int ratio, int hash,
                           int layer_id, char *err, size_t errcap);
void apus_layer_free(ApusLayer *L);

typedef struct {
    ApusAttnS attn;
} ApusLayerState;

void apus_layer_state_init(ApusLayerState *st, const ApusLayer *L);
void apus_layer_state_free(ApusLayerState *st, const ApusLayer *L);

/* Block intermediates (tests/m4b INTERM_KEYS). All optional; sizes:
 * hc_pre/post [s,4], hc_comb [s,16], *_norm_out/o_out/moe_* [s,dim],
 * post_attn_h [s,4,dim], router_scores/biased [s,E], router_idx/w [s,topk],
 * attention-level fields per ApusAttnInterm. */
typedef struct {
    float *attn_hc_pre, *attn_hc_post, *attn_hc_comb;
    float *attn_norm_out;
    ApusAttnInterm attn;
    float *post_attn_h;
    float *ffn_hc_pre, *ffn_hc_post, *ffn_hc_comb;
    float *ffn_norm_out;
    ApusMoeInterm moe;
    float *moe_out;         /* [s,dim] sublayer output before hc_post */
} ApusInterm;

/* Full block forward. h [s, hc_mult, dim] in/out (BF16 values in f32),
 * ids [s] input token ids, start_pos = 0 prefill / >0 decode (s == 1). */
void apus_block_forward(const ApusLayer *L, ApusLayerState *st,
                        float *h, const int64_t *ids, int s,
                        int64_t start_pos, ApusInterm *interm);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_LAYER_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mhc.h"

/* --- shared lazy rope tables -------------------------------------------------
 * One ApusRopeTab per (rope_head_dim, theta, YaRN params) class: SWA layers
 * share the plain-theta table, CSA/HCA layers share the compress-theta YaRN
 * table. Rows are computed on demand (geometric growth) via
 * apus_rope_precompute_range, so values are bit-identical to a full
 * apus_rope_precompute at max_pos while short contexts stay tiny. The
 * registry lives for the process lifetime (a few hundred KB per class);
 * layers hold a non-owning pointer. Compute-thread only (like the st.h
 * resolve cache). */
typedef struct ApusRopeTab {
    double base, factor;
    int osl, beta_fast, beta_slow, rd;
    int64_t cap;            /* rows built so far (geometric growth) */
    float *cos, *sin;       /* [cap, rd/2] */
    struct ApusRopeTab *next;
} ApusRopeTab;

static ApusRopeTab *apus_rope_tabs;

static ApusRopeTab *apus_rope_tab_get(const ApusCfg *cfg, int ratio) {
    double base = ratio ? cfg->compress_rope_theta : cfg->rope_theta;
    int osl = ratio ? cfg->original_seq_len : 0;   /* 0 disables YaRN (SWA) */
    for (ApusRopeTab *t = apus_rope_tabs; t; t = t->next)
        if (t->base == base && t->factor == cfg->rope_factor
            && t->osl == osl && t->beta_fast == cfg->beta_fast
            && t->beta_slow == cfg->beta_slow && t->rd == cfg->rope_head_dim)
            return t;
    ApusRopeTab *t = calloc(1, sizeof *t);
    t->base = base;
    t->factor = cfg->rope_factor;
    t->osl = osl;
    t->beta_fast = cfg->beta_fast;
    t->beta_slow = cfg->beta_slow;
    t->rd = cfg->rope_head_dim;
    t->next = apus_rope_tabs;
    apus_rope_tabs = t;
    return t;
}

static void apus_rope_tab_ensure(ApusRopeTab *t, int64_t rows) {
    if (rows <= t->cap) return;
    int64_t cap = t->cap ? t->cap : 1024;
    while (cap < rows) cap *= 2;
    int half = t->rd / 2;
    t->cos = realloc(t->cos, (size_t)cap * half * sizeof(float));
    t->sin = realloc(t->sin, (size_t)cap * half * sizeof(float));
    apus_rope_precompute_range(t->cos, t->sin, t->rd, t->cap, cap,
                               t->osl, t->base, t->factor,
                               t->beta_fast, t->beta_slow);
    t->cap = cap;
}

static float *apus_load_f32(ApusStSet *set, const char *name, size_t n,
                            char *err, size_t errcap) {
    const ApusStTensor *t = apus_st_set_get(set, name);
    if (!t || (t->dtype != APUS_ST_F32 && t->dtype != APUS_ST_BF16)
        || apus_st_nelem(t) != n) {
        snprintf(err, errcap, "layer: bad tensor %s", name);
        return NULL;
    }
    float *out = malloc(n * sizeof(float));
    apus_st_f32(t, out, n);
    return out;
}

int apus_layer_load(ApusLayer *L, ApusStSet *set, const ApusCfg *cfg,
                    int layer_idx, int ratio, int hash,
                    char *err, size_t errcap) {
    char prefix[32];
    snprintf(prefix, sizeof prefix, "layers.%d", layer_idx);
    return apus_layer_load_named(L, set, cfg, prefix, ratio, hash, layer_idx,
                                 err, errcap);
}

int apus_layer_load_named(ApusLayer *L, ApusStSet *set, const ApusCfg *cfg,
                          const char *prefix, int ratio, int hash,
                          int layer_id, char *err, size_t errcap) {
    memset(L, 0, sizeof *L);
    L->cfg = *cfg;
    L->ratio = ratio;
    L->hash = hash;
    int dim = cfg->dim, h = cfg->n_heads, d = cfg->head_dim;
    int ql = cfg->q_lora_rank, G = cfg->o_groups, ol = cfg->o_lora_rank;
    int E = cfg->n_routed_experts, topk = cfg->n_activated_experts;
    int hc = cfg->hc_mult, mix = (2 + hc) * hc;
    char n[192];
#define APUS_N(...) (snprintf(n, sizeof n, __VA_ARGS__), n)
#define APUS_LOAD(field, namestr, count) do { \
        L->field = apus_load_f32(set, namestr, (count), err, errcap); \
        if (!L->field) return -1; } while (0)

    /* attention dense (FP8 views) */
    if (apus_st_fp8w(set, APUS_N("%s.attn.wq_a", prefix), &L->aw.wq_a)
     || apus_st_fp8w(set, APUS_N("%s.attn.wq_b", prefix), &L->aw.wq_b)
     || apus_st_fp8w(set, APUS_N("%s.attn.wkv", prefix), &L->aw.wkv)
     || apus_st_fp8w(set, APUS_N("%s.attn.wo_b", prefix), &L->aw.wo_b)) {
        snprintf(err, errcap, "%s: fp8 attn weights", prefix);
        return -1;
    }
    /* wo_a: FP8 on disk, used as BF16 (A4): dequant + bf16 round.
     * M6c: stored as BF16 BITS (u16), halving the largest dense block
     * (5.8 GB -> 2.9 GB anon on the real model); widening at use is exact,
     * so values are bitwise identical to the old f32 storage. */
    {
        ApusFp8W wa;
        if (apus_st_fp8w(set, APUS_N("%s.attn.wo_a", prefix), &wa)) {
            snprintf(err, errcap, "%s: wo_a", prefix);
            return -1;
        }
        size_t cnt = (size_t)wa.O * wa.K;
        L->wo_a = malloc(cnt * sizeof(uint16_t));
        size_t nbk = (size_t)(wa.K + 127) / 128;
        for (int64_t o = 0; o < wa.O; o++)
            for (int64_t k = 0; k < wa.K; k++) {
                float sc = apus_ue8m0_f32(wa.scales[(o / 128) * nbk + k / 128]);
                float v = apus_e4m3_dequant_f32(wa.codes[o * wa.K + k]) * sc;
                L->wo_a[o * wa.K + k] = apus_bf16_bits(v);
            }
    }
    APUS_LOAD(q_norm, APUS_N("%s.attn.q_norm.weight", prefix), ql);
    APUS_LOAD(kv_norm, APUS_N("%s.attn.kv_norm.weight", prefix), d);
    APUS_LOAD(attn_sink, APUS_N("%s.attn.attn_sink", prefix), h);
    APUS_LOAD(attn_norm_w, APUS_N("%s.attn_norm.weight", prefix), dim);
    APUS_LOAD(ffn_norm_w, APUS_N("%s.ffn_norm.weight", prefix), dim);

    /* compressors */
    if (ratio) {
        int coff = ratio == 4 ? 2 : 1;
        APUS_LOAD(comp_wkv, APUS_N("%s.attn.compressor.wkv.weight", prefix), (size_t)coff * d * dim);
        APUS_LOAD(comp_wgate, APUS_N("%s.attn.compressor.wgate.weight", prefix), (size_t)coff * d * dim);
        APUS_LOAD(comp_ape, APUS_N("%s.attn.compressor.ape", prefix), (size_t)ratio * coff * d);
        APUS_LOAD(comp_norm, APUS_N("%s.attn.compressor.norm.weight", prefix), d);
        if (ratio == 4) {
            int idm = cfg->index_head_dim;
            if (apus_st_fp8w(set, APUS_N("%s.attn.indexer.wq_b", prefix), &L->aw.idx_wq_b)) {
                snprintf(err, errcap, "%s: indexer wq_b", prefix);
                return -1;
            }
            APUS_LOAD(idx_wproj, APUS_N("%s.attn.indexer.weights_proj.weight", prefix), (size_t)cfg->index_n_heads * dim);
            APUS_LOAD(idx_wkv, APUS_N("%s.attn.indexer.compressor.wkv.weight", prefix), (size_t)2 * idm * dim);
            APUS_LOAD(idx_wgate, APUS_N("%s.attn.indexer.compressor.wgate.weight", prefix), (size_t)2 * idm * dim);
            APUS_LOAD(idx_ape, APUS_N("%s.attn.indexer.compressor.ape", prefix), (size_t)ratio * 2 * idm);
            APUS_LOAD(idx_norm, APUS_N("%s.attn.indexer.compressor.norm.weight", prefix), idm);
        }
    }

    /* MoE */
    APUS_LOAD(gate_w, APUS_N("%s.ffn.gate.weight", prefix), (size_t)E * dim);
    if (hash) {
        const ApusStTensor *t =
            apus_st_set_get(set, APUS_N("%s.ffn.gate.tid2eid", prefix));
        if (!t || t->dtype != APUS_ST_I64) {
            snprintf(err, errcap, "%s: tid2eid", prefix);
            return -1;
        }
        L->tid2eid = malloc((size_t)cfg->vocab_size * topk * sizeof(int64_t));
        apus_st_i64(t, L->tid2eid, (size_t)cfg->vocab_size * topk);
    } else {
        APUS_LOAD(gate_bias, APUS_N("%s.ffn.gate.bias", prefix), E);
    }
    L->ew1 = calloc((size_t)E, sizeof(ApusFp4W));
    L->ew2 = calloc((size_t)E, sizeof(ApusFp4W));
    L->ew3 = calloc((size_t)E, sizeof(ApusFp4W));
    /* tiered mode (M6a, st->defer_experts): expert views stay empty and are
     * resolved on demand through the ApusMoeW hook (c/cache.h) — nothing
     * expert-sized is slurped. */
    if (!apus_st_set_deferred(set))
        for (int e = 0; e < E; e++)
            if (apus_st_fp4w(set, APUS_N("%s.ffn.experts.%d.w1", prefix, e), &L->ew1[e])
             || apus_st_fp4w(set, APUS_N("%s.ffn.experts.%d.w2", prefix, e), &L->ew2[e])
             || apus_st_fp4w(set, APUS_N("%s.ffn.experts.%d.w3", prefix, e), &L->ew3[e])) {
                snprintf(err, errcap, "%s expert %d", prefix, e);
                return -1;
            }
    if (apus_st_fp8w(set, APUS_N("%s.ffn.shared_experts.w1", prefix), &L->mw.sw1)
     || apus_st_fp8w(set, APUS_N("%s.ffn.shared_experts.w2", prefix), &L->mw.sw2)
     || apus_st_fp8w(set, APUS_N("%s.ffn.shared_experts.w3", prefix), &L->mw.sw3)) {
        snprintf(err, errcap, "%s: shared experts", prefix);
        return -1;
    }

    /* mHC (all F32) */
    APUS_LOAD(hc_attn_fn, APUS_N("%s.hc_attn_fn", prefix), (size_t)mix * hc * dim);
    APUS_LOAD(hc_attn_base, APUS_N("%s.hc_attn_base", prefix), mix);
    APUS_LOAD(hc_attn_scale, APUS_N("%s.hc_attn_scale", prefix), 3);
    APUS_LOAD(hc_ffn_fn, APUS_N("%s.hc_ffn_fn", prefix), (size_t)mix * hc * dim);
    APUS_LOAD(hc_ffn_base, APUS_N("%s.hc_ffn_base", prefix), mix);
    APUS_LOAD(hc_ffn_scale, APUS_N("%s.hc_ffn_scale", prefix), 3);
#undef APUS_LOAD
#undef APUS_N

    /* rope table: shared per theta class, rows built lazily on first use
     * (model.py:476-482): SWA plain theta, CSA/HCA + compressors + indexer
     * share the YaRN compress theta */
    L->rope = apus_rope_tab_get(cfg, ratio);
    int rd = cfg->rope_head_dim;

    /* sub-structs */
    L->acfg = (ApusAttnCfg){
        .dim = dim, .n_heads = h, .head_dim = d, .rope_dim = rd,
        .q_lora = ql, .o_groups = G, .o_lora = ol,
        .window = cfg->window_size, .ratio = ratio,
        .has_indexer = ratio == 4,
        .idx_heads = cfg->index_n_heads, .idx_dim = cfg->index_head_dim,
        .idx_topk = cfg->index_topk,
        .eps = cfg->norm_eps, .max_pos = cfg->max_pos,
    };
    L->aw.wo_a = L->wo_a;
    L->aw.q_norm = L->q_norm;
    L->aw.kv_norm = L->kv_norm;
    L->aw.sink = L->attn_sink;
    if (ratio) {
        L->aw.comp = (ApusCompW){ L->comp_wkv, L->comp_wgate, L->comp_ape,
                                  L->comp_norm, ratio, ratio == 4, d, rd, 0,
                                  cfg->norm_eps };
        if (ratio == 4)
            L->aw.idx_comp = (ApusCompW){ L->idx_wkv, L->idx_wgate, L->idx_ape,
                                          L->idx_norm, ratio, 1,
                                          cfg->index_head_dim, rd, 1,
                                          cfg->norm_eps };
        L->aw.idx_wproj = L->idx_wproj;
    }
    L->mw.E = E;
    L->mw.topk = topk;
    L->mw.inter = cfg->moe_inter_dim;
    L->mw.dim = dim;
    L->mw.hash = hash;
    L->mw.route_scale = cfg->route_scale;
    L->mw.limit = cfg->swiglu_limit;
    L->mw.gate_w = L->gate_w;
    L->mw.gate_bias = L->gate_bias;
    L->mw.tid2eid = L->tid2eid;
    L->mw.w1 = L->ew1;
    L->mw.w2 = L->ew2;
    L->mw.w3 = L->ew3;
    L->mw.layer_id = layer_id;
    return 0;
}

void apus_layer_free(ApusLayer *L) {
    free(L->wo_a); free(L->q_norm); free(L->kv_norm); free(L->attn_sink);
    free(L->attn_norm_w); free(L->ffn_norm_w);
    free(L->comp_wkv); free(L->comp_wgate); free(L->comp_ape); free(L->comp_norm);
    free(L->idx_wkv); free(L->idx_wgate); free(L->idx_ape); free(L->idx_norm);
    free(L->idx_wproj);
    free(L->gate_w); free(L->gate_bias); free(L->tid2eid);
    free(L->ew1); free(L->ew2); free(L->ew3);
    free(L->hc_attn_fn); free(L->hc_attn_base); free(L->hc_attn_scale);
    free(L->hc_ffn_fn); free(L->hc_ffn_base); free(L->hc_ffn_scale);
    memset(L, 0, sizeof *L);       /* L->rope is shared, not owned */
}

void apus_layer_state_init(ApusLayerState *st, const ApusLayer *L) {
    apus_attn_state_init(&st->attn, &L->acfg);
}

void apus_layer_state_free(ApusLayerState *st, const ApusLayer *L) {
    apus_attn_state_free(&st->attn, &L->acfg);
}

/* mHC pre (model.py:673-681): returns x [s,dim] BF16 + post/comb per token.
 * M6c note: kept on the SCALAR mhc kernels on purpose — the collapse output
 * feeds the compressor QAT paths whose m4c bitwise-diff bounds leave no room
 * for NEON reorder noise, and the mixes matmul is only ~1% of decode time.
 * M9d: the per-token loop is pooled over tokens at larger s — each token's
 * op sequence is exactly the serial loop's (token-independent scalar
 * kernels), so this is bitwise identical for any thread count. */
typedef struct {
    const ApusLayer *L;
    const float *h, *fn, *scale, *base;
    float *x, *post, *comb, *i_pre, *i_post, *i_comb;
} ApusHcPreJob;

static void apus_hc_pre_rows(void *vjob, size_t t0, size_t t1) {
    const ApusHcPreJob *j = vjob;
    int hc = j->L->cfg.hc_mult, dim = j->L->cfg.dim;
    float pre[APUS_MHC_MULT], po[APUS_MHC_MULT], cb[APUS_MHC_MULT * APUS_MHC_MULT];
    for (size_t t = t0; t < t1; t++) {
        const float *x4 = j->h + t * (size_t)hc * dim;
        apus_mhc_prepost_scalar(x4, (size_t)dim, (size_t)hc, j->fn, j->scale,
                                j->base, j->L->cfg.norm_eps, j->L->cfg.hc_eps,
                                j->L->cfg.hc_sinkhorn_iters, pre, po, cb, NULL);
        apus_mhc_collapse_scalar(x4, pre, j->x + t * (size_t)dim, (size_t)dim,
                                 (size_t)hc);
        for (int i = 0; i < dim; i++)
            j->x[t * (size_t)dim + i] = apus_bf16_round(j->x[t * (size_t)dim + i]);
        memcpy(j->post + t * (size_t)hc, po, (size_t)hc * sizeof(float));
        memcpy(j->comb + t * (size_t)hc * hc, cb, (size_t)hc * hc * sizeof(float));
        if (j->i_pre) memcpy(j->i_pre + t * (size_t)hc, pre, (size_t)hc * sizeof(float));
        if (j->i_post) memcpy(j->i_post + t * (size_t)hc, po, (size_t)hc * sizeof(float));
        if (j->i_comb) memcpy(j->i_comb + t * (size_t)hc * hc, cb,
                              (size_t)hc * hc * sizeof(float));
    }
}

static void apus_hc_pre(const ApusLayer *L, const float *h, int s,
                        const float *fn, const float *scale, const float *base,
                        float *x, float *post, float *comb,
                        float *i_pre, float *i_post, float *i_comb) {
    ApusHcPreJob j = { L, h, fn, scale, base, x, post, comb,
                       i_pre, i_post, i_comb };
    if (s >= 4)
        apus_pool_run((size_t)s, apus_hc_pre_rows, &j);
    else
        apus_hc_pre_rows(&j, 0, (size_t)s);
}

/* mHC post (model.py:683-686): h_j = post_j*x + sum_i comb[i,j]*res_i,
 * BF16-rounded. Uses c/mhc.h's apply, which implements the reference
 * comb[residual i][output j] indexing (fixed at M5 — see tests/m4c/README
 * note 1). M9d: pooled over tokens like apus_hc_pre (bitwise identical). */
typedef struct {
    const ApusLayer *L;
    const float *x, *h, *post, *comb;
    float *out;
} ApusHcPostJob;

static void apus_hc_post_rows(void *vjob, size_t t0, size_t t1) {
    const ApusHcPostJob *j = vjob;
    int hc = j->L->cfg.hc_mult, dim = j->L->cfg.dim;
    for (size_t t = t0; t < t1; t++) {
        apus_mhc_apply_scalar(j->x + t * (size_t)dim, j->h + t * (size_t)hc * dim,
                              j->post + t * (size_t)hc,
                              j->comb + t * (size_t)hc * hc,
                              j->out + t * (size_t)hc * dim, (size_t)dim,
                              (size_t)hc);
        float *y = j->out + t * (size_t)hc * dim;
        for (size_t i = 0; i < (size_t)hc * dim; i++)
            y[i] = apus_bf16_round(y[i]);
    }
}

static void apus_hc_post(const ApusLayer *L, const float *x, const float *h,
                         const float *post, const float *comb, float *out,
                         int s) {
    ApusHcPostJob j = { L, x, h, post, comb, out };
    if (s >= 4)
        apus_pool_run((size_t)s, apus_hc_post_rows, &j);
    else
        apus_hc_post_rows(&j, 0, (size_t)s);
}

/* M9d: per-token RMSNorm rows (attn/ffn norm in block_forward) — pooled
 * over tokens at larger s; each row is an independent apus_rms_norm call,
 * bitwise identical to the serial loop for any thread count. */
typedef struct {
    const float *x, *w;
    float eps;
    float *y;
    size_t n;
} ApusLayerNormJob;

static void apus_layer_norm_rows(void *vjob, size_t t0, size_t t1) {
    const ApusLayerNormJob *j = vjob;
    for (size_t t = t0; t < t1; t++)
        apus_rms_norm(j->x + t * j->n, j->w, j->eps, j->y + t * j->n, j->n);
}

static void apus_layer_norm(const ApusLayer *L, const float *w,
                            float *x, int s) {
    ApusLayerNormJob j = { x, w, L->cfg.norm_eps, x, (size_t)L->cfg.dim };
    if (s >= 8)
        apus_pool_run((size_t)s, apus_layer_norm_rows, &j);
    else
        apus_layer_norm_rows(&j, 0, (size_t)s);
}

void apus_block_forward(const ApusLayer *L, ApusLayerState *st,
                        float *h, const int64_t *ids, int s,
                        int64_t start_pos, ApusInterm *interm) {    int dim = L->cfg.dim, hc = L->cfg.hc_mult;
    ApusScratchMark mk = apus_scratch_mark();
    float *x = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    float *post = apus_scratch_alloc((size_t)s * hc * sizeof(float));
    float *comb = apus_scratch_alloc((size_t)s * hc * hc * sizeof(float));
    float *sub = apus_scratch_alloc((size_t)s * dim * sizeof(float));

    /* attention half (model.py:690-693) */
    apus_hc_pre(L, h, s, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base,
                x, post, comb,
                interm ? interm->attn_hc_pre : NULL,
                interm ? interm->attn_hc_post : NULL,
                interm ? interm->attn_hc_comb : NULL);
    apus_layer_norm(L, L->attn_norm_w, x, s);
    if (interm && interm->attn_norm_out)
        memcpy(interm->attn_norm_out, x, (size_t)s * dim * sizeof(float));
    /* rope rows needed this call: q/kv use start_pos..start_pos+s-1; the
     * compressor's entry positions never exceed that range */
    apus_rope_tab_ensure(L->rope, start_pos + s);
    apus_attention(&L->acfg, &L->aw, &st->attn, x, s, start_pos,
                   L->rope->cos, L->rope->sin, sub,
                   interm ? &interm->attn : NULL);
    float *hn = apus_scratch_alloc((size_t)s * hc * dim * sizeof(float));
    apus_hc_post(L, sub, h, post, comb, hn, s);
    memcpy(h, hn, (size_t)s * hc * dim * sizeof(float));
    if (interm && interm->post_attn_h)
        memcpy(interm->post_attn_h, h, (size_t)s * hc * dim * sizeof(float));
    /* M6b pilot: layer L+1's router lookahead on the post-attention state.
     * Read-only on h; never touches numerics. */
    if (L->pilot_post_attn)
        L->pilot_post_attn(L->pilot_ctx, L->mw.layer_id, h, s, start_pos);

    /* ffn half (model.py:696-699) */
    apus_hc_pre(L, h, s, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base,
                x, post, comb,
                interm ? interm->ffn_hc_pre : NULL,
                interm ? interm->ffn_hc_post : NULL,
                interm ? interm->ffn_hc_comb : NULL);
    apus_layer_norm(L, L->ffn_norm_w, x, s);
    if (interm && interm->ffn_norm_out)
        memcpy(interm->ffn_norm_out, x, (size_t)s * dim * sizeof(float));
    apus_moe_forward(&L->mw, x, ids, s, sub, interm ? &interm->moe : NULL);
    if (interm && interm->moe_out)
        memcpy(interm->moe_out, sub, (size_t)s * dim * sizeof(float));
    apus_hc_post(L, sub, h, post, comb, hn, s);
    memcpy(h, hn, (size_t)s * hc * dim * sizeof(float));

    apus_scratch_reset(mk);
}

#endif /* APUS_LAYER_IMPLEMENTATION */
#endif /* APUS_LAYER_H */
