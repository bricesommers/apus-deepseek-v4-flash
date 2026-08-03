/*
 * c/model.h — full DeepSeek-V4-Flash model state and forward pass (M5):
 * embedding -> N transformer blocks (c/layer.h) -> final hc_head 4->1
 * collapse -> RMSNorm -> LM head -> FP32 logits. C11.
 *
 * Normative reference: reference/inference/model.py Transformer.forward
 * (802-808), ParallelHead.forward/hc_head (714-735):
 *   h = embed(ids); h = repeat(h, hc_mult, axis=1)        # 4x replicate
 *   for layer in layers: h = block(h, start_pos, ids)
 *   y = hc_head(h)         # sigmoid-gated collapse, NO Sinkhorn, bf16 out
 *   yn = norm(y)           # RMSNorm, bf16 out
 *   logits = yn.float() @ head.weight.T                  # FP32, unrounded
 * The reference computes logits for the last position only; this header can
 * compute all positions (logits_all) for golden verification — same math
 * row-wise.
 *
 * Loader: config.json + model.safetensors.index.json in the model dir (or
 * its weights/ subdir). Both the fixture schema (tests/m5/fixtures/
 * config.json, SMALL_CFG keys) and the real checkpoint schema
 * (reference/config.json keys) are accepted. Embed/head stay zero-copy
 * BF16 shard views (widened per row at use); routed-expert tensors stay
 * lazy FP4 views via c/st.h exactly as in c/layer.h — nothing expert-sized
 * is loaded eagerly, so the same loader shape carries to the real
 * 160 GB container (M6 adds the tiering/pread path underneath st.h).
 *
 * Usage: #define APUS_MODEL_IMPLEMENTATION in exactly one TU (needs the
 * layer.h implementation TU as well — see tests/m5/test_full.c).
 */
#ifndef APUS_MODEL_H
#define APUS_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "layer.h"
#include "mhc.h"
#include "st.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 0731 DSpark speculative-decoding config (M11b): the mtp.* namespace
 * holds n_stages full DSpark stages (SWA blocks + stage glue), NOT the
 * classic M8 MTP block — n_mtp stays 0 and --spec runs c/dspark.h. */
typedef struct {
    int block_size;           /* dspark_block_size (0731: 5) */
    int noise_id;             /* dspark_noise_token_id (0731: 128799) */
    int n_stages;             /* DSpark stages (0731: 3 — the trailing
                                 compress_ratios entries past
                                 num_hidden_layers) */
    int n_targets;            /* join layers (0731: 3) */
    int targets[8];           /* dspark_target_layer_ids (0731: [40,41,42]) */
    int markov_rank;          /* dspark_markov_rank (0731: 256) */
} ApusDsparkCfg;

typedef struct {
    ApusCfg cfg;              /* shared per-layer config (see c/layer.h) */
    int n_layers;
    int bos_id, eos_id;       /* -1 if unset */
    int n_mtp;                /* MTP blocks (M8; 0 or 1 for V4-Flash) */
    int mtp_ratio, mtp_hash;  /* MTP block attention type (real: 0 = SWA,
                                 bias gate) — from compress_ratios' trailing
                                 entries / the fixture "mtp" array */
    int dspark;               /* 0731 DSpark config present (dspark_* keys /
                                 the fixture "dspark" object): mtp.* holds
                                 DSpark stages, NOT a classic MTP block —
                                 n_mtp stays 0 and --spec takes the c/dspark.h
                                 path (M11b) */
    ApusDsparkCfg dsc;        /* parsed DSpark config (valid iff dspark) */
    ApusStSet *set;           /* owned */
    ApusLayer *layers;        /* [n_layers], owned */
    /* top-level tensors: embed/head are zero-copy views into the shard
     * (BF16 or F32), [vocab, dim]; widened per row at use */
    const ApusStTensor *embed;
    const ApusStTensor *head;
    float *norm_w;            /* [dim] owned */
    float *hc_head_fn;        /* [hc, hc*dim] owned */
    float *hc_head_base;      /* [hc] owned */
    float hc_head_scale;
} ApusModel;

/* Load model dir (config.json + safetensors index, weights/ fallback).
 * Returns 0 on success. */
int  apus_model_load(ApusModel *m, const char *dir, char *err, size_t errcap);
/* Tiered variant (M6a): with tiered != 0, routed-expert tensors are NOT
 * resolved into shard views (the expert store c/cache.h owns expert reads;
 * attach it via apus_store_attach_moe per layer after load). */
int  apus_model_load_ex(ApusModel *m, const char *dir, int tiered,
                        char *err, size_t errcap);
void apus_model_free(ApusModel *m);

typedef struct {
    ApusLayerState *layers;   /* [n_layers] */
    int64_t pos;              /* next position (== tokens consumed) */
} ApusModelState;

void apus_model_state_init(ApusModelState *st, const ApusModel *m);
void apus_model_state_free(ApusModelState *st, const ApusModel *m);

/* Embed one token: out [hc*dim], the BF16 row replicated hc_mult times
 * (model.py:803-804). */
void apus_model_embed(const ApusModel *m, int64_t id, float *out);

/* Full forward. ids [s] (s > 1 prefill / s == 1 decode); state positions
 * are advanced by s. logits: [s, V] if logits_all, else [V] for the LAST
 * position only. FP32, unrounded. */
void apus_model_forward(const ApusModel *m, ApusModelState *st,
                        const int64_t *ids, int s,
                        float *logits, int logits_all);

/* M8 variant: additionally copies the post-block hidden state h
 * [s, hc*dim] (pre hc_head, BF16 values in f32) to h_out — the MTP draft
 * head's input (reference model.py:757: MTPBlock takes the previous 4x-dim
 * mHC hidden). NULL h_out == apus_model_forward. */
void apus_model_forward_h(const ApusModel *m, ApusModelState *st,
                          const int64_t *ids, int s,
                          float *logits, int logits_all, float *h_out);

/* M11b DSpark variant: additionally collects main_hidden — after each
 * dspark target layer, the per-position mean over the hc streams of the
 * post-block hidden (reference-0731 model.py:920-921: h.mean(dim=2), fp32
 * accumulate, BF16 result), concatenated over the targets ->
 * mh_out [s, n_targets*dim]. Requires m->dspark. NULL mh_out ==
 * apus_model_forward. */
void apus_model_forward_mh(const ApusModel *m, ApusModelState *st,
                           const int64_t *ids, int s,
                           float *logits, int logits_all, float *mh_out);

/* M6b measurement mode (tools/measure_router_locality.py): identical
 * forward math (logits [V], last position), but after each layer the
 * callback receives that layer's ACTUAL chosen routed experts
 * (router_idx [s, topk], == tid2eid rows for hash layers) and the
 * post-attention hidden state (post_attn_h [s, hc*dim]) the M6b pilot
 * predicts the next layer from. Numerics are untouched — the callback
 * only observes intermediates. */
typedef void (*ApusMeasureCb)(void *ctx, int layer, int64_t start_pos,
                              const int64_t *ids, int s,
                              const int32_t *router_idx,
                              const float *post_attn_h);
void apus_model_forward_measure(const ApusModel *m, ApusModelState *st,
                                const int64_t *ids, int s, float *logits,
                                ApusMeasureCb cb, void *cb_ctx);

/* LM-head GEMV on the raw shard view (BF16 or F32), FP32 accumulate, no
 * output rounding. Exposed for tests. */
void apus_head_gemv(const ApusStTensor *head, const float *x, float *out,
                    int64_t O, int64_t K);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MODEL_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

/* ---- config parsing (fixture schema + real checkpoint schema) ----------*/

static JVal *apus_cfg_key(JVal *root, const char *k1, const char *k2) {
    JVal *v = k1 ? json_obj_get(root, k1) : NULL;
    if (!v && k2) v = json_obj_get(root, k2);
    return v;
}

static int apus_cfg_num(JVal *root, const char *k1, const char *k2,
                        double *out) {
    JVal *v = apus_cfg_key(root, k1, k2);
    if (!v || json_type(v) != J_NUM) return -1;
    *out = json_num(v);
    return 0;
}

#define APUS_CFG_INT(field, k1, k2) do { \
        double v_; if (apus_cfg_num(root, k1, k2, &v_)) { \
            snprintf(err, errcap, "config: missing %s", k1); return -1; } \
        c->field = (int)v_; } while (0)
#define APUS_CFG_FLT(field, k1, k2) do { \
        double v_; if (apus_cfg_num(root, k1, k2, &v_)) { \
            snprintf(err, errcap, "config: missing %s", k1); return -1; } \
        c->field = (float)v_; } while (0)
#define APUS_CFG_DBL(field, k1, k2) do { \
        double v_; if (apus_cfg_num(root, k1, k2, &v_)) { \
            snprintf(err, errcap, "config: missing %s", k1); return -1; } \
        c->field = v_; } while (0)

/* Parse config.json. ratios/hash_out: malloc'd [n_layers]. */
static int apus_model_parse_config(ApusCfg *c, int *n_layers, int **ratios,
                                   int **hash_out, int *bos_id, int *eos_id,
                                   int *n_mtp, int *mtp_ratio, int *mtp_hash,
                                   int *dspark, ApusDsparkCfg *dsc,
                                   const char *path, char *err, size_t errcap) {
    JVal *root = json_parse_file(path, err, errcap);
    if (!root) return -1;
    memset(c, 0, sizeof *c);
    memset(dsc, 0, sizeof *dsc);
    *n_mtp = 0;
    *mtp_ratio = 0;
    *mtp_hash = 0;
    /* 0731 DSpark marker: any dspark_* config key (or the fixture's nested
     * "dspark" object) means the mtp.* namespace holds DSpark speculative
     * stages (mtp.0.main_proj, markov_head, ...), not the classic MTP block
     * our M8 loader implements. */
    JVal *dj = json_obj_get(root, "dspark");
    *dspark = (dj && json_type(dj) == J_OBJ)
           || json_obj_get(root, "dspark_block_size") != NULL
           || json_obj_get(root, "dspark_noise_token_id") != NULL
           || json_obj_get(root, "dspark_target_layer_ids") != NULL
           || json_obj_get(root, "dspark_markov_rank") != NULL;
    APUS_CFG_INT(dim, "dim", "hidden_size");
    APUS_CFG_INT(n_heads, "n_heads", "num_attention_heads");
    APUS_CFG_INT(head_dim, "head_dim", "head_dim");
    APUS_CFG_INT(rope_head_dim, "rope_head_dim", "qk_rope_head_dim");
    APUS_CFG_INT(q_lora_rank, "q_lora_rank", "q_lora_rank");
    APUS_CFG_INT(o_groups, "o_groups", "o_groups");
    APUS_CFG_INT(o_lora_rank, "o_lora_rank", "o_lora_rank");
    APUS_CFG_INT(window_size, "window_size", "sliding_window");
    APUS_CFG_INT(moe_inter_dim, "moe_inter_dim", "moe_intermediate_size");
    APUS_CFG_INT(n_routed_experts, "n_routed_experts", "n_routed_experts");
    APUS_CFG_INT(n_activated_experts, "n_activated_experts", "num_experts_per_tok");
    APUS_CFG_INT(index_n_heads, "index_n_heads", "index_n_heads");
    APUS_CFG_INT(index_head_dim, "index_head_dim", "index_head_dim");
    APUS_CFG_INT(index_topk, "index_topk", "index_topk");
    APUS_CFG_INT(hc_mult, "hc_mult", "hc_mult");
    APUS_CFG_INT(hc_sinkhorn_iters, "hc_sinkhorn_iters", "hc_sinkhorn_iters");
    APUS_CFG_INT(vocab_size, "vocab_size", "vocab_size");
    APUS_CFG_INT(max_pos, "max_pos", "max_position_embeddings");
    /* YaRN params: fixture schema has them top-level; the real checkpoint
     * keeps them inside rope_scaling (parsed below). Optional here. */
    { double v_;
        if (apus_cfg_num(root, "beta_fast", NULL, &v_) == 0) c->beta_fast = (int)v_;
        if (apus_cfg_num(root, "beta_slow", NULL, &v_) == 0) c->beta_slow = (int)v_;
        if (apus_cfg_num(root, "original_seq_len", NULL, &v_) == 0) c->original_seq_len = (int)v_;
        if (apus_cfg_num(root, "rope_factor", NULL, &v_) == 0) c->rope_factor = v_; }
    APUS_CFG_FLT(hc_eps, "hc_eps", "hc_eps");
    APUS_CFG_FLT(norm_eps, "norm_eps", "rms_norm_eps");
    APUS_CFG_FLT(route_scale, "route_scale", "routed_scaling_factor");
    APUS_CFG_FLT(swiglu_limit, "swiglu_limit", "swiglu_limit");
    APUS_CFG_DBL(rope_theta, "rope_theta", "rope_theta");
    APUS_CFG_DBL(compress_rope_theta, "compress_rope_theta", "compress_rope_theta");
    /* real config nests rope params under rope_scaling */
    {
        JVal *rs = json_obj_get(root, "rope_scaling");
        if (rs) {
            double v_;
            if (apus_cfg_num(rs, NULL, "factor", &v_) == 0) c->rope_factor = v_;
            if (apus_cfg_num(rs, NULL, "original_max_position_embeddings", &v_))
                c->original_seq_len = (int)v_;
            if (apus_cfg_num(rs, NULL, "beta_fast", &v_) == 0) c->beta_fast = (int)v_;
            if (apus_cfg_num(rs, NULL, "beta_slow", &v_) == 0) c->beta_slow = (int)v_;
        }
    }
    {
        double v_;
        *bos_id = apus_cfg_num(root, "bos_token_id", NULL, &v_) == 0 ? (int)v_ : -1;
        *eos_id = apus_cfg_num(root, "eos_token_id", NULL, &v_) == 0 ? (int)v_ : -1;
    }
    /* layer schedule: fixture "layers" array, else real compress_ratios +
     * num_hash_layers */
    JVal *layers = json_obj_get(root, "layers");
    JVal *cr = json_obj_get(root, "compress_ratios");
    if (layers && json_type(layers) == J_ARR) {
        *n_layers = (int)json_arr_len(layers);
        *ratios = malloc((size_t)*n_layers * sizeof(int));
        *hash_out = malloc((size_t)*n_layers * sizeof(int));
        for (int i = 0; i < *n_layers; i++) {
            JVal *L = json_arr_get(layers, (size_t)i);
            JVal *r = json_obj_get(L, "compress_ratio");
            JVal *h = json_obj_get(L, "hash");
            if (!r || json_type(r) != J_NUM) {
                snprintf(err, errcap, "config: layers[%d].compress_ratio", i);
                json_free(root);
                return -1;
            }
            (*ratios)[i] = (int)json_num(r);
            (*hash_out)[i] = h ? json_truthy(h) : 0;
        }
        /* M8: fixture MTP declaration — "mtp": [{"compress_ratio":0,...}];
         * the real checkpoint uses num_nextn_predict_layers (below). */
        JVal *mj = json_obj_get(root, "mtp");
        if (mj && json_type(mj) == J_ARR && json_arr_len(mj) > 0) {
            JVal *M = json_arr_get(mj, 0);
            JVal *r = json_obj_get(M, "compress_ratio");
            JVal *h = json_obj_get(M, "hash");
            *n_mtp = (int)json_arr_len(mj);
            *mtp_ratio = (r && json_type(r) == J_NUM) ? (int)json_num(r) : 0;
            *mtp_hash = h ? json_truthy(h) : 0;
        }
    } else if (cr && json_type(cr) == J_ARR) {
        double nh = 0, nextn = 0;
        int ncr = (int)json_arr_len(cr);
        (void)apus_cfg_num(root, "num_hidden_layers", NULL, &nh);
        (void)apus_cfg_num(root, "num_nextn_predict_layers", NULL, &nextn);
        /* The real checkpoint appends one ratio per speculative stage
         * (preview: 43 + 1 classic MTP). 0731 appends 3 DSpark stage
         * ratios (46 total with num_nextn_predict_layers = 1) — tolerate
         * a longer list: the main stack uses the first num_hidden_layers
         * entries, extras are the speculative stages. */
        if ((int)nh > 0 && ncr < (int)nh) {
            snprintf(err, errcap, "config: num_hidden_layers != compress_ratios");
            json_free(root);
            return -1;
        }
        *n_layers = (int)nh > 0 ? (int)nh : ncr;
        if (*dspark && (int)nh > 0)
            dsc->n_stages = ncr - (int)nh;   /* 0731: 46 - 43 = 3 stages */
        double nhl = 0;
        (void)apus_cfg_num(root, "num_hash_layers", NULL, &nhl);
        *ratios = malloc((size_t)*n_layers * sizeof(int));
        *hash_out = malloc((size_t)*n_layers * sizeof(int));
        for (int i = 0; i < *n_layers; i++) {
            (*ratios)[i] = (int)json_num(json_arr_get(cr, (size_t)i));
            (*hash_out)[i] = i < (int)nhl;
        }
        /* M8: MTP blocks are the trailing compress_ratios entries
         * (V4-Flash: one, ratio 0 = SWA, bias gate — never hash-routed).
         * 0731 DSpark: mtp.* is NOT a classic MTP block, so the engine's
         * effective n_mtp stays 0 (the M8 loader must not parse DSpark
         * mtp.* tensors; --spec is refused in engine_init). */
        if (!*dspark && (int)nextn > 0 && ncr >= (int)nh + (int)nextn) {
            *n_mtp = (int)nextn;
            *mtp_ratio = (int)json_num(json_arr_get(cr, (size_t)(int)nh));
            *mtp_hash = 0;
        }
    } else {
        snprintf(err, errcap, "config: no layer schedule (layers|compress_ratios)");
        json_free(root);
        return -1;
    }
    /* DSpark section (M11b): the fixture's nested "dspark" object
     * ({block_size, noise_token_id, target_layer_ids, markov_rank,
     * n_mtp_layers}) or the 0731 top-level dspark_* keys (n_stages from
     * the compress_ratios tail, filled above). */
    if (*dspark) {
        JVal *src = (dj && json_type(dj) == J_OBJ) ? dj : root;
        const char *k_block = src == dj ? "block_size" : "dspark_block_size";
        const char *k_noise = src == dj ? "noise_token_id" : "dspark_noise_token_id";
        const char *k_rank  = src == dj ? "markov_rank" : "dspark_markov_rank";
        const char *k_tgt   = src == dj ? "target_layer_ids" : "dspark_target_layer_ids";
        double v_;
        if (apus_cfg_num(src, NULL, k_block, &v_) == 0)
            dsc->block_size = (int)v_;
        if (apus_cfg_num(src, NULL, k_noise, &v_) == 0)
            dsc->noise_id = (int)v_;
        if (apus_cfg_num(src, NULL, k_rank, &v_) == 0)
            dsc->markov_rank = (int)v_;
        if (src == dj && apus_cfg_num(src, NULL, "n_mtp_layers", &v_) == 0)
            dsc->n_stages = (int)v_;
        JVal *tj = json_obj_get(src, k_tgt);
        if (tj && json_type(tj) == J_ARR) {
            dsc->n_targets = (int)json_arr_len(tj);
            if (dsc->n_targets > 8) dsc->n_targets = 8;
            for (int i = 0; i < dsc->n_targets; i++)
                dsc->targets[i] = (int)json_num(json_arr_get(tj, (size_t)i));
        }
        if (dsc->block_size <= 0 || dsc->n_stages <= 0
            || dsc->n_targets <= 0 || dsc->markov_rank <= 0) {
            snprintf(err, errcap,
                     "config: incomplete DSpark section (block_size %d, "
                     "stages %d, targets %d, markov_rank %d)",
                     dsc->block_size, dsc->n_stages, dsc->n_targets,
                     dsc->markov_rank);
            json_free(root);
            return -1;
        }
    }
    json_free(root);
    return 0;
}

/* ---- top-level tensor loading ------------------------------------------*/

static const ApusStTensor *apus_model_get_mat(ApusStSet *set, const char *name,
                                              int64_t rows, int64_t cols,
                                              char *err, size_t errcap) {
    const ApusStTensor *t = apus_st_set_get(set, name);
    if (!t || (t->dtype != APUS_ST_BF16 && t->dtype != APUS_ST_F32)
        || t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) {
        snprintf(err, errcap, "model: bad tensor %s", name);
        return NULL;
    }
    return t;
}

int apus_model_load(ApusModel *m, const char *dir, char *err, size_t errcap) {
    return apus_model_load_ex(m, dir, 0, err, errcap);
}

int apus_model_load_ex(ApusModel *m, const char *dir, int tiered,
                       char *err, size_t errcap) {
    memset(m, 0, sizeof *m);
    int *ratios = NULL, *hash = NULL;
    char path[1024];

    snprintf(path, sizeof path, "%s/config.json", dir);
    if (apus_model_parse_config(&m->cfg, &m->n_layers, &ratios, &hash,
                                &m->bos_id, &m->eos_id,
                                &m->n_mtp, &m->mtp_ratio, &m->mtp_hash,
                                &m->dspark, &m->dsc, path, err, errcap))
        return -1;

    /* shard set: <dir>/model.safetensors.index.json else <dir>/weights */
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        m->set = apus_st_set_open(dir, err, errcap);
    } else {
        snprintf(path, sizeof path, "%s/weights", dir);
        m->set = apus_st_set_open(path, err, errcap);
    }
    if (!m->set) { free(ratios); free(hash); return -1; }
    if (tiered) apus_st_set_defer_experts(m->set, 1);

    m->layers = calloc((size_t)m->n_layers, sizeof(ApusLayer));
    for (int i = 0; i < m->n_layers; i++) {
        if (apus_layer_load(&m->layers[i], m->set, &m->cfg, i,
                            ratios[i], hash[i], err, errcap)) {
            free(ratios); free(hash);
            return -1;
        }
    }
    free(ratios);
    free(hash);

    int dim = m->cfg.dim, V = m->cfg.vocab_size, hc = m->cfg.hc_mult;
    m->embed = apus_model_get_mat(m->set, "embed.weight", V, dim, err, errcap);
    if (!m->embed) return -1;
    m->head = apus_model_get_mat(m->set, "head.weight", V, dim, err, errcap);
    if (!m->head) return -1;
    {
        const ApusStTensor *t;
        t = apus_st_set_get(m->set, "norm.weight");
        if (!t || apus_st_nelem(t) != (size_t)dim) {
            snprintf(err, errcap, "model: bad tensor norm.weight");
            return -1;
        }
        m->norm_w = malloc((size_t)dim * sizeof(float));
        apus_st_f32(t, m->norm_w, (size_t)dim);
        t = apus_st_set_get(m->set, "hc_head_fn");
        if (!t || t->dtype != APUS_ST_F32
            || apus_st_nelem(t) != (size_t)hc * hc * dim) {
            snprintf(err, errcap, "model: bad tensor hc_head_fn");
            return -1;
        }
        m->hc_head_fn = malloc((size_t)hc * hc * dim * sizeof(float));
        apus_st_f32(t, m->hc_head_fn, (size_t)hc * hc * dim);
        t = apus_st_set_get(m->set, "hc_head_base");
        if (!t || apus_st_nelem(t) != (size_t)hc) {
            snprintf(err, errcap, "model: bad tensor hc_head_base");
            return -1;
        }
        m->hc_head_base = malloc((size_t)hc * sizeof(float));
        apus_st_f32(t, m->hc_head_base, (size_t)hc);
        t = apus_st_set_get(m->set, "hc_head_scale");
        if (!t || apus_st_nelem(t) != 1) {
            snprintf(err, errcap, "model: bad tensor hc_head_scale");
            return -1;
        }
        apus_st_f32(t, &m->hc_head_scale, 1);
    }
    return 0;
}

void apus_model_free(ApusModel *m) {
    if (!m) return;
    if (m->layers) {
        for (int i = 0; i < m->n_layers; i++) apus_layer_free(&m->layers[i]);
        free(m->layers);
    }
    if (m->set) apus_st_set_close(m->set);
    free(m->norm_w);
    free(m->hc_head_fn);
    free(m->hc_head_base);
    memset(m, 0, sizeof *m);
}

void apus_model_state_init(ApusModelState *st, const ApusModel *m) {
    st->layers = calloc((size_t)m->n_layers, sizeof(ApusLayerState));
    for (int i = 0; i < m->n_layers; i++)
        apus_layer_state_init(&st->layers[i], &m->layers[i]);
    st->pos = 0;
}

void apus_model_state_free(ApusModelState *st, const ApusModel *m) {
    if (!st->layers) return;
    for (int i = 0; i < m->n_layers; i++)
        apus_layer_state_free(&st->layers[i], &m->layers[i]);
    free(st->layers);
    st->layers = NULL;
}

/* ---- embedding + head ---------------------------------------------------*/

static float apus_st_elem_f32(const ApusStTensor *t, int64_t i) {
    if (t->dtype == APUS_ST_F32)
        return ((const float *)t->data)[i];
    /* BF16: widen (upper 16 bits) */
    uint32_t u = (uint32_t)((const uint16_t *)t->data)[i] << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void apus_model_embed(const ApusModel *m, int64_t id, float *out) {
    int dim = m->cfg.dim, hc = m->cfg.hc_mult;
    for (int j = 0; j < hc; j++)
        for (int i = 0; i < dim; i++)
            out[(size_t)j * dim + i] = apus_st_elem_f32(m->embed, id * dim + i);
}

/* M6c: rows [o0, o1) of the head GEMV — the same per-row code as the
 * sequential loop, so values are thread-count independent. */
typedef struct {
    const ApusStTensor *head;
    const float *x;
    float *out;
    int64_t K;
} ApusHeadJob;

static void apus_head_gemv_rows(void *vjob, size_t o0, size_t o1) {
    const ApusHeadJob *j = vjob;
    const ApusStTensor *head = j->head;
    const float *x = j->x;
    int64_t K = j->K;
    for (int64_t o = (int64_t)o0; o < (int64_t)o1; o++) {
        float acc = 0.0f;
        int64_t base = o * K;
        if (head->dtype == APUS_ST_F32) {
            const float *w = (const float *)head->data + base;
            int64_t k = 0;
#ifdef __ARM_NEON
            float32x4_t a4 = vdupq_n_f32(0.0f);
            for (; k + 4 <= K; k += 4)
                a4 = vfmaq_f32(a4, vld1q_f32(w + k), vld1q_f32(x + k));
            acc = vaddvq_f32(a4);
#endif
            for (; k < K; k++) acc += w[k] * x[k];
        } else {
            const uint16_t *w = (const uint16_t *)head->data + base;
            int64_t k = 0;
#ifdef __ARM_NEON
            float32x4_t a4 = vdupq_n_f32(0.0f);
            for (; k + 8 <= K; k += 8) {
                uint16x8_t b = vld1q_u16(w + k);
                uint32x4_t lo = vshll_n_u16(vget_low_u16(b), 16);
                uint32x4_t hi = vshll_high_n_u16(b, 16);
                a4 = vfmaq_f32(a4, vreinterpretq_f32_u32(lo),
                               vld1q_f32(x + k));
                a4 = vfmaq_f32(a4, vreinterpretq_f32_u32(hi),
                               vld1q_f32(x + k + 4));
            }
            acc = vaddvq_f32(a4);
#endif
            for (; k < K; k++) {
                uint32_t u = (uint32_t)w[k] << 16;
                float f;
                memcpy(&f, &u, 4);
                acc += f * x[k];
            }
        }
        j->out[o] = acc;
    }
}

void apus_head_gemv(const ApusStTensor *head, const float *x, float *out,
                    int64_t O, int64_t K) {
    /* M7b backend hook (BF16/F32 shard view on GPU, FP32 accumulate,
     * unrounded out — same contract as below). */
    if (apus_backend_hooks.head_gemv
        && apus_backend_hooks.head_gemv(head, x, out, O, K) == 0)
        return;
    ApusHeadJob job = { head, x, out, K };
    apus_pool_run((size_t)O, apus_head_gemv_rows, &job);
}

/* ---- forward ------------------------------------------------------------*/

static void apus_model_forward_impl(const ApusModel *m, ApusModelState *st,
                                    const int64_t *ids, int s,
                                    float *logits, int logits_all,
                                    float *h_out, float *mh_out,
                                    ApusMeasureCb cb, void *cb_ctx) {
    int dim = m->cfg.dim, hc = m->cfg.hc_mult, V = m->cfg.vocab_size;
    ApusScratchMark mk = apus_scratch_mark();
    float *h = apus_scratch_alloc((size_t)s * hc * dim * sizeof(float));
    for (int t = 0; t < s; t++)
        apus_model_embed(m, ids[t], h + (size_t)t * hc * dim);
    int64_t start_pos = st->pos;
    ApusInterm it;
    int32_t *midx = NULL;
    float *mpah = NULL;
    if (cb) {
        memset(&it, 0, sizeof it);
        midx = malloc((size_t)s * m->cfg.n_activated_experts * sizeof(int32_t));
        mpah = malloc((size_t)s * hc * dim * sizeof(float));
        it.moe.router_idx = midx;
        it.post_attn_h = mpah;
    }
    for (int i = 0; i < m->n_layers; i++) {
        apus_block_forward(&m->layers[i], &st->layers[i], h, ids, s,
                           start_pos, cb ? &it : NULL);
        if (cb) cb(cb_ctx, i, start_pos, ids, s, midx, mpah);
        /* M11b: DSpark main_hidden collection — after each target layer,
         * the per-position mean over the hc streams (reference-0731
         * model.py:920-921: h.mean(dim=2); fp32 accumulate, bf16 result).
         * Serial per element: deterministic for any thread count. */
        if (mh_out) {
            for (int tg = 0; tg < m->dsc.n_targets; tg++) {
                if (m->dsc.targets[tg] != i) continue;
                for (int r = 0; r < s; r++) {
                    const float *hr = h + (size_t)r * hc * dim;
                    float *o = mh_out + (size_t)r * m->dsc.n_targets * dim
                               + (size_t)tg * dim;
                    for (int cix = 0; cix < dim; cix++) {
                        float acc = 0.0f;
                        for (int j = 0; j < hc; j++)
                            acc += hr[(size_t)j * dim + cix];
                        o[cix] = apus_bf16_round(acc / (float)hc);
                    }
                }
            }
        }
    }
    st->pos += s;
    free(midx);
    free(mpah);
    /* M8: MTP draft input = the post-block mHC hidden, per position */
    if (h_out)
        memcpy(h_out, h, (size_t)s * hc * dim * sizeof(float));

    /* head: hc_head collapse -> bf16 -> RMSNorm (bf16) -> FP32 GEMV
     * (model.py:714-735; get_logits at 715) */
    float *y = apus_scratch_alloc((size_t)dim * sizeof(float));
    float *yn = apus_scratch_alloc((size_t)dim * sizeof(float));
    float *mixes = apus_scratch_alloc((size_t)hc * sizeof(float));
    int t0 = logits_all ? 0 : s - 1;
    for (int t = t0; t < s; t++) {
        apus_mhc_head_scalar(h + (size_t)t * hc * dim, (size_t)dim,
                             (size_t)hc, m->hc_head_fn, m->hc_head_scale,
                             m->hc_head_base, m->cfg.norm_eps, m->cfg.hc_eps,
                             y, mixes);
        for (int i = 0; i < dim; i++) y[i] = apus_bf16_round(y[i]);
        apus_rms_norm(y, m->norm_w, m->cfg.norm_eps, yn, (size_t)dim);
        apus_head_gemv(m->head, yn, logits + (size_t)(t - t0) * V, V, dim);
    }
    apus_scratch_reset(mk);
}

void apus_model_forward(const ApusModel *m, ApusModelState *st,
                        const int64_t *ids, int s,
                        float *logits, int logits_all) {
    apus_model_forward_impl(m, st, ids, s, logits, logits_all, NULL,
                            NULL, NULL, NULL);
}

void apus_model_forward_h(const ApusModel *m, ApusModelState *st,
                          const int64_t *ids, int s,
                          float *logits, int logits_all, float *h_out) {
    apus_model_forward_impl(m, st, ids, s, logits, logits_all, h_out,
                            NULL, NULL, NULL);
}

void apus_model_forward_mh(const ApusModel *m, ApusModelState *st,
                           const int64_t *ids, int s,
                           float *logits, int logits_all, float *mh_out) {
    apus_model_forward_impl(m, st, ids, s, logits, logits_all, NULL,
                            mh_out, NULL, NULL);
}

void apus_model_forward_measure(const ApusModel *m, ApusModelState *st,
                                const int64_t *ids, int s, float *logits,
                                ApusMeasureCb cb, void *cb_ctx) {
    apus_model_forward_impl(m, st, ids, s, logits, 0, NULL, NULL, cb, cb_ctx);
}

#endif /* APUS_MODEL_IMPLEMENTATION */
#endif /* APUS_MODEL_H */
