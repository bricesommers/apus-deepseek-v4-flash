/*
 * c/pilot.h — router-lookahead prefetch ("pilot", M6b): predict layer L+1's
 * routed experts from layer L's post-attention hidden state and warm the
 * M6a expert store before the demand resolve arrives. colibri pilot design
 * (docs/ARCHITECTURE.md §4/§7) adapted to DeepSeek-V4-Flash. C11, libc +
 * pthreads.
 *
 * Prediction math (reuse, never duplicate):
 *   - router input  = the layer's ffn mHC hc_pre collapse (BF16-rounded) +
 *                     ffn RMSNorm — the exact c/mhc.h + c/attn.h helpers
 *                     c/layer.h feeds the real router, applied to layer
 *                     L+1's OWN weights (hc_ffn_*, ffn_norm) on layer L's
 *                     post-attention state;
 *   - scores        = apus_router_score (c/moe.h) — the same sqrtsoftplus +
 *                     selection-only-bias code path as the real gate;
 *   - top-N         = apus_topk_stable (c/attn.h, A6 stable ties).
 * Depth dL = 1 only (cross-layer coupling tables are a separate offline
 * artifact, tools/measure_router_locality.py). Hash-routed targets are
 * never predicted — their experts are known exactly from tid2eid.
 *
 * Delivery: the compute thread (post-attention hook, c/layer.h) pushes
 * (pos, layer, eid) entries onto a bounded single-producer/single-consumer
 * lock-free ring; a dedicated pilot thread pops them and calls
 * apus_store_hint (c/cache.h — thread-safe, dedup'ed, eviction-guarded).
 * The compute thread NEVER blocks: a full ring drops the NEWEST entry
 * (justification: the ring is FIFO in issue order, which is also
 * time-to-need order — the oldest queued hint belongs to the soonest-
 * needed layer, so sacrificing the newest minimizes the expected damage;
 * hints that outlive their usefulness are dropped at the consumer by the
 * (pos, layer) watermark instead). Ordering: the consumer issues a hint
 * only if its (pos, layer) is not behind the compute thread's current
 * (pos, layer) — a hint for a layer the MoE already ran is wasted
 * bandwidth, not an error (the store dedups and never evicts a warm
 * demand-loaded expert for a speculation).
 *
 * Hash-layer prefetch: layers with gate.tid2eid (real model: 0-2) have
 * their expert sets known at tokenization time;
 * apus_pilot_prefetch_hash enqueues them before the forward starts.
 *
 * Prefill union lookahead (M9c, prefill_k > 0): during prefill (s > 1) the
 * post-attention hook runs the SAME prediction math batched over all s
 * tokens (pooled scalar-order matmul; per-token rsqrt after the dot — the
 * predicted sets equal s per-token apus_pilot_predict calls) and issues
 * one speculative store hint per unique expert in the union of the
 * per-token top-k sets, directly to the store (no ring). Layer L+1's
 * slabs therefore stream while layer L's MoE computes; the MoE's own
 * batch-union storm (demand class, c/cache.h hot jobs) overtakes queued
 * speculative loads for the ~10% of experts the union missed. Coverage vs
 * K measured on the real model in tests/m9c (89% at K=12, the default).
 *
 * Recall accounting (decode path, s == 1): the pilot wraps the MoE
 * hook_hint/hook_layer_end trampolines (attach AFTER apus_store_attach_moe)
 * and compares each layer's actual chosen experts against the pending
 * prediction — predictions_made / hints_issued / actual_hits counters make
 * recall = actual_hits / actual_experts measurable from
 * apus_pilot_stats alone. Numerics are never touched: the pilot only
 * reads hidden states and issues store hints.
 *
 * Usage: #define APUS_PILOT_IMPLEMENTATION in exactly one TU (needs the
 * moe/mhc/attn/cache implementations linked).
 */
#ifndef APUS_PILOT_H
#define APUS_PILOT_H

#include <stddef.h>
#include <stdint.h>

#include "cache.h"      /* ApusStore, apus_store_hint */
#include "model.h"      /* ApusModel (attach helper), layer.h router fields */

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only view of one layer's router + ffn-norm weights (owned by the
 * ApusLayer, not by the pilot). */
typedef struct {
    int            hash;        /* hash-routed (tid2eid) layer */
    const float   *gate_w;      /* [E, dim] f32 */
    const float   *gate_bias;   /* [E] f32 (non-hash) */
    const float   *ffn_norm_w;  /* [dim] f32 */
    const float   *hc_fn;       /* [(2+hc)*hc, hc*dim] f32 */
    const float   *hc_base;     /* [(2+hc)*hc] f32 */
    const float   *hc_scale;    /* [3] f32 */
    const int64_t *tid2eid;     /* [vocab, topk] (hash layers) */
} ApusPilotRouter;

typedef struct {
    ApusStore  *store;          /* hint target; NULL = predict-only */
    int         n_layers;       /* required */
    int         n_experts;      /* required (E) */
    int         topk;           /* required: router top-k */
    int         dim, hc_mult, sinkhorn_iters, vocab;
    float       norm_eps, hc_eps;
    int         enabled;        /* 0 = attach but never predict/prefetch */
    int         pilot_k;        /* top-N cap (APUS_PILOT_K); 0 = no router
                                   lookahead (hash prefetch still runs) */
    int         prefill_k;      /* M9c: batched union lookahead during
                                   prefill (s>1): predict target's top-N for
                                   ALL s tokens and hint the union directly
                                   to the store (speculative class).
                                   0 = APUS_PILOT_PREFILL_K env (default 12);
                                   <0 = off (M6b behavior: last-token ring
                                   prediction when prefill_last_only) */
    int         d2;             /* M9c: decode dL=2 lookahead — also predict
                                   layer L+2 from layer L's post-attention
                                   state (lower recall than dL=1 but ~2x the
                                   lead time). Default OFF (0 = env
                                   APUS_PILOT_D2, default 0): measured a net
                                   loss on the M1 (tests/m9c exp F/G/H);
                                   <0 = off */
    int         hash_prefetch;  /* enqueue tid2eid experts at tokenization */
    int         prefill_last_only; /* s>1: predict for the last token only */
    size_t      ring_entries;   /* ring capacity (rounded up to pow2) */
    const char *dump_path;      /* NULL = off; NDJSON P/A sets (decode) */
} ApusPilotCfg;

typedef struct {
    uint64_t predictions;         /* top-N sets computed */
    uint64_t pred_experts;        /* expert ids predicted (sum of N) */
    uint64_t hints_enqueued;      /* ring pushes accepted (pilot + hash) */
    uint64_t hints_dropped_full;  /* ring-full drops (drop-newest policy) */
    uint64_t hints_issued;        /* consumer -> apus_store_hint calls */
    uint64_t hints_dropped_stale; /* consumer-side (pos,layer) watermark */
    uint64_t hash_hints;          /* of enqueued: exact tid2eid prefetch */
    uint64_t actual_experts;      /* routed experts observed (s==1, non-hash
                                     layers with a pending prediction) */
    uint64_t actual_hits;         /* of those, present in the predicted set */
    uint64_t prefill_predictions; /* M9c: batched prefill predictions
                                     (tokens scored) */
    uint64_t prefill_hints;       /* M9c: unique union experts hinted
                                     (speculative) from prefill batches */
    uint64_t d2_predictions;      /* M9c: dL=2 predictions made (decode) */
    uint64_t d2_rescue;           /* M9c: actual experts MISSED by the dL=1
                                     pending set but present in the dL=2
                                     set (the marginal value of dL=2) */
    uint64_t d2_missed;           /* M9c: actual experts missed by dL=1
                                     (denominator for d2_rescue) */
} ApusPilotStats;

typedef struct ApusPilot ApusPilot;

ApusPilot *apus_pilot_create(const ApusPilotCfg *cfg);
void       apus_pilot_destroy(ApusPilot *p);   /* stop + join + free */

/* Register layer `layer`'s router view (call for every layer). */
void apus_pilot_attach_router(ApusPilot *p, int layer,
                              const ApusPilotRouter *r);

/* Wire the pilot into a loaded model: registers all router views, sets
 * every layer's post-attention hook, and wraps the MoE hint/layer_end
 * hooks (call AFTER apus_store_attach_moe). */
void apus_pilot_attach_model(ApusPilot *p, ApusModel *m);

/* Spawn the consumer thread (idempotent; destroy works without it). */
int  apus_pilot_start(ApusPilot *p);

/* Compute-thread entry points (never block): */
/* Hash-layer prefetch: enqueue tid2eid experts of all hash layers for
 * ids[0..n) occupying positions first_pos..first_pos+n-1. */
void apus_pilot_prefetch_hash(ApusPilot *p, const int64_t *ids, int n,
                              int64_t first_pos);

/* Pure prediction (no ring, no stats, no store) — shared by the runtime
 * hook, the --measure-locality dump, and unit tests.
 * apus_pilot_router_input: layer `target`'s router input (mHC ffn hc_pre
 * collapse + ffn RMSNorm) from the post-attention state h4 [hc*dim] of ONE
 * token; x_out [dim]. apus_pilot_predict: predicted top-n expert ids
 * (stable score order) for layer `target`. Both return -1 if the target
 * has no router view / is hash-routed. */
int  apus_pilot_router_input(const ApusPilot *p, int target,
                             const float *h4, float *x_out);
int  apus_pilot_predict(const ApusPilot *p, int target, const float *h4,
                        int32_t *idx, int n);

void apus_pilot_stats(ApusPilot *p, ApusPilotStats *out);

/* M9c batched dL=1 union prediction for prefill: predict target's top-k
 * for EVERY one of the s tokens whose post-attention states are h4
 * [s, hc*dim], and set the bit of every predicted expert in `seen`
 * ((E+63)/64 words, caller-zeroed). The math is the per-token
 * apus_pilot_router_input batched with the same per-element expressions:
 * the mixes matmul is apus_f32_linear (scalar sequential-k order per
 * output — the order apus_mhc_mixes_scalar uses) with the per-token rsqrt
 * applied AFTER the dot (reference order), so the predicted sets equal
 * what s per-token apus_pilot_predict calls produce. Returns 0, -1 if the
 * target has no router view / is hash-routed. */
int  apus_pilot_predict_union(ApusPilot *p, int target, const float *h4,
                              int s, int k, uint64_t *seen);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_PILOT_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <stdatomic.h>

#include "mhc.h"        /* apus_mhc_prepost_scalar / collapse_scalar */

#define APUS_PILOT_MAX_K 64    /* stack cap for a prediction set */

struct ApusPilot {
    ApusPilotCfg     cfg;
    ApusPilotRouter *routers;      /* [n_layers] */
    /* SPSC lock-free ring: producer = compute thread, consumer = pilot
     * thread. Entries pack (pos:32 | layer:16 | eid:16) into u64. */
    uint64_t        *ring;
    size_t           ring_mask;
    _Atomic uint64_t r_head;       /* consumer-owned index */
    _Atomic uint64_t r_tail;       /* producer-owned index */
    /* ordering watermark (producer-written, consumer-read) */
    _Atomic uint64_t cur_pos;
    _Atomic int      cur_layer;
    _Atomic int      cur_s;
    /* consumer thread */
    pthread_t        thread;
    int              started;
    pthread_mutex_t  mtx;          /* sleep/wakeup only, never held long */
    pthread_cond_t   cv;
    int              stopping;
    /* pending predictions for recall accounting (compute thread only) */
    int32_t         *pending;      /* [n_layers][pilot_k] */
    uint8_t         *pending_ok;   /* [n_layers] */
    int64_t         *pending_pos;  /* [n_layers] */
    /* M9c: pending dL=2 predictions (made at layer L-2 for layer L) */
    int32_t         *pending2;     /* [n_layers][pilot_k] */
    uint8_t         *pending2_ok;  /* [n_layers] */
    int64_t         *pending2_pos; /* [n_layers] */
    /* actual-set accumulation for the dump (compute thread only) */
    int32_t         *acc;          /* [n_layers][acc_cap] */
    int             *acc_n;        /* [n_layers] */
    int              acc_cap;
    FILE            *dump;
    /* saved MoE hooks (the store's trampolines) */
    void            *saved_ctx;
    void           (*saved_hint)(void *, int, int);
    void           (*saved_layer_end)(void *, int);
    /* stats (atomics: written by both threads) */
    _Atomic uint64_t st_predictions, st_pred_experts;
    _Atomic uint64_t st_enqueued, st_dropped_full;
    _Atomic uint64_t st_issued, st_dropped_stale;
    _Atomic uint64_t st_hash;
    _Atomic uint64_t st_actual, st_hits;
    _Atomic uint64_t st_pf_preds, st_pf_hints;
    _Atomic uint64_t st_d2_preds, st_d2_rescue, st_d2_missed;
};

/* --- ring ------------------------------------------------------------------*/

static uint64_t apus_pilot_pack(int64_t pos, int layer, int eid) {
    return ((uint64_t)(uint32_t)pos << 32)
         | ((uint64_t)(uint16_t)layer << 16) | (uint16_t)eid;
}

/* Producer-side push; drop-newest when full. Returns 1 if enqueued. */
static int apus_pilot_push(ApusPilot *p, int64_t pos, int layer, int eid) {
    uint64_t tail = atomic_load_explicit(&p->r_tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&p->r_head, memory_order_acquire);
    if (tail - head > p->ring_mask) {      /* full (cap = mask+1) */
        atomic_fetch_add(&p->st_dropped_full, 1);
        return 0;
    }
    p->ring[tail & p->ring_mask] = apus_pilot_pack(pos, layer, eid);
    atomic_store_explicit(&p->r_tail, tail + 1, memory_order_release);
    atomic_fetch_add(&p->st_enqueued, 1);
    return 1;
}

/* Consumer-side pop. Returns 1 and fills out-params, 0 when empty. */
static int apus_pilot_pop(ApusPilot *p, int64_t *pos, int *layer, int *eid) {
    uint64_t head = atomic_load_explicit(&p->r_head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&p->r_tail, memory_order_acquire);
    if (head == tail) return 0;
    uint64_t e = p->ring[head & p->ring_mask];
    atomic_store_explicit(&p->r_head, head + 1, memory_order_release);
    *pos = (int64_t)(uint32_t)(e >> 32);
    *layer = (int)(uint16_t)(e >> 16);
    *eid = (int)(uint16_t)e;
    return 1;
}

static void apus_pilot_kick(ApusPilot *p) {
    pthread_mutex_lock(&p->mtx);
    pthread_cond_signal(&p->cv);
    pthread_mutex_unlock(&p->mtx);
}

/* --- consumer thread ---------------------------------------------------------*/

/* A hint is worth issuing only if its (pos, layer) is not strictly behind
 * the compute thread's watermark: an earlier token's layer, or an earlier
 * layer of the current token, has already run its MoE. */
static int apus_pilot_stale(ApusPilot *p, int64_t pos, int layer) {
    int64_t cp = (int64_t)atomic_load_explicit(&p->cur_pos,
                                               memory_order_acquire);
    int cl = atomic_load_explicit(&p->cur_layer, memory_order_acquire);
    return pos < cp || (pos == cp && layer < cl);
}

static void *apus_pilot_consumer(void *arg) {
    ApusPilot *p = arg;
    for (;;) {
        pthread_mutex_lock(&p->mtx);
        while (!p->stopping
               && atomic_load_explicit(&p->r_head, memory_order_acquire)
                  == atomic_load_explicit(&p->r_tail, memory_order_acquire))
            pthread_cond_wait(&p->cv, &p->mtx);
        if (p->stopping) {
            pthread_mutex_unlock(&p->mtx);
            return NULL;
        }
        pthread_mutex_unlock(&p->mtx);
        int64_t pos;
        int layer, eid;
        if (!apus_pilot_pop(p, &pos, &layer, &eid)) continue;
        if (apus_pilot_stale(p, pos, layer)) {
            atomic_fetch_add(&p->st_dropped_stale, 1);
            continue;
        }
        if (p->cfg.store) apus_store_hint(p->cfg.store, layer, eid);
        atomic_fetch_add(&p->st_issued, 1);
    }
}

/* --- prediction (shared math) --------------------------------------------------*/

int apus_pilot_router_input(const ApusPilot *p, int target,
                            const float *h4, float *x_out) {
    if (!p || target < 0 || target >= p->cfg.n_layers) return -1;
    const ApusPilotRouter *r = &p->routers[target];
    if (!r->gate_w || r->hash) return -1;
    size_t hc = (size_t)p->cfg.hc_mult, dim = (size_t)p->cfg.dim;
    ApusScratchMark mk = apus_scratch_mark();
    float *pre = apus_scratch_alloc(hc * sizeof(float));
    float *post = apus_scratch_alloc(hc * sizeof(float));
    float *comb = apus_scratch_alloc(hc * hc * sizeof(float));
    float *mixes = apus_scratch_alloc((2 + hc) * hc * sizeof(float));
    /* identical to c/layer.h apus_hc_pre (same c/mhc.h helpers, same
     * order) + the ffn RMSNorm the real router input goes through */
    apus_mhc_prepost_scalar(h4, dim, hc, r->hc_fn, r->hc_scale, r->hc_base,
                            p->cfg.norm_eps, p->cfg.hc_eps,
                            p->cfg.sinkhorn_iters, pre, post, comb, mixes);
    apus_mhc_collapse_scalar(h4, pre, x_out, dim, hc);
    for (size_t i = 0; i < dim; i++) x_out[i] = apus_bf16_round(x_out[i]);
    apus_rms_norm(x_out, r->ffn_norm_w, p->cfg.norm_eps, x_out, dim);
    apus_scratch_reset(mk);
    return 0;
}

int apus_pilot_predict(const ApusPilot *p, int target, const float *h4,
                       int32_t *idx, int n) {
    if (!p || target < 0 || target >= p->cfg.n_layers || n <= 0) return -1;
    const ApusPilotRouter *r = &p->routers[target];
    if (!r->gate_w || r->hash) return -1;
    int E = p->cfg.n_experts, dim = p->cfg.dim;
    if (n > E) n = E;
    ApusScratchMark mk = apus_scratch_mark();
    float *x = apus_scratch_alloc((size_t)dim * sizeof(float));
    float *sp = apus_scratch_alloc((size_t)E * sizeof(float));
    float *biased = apus_scratch_alloc((size_t)E * sizeof(float));
    if (apus_pilot_router_input(p, target, h4, x)) {
        apus_scratch_reset(mk);
        return -1;
    }
    /* the real router's own scoring path (c/moe.h) + the same stable
     * top-k the gate uses (c/attn.h, A6) */
    ApusMoeW mw;
    memset(&mw, 0, sizeof mw);
    mw.E = E;
    mw.dim = dim;
    mw.gate_w = r->gate_w;
    mw.gate_bias = r->gate_bias;
    apus_router_score(&mw, x, 1, sp, biased);
    apus_topk_stable(biased, E, n, idx);
    apus_scratch_reset(mk);
    return 0;
}

/* --- compute-thread hooks --------------------------------------------------------*/

static void apus_pilot_dump_set(ApusPilot *p, char type, int64_t pos,
                                int layer, const int32_t *eids, int n) {
    fprintf(p->dump, "{\"type\":\"%c\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
            type, (long long)pos, layer);
    for (int j = 0; j < n; j++) fprintf(p->dump, "%s%d", j ? "," : "", eids[j]);
    fprintf(p->dump, "]}\n");
}

int apus_pilot_predict_union(ApusPilot *p, int target, const float *h4,
                             int s, int k, uint64_t *seen) {
    if (!p || target < 0 || target >= p->cfg.n_layers || s <= 0 || k <= 0)
        return -1;
    const ApusPilotRouter *r = &p->routers[target];
    if (!r->gate_w || r->hash) return -1;
    int E = p->cfg.n_experts, dim = p->cfg.dim;
    if (k > E) k = E;
    size_t hc = (size_t)p->cfg.hc_mult, nx = hc * (size_t)dim;
    size_t nmix = (2 + hc) * hc;
    if (hc > APUS_MHC_MULT) return -1;   /* stack pre[] cap (real model: 4) */
    ApusScratchMark mk = apus_scratch_mark();
    float *x  = apus_scratch_alloc((size_t)s * dim * sizeof(float));
    float *mx = apus_scratch_alloc((size_t)s * nmix * sizeof(float));
    float *rs = apus_scratch_alloc((size_t)s * sizeof(float));
    float *sp = apus_scratch_alloc((size_t)s * E * sizeof(float));
    float *bi = apus_scratch_alloc((size_t)s * E * sizeof(float));
    for (int t = 0; t < s; t++)
        rs[t] = apus_mhc_rsqrt_scalar(h4 + (size_t)t * nx, nx,
                                      p->cfg.norm_eps);
    apus_f32_linear(r->hc_fn, h4, mx, s, (int)nx, (int)nmix);
    for (int t = 0; t < s; t++) {
        const float *h4t = h4 + (size_t)t * nx;
        float *xt = x + (size_t)t * dim;
        float pre[APUS_MHC_MULT];
        for (size_t j = 0; j < hc; j++)
            pre[j] = apus_mhc_sigmoid(mx[(size_t)t * nmix + j] * rs[t]
                                      * r->hc_scale[0] + r->hc_base[j])
                     + p->cfg.hc_eps;
        apus_mhc_collapse_scalar(h4t, pre, xt, (size_t)dim, hc);
        for (int i = 0; i < dim; i++) xt[i] = apus_bf16_round(xt[i]);
        apus_rms_norm(xt, r->ffn_norm_w, p->cfg.norm_eps, xt, (size_t)dim);
    }
    ApusMoeW mw;
    memset(&mw, 0, sizeof mw);
    mw.E = E;
    mw.dim = dim;
    mw.gate_w = r->gate_w;
    mw.gate_bias = r->gate_bias;
    apus_router_score(&mw, x, s, sp, bi);
    for (int t = 0; t < s; t++) {
        int32_t idx[APUS_PILOT_MAX_K];
        apus_topk_stable(bi + (size_t)t * E, E, k, idx);
        atomic_fetch_add(&p->st_pf_preds, 1);
        for (int j = 0; j < k; j++)
            seen[idx[j] >> 6] |= 1ull << (idx[j] & 63);
    }
    apus_scratch_reset(mk);
    return 0;
}

/* Post-attention hook (c/layer.h): watermark update + layer L+1 router
 * lookahead. Read-only on the hidden state; never blocks. */
static void apus_pilot_tr_post_attn(void *ctx, int layer, const float *h,
                                    int s, int64_t start_pos) {
    ApusPilot *p = ctx;
    atomic_store_explicit(&p->cur_pos, (uint64_t)start_pos,
                          memory_order_release);
    atomic_store_explicit(&p->cur_layer, layer, memory_order_release);
    atomic_store_explicit(&p->cur_s, s, memory_order_release);
    if (!p->cfg.enabled) return;
    int target = layer + 1;
    if (target >= p->cfg.n_layers) return;
    const ApusPilotRouter *r = &p->routers[target];
    if (r->hash || !r->gate_w) return;      /* hash targets: exact prefetch */
    /* M9c: batched union lookahead replaces the per-token ring path for
     * prefill when prefill_k > 0. Hints are speculative; the MoE's own
     * batch-union storm (demand class, c/cache.h hot jobs) overtakes them
     * for the experts the union missed. */
    if (s > 1 && p->cfg.prefill_k > 0) {
        int E = p->cfg.n_experts;
        ApusScratchMark mk = apus_scratch_mark();
        uint64_t *seen =
            apus_scratch_alloc(((size_t)E + 63) / 64 * sizeof(uint64_t));
        memset(seen, 0, ((size_t)E + 63) / 64 * sizeof(uint64_t));
        if (apus_pilot_predict_union(p, target, h, s, p->cfg.prefill_k,
                                     seen) == 0
            && p->cfg.store)
            for (int e = 0; e < E; e++)
                if (seen[e >> 6] & (1ull << (e & 63))) {
                    apus_store_hint(p->cfg.store, target, e);
                    atomic_fetch_add(&p->st_pf_hints, 1);
                }
        apus_scratch_reset(mk);
        return;
    }
    if (p->cfg.pilot_k <= 0) return;
    int k = p->cfg.pilot_k;
    size_t hcd = (size_t)p->cfg.hc_mult * p->cfg.dim;
    int t0 = (p->cfg.prefill_last_only && s > 1) ? s - 1 : 0;
    for (int t = t0; t < s; t++) {
        const float *ht = h + (size_t)t * hcd;
        int32_t idx[APUS_PILOT_MAX_K];
        if (apus_pilot_predict(p, target, ht, idx, k))
            continue;
        int64_t pos = start_pos + t;
        atomic_fetch_add(&p->st_predictions, 1);
        atomic_fetch_add(&p->st_pred_experts, (uint64_t)k);
        memcpy(p->pending + (size_t)target * k, idx,
               (size_t)k * sizeof(int32_t));
        p->pending_ok[target] = 1;
        p->pending_pos[target] = pos;
        if (p->dump) apus_pilot_dump_set(p, 'P', pos, target, idx, k);
        for (int j = 0; j < k; j++) apus_pilot_push(p, pos, target, idx[j]);
        /* M9c dL=2: also predict layer L+2 from the same post-attention
         * state. Recall is lower than dL=1, but the hint lands ~2 layers
         * ahead of the resolve instead of ~1 — that lead time is what the
         * just-in-time decode stalls need (tests/m9c measurements). */
        int target2 = target + 1;
        if (p->cfg.d2 > 0 && target2 < p->cfg.n_layers
            && !p->routers[target2].hash && p->routers[target2].gate_w) {
            int32_t idx2[APUS_PILOT_MAX_K];
            if (apus_pilot_predict(p, target2, ht, idx2, k) == 0) {
                atomic_fetch_add(&p->st_d2_preds, 1);
                memcpy(p->pending2 + (size_t)target2 * k, idx2,
                       (size_t)k * sizeof(int32_t));
                p->pending2_ok[target2] = 1;
                p->pending2_pos[target2] = pos;
                if (p->dump)
                    apus_pilot_dump_set(p, 'Q', pos, target2, idx2, k);
                for (int j = 0; j < k; j++)
                    apus_pilot_push(p, pos, target2, idx2[j]);
            }
        }
    }
    apus_pilot_kick(p);
}

/* Wrapped MoE hook_hint (recall accounting + A-set dump), then chain to
 * the store's trampoline. Compute thread only. */
static void apus_pilot_tr_hint(void *ctx, int layer, int eid) {
    ApusPilot *p = ctx;
    if (atomic_load_explicit(&p->cur_s, memory_order_acquire) == 1) {
        if (p->dump && p->acc_n[layer] < p->acc_cap)
            p->acc[(size_t)layer * p->acc_cap + p->acc_n[layer]++] = eid;
        if (!p->routers[layer].hash && p->pending_ok[layer]
            && p->pending_pos[layer]
               == (int64_t)atomic_load_explicit(&p->cur_pos,
                                                memory_order_acquire)) {
            int k = p->cfg.pilot_k;
            const int32_t *pd = p->pending + (size_t)layer * k;
            atomic_fetch_add(&p->st_actual, 1);
            int found = 0;
            for (int j = 0; j < k; j++)
                if (pd[j] == eid) {
                    atomic_fetch_add(&p->st_hits, 1);
                    found = 1;
                    break;
                }
            /* M9c: dL=1 missed — did the dL=2 set (made two layers back)
             * rescue it? */
            if (!found && p->cfg.d2 > 0) {
                atomic_fetch_add(&p->st_d2_missed, 1);
                if (p->pending2_ok[layer]
                    && p->pending2_pos[layer] == p->pending_pos[layer]) {
                    const int32_t *pd2 = p->pending2 + (size_t)layer * k;
                    for (int j = 0; j < k; j++)
                        if (pd2[j] == eid) {
                            atomic_fetch_add(&p->st_d2_rescue, 1);
                            break;
                        }
                }
            }
        }
    }
    if (p->saved_hint) p->saved_hint(p->saved_ctx, layer, eid);
}

static void apus_pilot_tr_layer_end(void *ctx, int layer) {
    ApusPilot *p = ctx;
    if (p->dump && p->acc_n[layer] > 0) {
        apus_pilot_dump_set(p, 'A',
                            (int64_t)atomic_load_explicit(&p->cur_pos,
                                                          memory_order_acquire),
                            layer,
                            p->acc + (size_t)layer * p->acc_cap,
                            p->acc_n[layer]);
        p->acc_n[layer] = 0;
        fflush(p->dump);
    }
    if (p->saved_layer_end) p->saved_layer_end(p->saved_ctx, layer);
}

void apus_pilot_prefetch_hash(ApusPilot *p, const int64_t *ids, int n,
                              int64_t first_pos) {
    if (!p || !p->cfg.enabled || !p->cfg.hash_prefetch) return;
    for (int l = 0; l < p->cfg.n_layers; l++) {
        const ApusPilotRouter *r = &p->routers[l];
        if (!r->hash || !r->tid2eid) continue;
        for (int t = 0; t < n; t++) {
            int64_t id = ids[t];
            if (id < 0 || id >= p->cfg.vocab) continue;
            for (int j = 0; j < p->cfg.topk; j++) {
                int eid = (int)r->tid2eid[id * p->cfg.topk + j];
                if (apus_pilot_push(p, first_pos + t, l, eid))
                    atomic_fetch_add(&p->st_hash, 1);
            }
        }
    }
    apus_pilot_kick(p);
}

/* --- lifecycle ------------------------------------------------------------------*/

ApusPilot *apus_pilot_create(const ApusPilotCfg *cfg) {
    if (!cfg || cfg->n_layers <= 0 || cfg->n_experts <= 0 || cfg->topk <= 0)
        return NULL;
    ApusPilot *p = calloc(1, sizeof *p);
    p->cfg = *cfg;
    if (p->cfg.pilot_k > APUS_PILOT_MAX_K) p->cfg.pilot_k = APUS_PILOT_MAX_K;
    /* M9c: prefill union lookahead. field 0 -> env -> default 12 (the
     * tests/m9c coverage curve: 89% of the actual unique-expert union at
     * ~96 slabs/layer on the real model); field/env <0 -> off (the M6b
     * last-token ring path is kept when off). */
    if (p->cfg.prefill_k == 0)
        p->cfg.prefill_k = apus_env_int("APUS_PILOT_PREFILL_K", 12);
    if (p->cfg.prefill_k > APUS_PILOT_MAX_K)
        p->cfg.prefill_k = APUS_PILOT_MAX_K;
    /* M9c: decode dL=2 lookahead. field 0 -> env -> default 0 (OFF): it
     * does cut just-in-time waits (8.7 -> 5.0 s/24 tok) but the extra
     * speculative slabs cost more compute-contention than the waits save
     * (tests/m9c exp F/G/H). Opt in with APUS_PILOT_D2=1. */
    if (p->cfg.d2 == 0)
        p->cfg.d2 = apus_env_int("APUS_PILOT_D2", 0);
    size_t cap = p->cfg.ring_entries ? p->cfg.ring_entries : 4096;
    size_t pow2 = 16;
    while (pow2 < cap) pow2 <<= 1;
    p->ring = malloc(pow2 * sizeof(uint64_t));
    p->ring_mask = pow2 - 1;
    atomic_store(&p->cur_pos, 0);
    atomic_store(&p->cur_layer, -1);
    atomic_store(&p->cur_s, 1);
    pthread_mutex_init(&p->mtx, NULL);
    pthread_cond_init(&p->cv, NULL);
    int nl = p->cfg.n_layers, k = p->cfg.pilot_k ? p->cfg.pilot_k : 1;
    p->routers = calloc((size_t)nl, sizeof *p->routers);
    p->pending = calloc((size_t)nl * k, sizeof(int32_t));
    p->pending_ok = calloc((size_t)nl, 1);
    p->pending_pos = calloc((size_t)nl, sizeof(int64_t));
    p->pending2 = calloc((size_t)nl * k, sizeof(int32_t));
    p->pending2_ok = calloc((size_t)nl, 1);
    p->pending2_pos = calloc((size_t)nl, sizeof(int64_t));
    p->acc_cap = 4096;
    p->acc = calloc((size_t)nl * p->acc_cap, sizeof(int32_t));
    p->acc_n = calloc((size_t)nl, sizeof(int));
    if (p->cfg.dump_path) {
        p->dump = fopen(p->cfg.dump_path, "w");
        if (!p->dump)
            fprintf(stderr, "pilot: cannot open dump %s\n", p->cfg.dump_path);
    }
    return p;
}

void apus_pilot_attach_router(ApusPilot *p, int layer,
                              const ApusPilotRouter *r) {
    if (!p || layer < 0 || layer >= p->cfg.n_layers) return;
    p->routers[layer] = *r;
}

void apus_pilot_attach_model(ApusPilot *p, ApusModel *m) {
    if (!p || !m) return;
    for (int i = 0; i < m->n_layers && i < p->cfg.n_layers; i++) {
        ApusLayer *L = &m->layers[i];
        ApusPilotRouter r = {
            .hash = L->hash,
            .gate_w = L->gate_w,
            .gate_bias = L->gate_bias,
            .ffn_norm_w = L->ffn_norm_w,
            .hc_fn = L->hc_ffn_fn,
            .hc_base = L->hc_ffn_base,
            .hc_scale = L->hc_ffn_scale,
            .tid2eid = L->tid2eid,
        };
        apus_pilot_attach_router(p, i, &r);
        L->pilot_ctx = p;
        L->pilot_post_attn = apus_pilot_tr_post_attn;
        /* wrap the store's MoE hooks for recall accounting (the saved
         * trampolines are identical across layers; save once) */
        if (L->mw.hook_hint && !p->saved_hint) {
            p->saved_ctx = L->mw.hook_ctx;
            p->saved_hint = L->mw.hook_hint;
            p->saved_layer_end = L->mw.hook_layer_end;
        }
        if (p->saved_hint) {
            /* resolve keeps hook_ctx (the store); hint/layer_end get the
             * pilot's own ctx via the M6b per-hook contexts */
            L->mw.hook_hint_ctx = p;
            L->mw.hook_hint = apus_pilot_tr_hint;
            L->mw.hook_layer_end_ctx = p;
            L->mw.hook_layer_end = apus_pilot_tr_layer_end;
        }
    }
}

int apus_pilot_start(ApusPilot *p) {
    if (!p || p->started) return 0;
    if (pthread_create(&p->thread, NULL, apus_pilot_consumer, p)) return -1;
    p->started = 1;
    return 0;
}

void apus_pilot_destroy(ApusPilot *p) {
    if (!p) return;
    if (p->started) {
        pthread_mutex_lock(&p->mtx);
        p->stopping = 1;
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mtx);
        pthread_join(p->thread, NULL);
    }
    if (p->dump) fclose(p->dump);
    pthread_mutex_destroy(&p->mtx);
    pthread_cond_destroy(&p->cv);
    free(p->ring);
    free(p->routers);
    free(p->pending);
    free(p->pending_ok);
    free(p->pending_pos);
    free(p->pending2);
    free(p->pending2_ok);
    free(p->pending2_pos);
    free(p->acc);
    free(p->acc_n);
    free(p);
}

void apus_pilot_stats(ApusPilot *p, ApusPilotStats *out) {
    out->predictions = atomic_load(&p->st_predictions);
    out->pred_experts = atomic_load(&p->st_pred_experts);
    out->hints_enqueued = atomic_load(&p->st_enqueued);
    out->hints_dropped_full = atomic_load(&p->st_dropped_full);
    out->hints_issued = atomic_load(&p->st_issued);
    out->hints_dropped_stale = atomic_load(&p->st_dropped_stale);
    out->hash_hints = atomic_load(&p->st_hash);
    out->actual_experts = atomic_load(&p->st_actual);
    out->actual_hits = atomic_load(&p->st_hits);
    out->prefill_predictions = atomic_load(&p->st_pf_preds);
    out->prefill_hints = atomic_load(&p->st_pf_hints);
    out->d2_predictions = atomic_load(&p->st_d2_preds);
    out->d2_rescue = atomic_load(&p->st_d2_rescue);
    out->d2_missed = atomic_load(&p->st_d2_missed);
}

#endif /* APUS_PILOT_IMPLEMENTATION */
#endif /* APUS_PILOT_H */
