/*
 * tests/m6b/test_recall.c — end-to-end pilot recall measurement on the
 * fixture model. Two dumps of the SAME greedy run (deterministic):
 *
 *   runtime dump  (arg 1): written by the live pilot itself — P lines from
 *                  the post-attention hook predictions, A lines from the
 *                  wrapped MoE hint hook (decode path, s == 1). The pilot's
 *                  own stats counters must agree with a Python recomputation
 *                  over this file (tests/m6b/check_recall.py).
 *   measure dump  (arg 2): written through apus_model_forward_measure +
 *                  apus_pilot_predict (the --measure-locality machinery the
 *                  §8 tool consumes). For decode positions its P/A sets
 *                  must be IDENTICAL to the runtime dump's — the two paths
 *                  share the prediction code, this proves the plumbing.
 *
 * Prints one machine-readable line:
 *   RECALL hits=H actuals=A predictions=P enqueued=E issued=I hash=G dropped_full=DF dropped_stale=DS tokens=T0,T1,...
 *
 * NOTE: the fixture model has RANDOM weights; the recall NUMBER here is
 * not meaningful (near-uniform router, bias-dominated selection). What is
 * validated is the machinery: counts, dump consistency, Python-vs-C recall
 * agreement. Exit 0 iff the run completed and counters are self-consistent
 * (0 < hits <= actuals, actuals == decode_tokens * non-hash layers * topk).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_FP4_IMPLEMENTATION
#define APUS_FP8_IMPLEMENTATION
#define APUS_MHC_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#define APUS_PILOT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
#include "cache.h"
#include "pilot.h"

#define FIX "tests/m6b/fixtures"
#define N_PROMPT 8
#define N_GEN 24
#define MEASURE_N 24

static const int64_t PROMPT[N_PROMPT] = {3, 41, 7, 200, 511, 0, 128, 65};

/* measure-dump callback (same as c/apus.c --measure-locality) */
typedef struct {
    FILE *f;
    ApusPilot *pilot;
    int topk;
} MD;

static void measure_cb(void *ctx, int layer, int64_t start_pos,
                       const int64_t *ids, int s,
                       const int32_t *router_idx, const float *post_attn_h) {
    MD *md = ctx;
    int hc_dim = md->pilot->cfg.hc_mult * md->pilot->cfg.dim;
    (void)ids;
    for (int t = 0; t < s; t++) {
        fprintf(md->f, "{\"type\":\"A\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                (long long)(start_pos + t), layer);
        for (int j = 0; j < md->topk; j++)
            fprintf(md->f, "%s%d", j ? "," : "",
                    router_idx[(size_t)t * md->topk + j]);
        fprintf(md->f, "]}\n");
        if (layer + 1 < md->pilot->cfg.n_layers) {
            int32_t idx[MEASURE_N];
            if (apus_pilot_predict(md->pilot, layer + 1,
                                   post_attn_h + (size_t)t * hc_dim,
                                   idx, MEASURE_N) == 0) {
                fprintf(md->f,
                        "{\"type\":\"P\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                        (long long)(start_pos + t), layer + 1);
                for (int j = 0; j < MEASURE_N; j++)
                    fprintf(md->f, "%s%d", j ? "," : "", idx[j]);
                fprintf(md->f, "]}\n");
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test_recall RUNTIME_DUMP MEASURE_DUMP\n");
        return 2;
    }
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, FIX, 1, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    ApusStoreCfg sc = {0};
    sc.n_layers = m.n_layers;
    sc.n_experts = m.cfg.n_routed_experts;
    sc.slots_per_layer = 16;
    sc.io_threads = 4;
    sc.usage_path = "";
    ApusStore *st = apus_store_open(FIX, &sc, err, sizeof err);
    if (!st) { fprintf(stderr, "store: %s\n", err); return 1; }
    for (int i = 0; i < m.n_layers; i++)
        apus_store_attach_moe(st, &m.layers[i].mw);

    ApusPilotCfg pc = {
        .store = st,
        .n_layers = m.n_layers,
        .n_experts = m.cfg.n_routed_experts,
        .topk = m.cfg.n_activated_experts,
        .dim = m.cfg.dim,
        .hc_mult = m.cfg.hc_mult,
        .sinkhorn_iters = m.cfg.hc_sinkhorn_iters,
        .vocab = m.cfg.vocab_size,
        .norm_eps = m.cfg.norm_eps,
        .hc_eps = m.cfg.hc_eps,
        .enabled = 1,
        .pilot_k = 8,
        .prefill_k = -1,            /* M6b scenario: last-token ring path */
        .d2 = -1,                  /* M6b scenario: dL=1 only */
        .hash_prefetch = 1,
        .prefill_last_only = 1,
        .ring_entries = 4096,
        .dump_path = argv[1],
    };
    ApusPilot *pi = apus_pilot_create(&pc);
    apus_pilot_attach_model(pi, &m);
    apus_pilot_start(pi);

    FILE *mf = fopen(argv[2], "w");
    if (!mf) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    MD md = { mf, pi, m.cfg.n_activated_experts };

    int V = m.cfg.vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    ApusModelState stt;
    apus_model_state_init(&stt, &m);
    int tokens[N_GEN];

    /* one deterministic greedy run; BOTH dumps record it */
    apus_pilot_prefetch_hash(pi, PROMPT, N_PROMPT, 0);
    apus_model_forward_measure(&m, &stt, PROMPT, N_PROMPT, logits,
                               measure_cb, &md);
    for (int t = 0; t < N_GEN; t++) {
        int tok = apus_sample_argmax(logits, (size_t)V);
        tokens[t] = tok;
        if (t + 1 < N_GEN) {
            int64_t next = tok;
            apus_pilot_prefetch_hash(pi, &next, 1, stt.pos);
            apus_model_forward_measure(&m, &stt, &next, 1, logits,
                                       measure_cb, &md);
        }
    }
    fclose(mf);

    ApusPilotStats ps;
    apus_pilot_stats(pi, &ps);
    int fails = 0;
    /* counter self-consistency (decode path: 23 forwarded tokens x 3
     * non-hash layers x topk 4 actuals) */
    uint64_t want_actuals = (uint64_t)(N_GEN - 1) * 3 * m.cfg.n_activated_experts;
    if (ps.actual_experts != want_actuals) {
        fprintf(stderr, "FAIL: actual_experts %llu != %llu\n",
                (unsigned long long)ps.actual_experts,
                (unsigned long long)want_actuals);
        fails++;
    }
    if (ps.actual_hits == 0 || ps.actual_hits > ps.actual_experts) {
        fprintf(stderr, "FAIL: hits %llu out of range\n",
                (unsigned long long)ps.actual_hits);
        fails++;
    }
    if (ps.predictions != (uint64_t)(N_GEN - 1) * 3 + 3) {   /* +3 prefill */
        fprintf(stderr, "FAIL: predictions %llu != %d\n",
                (unsigned long long)ps.predictions, (N_GEN - 1) * 3 + 3);
        fails++;
    }
    printf("RECALL hits=%llu actuals=%llu predictions=%llu enqueued=%llu "
           "issued=%llu hash=%llu dropped_full=%llu dropped_stale=%llu "
           "tokens=",
           (unsigned long long)ps.actual_hits,
           (unsigned long long)ps.actual_experts,
           (unsigned long long)ps.predictions,
           (unsigned long long)ps.hints_enqueued,
           (unsigned long long)ps.hints_issued,
           (unsigned long long)ps.hash_hints,
           (unsigned long long)ps.hints_dropped_full,
           (unsigned long long)ps.hints_dropped_stale);
    for (int t = 0; t < N_GEN; t++) printf("%s%d", t ? "," : "", tokens[t]);
    printf("\n");

    free(logits);
    apus_model_state_free(&stt, &m);
    apus_pilot_destroy(pi);
    apus_store_close(st);
    apus_model_free(&m);
    printf("test_recall: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
