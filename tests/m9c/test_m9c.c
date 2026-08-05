/*
 * tests/m9c/test_m9c.c — M9c expert-I/O pipelining tests:
 *
 *   1. Batched union prediction (c/pilot.h apus_pilot_predict_union): the
 *      union bitmap over s tokens is EXACTLY the union of the per-token
 *      apus_pilot_predict top-k sets (same math, batched — a deviation is
 *      a pilot bug that would silently cost coverage).
 *   2. Demand-boost I/O queue (c/cache.h): with a backed-up speculative
 *      queue, a demand-class hint's load completes before the speculative
 *      backlog; a resolve blocking on a speculative load boosts it.
 *   3. Wait instrumentation: waits/wait_ns count only blocking resolves.
 *   4. Quality invariance with the prefill union lookahead ON: greedy
 *      prefill+decode, pilot prefill_k=12 vs pilot OFF vs eager — token
 *      streams and all logits BITWISE identical across cache budgets.
 *      (The M6b invariance gate pins the old last-token path; this pins
 *      the new default path.)
 *   5. Thread-count independence: an FNV-1a digest of the invariance
 *      logits is printed; the Makefile diffs it across APUS_THREADS=1/4/8.
 *
 * Run from the repository root. Exit 0 iff all checks pass.
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
#include <time.h>
#include <unistd.h>

#include "model.h"
#include "cache.h"
#include "pilot.h"
#include "sample.h"

#define FIX "tests/m6b/fixtures"
#define TMP "tests/m9c/tmp"

static long g_checks;
static int g_fails;

#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { \
        g_fails++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static uint64_t g_fnv;
static void fnv_mix(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        g_fnv ^= b[i];
        g_fnv *= 1099511628211ull;
    }
}

static ApusPilotCfg test_pcfg(ApusStore *store, const ApusModel *m) {
    ApusPilotCfg c = {
        .store = store,
        .n_layers = m->n_layers,
        .n_experts = m->cfg.n_routed_experts,
        .topk = m->cfg.n_activated_experts,
        .dim = m->cfg.dim,
        .hc_mult = m->cfg.hc_mult,
        .sinkhorn_iters = m->cfg.hc_sinkhorn_iters,
        .vocab = m->cfg.vocab_size,
        .norm_eps = m->cfg.norm_eps,
        .hc_eps = m->cfg.hc_eps,
        .enabled = 1,
        .pilot_k = 8,
        .prefill_k = 12,            /* the new default path under test */
        .hash_prefetch = 1,
        .prefill_last_only = 1,
        .ring_entries = 4096,
    };
    return c;
}

/* --- 1. batched union == per-token union -----------------------------------*/

static void test_union_equivalence(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, FIX, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        g_checks++; g_fails++;
        return;
    }
    ApusPilotCfg pc = test_pcfg(NULL, &m);
    ApusPilot *p = apus_pilot_create(&pc);
    apus_pilot_attach_model(p, &m);

    int dim = m.cfg.dim, hc = m.cfg.hc_mult, E = m.cfg.n_routed_experts;
    int topk = m.cfg.n_activated_experts;
    int s = 8;
    float *h = malloc((size_t)s * hc * dim * sizeof(float));
    srand(999);
    for (int i = 0; i < s * hc * dim; i++)
        h[i] = apus_bf16_round(((float)(rand() % 2001) - 1000.0f) / 700.0f);

    int nw = (E + 63) / 64;
    for (int target = 3; target < m.n_layers; target++) {   /* non-hash */
        for (int k = topk; k <= 12; k += 4) {
            uint64_t *ub = calloc((size_t)nw, sizeof(uint64_t));
            uint64_t *ur = calloc((size_t)nw, sizeof(uint64_t));
            int rc = apus_pilot_predict_union(p, target, h, s, k, ub);
            CHECK(rc == 0, "union: target %d rc=%d", target, rc);
            for (int t = 0; t < s; t++) {
                int32_t idx[16];
                if (apus_pilot_predict(p, target,
                                       h + (size_t)t * hc * dim, idx, k))
                    continue;
                for (int j = 0; j < k; j++)
                    ur[idx[j] >> 6] |= 1ull << (idx[j] & 63);
            }
            int diff = 0;
            for (int i = 0; i < nw; i++)
                if (ub[i] != ur[i]) diff = 1;
            CHECK(!diff, "union: target %d k=%d bitmap != per-token union",
                  target, k);
            free(ub);
            free(ur);
        }
    }
    /* hash target and out-of-range must fail */
    uint64_t dummy[1] = {0};
    CHECK(apus_pilot_predict_union(p, 0, h, s, 8, dummy) == -1,
          "union: hash target must fail");
    CHECK(apus_pilot_predict_union(p, m.n_layers, h, s, 8, dummy) == -1,
          "union: out-of-range target must fail");
    free(h);
    apus_pilot_destroy(p);
    apus_model_free(&m);
}

/* --- 2/3. demand-boost priority + wait instrumentation ----------------------*/

/* Slow every job's claim by 5 ms (worker thread, mu NOT held) so queue
 * ordering is observable with the tiny fixture slabs. */
static void slow_claim(ApusStore *st, int layer, int32_t eid, uint64_t gen) {
    (void)st; (void)layer; (void)eid; (void)gen;
    struct timespec ts = {0, 5000000};
    nanosleep(&ts, NULL);
}

static void test_boost(void) {
    char err[256];
    (void)system("mkdir -p " TMP);
    ApusStoreCfg sc = {0};
    sc.n_layers = 6;
    sc.n_experts = 64;
    sc.slots_per_layer = 8;
    sc.io_threads = 1;              /* single worker: strict order */
    sc.usage_path = "";
    ApusStore *st = apus_store_open(FIX, &sc, err, sizeof err);
    CHECK(st != NULL, "boost: store open: %s", err);
    if (!st) return;
    apus_store_debug_set_pre_claim(st, slow_claim);

    /* speculative backlog: 24 experts of layer 1 */
    for (int e = 0; e < 24; e++) apus_store_hint(st, 1, e);
    /* demand-class hint for one expert of layer 1, submitted AFTER the
     * speculative backlog: must jump the queue */
    int hot_e = 40;
    apus_store_hint_demand(st, 1, hot_e);
    ApusFp4W w1, w2, w3;
    /* wait for the hot expert to be READY; then count how many of the
     * speculative experts have COMPLETED (READY) too. With boost: the
     * in-flight job + timing slack at most (<= 4); without: all 24. */
    CHECK(apus_store_resolve(st, 1, hot_e, &w1, &w2, &w3) == 0,
          "boost: resolve hot expert");
    int spec_ready = 0;
    for (int e = 0; e < 24; e++)
        spec_ready += apus_store_debug_ready(st, 1, e);
    CHECK(spec_ready <= 4,
          "boost: %d/24 speculative experts done before the demand one "
          "(want <= 4)", spec_ready);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.hint_loads == 25, "boost: hint_loads %llu != 25",
          (unsigned long long)ss.hint_loads);
    CHECK(ss.demand_loads == 0, "boost: demand_loads %llu != 0",
          (unsigned long long)ss.demand_loads);
    apus_store_close(st);

    /* resolve blocked on a speculative load boosts it + wait accounting:
     * fresh store, speculative-hint 24 experts, then resolve the LAST
     * queued one — it must complete without waiting for the whole queue */
    sc.io_threads = 1;
    ApusStore *st2 = apus_store_open(FIX, &sc, err, sizeof err);
    CHECK(st2 != NULL, "boost2: store open: %s", err);
    if (!st2) return;
    apus_store_debug_set_pre_claim(st2, slow_claim);
    for (int e = 0; e < 24; e++) apus_store_hint(st2, 2, e);
    int last_e = 23;
    CHECK(apus_store_resolve(st2, 2, last_e, &w1, &w2, &w3) == 0,
          "boost2: resolve tail expert");
    int ready = 0;
    for (int e = 0; e < last_e; e++)
        ready += apus_store_debug_ready(st2, 2, e);
    CHECK(ready <= 4,
          "boost2: %d/23 queued experts done before the boosted resolve "
          "(want <= 4)", ready);
    apus_store_stats(st2, &ss);
    CHECK(ss.waits >= 1, "boost2: waits %llu must be >= 1",
          (unsigned long long)ss.waits);
    CHECK(ss.wait_ns > 0, "boost2: wait_ns must be > 0");
    apus_store_close(st2);

    /* a resolve served from the resident working set does NOT count a wait */
    ApusStoreCfg sc3 = sc;
    sc3.io_threads = -1;            /* synchronous: hint completes inline */
    ApusStore *st3 = apus_store_open(FIX, &sc3, err, sizeof err);
    CHECK(st3 != NULL, "waitsync: store open: %s", err);
    if (!st3) return;
    apus_store_hint(st3, 0, 7);
    CHECK(apus_store_resolve(st3, 0, 7, &w1, &w2, &w3) == 0,
          "waitsync: resolve 7");
    CHECK(apus_store_resolve(st3, 0, 7, &w1, &w2, &w3) == 0,
          "waitsync: re-resolve 7");
    apus_store_stats(st3, &ss);
    CHECK(ss.waits == 0, "waitsync: waits %llu != 0 (sync + resident)",
          (unsigned long long)ss.waits);
    CHECK(ss.wait_ns == 0, "waitsync: wait_ns != 0");
    apus_store_close(st3);
}

/* --- 4/5. invariance with prefill union lookahead ON -------------------------*/

#define N_GEN 16

typedef struct {
    int tiered;
    int64_t tokens[N_GEN];
    float *logits;      /* [N_GEN][V] */
    int V;
} Run;

static int run_greedy(Run *r, int slots, int prefill_k, const char *usage) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, FIX, slots >= 0, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        return -1;
    }
    ApusStore *st = NULL;
    ApusPilot *pi = NULL;
    if (slots >= 0) {
        ApusStoreCfg sc = {0};
        sc.n_layers = m.n_layers;
        sc.n_experts = m.cfg.n_routed_experts;
        sc.slots_per_layer = slots;
        sc.pins_per_layer = 0;
        sc.io_threads = 4;
        sc.usage_path = usage;
        st = apus_store_open(FIX, &sc, err, sizeof err);
        if (!st) { fprintf(stderr, "store: %s\n", err); return -1; }
        for (int i = 0; i < m.n_layers; i++)
            apus_store_attach_moe(st, &m.layers[i].mw);
        ApusPilotCfg pc = test_pcfg(st, &m);
        pc.prefill_k = prefill_k;
        pi = apus_pilot_create(&pc);
        apus_pilot_attach_model(pi, &m);
        apus_pilot_start(pi);
    }
    static const int64_t PROMPT[8] = {3, 41, 7, 200, 511, 0, 128, 65};
    int V = m.cfg.vocab_size;
    r->V = V;
    r->logits = malloc((size_t)N_GEN * V * sizeof(float));
    float *lg = malloc((size_t)V * sizeof(float));
    ApusRng rng;
    apus_rng_seed(&rng, 42);
    void *smp = malloc(apus_sample_scratch_size((size_t)V));
    ApusModelState mst;
    apus_model_state_init(&mst, &m);
    if (pi) apus_pilot_prefetch_hash(pi, PROMPT, 8, 0);
    apus_model_forward(&m, &mst, PROMPT, 8, lg, 0);
    for (int t = 0; t < N_GEN; t++) {
        int tok = apus_sample(lg, (size_t)V, 0.0f, 1.0f, &rng, smp);
        r->tokens[t] = tok;
        int64_t next = tok;
        if (pi) apus_pilot_prefetch_hash(pi, &next, 1, mst.pos);
        apus_model_forward(&m, &mst, &next, 1, lg, 0);
        memcpy(r->logits + (size_t)t * V, lg, (size_t)V * sizeof(float));
    }
    free(lg);
    free(smp);
    apus_model_state_free(&mst, &m);
    if (pi) apus_pilot_destroy(pi);
    if (st) apus_store_close(st);
    r->tiered = slots >= 0;
    apus_model_free(&m);
    return 0;
}

static int cmp_runs(const char *name, const Run *ref, const Run *b) {
    int tok_diff = 0;
    for (int t = 0; t < N_GEN; t++)
        if (ref->tokens[t] != b->tokens[t]) tok_diff++;
    size_t bit = 0;
    for (size_t i = 0; i < (size_t)N_GEN * ref->V; i++) {
        uint32_t ua, ub;
        memcpy(&ua, &ref->logits[i], 4);
        memcpy(&ub, &b->logits[i], 4);
        if (ua != ub) bit++;
    }
    int ok = tok_diff == 0 && bit == 0;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-52s tokdiff=%d/%d logit-bitdiff=%zu/%zu\n",
           ok ? "ok  " : "FAIL", name, tok_diff, N_GEN, bit,
           (size_t)N_GEN * ref->V);
    return ok;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    g_fnv = 1469598103934665603ull;
    printf("test_m9c: M9c expert-I/O pipelining\n");
    test_union_equivalence();
    test_boost();

    Run eager, off, on_full, on_min;
    if (run_greedy(&eager, -1, 0, NULL)) return 1;
    if (run_greedy(&off, 64, -1, "")) return 1;
    if (run_greedy(&on_full, 64, 12, "")) return 1;
    if (run_greedy(&on_min, 2, 12, "")) return 1;
    cmp_runs("prefill-union OFF (64 slots) vs eager", &eager, &off);
    cmp_runs("prefill-union ON k=12 (64 slots) vs eager", &eager, &on_full);
    cmp_runs("prefill-union ON k=12 (2 slots) vs eager", &eager, &on_min);
    cmp_runs("prefill-union ON vs OFF (64 slots)", &off, &on_full);

    fnv_mix(eager.tokens, sizeof eager.tokens);
    fnv_mix(on_full.tokens, sizeof on_full.tokens);
    fnv_mix(on_min.tokens, sizeof on_min.tokens);
    fnv_mix(on_full.logits, (size_t)N_GEN * on_full.V * sizeof(float));
    fnv_mix(on_min.logits, (size_t)N_GEN * on_min.V * sizeof(float));
    printf("  digest %016llx\n", (unsigned long long)g_fnv);
    free(eager.logits);
    free(off.logits);
    free(on_full.logits);
    free(on_min.logits);
    printf("test_m9c: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
