/*
 * tests/m6b/test_pilot.c — M6b pilot unit tests:
 *
 *   1. Router-input equivalence: apus_pilot_router_input on a layer's
 *      post-attention hidden state is BITWISE identical to the ffn_norm
 *      output the real router consumes in apus_block_forward, and
 *      apus_pilot_predict's top-k prefix equals the MoE's actual
 *      router_idx (same sqrtsoftplus + selection-bias + stable top-k code
 *      paths — any deviation is a semantic bug).
 *   2. Ring: bounded, drop-newest when full, drains completely, clean
 *      wraparound; consumer survives stop/destroy with a non-empty ring.
 *   3. Thread lifetime: repeated start/stop/destroy, no leaks (UBSan).
 *   4. Hash-layer prefetch end-to-end: with a drained ring + synchronous
 *      store, EVERY hash-layer (0-2) expert the MoE asks for is already
 *      present (100% coverage), and no resolve ever submits a demand load.
 *   5. Usage-history heat decay: decay 0.5 halves old counts at save;
 *      decay 1.0 (default) preserves M6a cumulative behavior; pin seeding
 *      follows the decayed file.
 *   6. Store attribution: hint-then-resolve => demand_loads untouched.
 *
 * Exit 0 iff all checks pass. Run from the repository root.
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
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#define APUS_PILOT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "cache.h"
#include "pilot.h"

#define FIX "tests/m6b/fixtures"
#define TMP "tests/m6b/tmp"

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

static void pilot_drain(ApusPilot *p) {
    /* 1 ms polls: the same ~20 s wall-clock budget as before, but immune
     * to platforms whose nanosleep rounds sub-ms requests down to ~0
     * (Windows) — a fast-spinning drain would otherwise give up while
     * the consumer is simply a few ms behind. */
    struct timespec ts = {0, 1000000};
    for (int i = 0; i < 20000; i++) {
        ApusPilotStats s;
        apus_pilot_stats(p, &s);
        if (s.hints_issued + s.hints_dropped_stale >= s.hints_enqueued)
            return;
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "pilot_drain: timeout\n");
    g_checks++;
    g_fails++;
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
        .prefill_k = -1,            /* M6b scenario: last-token ring path */
        .d2 = -1,                  /* M6b scenario: dL=1 only */
        .hash_prefetch = 1,
        .prefill_last_only = 1,
        .ring_entries = 4096,
    };
    return c;
}

/* --- 1. router-input / prediction equivalence ----------------------------*/

static void test_router_equivalence(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, FIX, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        g_checks++; g_fails++;
        return;
    }
    ApusPilotCfg pc = test_pcfg(NULL, &m);
    pc.enabled = 0;
    ApusPilot *p = apus_pilot_create(&pc);
    apus_pilot_attach_model(p, &m);

    int dim = m.cfg.dim, hc = m.cfg.hc_mult, E = m.cfg.n_routed_experts;
    int topk = m.cfg.n_activated_experts;
    /* deterministic bf16-valued hidden state */
    float *h = malloc((size_t)hc * dim * sizeof(float));
    srand(12345);
    for (int i = 0; i < hc * dim; i++)
        h[i] = apus_bf16_round(((float)(rand() % 2001) - 1000.0f) / 500.0f);
    int64_t id = 42;

    for (int L = 3; L < m.n_layers; L++) {   /* non-hash layers */
        ApusLayerState lst;
        apus_layer_state_init(&lst, &m.layers[L]);
        float *pah = malloc((size_t)hc * dim * sizeof(float));
        float *fno = malloc((size_t)dim * sizeof(float));
        float *rsb = malloc((size_t)E * sizeof(float));
        int32_t *ridx = malloc((size_t)topk * sizeof(int32_t));
        ApusInterm it;
        memset(&it, 0, sizeof it);
        it.post_attn_h = pah;
        it.ffn_norm_out = fno;
        it.moe.router_scores_biased = rsb;
        it.moe.router_idx = ridx;
        float *hcopy = malloc((size_t)hc * dim * sizeof(float));
        memcpy(hcopy, h, (size_t)hc * dim * sizeof(float));
        apus_block_forward(&m.layers[L], &lst, hcopy, &id, 1, 0, &it);

        /* router input: bitwise vs the real ffn path */
        float *x = malloc((size_t)dim * sizeof(float));
        int rc = apus_pilot_router_input(p, L, pah, x);
        CHECK(rc == 0, "router_input layer %d rc=%d", L, rc);
        size_t bd = 0;
        for (int i = 0; i < dim; i++) {
            uint32_t a, b;
            memcpy(&a, &x[i], 4);
            memcpy(&b, &fno[i], 4);
            if (a != b) bd++;
        }
        CHECK(bd == 0, "router_input layer %d: %zu/%d bitwise diffs", L, bd,
              dim);

        /* prediction: top-k prefix == actual router_idx */
        int32_t idx[8];
        rc = apus_pilot_predict(p, L, pah, idx, 8);
        CHECK(rc == 0, "predict layer %d rc=%d", L, rc);
        int eq = 1;
        for (int j = 0; j < topk; j++)
            if (idx[j] != ridx[j]) eq = 0;
        CHECK(eq, "predict layer %d top-%d prefix != router_idx", L, topk);

        /* shared scoring path: apus_router_score == interm biased scores */
        float *sp = malloc((size_t)E * sizeof(float));
        float *bi = malloc((size_t)E * sizeof(float));
        apus_router_score(&m.layers[L].mw, fno, 1, sp, bi);
        size_t bs = 0;
        for (int i = 0; i < E; i++) {
            uint32_t a, b;
            memcpy(&a, &bi[i], 4);
            memcpy(&b, &rsb[i], 4);
            if (a != b) bs++;
        }
        CHECK(bs == 0, "router_score layer %d: %zu/%d bitwise diffs", L, bs,
              E);

        free(x); free(sp); free(bi);
        free(pah); free(fno); free(rsb); free(ridx); free(hcopy);
        apus_layer_state_free(&lst, &m.layers[L]);
    }
    /* hash targets are never predicted */
    CHECK(apus_pilot_predict(p, 0, h, (int32_t[]){0}, 4) == -1,
          "predict on hash target must fail");
    CHECK(apus_pilot_predict(p, m.n_layers, h, (int32_t[]){0}, 4) == -1,
          "predict beyond last layer must fail");
    free(h);
    apus_pilot_destroy(p);
    apus_model_free(&m);
}

/* --- 2/3. ring behavior + thread lifetime ---------------------------------*/

static void test_ring(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, FIX, err, sizeof err)) {
        g_checks++; g_fails++;
        return;
    }
    ApusPilotCfg pc = test_pcfg(NULL, &m);   /* store NULL: consumer no-ops */
    pc.ring_entries = 16;
    ApusPilot *p = apus_pilot_create(&pc);
    apus_pilot_attach_model(p, &m);
    ApusPilotStats s;

    /* thread NOT started: pushes are deterministic */
    int64_t ids[8] = {1, 2, 3, 4, 5, 6, 7, 8};   /* 8 x 3 layers x 4 = 96 */
    apus_pilot_prefetch_hash(p, ids, 8, 0);
    apus_pilot_stats(p, &s);
    CHECK(s.hints_enqueued == 16, "ring: enqueued %llu != 16",
          (unsigned long long)s.hints_enqueued);
    CHECK(s.hints_dropped_full == 80, "ring: dropped-full %llu != 80",
          (unsigned long long)s.hints_dropped_full);
    CHECK(s.hash_hints == 16, "ring: hash hints %llu != 16",
          (unsigned long long)s.hash_hints);
    CHECK(s.hints_issued == 0, "ring: issued %llu != 0 (no consumer)",
          (unsigned long long)s.hints_issued);

    /* start the consumer: drains all 16 */
    CHECK(apus_pilot_start(p) == 0, "ring: start failed");
    pilot_drain(p);
    apus_pilot_stats(p, &s);
    CHECK(s.hints_issued + s.hints_dropped_stale == 16,
          "ring: drained %llu + %llu != 16",
          (unsigned long long)s.hints_issued,
          (unsigned long long)s.hints_dropped_stale);

    /* wraparound: two more drained batches, each smaller than the ring, so
     * every push is accepted and the indices wrap past the capacity */
    int64_t one[1] = {9};                    /* 1 x 3 layers x 4 = 12 < 16 */
    apus_pilot_prefetch_hash(p, one, 1, 100);
    pilot_drain(p);
    apus_pilot_prefetch_hash(p, one, 1, 200);
    pilot_drain(p);
    apus_pilot_stats(p, &s);
    CHECK(s.hints_enqueued == 40, "ring: enqueued %llu != 40 after wrap",
          (unsigned long long)s.hints_enqueued);
    CHECK(s.hints_issued + s.hints_dropped_stale == 40,
          "ring: consumed %llu != 40 after wrap",
          (unsigned long long)(s.hints_issued + s.hints_dropped_stale));
    CHECK(s.hints_dropped_full == 80,
          "ring: dropped-full changed to %llu (must stay 80)",
          (unsigned long long)s.hints_dropped_full);
    apus_pilot_destroy(p);

    /* lifetime: repeated start/stop with traffic, destroy with backlog */
    for (int rep = 0; rep < 3; rep++) {
        ApusPilot *q = apus_pilot_create(&pc);
        apus_pilot_attach_model(q, &m);
        apus_pilot_start(q);
        apus_pilot_prefetch_hash(q, ids, 8, 0);
        if (rep == 1) pilot_drain(q);
        apus_pilot_destroy(q);      /* backlog discarded on rep 0/2 */
    }
    /* destroy without start */
    ApusPilot *q = apus_pilot_create(&pc);
    apus_pilot_prefetch_hash(q, ids, 2, 0);
    apus_pilot_destroy(q);
    g_checks++;   /* reaching here without hang/crash is the assertion */
    apus_model_free(&m);
}

/* --- 4. hash-layer prefetch coverage --------------------------------------*/

static ApusStore *g_store;
static long g_asked, g_present;

static int test_resolve(void *ctx, int layer, int eid,
                        ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3) {
    (void)ctx;
    if (layer < 3) {   /* hash layers */
        g_asked++;
        g_present += apus_store_debug_present(g_store, layer, eid);
    }
    return apus_store_resolve(g_store, layer, eid, w1, w2, w3);
}

static void test_hash_prefetch(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, FIX, 1, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        g_checks++; g_fails++;
        return;
    }
    ApusStoreCfg sc = {0};
    sc.n_layers = m.n_layers;
    sc.n_experts = m.cfg.n_routed_experts;
    sc.slots_per_layer = 8;
    sc.pins_per_layer = 0;
    sc.io_threads = -1;             /* synchronous: hint completes inline */
    sc.usage_path = "";
    ApusStore *st = apus_store_open(FIX, &sc, err, sizeof err);
    if (!st) { fprintf(stderr, "store: %s\n", err); g_checks++; g_fails++; return; }
    g_store = st;
    for (int i = 0; i < m.n_layers; i++) {
        apus_store_attach_moe(st, &m.layers[i].mw);
        m.layers[i].mw.hook_resolve = test_resolve;   /* observation only */
    }
    ApusPilotCfg pc = test_pcfg(st, &m);
    pc.pilot_k = 0;                 /* router lookahead OFF: hash only */
    ApusPilot *p = apus_pilot_create(&pc);
    apus_pilot_attach_model(p, &m);
    apus_pilot_start(p);

    static const int64_t PROMPT[8] = {3, 41, 7, 200, 511, 0, 128, 65};
    int V = m.cfg.vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    ApusModelState mst;
    apus_model_state_init(&mst, &m);

    apus_pilot_prefetch_hash(p, PROMPT, 8, 0);
    pilot_drain(p);
    apus_model_forward(&m, &mst, PROMPT, 8, logits, 0);
    for (int t = 0; t < 8; t++) {
        int64_t next = (int64_t)((t * 37 + 11) % V);
        apus_pilot_prefetch_hash(p, &next, 1, mst.pos);
        pilot_drain(p);
        apus_model_forward(&m, &mst, &next, 1, logits, 0);
    }

    /* every hash-layer ask found the expert already present: 100%.
     * Expected asks: M9b batches prefill routed experts per UNIQUE expert
     * per layer (was: per (t,j)), so the prefill side asks once per unique
     * (layer, expert) pair over the prompt; decode stays per (t,j). */
    long uniq = 0;
    for (int L = 0; L < 3; L++) {
        const int64_t *t2e = m.layers[L].mw.tid2eid;
        int topk = m.layers[L].mw.topk;
        uint64_t seen[4] = {0, 0, 0, 0};   /* E <= 256 */
        for (int t = 0; t < 8; t++)
            for (int j = 0; j < topk; j++) {
                int e = (int)t2e[PROMPT[t] * topk + j];
                if (!(seen[e >> 6] & (1ull << (e & 63)))) {
                    seen[e >> 6] |= 1ull << (e & 63);
                    uniq++;
                }
            }
    }
    CHECK(g_asked == uniq + 8 * 3 * 4,
          "hash: asked %ld != %ld (unique prefill + decode)", g_asked,
          uniq + 8 * 3 * 4);
    CHECK(g_present == g_asked, "hash: coverage %ld/%ld < 100%%",
          g_present, g_asked);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.demand_loads == 0, "hash: %llu demand loads (must be 0)",
          (unsigned long long)ss.demand_loads);
    ApusPilotStats ps;
    apus_pilot_stats(p, &ps);
    CHECK(ps.hash_hints == ps.hints_enqueued,
          "hash: %llu hash != %llu enqueued (pilot_k=0)",
          (unsigned long long)ps.hash_hints,
          (unsigned long long)ps.hints_enqueued);
    printf("  hash prefetch: %ld/%ld present at ask (100%%), "
           "%llu preads, 0 demand loads\n",
           g_present, g_asked, (unsigned long long)ss.preads);

    free(logits);
    apus_model_state_free(&mst, &m);
    apus_pilot_destroy(p);
    apus_store_close(st);
    apus_model_free(&m);
}

/* --- 5. usage-history heat decay -------------------------------------------*/

static int read_usage_count(const char *path, int want_l, int want_e) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int L, e, c, out = 0;
    while (fscanf(f, "%d %d %d", &L, &e, &c) == 3)
        if (L == want_l && e == want_e) out = c;
    fclose(f);
    return out;
}

static void test_heat_decay(void) {
    char err[256];
    (void)apus_sys_mkdir_p(TMP);   /* M15 */
    const char *up = TMP "/usage_decay.txt";
    FILE *f = fopen(up, "w");
    fprintf(f, "0 5 100\n0 6 50\n1 7 33\n");
    fclose(f);

    /* decay 0.5: halves the old counts at save */
    ApusStoreCfg sc = {0};
    sc.n_layers = 6;
    sc.n_experts = 64;
    sc.slots_per_layer = 4;
    sc.pins_per_layer = 1;
    sc.io_threads = -1;
    sc.usage_path = up;
    sc.usage_decay = 0.5;
    ApusStore *st = apus_store_open(FIX, &sc, err, sizeof err);
    CHECK(st != NULL, "decay: store open: %s", err);
    CHECK(apus_store_save_usage(st) == 0, "decay: save failed");
    /* pin seeded from the file: hottest = eid 5 */
    int32_t pins[1] = {-1};
    apus_store_debug_layer(st, 0, NULL, 0, pins, 1);
    CHECK(pins[0] == 5, "decay: pin eid %d != 5", pins[0]);
    apus_store_close(st);
    CHECK(read_usage_count(up, 0, 5) == 50, "decay: count %d != 50",
          read_usage_count(up, 0, 5));
    CHECK(read_usage_count(up, 0, 6) == 25, "decay: count %d != 25",
          read_usage_count(up, 0, 6));
    CHECK(read_usage_count(up, 1, 7) == 16, "decay: count %d != 16",
          read_usage_count(up, 1, 7));

    /* decay 1.0 (default): cumulative, M6a behavior preserved */
    sc.usage_decay = 1.0;
    st = apus_store_open(FIX, &sc, err, sizeof err);
    CHECK(st != NULL, "decay1.0: store open: %s", err);
    apus_store_save_usage(st);
    apus_store_close(st);
    CHECK(read_usage_count(up, 0, 5) == 50, "decay1.0: count %d != 50",
          read_usage_count(up, 0, 5));
}

/* --- 6. store hint/demand attribution ---------------------------------------*/

static void test_attribution(void) {
    char err[256];
    ApusStoreCfg sc = {0};
    sc.n_layers = 6;
    sc.n_experts = 64;
    sc.slots_per_layer = 8;
    sc.io_threads = -1;
    sc.usage_path = "";
    ApusStore *st = apus_store_open(FIX, &sc, err, sizeof err);
    CHECK(st != NULL, "attr: store open: %s", err);
    ApusFp4W w1, w2, w3;
    apus_store_hint(st, 0, 7);                 /* prefetch */
    CHECK(apus_store_resolve(st, 0, 7, &w1, &w2, &w3) == 0, "attr: resolve 7");
    CHECK(apus_store_resolve(st, 0, 8, &w1, &w2, &w3) == 0, "attr: resolve 8");
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.hint_loads == 1, "attr: hint_loads %llu != 1",
          (unsigned long long)ss.hint_loads);
    CHECK(ss.demand_loads == 1, "attr: demand_loads %llu != 1",
          (unsigned long long)ss.demand_loads);
    CHECK(ss.preads == 2, "attr: preads %llu != 2",
          (unsigned long long)ss.preads);
    /* after a resolve the expert sits in the working set: present == 1;
     * an untouched expert is not */
    CHECK(apus_store_debug_present(st, 0, 7) == 1,
          "attr: eid 7 must be present after resolve");
    CHECK(apus_store_debug_present(st, 0, 9) == 0,
          "attr: eid 9 must NOT be present");
    apus_store_close(st);
}

int main(void) {
    printf("test_pilot: M6b pilot unit tests\n");
    test_router_equivalence();
    test_ring();
    test_hash_prefetch();
    test_heat_decay();
    test_attribution();
    printf("test_pilot: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
