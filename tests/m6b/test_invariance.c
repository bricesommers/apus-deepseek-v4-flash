/*
 * tests/m6b/test_invariance.c — THE M6b hard test: the pilot (router-
 * lookahead prefetch + hash-layer prefetch + a live pilot thread issuing
 * real loads through the store) must NEVER change numerics. Identical
 * prompt, greedy decode, full forward:
 *
 *   (eager)     legacy slurp path (reference)
 *   (off-full)  store 64 slots/layer, pilot OFF
 *   (on-full)   store 64 slots/layer, pilot ON (K=8, hash, ring, 4 I/O)
 *   (on-25%)    store 16 slots/layer, pilot ON
 *   (on-min)    store 2 slots/layer, pilot ON (hint thrash + eviction guard)
 *   (on-guard)  store 2 slots/layer + RSS budget 1 byte, pilot ON
 *   (on-pinned) store 4 slots + 2 pins/layer (usage file), pilot ON
 *
 * Token streams IDENTICAL and logits BITWISE identical across all seven.
 * The pilot only changes WHEN/WHETHER an expert is in RAM; the store
 * serves the same slab bytes to the same kernels in the same order, so
 * any difference is a pilot/store bug, not noise.
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

static long g_checks;
static int g_fails;

#define N_PROMPT 8
#define N_GEN 24

static const int64_t PROMPT[N_PROMPT] = {3, 41, 7, 200, 511, 0, 128, 65};

typedef struct {
    int tokens[N_GEN];
    float *logits;
    int V;
    ApusStoreStats store;
    ApusPilotStats pilot;
    int tiered;
} Run;

/* One greedy run. slots < 0 -> legacy eager path (no store, no pilot). */
static int run_greedy(Run *r, int slots, int pins, uint64_t rss_budget,
                      int pilot_on, const char *usage_path) {
    char err[256];
    ApusModel m;
    int tiered = slots >= 0;
    if (apus_model_load_ex(&m, FIX, tiered, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        return -1;
    }
    ApusStore *st = NULL;
    ApusPilot *pi = NULL;
    if (tiered) {
        ApusStoreCfg c = {0};
        c.n_layers = m.n_layers;
        c.n_experts = m.cfg.n_routed_experts;
        c.slots_per_layer = slots;
        c.pins_per_layer = pins;
        c.rss_budget_bytes = rss_budget;
        c.io_threads = 4;           /* async pool: full concurrency */
        c.usage_path = usage_path;
        st = apus_store_open(FIX, &c, err, sizeof err);
        if (!st) {
            fprintf(stderr, "store: %s\n", err);
            return -1;
        }
        for (int i = 0; i < m.n_layers; i++)
            apus_store_attach_moe(st, &m.layers[i].mw);
        if (pilot_on) {
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
                .prefill_k = -1,        /* M6b scenario: last-token path */
                .d2 = -1,              /* M6b scenario: dL=1 only */
                .hash_prefetch = 1,
                .prefill_last_only = 1,
                .ring_entries = 4096,
            };
            pi = apus_pilot_create(&pc);
            apus_pilot_attach_model(pi, &m);
            apus_pilot_start(pi);
        }
    }
    int V = m.cfg.vocab_size;
    r->V = V;
    r->logits = malloc((size_t)N_GEN * V * sizeof(float));
    ApusModelState stt;
    apus_model_state_init(&stt, &m);
    float *logits = malloc((size_t)V * sizeof(float));
    if (pi) apus_pilot_prefetch_hash(pi, PROMPT, N_PROMPT, 0);
    apus_model_forward(&m, &stt, PROMPT, N_PROMPT, logits, 0);
    for (int t = 0; t < N_GEN; t++) {
        int tok = apus_sample_argmax(logits, (size_t)V);
        r->tokens[t] = tok;
        memcpy(r->logits + (size_t)t * V, logits, (size_t)V * sizeof(float));
        if (t + 1 < N_GEN) {
            int64_t next = tok;
            if (pi) apus_pilot_prefetch_hash(pi, &next, 1, stt.pos);
            apus_model_forward(&m, &stt, &next, 1, logits, 0);
        }
    }
    free(logits);
    apus_model_state_free(&stt, &m);
    if (st) apus_store_stats(st, &r->store);
    else memset(&r->store, 0, sizeof r->store);
    if (pi) {
        apus_pilot_stats(pi, &r->pilot);
        apus_pilot_destroy(pi);
    } else {
        memset(&r->pilot, 0, sizeof r->pilot);
    }
    if (st) apus_store_close(st);
    r->tiered = tiered;
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
    printf("  [%s] %-46s tokdiff=%d/%d logit-bitdiff=%zu/%zu\n",
           ok ? "ok  " : "FAIL", name, tok_diff, N_GEN, bit,
           (size_t)N_GEN * ref->V);
    return ok;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("test_invariance: M6b pilot quality invariance (greedy, bitwise)\n");
    Run eager, off_full, on_full, on_q25, on_min, on_guard, on_pinned;
    if (run_greedy(&eager, -1, 0, 0, 0, NULL)) return 1;
    if (run_greedy(&off_full, 64, 0, 1ull << 40, 0, "")) return 1;
    if (run_greedy(&on_full, 64, 0, 1ull << 40, 1, "")) return 1;
    if (run_greedy(&on_q25, 16, 0, 1ull << 40, 1, "")) return 1;
    if (run_greedy(&on_min, 2, 0, 1ull << 40, 1, "")) return 1;
    if (run_greedy(&on_guard, 2, 0, 1, 1, "")) return 1;

    printf("  pilot-on traffic: full %llu preads (%llu hint) | 25%% %llu | "
           "min %llu | guard %llu (+%llu rss drops); recall %llu/%llu\n",
           (unsigned long long)on_full.store.preads,
           (unsigned long long)on_full.store.hint_loads,
           (unsigned long long)on_q25.store.preads,
           (unsigned long long)on_min.store.preads,
           (unsigned long long)on_guard.store.preads,
           (unsigned long long)on_guard.store.rss_drops,
           (unsigned long long)on_full.pilot.actual_hits,
           (unsigned long long)on_full.pilot.actual_experts);

    cmp_runs("store only (64 slots) vs eager", &eager, &off_full);
    cmp_runs("pilot ON, full cache (64) vs eager", &eager, &on_full);
    cmp_runs("pilot ON, 25% cache (16) vs eager", &eager, &on_q25);
    cmp_runs("pilot ON, minimal cache (2) vs eager", &eager, &on_min);
    cmp_runs("pilot ON, rss-guard pressure vs eager", &eager, &on_guard);
    cmp_runs("pilot ON vs pilot OFF (same 64-slot store)", &off_full,
             &on_full);

    /* pins + usage history with the pilot on */
    {
        (void)apus_sys_mkdir_p("tests/m6b/tmp");   /* M15 */
        FILE *uf = fopen("tests/m6b/tmp/usage_invariance.txt", "w");
        for (int l = 0; l < 6; l++)
            fprintf(uf, "%d %d %d\n", l, (l * 7) % 64, 900 - l);
        fclose(uf);
        if (run_greedy(&on_pinned, 4, 2, 1ull << 40, 1,
                       "tests/m6b/tmp/usage_invariance.txt"))
            return 1;
        cmp_runs("pilot ON, pinned (4 slots + 2 pins) vs eager", &eager,
                 &on_pinned);
        free(on_pinned.logits);
    }

    printf("  greedy tokens: ");
    for (int t = 0; t < N_GEN; t++) printf("%d ", eager.tokens[t]);
    printf("\n");

    free(eager.logits);
    free(off_full.logits);
    free(on_full.logits);
    free(on_q25.logits);
    free(on_min.logits);
    free(on_guard.logits);
    printf("test_invariance: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
