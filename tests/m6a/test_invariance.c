/*
 * tests/m6a/test_invariance.c — THE M6a hard test: tiering must never change
 * numerics. Identical prompt, greedy decode, full forward, five storage
 * configurations:
 *
 *   (eager)  legacy slurp path (apus_model_load, all expert views resident)
 *   (full)   expert store, 64 slots/layer — every expert fits
 *   (25%)    expert store, 16 slots/layer
 *   (min)    expert store, 2 slots/layer (heavy churn)
 *   (guard)  2 slots/layer + RSS budget 1 byte (guard drops payloads
 *            constantly — maximal pressure)
 *
 * The token streams must be IDENTICAL and the logits BITWISE identical
 * across all five. There is no tolerance budget: the store serves the same
 * slab bytes to the same kernels in the same accumulation order, so any
 * difference is a bug in the store, not noise.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
#include "cache.h"

#define FIX "tests/m6a/fixtures"

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

#define N_PROMPT 8
#define N_GEN 24

static const int64_t PROMPT[N_PROMPT] = {3, 41, 7, 200, 511, 0, 128, 65};

typedef struct {
    int tokens[N_GEN];
    float *logits;      /* [N_GEN, V] per-step pre-sample logits */
    int V;
    ApusStoreStats store;
    int tiered;
} Run;

/* One greedy run. slots < 0 -> legacy eager path (no store). */
static int run_greedy(Run *r, int slots, int pins, uint64_t rss_budget,
                      int io_threads, const char *usage_path) {
    char err[256];
    ApusModel m;
    int tiered = slots >= 0;
    if (apus_model_load_ex(&m, FIX, tiered, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        return -1;
    }
    ApusStore *st = NULL;
    if (tiered) {
        ApusStoreCfg c = {0};
        c.n_layers = m.n_layers;
        c.n_experts = m.cfg.n_routed_experts;
        c.slots_per_layer = slots;
        c.pins_per_layer = pins;
        c.rss_budget_bytes = rss_budget;
        c.io_threads = io_threads;
        c.usage_path = usage_path;      /* "" = no history */
        st = apus_store_open(FIX, &c, err, sizeof err);
        if (!st) {
            fprintf(stderr, "store: %s\n", err);
            return -1;
        }
        for (int i = 0; i < m.n_layers; i++)
            apus_store_attach_moe(st, &m.layers[i].mw);
    }
    int V = m.cfg.vocab_size;
    r->V = V;
    r->logits = malloc((size_t)N_GEN * V * sizeof(float));
    ApusModelState stt;
    apus_model_state_init(&stt, &m);
    float *logits = malloc((size_t)V * sizeof(float));
    apus_model_forward(&m, &stt, PROMPT, N_PROMPT, logits, 0);
    for (int t = 0; t < N_GEN; t++) {
        int tok = apus_sample_argmax(logits, (size_t)V);
        r->tokens[t] = tok;
        memcpy(r->logits + (size_t)t * V, logits, (size_t)V * sizeof(float));
        if (t + 1 < N_GEN) {
            int64_t next = tok;
            apus_model_forward(&m, &stt, &next, 1, logits, 0);
        }
    }
    free(logits);
    apus_model_state_free(&stt, &m);
    if (st) {
        apus_store_stats(st, &r->store);
        apus_store_close(st);
    } else {
        memset(&r->store, 0, sizeof r->store);
    }
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
    printf("  [%s] %-52s tokdiff=%d/%d logit-bitdiff=%zu/%zu\n",
           ok ? "ok  " : "FAIL", name, tok_diff, N_GEN, bit,
           (size_t)N_GEN * ref->V);
    return ok;
}

int main(void) {
    printf("test_invariance: M6a tiering quality invariance (greedy, "
           "bitwise)\n");
    Run eager, full, q25, min2, guard;
    if (run_greedy(&eager, -1, 0, 0, 0, NULL)) return 1;
    if (run_greedy(&full, 64, 0, 1ull << 40, 4, "")) return 1;
    if (run_greedy(&q25, 16, 0, 1ull << 40, 4, "")) return 1;
    if (run_greedy(&min2, 2, 0, 1ull << 40, 4, "")) return 1;
    if (run_greedy(&guard, 2, 0, 1, 4, "")) return 1;

    printf("  store traffic: full %llu preads | 25%% %llu | min %llu | "
           "guard %llu (+%llu rss drops)\n",
           (unsigned long long)full.store.preads,
           (unsigned long long)q25.store.preads,
           (unsigned long long)min2.store.preads,
           (unsigned long long)guard.store.preads,
           (unsigned long long)guard.store.rss_drops);

    /* the eager legacy path is the reference */
    cmp_runs("full cache (64 slots/layer) vs eager", &eager, &full);
    cmp_runs("25% cache (16 slots/layer) vs eager", &eager, &q25);
    cmp_runs("minimal cache (2 slots/layer) vs eager", &eager, &min2);
    cmp_runs("rss-guard pressure (budget 1B) vs eager", &eager, &guard);

    /* pins enabled: pins change residency, never numerics. Seed the pin
     * store from a synthetic usage-history file. */
    {
        (void)apus_sys_mkdir_p("tests/m6a/tmp");   /* M15 */
        FILE *uf = fopen("tests/m6a/tmp/usage_invariance.txt", "w");
        for (int l = 0; l < 6; l++)
            fprintf(uf, "%d %d %d\n", l, (l * 7) % 64, 900 - l);
        fclose(uf);
        Run pinned;
        if (run_greedy(&pinned, 4, 2, 1ull << 40, 4,
                       "tests/m6a/tmp/usage_invariance.txt"))
            return 1;
        cmp_runs("pinned (4 slots + 2 pins/layer) vs eager", &eager,
                 &pinned);
        free(pinned.logits);
    }

    printf("  greedy tokens: ");
    for (int t = 0; t < N_GEN; t++) printf("%d ", eager.tokens[t]);
    printf("\n");

    free(eager.logits);
    free(full.logits);
    free(q25.logits);
    free(min2.logits);
    free(guard.logits);
    printf("test_invariance: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
