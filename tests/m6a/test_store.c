/*
 * tests/m6a/test_store.c — M6a expert-store unit tests (c/cache.h on the
 * tests/m6a synthetic container: 6 layers x 64 experts, 52,224 B coalesced
 * slabs).
 *
 *   1. slab derivation: addressing (layer,eid) -> one coalesced slab;
 *      one pread per expert (instrumented read counter); view dims.
 *   2. hit/miss accounting.
 *   3. working-set + end-of-block promotion: mid-block overflow stays
 *      usable, promotion swaps with the coldest slots.
 *   4. LRU recency: a hit protects a slot from eviction.
 *   5. hot pins seeded from a usage-history file; never evicted.
 *   6. pin persistence across a simulated restart (save -> reseed).
 *   7. LFRU REPIN with 25%+4 hysteresis.
 *   8. RSS guard frees only LRU payloads (pins untouched, identity kept).
 *   9. generation-tag straggler safety (synthetic race via pre-claim hook).
 *  10. miss-overlap correctness: concurrent loads byte-identical to serial.
 *  11. speculative-hint eviction guard: a hint-only load never evicts a
 *      warm demand-loaded expert (M6b pilot surface).
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
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cache.h"

#define FIX "tests/m6a/fixtures"
#define TMP "tests/m6a/tmp"
#define NLAYERS 6
#define NEXP 64
#define SLAB 52224

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

#define RSS_HUGE (1ull << 40)

static ApusStore *open_store(int slots, int pins, const char *usage,
                             uint64_t rss_budget, int io_threads) {
    ApusStoreCfg c = {0};
    c.n_layers = NLAYERS;
    c.n_experts = NEXP;
    c.slots_per_layer = slots;
    c.pins_per_layer = pins;
    c.rss_budget_bytes = rss_budget;
    c.io_threads = io_threads;
    c.usage_path = usage;
    char err[256];
    ApusStore *st = apus_store_open(FIX, &c, err, sizeof err);
    if (!st) fprintf(stderr, "store open: %s\n", err);
    return st;
}

static int has_eid(const int32_t *a, int n, int32_t e) {
    for (int i = 0; i < n; i++) if (a[i] == e) return 1;
    return 0;
}

/* --- 1/2. slab derivation, one pread per expert, hit accounting -----------*/

static void test_slab_and_reads(void) {
    ApusStore *st = open_store(64, 0, "", RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    CHECK(apus_store_slab_bytes(st) == SLAB, "slab bytes %zu != %d",
          apus_store_slab_bytes(st), SLAB);

    ApusFp4W w1, w2, w3;
    for (int e = 0; e < 10; e++)
        CHECK(apus_store_resolve(st, 0, e, &w1, &w2, &w3) == 0,
              "resolve L0 e%d", e);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.preads == 10, "one pread per expert: %llu != 10",
          (unsigned long long)ss.preads);
    CHECK(ss.bytes_read == 10 * (uint64_t)SLAB, "bytes %llu != %d",
          (unsigned long long)ss.bytes_read, 10 * SLAB);
    CHECK(ss.misses == 10, "misses %llu != 10", (unsigned long long)ss.misses);

    /* view dims from the shard headers: w1/w3 [128,256], w2 [256,128] */
    CHECK(apus_store_resolve(st, 0, 3, &w1, &w2, &w3) == 0, "resolve e3");
    CHECK(w1.O == 128 && w1.K == 256, "w1 dims %lldx%lld",
          (long long)w1.O, (long long)w1.K);
    CHECK(w2.O == 256 && w2.K == 128, "w2 dims %lldx%lld",
          (long long)w2.O, (long long)w2.K);
    CHECK(w3.O == 128 && w3.K == 256, "w3 dims %lldx%lld",
          (long long)w3.O, (long long)w3.K);
    /* zero-copy: the three weight views lie inside one slab window */
    CHECK(w1.scales == w1.packed + 16384, "w1 scale view not coalesced");
    CHECK(w2.packed == w1.scales + 1024, "w2 view not coalesced");

    apus_store_stats(st, &ss);
    uint64_t preads = ss.preads;
    CHECK(apus_store_resolve(st, 0, 3, &w1, &w2, &w3) == 0, "re-resolve e3");
    apus_store_stats(st, &ss);
    CHECK(ss.preads == preads && ss.hits >= 2, "hit served without a read");
    apus_store_close(st);
}

/* --- 3. working set + end-of-block promotion -------------------------------*/

static void test_promotion(void) {
    ApusStore *st = open_store(4, 0, "", RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    ApusFp4W w1, w2, w3;
    /* one block: 6 distinct experts into a 4-slot LRU */
    for (int e = 0; e < 6; e++)
        apus_store_resolve(st, 0, e, &w1, &w2, &w3);
    /* mid-block: all six still resolvable from the working set (no reads) */
    ApusStoreStats s0, s1;
    apus_store_stats(st, &s0);
    for (int e = 0; e < 6; e++)
        CHECK(apus_store_resolve(st, 0, e, &w1, &w2, &w3) == 0,
              "mid-block re-resolve e%d", e);
    apus_store_stats(st, &s1);
    CHECK(s1.preads == s0.preads, "mid-block re-resolve caused reads");

    apus_store_layer_end(st, 0);
    apus_store_stats(st, &s1);
    CHECK(s1.evictions == 2, "evictions %llu != 2",
          (unsigned long long)s1.evictions);
    /* LRU keeps the 4 most recent: e2..e5; e0/e1 dropped */
    int32_t lru[4];
    apus_store_debug_layer(st, 0, lru, 4, NULL, 0);
    CHECK(has_eid(lru, 4, 2) && has_eid(lru, 4, 3)
          && has_eid(lru, 4, 4) && has_eid(lru, 4, 5),
          "LRU holds e2..e5 after promotion");
    CHECK(!has_eid(lru, 4, 0) && !has_eid(lru, 4, 1), "e0/e1 evicted");
    /* e2 is a hit now, e0 a miss */
    apus_store_stats(st, &s0);
    apus_store_resolve(st, 0, 2, &w1, &w2, &w3);
    apus_store_stats(st, &s1);
    CHECK(s1.preads == s0.preads, "e2 hit after promotion");
    apus_store_resolve(st, 0, 0, &w1, &w2, &w3);
    apus_store_stats(st, &s1);
    CHECK(s1.preads == s0.preads + 1, "e0 miss after eviction");
    apus_store_close(st);
}

/* --- 4. LRU recency -----------------------------------------------------------*/

static void test_recency(void) {
    ApusStore *st = open_store(4, 0, "", RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    ApusFp4W w1, w2, w3;
    for (int e = 0; e < 4; e++) apus_store_resolve(st, 0, e, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    /* touch e0 so it is no longer the coldest */
    apus_store_resolve(st, 0, 0, &w1, &w2, &w3);
    /* next block: two new experts evict e1/e2 (coldest), not e0 */
    apus_store_resolve(st, 0, 10, &w1, &w2, &w3);
    apus_store_resolve(st, 0, 11, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    int32_t lru[4];
    apus_store_debug_layer(st, 0, lru, 4, NULL, 0);
    CHECK(has_eid(lru, 4, 0), "recently-used e0 survived");
    CHECK(has_eid(lru, 4, 10) && has_eid(lru, 4, 11), "new experts promoted");
    CHECK(has_eid(lru, 4, 3), "e3 survived");
    CHECK(!has_eid(lru, 4, 1) && !has_eid(lru, 4, 2), "coldest e1/e2 evicted");
    apus_store_close(st);
}

/* --- 5/6. pins: seed from usage file + persistence across restart ------------*/

static void write_usage(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    fputs(text, f);
    fclose(f);
}

static void test_pins(void) {
    char path[256];
    snprintf(path, sizeof path, TMP "/usage_pins.txt");
    write_usage(path, "0 5 100\n0 9 50\n1 7 42\n");
    ApusStore *st = open_store(4, 2, path, RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    int32_t pins[2];
    apus_store_debug_layer(st, 0, NULL, 0, pins, 2);
    CHECK(has_eid(pins, 2, 5) && has_eid(pins, 2, 9),
          "pins seeded from usage history (got %d,%d)", pins[0], pins[1]);
    int32_t pins1[2];
    apus_store_debug_layer(st, 1, NULL, 0, pins1, 2);
    CHECK(has_eid(pins1, 2, 7), "layer 1 pin seeded");

    /* pin loads on first use, then hits */
    ApusFp4W w1, w2, w3;
    apus_store_resolve(st, 0, 5, &w1, &w2, &w3);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.pin_loads == 1, "pin first-touch load (%llu)",
          (unsigned long long)ss.pin_loads);
    uint64_t preads = ss.preads;
    apus_store_resolve(st, 0, 5, &w1, &w2, &w3);
    apus_store_stats(st, &ss);
    CHECK(ss.preads == preads, "pin hit without a read");

    /* RSS pressure: pins never dropped (guard with a 1-byte budget) */
    ApusStore *st2 = open_store(4, 2, path, 1, 4);
    CHECK(st2 != NULL, "open (rss budget 1)");
    if (st2) {
        apus_store_resolve(st2, 0, 5, &w1, &w2, &w3);   /* pin */
        apus_store_resolve(st2, 0, 20, &w1, &w2, &w3);  /* LRU */
        apus_store_layer_end(st2, 0);                    /* guard fires */
        apus_store_stats(st2, &ss);
        CHECK(ss.rss_drops >= 1, "rss guard dropped LRU payloads");
        uint64_t p0 = ss.preads;
        apus_store_resolve(st2, 0, 5, &w1, &w2, &w3);
        apus_store_stats(st2, &ss);
        CHECK(ss.preads == p0, "pin survived the rss guard");
        size_t res = apus_store_resident_bytes(st2);
        CHECK(res <= 2 * (size_t)SLAB + 0, "resident %zu > pin set", res);
        /* dropped expert keeps identity but reloads on demand */
        apus_store_resolve(st2, 0, 20, &w1, &w2, &w3);
        apus_store_stats(st2, &ss);
        CHECK(ss.preads == p0 + 1, "dropped expert reloaded on demand");
        apus_store_close(st2);
    }
    apus_store_close(st);
}

static void test_pin_persistence(void) {
    char path[256];
    snprintf(path, sizeof path, TMP "/usage_persist.txt");
    remove(path);
    ApusFp4W w1, w2, w3;
    /* session A: e3 x10, e7 x5, e1 x1 */
    ApusStore *a = open_store(64, 0, path, RSS_HUGE, 4);
    CHECK(a != NULL, "open A");
    if (!a) return;
    for (int i = 0; i < 10; i++) apus_store_resolve(a, 0, 3, &w1, &w2, &w3);
    for (int i = 0; i < 5; i++) apus_store_resolve(a, 0, 7, &w1, &w2, &w3);
    apus_store_resolve(a, 0, 1, &w1, &w2, &w3);
    apus_store_layer_end(a, 0);
    CHECK(apus_store_save_usage(a) == 0, "save usage");
    apus_store_close(a);
    /* session B ("restart"): pins reseed from the file */
    ApusStore *b = open_store(4, 2, path, RSS_HUGE, 4);
    CHECK(b != NULL, "open B");
    if (b) {
        int32_t pins[2];
        apus_store_debug_layer(b, 0, NULL, 0, pins, 2);
        CHECK(has_eid(pins, 2, 3) && has_eid(pins, 2, 7),
              "pins persisted across restart (got %d,%d)", pins[0], pins[1]);
        apus_store_close(b);
    }
}

/* --- 7. LFRU REPIN hysteresis -------------------------------------------------*/

static void test_repin(void) {
    char path[256];
    snprintf(path, sizeof path, TMP "/usage_repin.txt");
    write_usage(path, "0 10 10\n");         /* pin e10 with freq 10 */
    ApusStore *st = open_store(8, 1, path, RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    ApusFp4W w1, w2, w3;
    /* e0 used 20 times: 20 >= 10 + 10/4 + 4 = 16 -> swap on REPIN */
    for (int i = 0; i < 20; i++) apus_store_resolve(st, 0, 0, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    apus_store_repin(st);
    int32_t pin;
    apus_store_debug_layer(st, 0, NULL, 0, &pin, 1);
    CHECK(pin == 0, "repin: hot e0 replaced cold pin (got %d)", pin);
    /* e1 used 25 times: 25 < 20 + 20/4 + 4 = 29 -> hysteresis holds */
    for (int i = 0; i < 25; i++) apus_store_resolve(st, 0, 1, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    apus_store_repin(st);
    apus_store_debug_layer(st, 0, NULL, 0, &pin, 1);
    CHECK(pin == 0, "repin hysteresis: 25 < 29 keeps e0 (got %d)", pin);
    /* e1 to 29 uses -> swap */
    for (int i = 0; i < 4; i++) apus_store_resolve(st, 0, 1, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    apus_store_repin(st);
    apus_store_debug_layer(st, 0, NULL, 0, &pin, 1);
    CHECK(pin == 1, "repin at 29 >= 29 swaps to e1 (got %d)", pin);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.repin_swaps == 2, "repin_swaps %llu != 2",
          (unsigned long long)ss.repin_swaps);
    apus_store_close(st);
}

/* --- 9. generation-tag straggler (synthetic race) -------------------------------*/

static int g_stale_fired;

static void pre_claim_stale(ApusStore *st, int layer, int32_t eid,
                            uint64_t gen) {
    (void)gen;
    if (!g_stale_fired) {
        g_stale_fired = 1;
        /* simulate: the slot got recycled by a newer generation while this
         * job was in flight */
        CHECK(apus_store_debug_stale_gen(st, layer, eid) == 0,
              "stale_gen on loading slot");
    }
}

static void test_straggler(void) {
    ApusStore *st = open_store(4, 0, "", RSS_HUGE, 1);
    CHECK(st != NULL, "open");
    if (!st) return;
    g_stale_fired = 0;
    apus_store_debug_set_pre_claim(st, pre_claim_stale);
    ApusFp4W w1, w2, w3;
    CHECK(apus_store_resolve(st, 0, 33, &w1, &w2, &w3) == 0,
          "resolve recovers from a stale claim");
    CHECK(g_stale_fired, "synthetic race actually fired");
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.preads == 2, "stale payload discarded, reloaded (%llu preads)",
          (unsigned long long)ss.preads);
    /* content must be the real slab, not the straggler's: byte-compare
     * against a clean store */
    ApusStore *ref = open_store(4, 0, "", RSS_HUGE, -1);
    ApusFp4W r1, r2, r3;
    apus_store_resolve(ref, 0, 33, &r1, &r2, &r3);
    CHECK(memcmp(w1.packed, r1.packed, 16384) == 0
          && memcmp(w2.packed, r2.packed, 16384) == 0
          && memcmp(w3.packed, r3.packed, 16384) == 0,
          "slab content correct after straggler");
    apus_store_close(ref);
    apus_store_close(st);
}

/* --- 10. concurrent vs serial byte-identical ------------------------------------*/

static void test_concurrent_identical(void) {
    static const int es[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    enum { N = 12 };
    ApusStore *conc = open_store(64, 0, "", RSS_HUGE, 4);
    ApusStore *ser = open_store(64, 0, "", RSS_HUGE, -1);
    CHECK(conc && ser, "open both");
    if (!conc || !ser) return;
    for (int i = 0; i < N; i++) apus_store_hint(conc, 2, es[i]);
    for (int i = 0; i < N; i++) {
        ApusFp4W a1, a2, a3, b1, b2, b3;
        apus_store_resolve(conc, 2, es[i], &a1, &a2, &a3);
        apus_store_resolve(ser, 2, es[i], &b1, &b2, &b3);
        CHECK(memcmp(a1.packed, b1.packed, 16384) == 0
              && memcmp(a1.scales, b1.scales, 1024) == 0
              && memcmp(a2.packed, b2.packed, 16384) == 0
              && memcmp(a2.scales, b2.scales, 1024) == 0
              && memcmp(a3.packed, b3.packed, 16384) == 0
              && memcmp(a3.scales, b3.scales, 1024) == 0,
              "concurrent slab e%d byte-identical to serial", es[i]);
    }
    ApusStoreStats ss;
    apus_store_stats(conc, &ss);
    CHECK(ss.preads == N, "concurrent preads %llu != %d",
          (unsigned long long)ss.preads, N);
    apus_store_close(conc);
    apus_store_close(ser);
}

/* --- 11. speculative-hint eviction guard --------------------------------------
 * A hint that is never consumed by the forward (a pure speculation, the M6b
 * pilot case) has no use clock: it may take a FREE slot but must never
 * evict a warm demand-loaded expert. Once consumed by a resolve it becomes
 * an ordinary demand load. */

static void test_hint_guard(void) {
    ApusFp4W w1, w2, w3;
    /* (a) full warm LRU: an unconsumed hint never evicts */
    ApusStore *st = open_store(4, 0, "", RSS_HUGE, 4);
    CHECK(st != NULL, "open");
    if (!st) return;
    for (int e = 0; e < 4; e++) apus_store_resolve(st, 0, e, &w1, &w2, &w3);
    apus_store_layer_end(st, 0);
    apus_store_hint(st, 0, 40);                 /* speculation, never used */
    apus_store_layer_end(st, 0);
    apus_store_layer_end(st, 0);                /* (covers in-flight timing) */
    int32_t lru[4];
    apus_store_debug_layer(st, 0, lru, 4, NULL, 0);
    CHECK(has_eid(lru, 4, 0) && has_eid(lru, 4, 1)
          && has_eid(lru, 4, 2) && has_eid(lru, 4, 3),
          "unconsumed hint never evicts warm experts");
    apus_store_close(st);

    /* (b) free slot: the hint takes it and a later resolve is a hit */
    ApusStore *st2 = open_store(5, 0, "", RSS_HUGE, 4);
    CHECK(st2 != NULL, "open 2");
    if (!st2) return;
    for (int e = 0; e < 4; e++) apus_store_resolve(st2, 0, e, &w1, &w2, &w3);
    apus_store_layer_end(st2, 0);
    apus_store_hint(st2, 0, 40);
    apus_store_resolve(st2, 0, 40, &w1, &w2, &w3);   /* waits for the load */
    ApusStoreStats s0, s1;
    apus_store_stats(st2, &s0);
    apus_store_resolve(st2, 0, 40, &w1, &w2, &w3);   /* must be a pure hit */
    apus_store_stats(st2, &s1);
    CHECK(s1.preads == s0.preads, "hinted expert resident (free-slot case)");
    apus_store_layer_end(st2, 0);
    int32_t lru2[5];
    apus_store_debug_layer(st2, 0, lru2, 5, NULL, 0);
    CHECK(has_eid(lru2, 5, 40), "hinted expert promoted into the free slot");
    apus_store_close(st2);
}

int main(void) {
    printf("test_store: M6a expert-store unit tests\n");
    (void)apus_sys_mkdir_p(TMP);   /* M15: portable (no cmd.exe) */
    test_slab_and_reads();
    test_promotion();
    test_recency();
    test_pins();
    test_pin_persistence();
    test_repin();
    test_straggler();
    test_concurrent_identical();
    test_hint_guard();
    printf("test_store: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
