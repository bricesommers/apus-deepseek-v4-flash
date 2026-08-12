/*
 * c/cache.h — expert store (M6a): demand-loading of routed-expert slabs
 * from NVMe through a bounded RAM cache. colibri tiering design (ESlot slab
 * machinery, per-layer LRU with end-of-block promotion, LFRU pins with
 * hysteresis, RSS guard, generation-tagged miss overlap) adapted for the
 * DeepSeek-V4-Flash apus container. C11, libc + pthreads.
 *
 * Addressing: (layer, eid) -> slab record. The container stores each
 * expert's six tensors {w1,w1_scale,w2,w2_scale,w3,w3_scale} contiguously in
 * one shard (tools/convert.py SLAB_MEMBERS order), so a cache miss is ONE
 * pread of the whole slab into a single 4 KiB-aligned buffer, and the
 * ApusFp4W views are zero-copy pointers into it. Slab records are derived
 * from the shard headers at open (apus.index.json manifest support is not
 * required — headers suffice); open fails loudly if the coalescing
 * invariant is violated.
 *
 * Cache policy (docs/ARCHITECTURE.md §7):
 *   - Per-layer LRU slot arrays. Misses load into a small per-forward
 *     working set (never directly into the LRU), promoted at layer end by
 *     swapping with the coldest slots; hits bump an atomic clock.
 *   - Hot-pin store: per-layer pinned slots, never evicted, seeded from a
 *     persistent usage-history file (plain text "layer eid count",
 *     atomically rewritten — colibri's .coli_usage pattern). Between-turns
 *     REPIN pass with LFRU score (frequency primary, recency tiebreak) and
 *     25%+4 hysteresis.
 *   - RSS guard: when the measured process RSS (mach task_info) exceeds the
 *     budget, LRU payloads are freed in place (slots keep their identity,
 *     payloads are dropped) — pins are never touched, and the guard only
 *     runs at block boundaries (never while a forward holds slab pointers).
 *   - Miss overlap: a pthread I/O pool pulls miss jobs; each job is
 *     generation-tagged so a straggler completing after its slot was
 *     recycled cannot corrupt a newer generation (claim check at hand-over).
 *     The compute thread never does I/O in pool mode — it waits
 *     just-in-time on a per-slot condvar. F_NOCACHE streaming reads keep
 *     expert traffic out of the page cache.
 *   - M9c job priority: loads are demand-class (MoE batch-union storm via
 *     apus_store_hint_demand, resolve re-submits, and any LOADING slot a
 *     resolve blocks on) or speculative (apus_store_hint — the pilot
 *     surface). Workers pop the first demand-class job in the queue before
 *     FIFO speculative ones, so deep next-layer prefetch can never delay
 *     the current sublayer's experts. Just-in-time resolve waits are timed
 *     (stats.waits/wait_ns).
 *
 * Budgets (env, sane defaults for a 32 GB Mac; all overridable via
 * ApusStoreCfg): APUS_EXPERT_CACHE_MB (default 4096), APUS_PIN_MB
 * (512), APUS_RSS_GUARD_MB (26624 — emergency brake, normally inactive),
 * APUS_IO_THREADS (4). The usage-history file defaults to
 * <model_dir>/apus.usage. (M6c retune: the M6a-era 12288/2048 defaults
 * pushed the real-model footprint to ~30 GB — past physical RAM with the
 * OS — and the VM compressor then cost ~2.5 cores of kernel time in
 * anonymous-page reclaim/decompress churn. 4096/512 keeps the footprint
 * ~17 GB and RSS ≤ ~14 GB; miss counts rise slightly but all loads stay
 * pilot-covered (0 demand loads) and pread time is only ~2% of decode.)
 *
 * Threading contract: resolve/hint/layer_end are called from the compute
 * thread (c/moe.h hooks); all entry points are internally locked, so a
 * prefetch/pilot thread (M6b) may call apus_store_hint from elsewhere.
 * repin/save_usage/rss_guard must be called between forwards (or at layer
 * end from the compute thread) — never while slab views are in use.
 *
 * Usage: #define APUS_CACHE_IMPLEMENTATION in exactly one TU (also needs
 * the st/json/compat implementations).
 */
#ifndef APUS_CACHE_H
#define APUS_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "moe.h"      /* ApusFp4W, ApusMoeW (via st.h) */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    n_layers;          /* required: TOTAL expert-bearing layers,
                                 main + n_mtp (M8: the MTP block's experts
                                 are store layer n_main + mtp_idx) */
    int    n_experts;         /* required: routed experts per layer (E) */
    int    n_mtp;             /* M8: MTP blocks (their "mtp.K.ffn.experts.*"
                                 tensors map to store layers n_main+K);
                                 0 = no MTP (default) */
    size_t cache_bytes;       /* LRU budget; 0 = APUS_EXPERT_CACHE_MB env */
    size_t pin_bytes;         /* pin budget; 0 = APUS_PIN_MB env */
    int    slots_per_layer;   /* explicit LRU slots; 0 = derive from budget */
    int    pins_per_layer;    /* explicit pin slots; 0 = derive from budget */
    size_t rss_budget_bytes;  /* 0 = APUS_RSS_GUARD_MB env */
    int    io_threads;        /* 0 = env/default (4); <0 = synchronous mode */
    int    nocache;           /* >0 = F_NOCACHE reads, <0 = cached fds,
                                 0 = APUS_NOCACHE env (default 1) */
    const char *usage_path;   /* NULL = <model_dir>/apus.usage; "" = off */
    double usage_decay;       /* M6b heat decay: old usage-file counts are
                                 multiplied by this at save (0.5 = halve-on-
                                 save, pins track drift). 0 = APUS_USAGE_DECAY
                                 env (default 1.0 = cumulative, M6a behavior) */
} ApusStoreCfg;

typedef struct ApusStore ApusStore;

typedef struct {
    uint64_t hits;            /* resolves served without a read */
    uint64_t misses;          /* resolves that loaded a slab */
    uint64_t preads;          /* apus_st_lazy_pread calls (1 per slab load) */
    uint64_t bytes_read;
    uint64_t evictions;       /* LRU payloads replaced at promotion */
    uint64_t rss_drops;       /* LRU payloads freed by the RSS guard */
    uint64_t pin_loads;       /* first-touch pin slab loads */
    uint64_t repin_swaps;     /* LFRU REPIN pin<->LRU swaps */
    uint64_t hint_loads;      /* slab loads submitted via apus_store_hint
                                 (M6b: pilot/hash prefetch attribution) */
    uint64_t demand_loads;    /* slab loads submitted via apus_store_resolve
                                 (M6b: prefetch coverage = 1 - demand/all) */
    uint64_t waits;           /* M9c: resolves that blocked on an in-flight
                                 or re-submitted load (just-in-time stalls) */
    uint64_t wait_ns;         /* M9c: total ns the compute thread blocked in
                                 apus_store_wait_ready */
    uint64_t pread_ns;        /* M9c: total ns inside apus_st_lazy_pread
                                 (summed over I/O workers; divide by wall to
                                 get the pool's read duty cycle) */
} ApusStoreStats;

/* Open the store over a model dir (config-less: uses
 * model.safetensors.index.json in dir or dir/weights, like c/model.h).
 * Derives slab records from shard headers; errors if the per-expert
 * 6-tensor coalescing invariant is violated. Seeds pins from the
 * usage-history file if present. Returns NULL on error. */
ApusStore *apus_store_open(const char *model_dir, const ApusStoreCfg *cfg,
                           char *err, size_t errcap);
void       apus_store_close(ApusStore *st);

/* Resolve expert (layer, eid): fills w1/w2/w3 with zero-copy views into a
 * cache slot, waiting just-in-time if a load is in flight. Returns 0. */
int  apus_store_resolve(ApusStore *st, int layer, int eid,
                        ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3);

/* Non-blocking prefetch hint: submits the miss job without waiting. Safe
 * from any thread (the M6b pilot surface). Deduplicates against slots,
 * working set, and in-flight jobs. This is the SPECULATIVE class: it never
 * re-prioritizes an already-queued load (M9c). */
void apus_store_hint(ApusStore *st, int layer, int eid);

/* Demand-class variant (M9c): identical submission semantics to
 * apus_store_hint, but the load is marked hot — the I/O pool serves hot
 * jobs before queued speculative ones. Used for the MoE batch-union storm,
 * whose experts are resolved within the same sublayer. */
void apus_store_hint_demand(ApusStore *st, int layer, int eid);

/* End-of-block: promote this layer's working set into the LRU (swap with
 * the coldest slots), advance the load generation, run the RSS guard.
 * Called by the moe hook after each MoE sublayer. */
void apus_store_layer_end(ApusStore *st, int layer);

/* Between-turns REPIN: LFRU score (frequency primary, recency tiebreak)
 * with 25%+4 hysteresis; swaps cold pins with hotter unpinned experts. */
void apus_store_repin(ApusStore *st);

/* Persist usage history ("layer eid count" per line), atomically
 * (tmp + fsync + rename). Returns 0 on success. */
int  apus_store_save_usage(ApusStore *st);

/* RSS guard: if RSS > budget, free coldest LRU payloads in place (slots
 * keep eid/freq identity; pins and in-flight loads untouched). Safe to call
 * any time no forward holds slab views; layer_end calls it automatically. */
void apus_store_rss_guard(ApusStore *st);

void     apus_store_stats(const ApusStore *st, ApusStoreStats *out);
size_t   apus_store_slab_bytes(const ApusStore *st);
size_t   apus_store_resident_bytes(ApusStore *st);   /* live payload bytes */

/* Wire a layer's MoE to this store (sets the ApusMoeW resolve hooks; uses
 * mw->layer_id set by c/layer.h). Call once per layer after model load. */
void apus_store_attach_moe(ApusStore *st, ApusMoeW *mw);

/* --- introspection / test hooks -------------------------------------------*/

/* Snapshot of one layer's cache: LRU slot eids (-1 = empty) and pin eids.
 * Arrays may be NULL. Returns 0. */
int  apus_store_debug_layer(ApusStore *st, int layer,
                            int32_t *lru_eids, int n_lru,
                            int32_t *pin_eids, int n_pins);

/* Test hook: invoked by an I/O worker after its pread, before claiming the
 * slot. Lets tests deterministically simulate a generation-straggler race. */
void apus_store_debug_set_pre_claim(ApusStore *st,
    void (*fn)(ApusStore *st, int layer, int32_t eid, uint64_t gen));

/* Test helper: force the loading slot for (layer, eid) to a stale
 * generation (simulates the slot being recycled under an in-flight job). */
int  apus_store_debug_stale_gen(ApusStore *st, int layer, int32_t eid);

/* Test/introspection helper (M6b): 1 iff (layer, eid) is already present —
 * pin/LRU resident or in the working set (any state, incl. LOADING) — i.e.
 * a demand resolve right now would NOT submit a fresh load. Used to assert
 * prefetch coverage ("hit") at the moment the MoE first asks. */
int  apus_store_debug_present(ApusStore *st, int layer, int eid);

/* Test helper (M9c): 1 iff (layer, eid) is READY with a live payload
 * (stricter than debug_present, which also counts in-flight loads). Used
 * to assert I/O queue ordering ("which loads have COMPLETED"). */
int  apus_store_debug_ready(ApusStore *st, int layer, int eid);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_CACHE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <unistd.h>

#include "compat.h"
#include "json.h"

/* --- slab records ----------------------------------------------------------*/

typedef struct {
    ApusStLazy *lz;
    uint64_t    off;        /* absolute file offset of the slab */
    uint64_t    len;        /* slab bytes */
    uint32_t    rel[6];     /* per-member offset within slab, canonical
                               order w1.w, w1.s, w2.w, w2.s, w3.w, w3.s */
    int64_t     O[3], K[3]; /* w1/w2/w3 logical dims */
} ApusSlabRec;

/* --- slots -----------------------------------------------------------------*/

enum { APUS_SLOT_EMPTY = 0, APUS_SLOT_LOADING = 1, APUS_SLOT_READY = 2 };

typedef struct {
    int32_t  eid;           /* -1 = unassigned */
    uint8_t *buf;           /* slab payload, NULL when dropped/unloaded */
    uint64_t last;          /* LRU clock of last use */
    uint64_t freq;          /* uses since load (LFRU frequency) */
    int      state;
    uint64_t gen;           /* generation tag of the in-flight load */
    uint8_t  hot;           /* M9c: demand-class — the I/O pool serves this
                               slot's queued load before speculative ones */
} ApusSlot;

typedef struct {
    ApusSlot  *slots;       /* LRU slots [n_slots] */
    int        n_slots;
    ApusSlot  *pins;        /* pinned slots [n_pins] */
    int        n_pins;
    ApusSlot **ws;          /* per-forward working set (heap slots) */
    int        ws_n, ws_cap;
} ApusLayerCache;

typedef struct {
    int          layer;
    ApusSlot    *slot;
    ApusSlabRec *rec;
    int          is_pin;
    uint64_t     gen;
} ApusJob;

struct ApusStore {
    int            n_layers, E;
    int            n_main;    /* main-model layers (M8: mtp.K -> n_main+K) */
    size_t         slab_bytes;
    ApusSlabRec   *recs;        /* [n_layers * E] */
    ApusLayerCache *lc;         /* [n_layers] */
    /* shards (lazy readers), deduped by file name */
    struct { char *name; ApusStLazy *lz; } *shards;
    int            shards_n, shards_cap;
    /* sync */
    pthread_mutex_t mu;
    pthread_cond_t  cv;         /* slot completions */
    /* I/O pool */
    pthread_t      *threads;
    int             n_threads;  /* 0 = synchronous mode */
    ApusJob        *jobs;
    int             jq_head, jq_n, jq_cap;
    pthread_cond_t  jq_cv;
    int             stopping;
    int             boost;      /* M9c: hot-first pops (demand class ahead of
                                   speculative); 0 via APUS_STORE_BOOST=0 —
                                   ablation/debug only */
    /* M6c payload-buffer recycling: evicted/dropped slab buffers are pushed
     * on a free list (capacity slab_bytes each) and popped by the next load
     * instead of free()+posix_memalign — a 13 MB mmap/munmap plus zero-fill
     * soft faults per expert load was a major kernel-time cost on the real
     * model. RSS-guard drops still really free() (their purpose is memory
     * relief). All under mu. */
    uint8_t       **buf_free;
    int             buf_free_n, buf_free_cap;
    /* policy state (all under mu) */
    uint64_t        clock;
    uint64_t        gen;
    size_t          rss_budget;
    ApusStoreStats  stats;
    char            usage_path[1200];
    int             usage_enabled;
    double          usage_decay;    /* applied to old file counts at save */
    void          (*test_pre_claim)(ApusStore *, int, int32_t, uint64_t);
};

/* --- small utilities --------------------------------------------------------*/

static uint64_t apus_clock_tick(ApusStore *st) { return ++st->clock; }

static void *apus_slab_alloc(size_t n) {
    /* M15: apus_aligned_alloc pairs with apus_aligned_free everywhere a
     * slab buffer is released (Windows _aligned_malloc storage must not
     * pass through free()). */
    return apus_aligned_alloc(4096, n);
}

#define APUS_BUF_FREE_MAX 64   /* <= 64 slabs retained (~855 MB at 13.4 MB) */

/* Pop a recycled slab buffer (exact slab_bytes class) or allocate fresh.
 * Called from I/O workers — takes mu briefly. */
static uint8_t *apus_store_buf_get(ApusStore *st, size_t n) {
    if (n == st->slab_bytes) {
        pthread_mutex_lock(&st->mu);
        if (st->buf_free_n > 0) {
            uint8_t *b = st->buf_free[--st->buf_free_n];
            pthread_mutex_unlock(&st->mu);
            return b;
        }
        pthread_mutex_unlock(&st->mu);
    }
    return apus_slab_alloc(n);
}

/* Recycle a slab buffer onto the free list (mu held); really free it when
 * the list is full. */
static void apus_store_buf_put(ApusStore *st, uint8_t *b) {
    if (!b) return;
    if (st->buf_free_n < st->buf_free_cap) {
        st->buf_free[st->buf_free_n++] = b;
    } else {
        apus_aligned_free(b);
    }
}

static ApusSlabRec *apus_store_rec(ApusStore *st, int layer, int eid) {
    return &st->recs[(size_t)layer * st->E + eid];
}

static void apus_slot_views(const ApusSlabRec *rec, const uint8_t *buf,
                            ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3) {
    w1->packed = buf + rec->rel[0];
    w1->scales = buf + rec->rel[1];
    w1->O = rec->O[0];
    w1->K = rec->K[0];
    w2->packed = buf + rec->rel[2];
    w2->scales = buf + rec->rel[3];
    w2->O = rec->O[1];
    w2->K = rec->K[1];
    w3->packed = buf + rec->rel[4];
    w3->scales = buf + rec->rel[5];
    w3->O = rec->O[2];
    w3->K = rec->K[2];
}

/* --- open: weight_map scan + slab derivation --------------------------------*/

typedef struct {
    int layer, eid, w;      /* w: 1..3; layer: store index (mtp.K -> n_main+K) */
    int is_scale;
    int is_mtp;             /* M8: name was "mtp.K.ffn.experts.*" */
    char shard[256];
} ApusExpertTensorRef;

static int apus_ref_cmp(const void *a, const void *b) {
    const ApusExpertTensorRef *x = a, *y = b;
    if (x->layer != y->layer) return x->layer - y->layer;
    if (x->eid != y->eid) return x->eid - y->eid;
    if (x->w != y->w) return x->w - y->w;
    return x->is_scale - y->is_scale;   /* weight before scale */
}

static int apus_parse_expert_name(const char *name, int *layer, int *eid,
                                  int *w, int *is_scale, int *is_mtp) {
    /* layers.{L}.ffn.experts.{E}.w{1,2,3}.{weight|scale}
     * M8: mtp.{K}.ffn.experts.{E}.w{1,2,3}.{weight|scale} (K returned in
     * *layer; *is_mtp distinguishes — the caller adds its n_main base). */
    int L, E, W;
    char kind[16], chk[200];
    *is_mtp = 0;
    if (sscanf(name, "layers.%d.ffn.experts.%d.w%d.%15s", &L, &E, &W, kind) == 4) {
        if (W < 1 || W > 3) return -1;
        if (strcmp(kind, "weight") && strcmp(kind, "scale")) return -1;
        /* reject suffix garbage like "w1.weightx" */
        snprintf(chk, sizeof chk, "layers.%d.ffn.experts.%d.w%d.%s", L, E, W, kind);
        if (strcmp(chk, name)) return -1;
    } else if (sscanf(name, "mtp.%d.ffn.experts.%d.w%d.%15s", &L, &E, &W, kind) == 4) {
        if (W < 1 || W > 3) return -1;
        if (strcmp(kind, "weight") && strcmp(kind, "scale")) return -1;
        snprintf(chk, sizeof chk, "mtp.%d.ffn.experts.%d.w%d.%s", L, E, W, kind);
        if (strcmp(chk, name)) return -1;
        *is_mtp = 1;
    } else {
        return -1;
    }
    *layer = L; *eid = E; *w = W;
    *is_scale = !strcmp(kind, "scale");
    return 0;
}

static ApusStLazy *apus_store_shard(ApusStore *st, const char *dir,
                                    const char *name, int nocache,
                                    char *err, size_t errcap) {
    for (int i = 0; i < st->shards_n; i++)
        if (!strcmp(st->shards[i].name, name)) return st->shards[i].lz;
    if (st->shards_n == st->shards_cap) {
        st->shards_cap = st->shards_cap ? 2 * st->shards_cap : 8;
        st->shards = realloc(st->shards,
                             (size_t)st->shards_cap * sizeof *st->shards);
    }
    char path[1400];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    ApusStLazy *lz = apus_st_lazy_open(path, nocache, err, errcap);
    if (!lz) return NULL;
    st->shards[st->shards_n].name = strdup(name);
    st->shards[st->shards_n].lz = lz;
    st->shards_n++;
    return lz;
}

/* Derive one slab record: the six member tensors must live in one shard and
 * tile a contiguous byte range. */
static int apus_store_derive_slab(ApusStore *st, const char *dir,
                                  ApusExpertTensorRef *refs /* [6] */,
                                  int nocache, char *err, size_t errcap) {
    int layer = refs[0].layer, eid = refs[0].eid;
    ApusSlabRec *rec = apus_store_rec(st, layer, eid);
    memset(rec, 0, sizeof *rec);
    /* canonical member order: w1.w, w1.s, w2.w, w2.s, w3.w, w3.s */
    const ApusStLazyTensor *mem[6];
    ApusStLazy *lz = NULL;
    for (int i = 0; i < 6; i++) {
        lz = apus_store_shard(st, dir, refs[i].shard, nocache, err, errcap);
        if (!lz) return -1;
        if (i && lz != apus_store_shard(st, dir, refs[0].shard, nocache,
                                        err, errcap)) {
            snprintf(err, errcap,
                     "store: expert %d.%d split across shards", layer, eid);
            return -1;
        }
        char tname[200];
        if (refs[i].is_mtp)
            snprintf(tname, sizeof tname, "mtp.%d.ffn.experts.%d.w%d.%s",
                     layer - st->n_main, eid, refs[i].w,
                     refs[i].is_scale ? "scale" : "weight");
        else
            snprintf(tname, sizeof tname, "layers.%d.ffn.experts.%d.w%d.%s",
                     layer, eid, refs[i].w,
                     refs[i].is_scale ? "scale" : "weight");
        mem[i] = apus_st_lazy_find(lz, tname);
        if (!mem[i]) {
            snprintf(err, errcap, "store: missing tensor %s", tname);
            return -1;
        }
    }
    /* contiguity: sorted member intervals must tile exactly */
    int ord[6] = {0, 1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 6; j++)
            if (mem[ord[j]]->file_off < mem[ord[i]]->file_off) {
                int t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    uint64_t base = mem[ord[0]]->file_off, cur = base;
    for (int i = 0; i < 6; i++) {
        if (mem[ord[i]]->file_off != cur) {
            snprintf(err, errcap,
                     "store: expert %d.%d slab not contiguous "
                     "(gap at member %d)", layer, eid, ord[i]);
            return -1;
        }
        cur += mem[ord[i]]->nbytes;
    }
    rec->lz = lz;
    rec->off = base;
    rec->len = cur - base;
    for (int i = 0; i < 6; i++)
        rec->rel[i] = (uint32_t)(mem[i]->file_off - base);
    for (int t = 0; t < 3; t++) {
        const ApusStLazyTensor *wt = mem[2 * t], *sc = mem[2 * t + 1];
        if (wt->dtype != APUS_ST_I8 || sc->dtype != APUS_ST_F8_E8M0
            || wt->ndim != 2 || sc->ndim != 2
            || wt->shape[0] != sc->shape[0]
            || wt->shape[1] * 2 != sc->shape[1] * 32) {
            snprintf(err, errcap,
                     "store: expert %d.%d w%d shape/dtype mismatch",
                     layer, eid, t + 1);
            return -1;
        }
        rec->O[t] = wt->shape[0];
        rec->K[t] = wt->shape[1] * 2;
    }
    if (st->slab_bytes == 0) st->slab_bytes = (size_t)rec->len;
    if (rec->len != st->slab_bytes) {
        snprintf(err, errcap,
                 "store: expert %d.%d slab %llu != slab_bytes %zu",
                 layer, eid, (unsigned long long)rec->len, st->slab_bytes);
        return -1;
    }
    return 0;
}

/* --- usage history -----------------------------------------------------------*/

static int apus_store_load_usage(ApusStore *st) {
    FILE *f = fopen(st->usage_path, "r");
    if (!f) return -1;
    /* per-layer counts: E-sized freq table */
    uint64_t *cnt = calloc((size_t)st->n_layers * st->E, sizeof(uint64_t));
    int L, e;
    unsigned long long c;
    while (fscanf(f, "%d %d %llu", &L, &e, &c) == 3)
        if (L >= 0 && L < st->n_layers && e >= 0 && e < st->E)
            cnt[(size_t)L * st->E + e] += c;
    fclose(f);
    /* seed pins: top pins_per_layer by count (eid tiebreak for determinism) */
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int p = 0; p < lc->n_pins; p++) {
            int best = -1;
            for (int e2 = 0; e2 < st->E; e2++) {
                if (!cnt[(size_t)l * st->E + e2]) continue;
                int taken = 0;
                for (int q = 0; q < p; q++)
                    if (lc->pins[q].eid == e2) { taken = 1; break; }
                if (taken) continue;
                if (best < 0
                    || cnt[(size_t)l * st->E + e2] > cnt[(size_t)l * st->E + best])
                    best = e2;
            }
            if (best < 0) break;
            lc->pins[p].eid = best;      /* reserved; payload lazy on 1st use */
            /* LFRU seed: the historical count, scaled by the same heat
             * decay the file gets at save (M6b) — otherwise the seeded
             * frequency merges back at full strength and halve-on-save
             * never decays pinned experts. decay=1.0: M6a behavior. */
            lc->pins[p].freq =
                (uint64_t)((double)cnt[(size_t)l * st->E + best]
                           * st->usage_decay);
        }
    }
    free(cnt);
    return 0;
}

int apus_store_save_usage(ApusStore *st) {
    if (!st->usage_enabled) return 0;
    /* merge-with-max against the existing file: live counts are cumulative
     * and monotonic. M6b heat decay: the OLD file counts are first scaled
     * by usage_decay (halve-on-save tracks routing drift; 1.0 = pure
     * cumulative, the M6a behavior). Decayed-to-zero entries are dropped. */
    uint64_t *cnt = calloc((size_t)st->n_layers * st->E, sizeof(uint64_t));
    FILE *f = fopen(st->usage_path, "r");
    if (f) {
        int L, e;
        unsigned long long c;
        while (fscanf(f, "%d %d %llu", &L, &e, &c) == 3)
            if (L >= 0 && L < st->n_layers && e >= 0 && e < st->E) {
                c = (unsigned long long)((double)c * st->usage_decay);
                if (c > cnt[(size_t)L * st->E + e])
                    cnt[(size_t)L * st->E + e] = c;
            }
        fclose(f);
    }
    pthread_mutex_lock(&st->mu);
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int i = 0; i < lc->n_slots; i++)
            if (lc->slots[i].eid >= 0
                && lc->slots[i].freq > cnt[(size_t)l * st->E + lc->slots[i].eid])
                cnt[(size_t)l * st->E + lc->slots[i].eid] = lc->slots[i].freq;
        for (int i = 0; i < lc->n_pins; i++)
            if (lc->pins[i].eid >= 0
                && lc->pins[i].freq > cnt[(size_t)l * st->E + lc->pins[i].eid])
                cnt[(size_t)l * st->E + lc->pins[i].eid] = lc->pins[i].freq;
    }
    pthread_mutex_unlock(&st->mu);
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s.tmp", st->usage_path);
    f = fopen(tmp, "w");
    if (!f) { free(cnt); return -1; }
    for (int l = 0; l < st->n_layers; l++)
        for (int e = 0; e < st->E; e++)
            if (cnt[(size_t)l * st->E + e])
                fprintf(f, "%d %d %llu\n", l, e,
                        (unsigned long long)cnt[(size_t)l * st->E + e]);
    free(cnt);
    fflush(f);
    apus_sys_fsync(f);   /* M15: _commit on Windows */
    if (fclose(f)) { remove(tmp); return -1; }
    if (rename(tmp, st->usage_path)) { remove(tmp); return -1; }
    return 0;
}

/* --- I/O pool ------------------------------------------------------------------*/

static void apus_store_job_push(ApusStore *st, ApusJob j) {
    if (st->jq_n == st->jq_cap) {
        /* grow: re-lay-out the ring linearly — a plain realloc keeps the
         * bytes but the modulo indexing changes with the new capacity, so
         * wrapped entries would be read back as garbage (latent M6a bug,
         * exposed by M6b pilot hint bursts filling the queue) */
        int ncap = st->jq_cap ? 2 * st->jq_cap : 32;
        ApusJob *nj = malloc((size_t)ncap * sizeof *nj);
        for (int i = 0; i < st->jq_n; i++)
            nj[i] = st->jobs[(st->jq_head + i) % st->jq_cap];
        free(st->jobs);
        st->jobs = nj;
        st->jq_cap = ncap;
        st->jq_head = 0;
    }
    int tail = (st->jq_head + st->jq_n) % st->jq_cap;
    st->jobs[tail] = j;
    st->jq_n++;
    pthread_cond_signal(&st->jq_cv);
}

/* Worker-side load: pread into a private buffer, then claim the slot only if
 * the generation tag still matches (straggler safety). The buffer is freed
 * on a failed claim — it can never alias a newer generation's slot. */
static void apus_store_run_job(ApusStore *st, ApusJob j) {
    uint8_t *buf = apus_store_buf_get(st, (size_t)j.rec->len);
    if (!buf) {
        pthread_mutex_lock(&st->mu);
        j.slot->state = APUS_SLOT_EMPTY;
        j.slot->eid = -1;
        pthread_cond_broadcast(&st->cv);
        pthread_mutex_unlock(&st->mu);
        return;
    }
    struct timespec pt0, pt1;
    clock_gettime(CLOCK_MONOTONIC, &pt0);
    apus_st_lazy_pread(j.rec->lz, j.rec->off, buf, (size_t)j.rec->len);
    clock_gettime(CLOCK_MONOTONIC, &pt1);
    uint64_t pns = (uint64_t)(pt1.tv_sec - pt0.tv_sec) * 1000000000ull
                 + (uint64_t)(pt1.tv_nsec - pt0.tv_nsec);
    apus_fadvise_dontneed(-1, j.rec->off, j.rec->len);   /* F_NOCACHE covers */
    if (st->test_pre_claim)
        st->test_pre_claim(st, j.layer, j.slot->eid, j.gen);
    pthread_mutex_lock(&st->mu);
    st->stats.preads++;
    st->stats.bytes_read += j.rec->len;
    st->stats.pread_ns += pns;
    if (j.slot->state == APUS_SLOT_LOADING && j.slot->gen == j.gen) {
        j.slot->buf = buf;
        j.slot->state = APUS_SLOT_READY;
        j.slot->hot = 0;                /* served; boost no longer needed */
        if (j.is_pin) st->stats.pin_loads++;
    } else {
        /* stale generation: slot was recycled under us; drop the payload
         * and reset the slot so a waiter re-submits a fresh job */
        apus_store_buf_put(st, buf);
        j.slot->state = APUS_SLOT_EMPTY;
        j.slot->hot = 0;
    }
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
}

static void *apus_store_worker(void *arg) {
    ApusStore *st = arg;
    for (;;) {
        pthread_mutex_lock(&st->mu);
        while (!st->jq_n && !st->stopping)
            pthread_cond_wait(&st->jq_cv, &st->mu);
        if (st->stopping && !st->jq_n) {
            pthread_mutex_unlock(&st->mu);
            return NULL;
        }
        /* M9c: hot-first pop. A job is hot while its slot is demand-class
         * (MoE batch-union storm or a resolve already waiting on it). Scan
         * from the head for the first hot job; fall back to plain FIFO.
         * The scan is O(queue depth) under mu — depths are a few hundred
         * at most (pilot bursts), and serving the about-to-be-resolved
         * slab first is exactly the pipeline's priority fix: speculative
         * next-layer loads must not delay this layer's demand. */
        int pick = 0;
        if (st->boost)
            for (int i = 0; i < st->jq_n; i++) {
                ApusJob *c = &st->jobs[(st->jq_head + i) % st->jq_cap];
                if (c->slot->hot) { pick = i; break; }
            }
        ApusJob j = st->jobs[(st->jq_head + pick) % st->jq_cap];
        /* remove entry `pick`, preserving the FIFO order of the rest:
         * shift entries [0, pick) one slot away from the head */
        for (int i = pick; i > 0; i--)
            st->jobs[(st->jq_head + i) % st->jq_cap] =
                st->jobs[(st->jq_head + i - 1) % st->jq_cap];
        st->jq_head = (st->jq_head + 1) % st->jq_cap;
        st->jq_n--;
        pthread_mutex_unlock(&st->mu);
        apus_store_run_job(st, j);
    }
}

/* Submit a load for slot (caller holds mu, slot marked LOADING first). */
static void apus_store_submit(ApusStore *st, int layer, ApusSlot *slot,
                              ApusSlabRec *rec, int is_pin) {
    slot->state = APUS_SLOT_LOADING;
    slot->gen = st->gen;
    if (st->n_threads > 0) {
        ApusJob j = { layer, slot, rec, is_pin, st->gen };
        apus_store_job_push(st, j);
    } else {
        /* synchronous mode: run inline after dropping the lock (see
         * apus_store_wait_ready) — emulate by running with mu released. */
        ApusJob j = { layer, slot, rec, is_pin, st->gen };
        pthread_mutex_unlock(&st->mu);
        apus_store_run_job(st, j);
        pthread_mutex_lock(&st->mu);
    }
}

/* Wait until slot is READY; re-submit if a stale claim reset it (mu held).
 * Resolve path only — re-submits count as demand loads. M9c: the wait is
 * timed (stats.waits/wait_ns) and a LOADING slot the compute thread is
 * about to block on is boosted to demand-class so the I/O pool serves it
 * ahead of queued speculative (pilot) loads. */
static int apus_store_wait_ready(ApusStore *st, int layer, ApusSlot *slot,
                                 ApusSlabRec *rec, int is_pin) {
    struct timespec t0, t1;
    int timed = 0;
    while (slot->state != APUS_SLOT_READY) {
        if (!timed) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            timed = 1;
            if (slot->state == APUS_SLOT_LOADING)
                slot->hot = 1;      /* resolve is blocked on this load */
        }
        if (slot->state == APUS_SLOT_EMPTY) {
            if (slot->eid < 0) return -1;   /* alloc failure earlier */
            st->stats.demand_loads++;
            slot->hot = 1;
            apus_store_submit(st, layer, slot, rec, is_pin);
        } else {
            pthread_cond_wait(&st->cv, &st->mu);
        }
    }
    if (timed) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        st->stats.waits++;
        st->stats.wait_ns += (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull
                           + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    }
    return 0;
}

/* --- resolve / hint ------------------------------------------------------------*/

/* Find eid in one slot array; idx_out = index or -1. (mu held) */
static int apus_slot_find(const ApusSlot *slots, int n, int32_t eid) {
    for (int i = 0; i < n; i++)
        if (slots[i].eid == eid) return i;
    return -1;
}

static ApusSlot *apus_ws_find(ApusLayerCache *lc, int32_t eid) {
    for (int i = 0; i < lc->ws_n; i++)
        if (lc->ws[i]->eid == eid) return lc->ws[i];
    return NULL;
}

/* kind_out: 1 = pin, 2 = working set, 3 = LRU. (mu held) */
static int apus_store_lookup(ApusStore *st, int layer, int eid,
                             ApusSlot **slot_out, int *kind_out) {
    ApusLayerCache *lc = &st->lc[layer];
    int i = apus_slot_find(lc->pins, lc->n_pins, eid);
    if (i >= 0) { *slot_out = &lc->pins[i]; *kind_out = 1; return 0; }
    ApusSlot *w = apus_ws_find(lc, eid);
    if (w) { *slot_out = w; *kind_out = 2; return 0; }
    i = apus_slot_find(lc->slots, lc->n_slots, eid);
    if (i >= 0 && lc->slots[i].buf) {
        *slot_out = &lc->slots[i];
        *kind_out = 3;
        return 0;
    }
    return -1;
}

int apus_store_resolve(ApusStore *st, int layer, int eid,
                       ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return -1;
    ApusSlabRec *rec = apus_store_rec(st, layer, eid);
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot = NULL;
    int kind = 0;
    if (apus_store_lookup(st, layer, eid, &slot, &kind) == 0) {
        /* hit := the slab was already cached (pin/LRU resident, or a
         * working-set entry this block already consumed). A first consume
         * of a hint-loaded working-set entry is a miss — the slab came
         * from disk for this block. */
        int resident = !(kind == 2 && slot->freq == 0)
                       && slot->state == APUS_SLOT_READY && slot->buf != NULL;
        if (apus_store_wait_ready(st, layer, slot, rec, kind == 1)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        slot->freq++;
        slot->last = apus_clock_tick(st);
        if (resident) st->stats.hits++;
        else st->stats.misses++;
    } else {
        /* miss: load into the per-forward working set */
        ApusLayerCache *lc = &st->lc[layer];
        ApusSlot *w = calloc(1, sizeof *w);
        w->eid = eid;
        w->freq = 1;
        w->hot = 1;                     /* demand load (M9c) */
        if (lc->ws_n == lc->ws_cap) {
            lc->ws_cap = lc->ws_cap ? 2 * lc->ws_cap : 8;
            lc->ws = realloc(lc->ws, (size_t)lc->ws_cap * sizeof *lc->ws);
        }
        lc->ws[lc->ws_n++] = w;
        st->stats.demand_loads++;
        apus_store_submit(st, layer, w, rec, 0);
        if (apus_store_wait_ready(st, layer, w, rec, 0)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        w->last = apus_clock_tick(st);
        st->stats.misses++;
        slot = w;
    }
    apus_slot_views(rec, slot->buf, w1, w2, w3);
    pthread_mutex_unlock(&st->mu);
    return 0;
}

/* Shared hint body (mu NOT held). demand: mark the load hot so the I/O
 * pool serves it ahead of queued speculative loads (M9c). */
static void apus_store_hint_impl(ApusStore *st, int layer, int eid,
                                 int demand) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    if (apus_store_lookup(st, layer, eid, &slot, &kind) == 0) {
        if (slot->state == APUS_SLOT_EMPTY && slot->eid >= 0) {
            slot->hot = demand ? 1 : slot->hot;
            st->stats.hint_loads++;
            apus_store_submit(st, layer, slot, apus_store_rec(st, layer, eid),
                              kind == 1);
        } else if (demand && slot->state == APUS_SLOT_LOADING) {
            slot->hot = 1;      /* boost the in-flight/queued load */
        }
        pthread_mutex_unlock(&st->mu);
        return;
    }
    ApusLayerCache *lc = &st->lc[layer];
    ApusSlot *w = calloc(1, sizeof *w);
    w->eid = eid;
    w->freq = 0;
    w->hot = demand ? 1 : 0;
    if (lc->ws_n == lc->ws_cap) {
        lc->ws_cap = lc->ws_cap ? 2 * lc->ws_cap : 8;
        lc->ws = realloc(lc->ws, (size_t)lc->ws_cap * sizeof *lc->ws);
    }
    lc->ws[lc->ws_n++] = w;
    st->stats.hint_loads++;
    apus_store_submit(st, layer, w, apus_store_rec(st, layer, eid), 0);
    pthread_mutex_unlock(&st->mu);
}

void apus_store_hint(ApusStore *st, int layer, int eid) {
    apus_store_hint_impl(st, layer, eid, 0);
}

void apus_store_hint_demand(ApusStore *st, int layer, int eid) {
    apus_store_hint_impl(st, layer, eid, 1);
}

/* --- end-of-block promotion -----------------------------------------------------*/

/* Pick the LRU victim slot: prefer a truly empty slot, else the coldest.
 * Never returns a LOADING slot. (mu held) */
static ApusSlot *apus_lru_victim(ApusLayerCache *lc) {
    ApusSlot *best = NULL;
    for (int i = 0; i < lc->n_slots; i++) {
        ApusSlot *s = &lc->slots[i];
        if (s->state == APUS_SLOT_LOADING) continue;
        if (s->eid < 0) return s;           /* empty: free real estate */
        if (!best || s->last < best->last) best = s;
    }
    return best;
}

void apus_store_layer_end(ApusStore *st, int layer) {
    if (!st || layer < 0 || layer >= st->n_layers) return;
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    int out = 0;
    for (int i = 0; i < lc->ws_n; i++) {
        ApusSlot *w = lc->ws[i];
        if (w->state != APUS_SLOT_READY) {
            /* in-flight (pilot hint not yet consumed): keep in the working
             * set; promoted by a later layer_end once READY */
            lc->ws[out++] = w;
            continue;
        }
        ApusSlot *v = apus_lru_victim(lc);
        if (v && v->last <= w->last) {
            if (v->eid >= 0 && v->buf) st->stats.evictions++;
            apus_store_buf_put(st, v->buf);
            v->eid = w->eid;
            v->buf = w->buf;
            v->freq = w->freq;
            v->last = w->last;
            v->state = APUS_SLOT_READY;
            v->hot = 0;
            free(w);
        } else {
            /* LRU full of warmer entries (batch-union overflow): the slab
             * served this block and is dropped without entering the cache */
            apus_store_buf_put(st, w->buf);
            free(w);
        }
    }
    lc->ws_n = out;
    st->gen++;
    pthread_mutex_unlock(&st->mu);
    apus_store_rss_guard(st);
}

/* --- RSS guard ------------------------------------------------------------------*/

void apus_store_rss_guard(ApusStore *st) {
    uint64_t rss = apus_rss_bytes();
    if (!rss || rss <= st->rss_budget) return;
    uint64_t excess = rss - st->rss_budget;
    pthread_mutex_lock(&st->mu);
    /* collect droppable LRU payloads (READY, has buf; pins/ws/LOADING never) */
    size_t total = 0;
    for (int l = 0; l < st->n_layers; l++)
        total += (size_t)st->lc[l].n_slots;
    ApusSlot **ord = malloc(total * sizeof *ord);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++)
        for (int i = 0; i < st->lc[l].n_slots; i++) {
            ApusSlot *s = &st->lc[l].slots[i];
            if (s->state == APUS_SLOT_READY && s->buf) ord[n++] = s;
        }
    for (size_t i = 0; i + 1 < n; i++)
        for (size_t j = i + 1; j < n; j++)
            if (ord[j]->last < ord[i]->last) {
                ApusSlot *t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    uint64_t freed = 0;
    for (size_t i = 0; i < n && freed < excess; i++) {
        apus_aligned_free(ord[i]->buf);
        ord[i]->buf = NULL;
        ord[i]->state = APUS_SLOT_EMPTY;   /* slot keeps eid/freq identity */
        freed += st->slab_bytes;
        st->stats.rss_drops++;
    }
    free(ord);
    pthread_mutex_unlock(&st->mu);
}

/* --- LFRU REPIN ------------------------------------------------------------------*/

/* LFRU score: frequency primary, recency tiebreak (colibri tier.h). Compare
 * as (freq, last) lexicographic; hysteresis: challenger must beat the pin's
 * frequency by 25% + 4 (or, on equal frequency, recency decides). */

static int apus_lfru_beats(uint64_t c_freq, uint64_t c_last,
                           uint64_t p_freq, uint64_t p_last) {
    uint64_t need = p_freq + p_freq / 4 + 4;      /* 25% + 4 hysteresis */
    if (c_freq >= need) return 1;
    if (c_freq == p_freq && c_last > p_last && c_freq >= need / 2) return 1;
    return 0;
}

void apus_store_repin(ApusStore *st) {
    pthread_mutex_lock(&st->mu);
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        if (!lc->n_pins) continue;
        for (int iter = 0; iter < lc->n_pins; iter++) {
            /* coldest pin */
            ApusSlot *pin = NULL;
            for (int i = 0; i < lc->n_pins; i++) {
                ApusSlot *p = &lc->pins[i];
                if (p->eid < 0) continue;
                if (!pin || p->freq < pin->freq
                    || (p->freq == pin->freq && p->last < pin->last))
                    pin = p;
            }
            /* hottest unpinned resident expert */
            ApusSlot *cand = NULL;
            for (int i = 0; i < lc->n_slots; i++) {
                ApusSlot *s = &lc->slots[i];
                if (s->eid < 0 || s->state != APUS_SLOT_READY || !s->freq)
                    continue;
                if (!cand || s->freq > cand->freq
                    || (s->freq == cand->freq && s->last > cand->last))
                    cand = s;
            }
            if (!pin || !cand
                || !apus_lfru_beats(cand->freq, cand->last,
                                    pin->freq, pin->last))
                break;
            /* swap identities: candidate becomes the pin, old pin demoted */
            int32_t te = pin->eid;
            uint8_t *tb = pin->buf;
            uint64_t tf = pin->freq, tl = pin->last;
            int tst = pin->state;
            pin->eid = cand->eid; pin->buf = cand->buf;
            pin->freq = cand->freq; pin->last = cand->last;
            pin->state = cand->state;
            cand->eid = te; cand->buf = tb;
            cand->freq = tf; cand->last = tl;
            cand->state = tst;
            st->stats.repin_swaps++;
        }
    }
    pthread_mutex_unlock(&st->mu);
}

/* --- open / close ------------------------------------------------------------------*/

ApusStore *apus_store_open(const char *model_dir, const ApusStoreCfg *cfg,
                           char *err, size_t errcap) {
    ApusStoreCfg c = cfg ? *cfg : (ApusStoreCfg){0};
    if (c.n_layers <= 0 || c.n_experts <= 0) {
        snprintf(err, errcap, "store: n_layers/n_experts required");
        return NULL;
    }
    /* locate the shard-set dir (same rule as c/model.h) */
    char dir[1024], path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        snprintf(dir, sizeof dir, "%s", model_dir);
    } else {
        snprintf(dir, sizeof dir, "%s/weights", model_dir);
    }
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    char jerr[128];
    JVal *idx = json_parse_file(path, jerr, sizeof jerr);
    if (!idx) { snprintf(err, errcap, "store: %s", jerr); return NULL; }
    JVal *wm = json_obj_get(idx, "weight_map");
    if (!wm || json_type(wm) != J_OBJ) {
        json_free(idx);
        snprintf(err, errcap, "store: no weight_map in %s", path);
        return NULL;
    }

    ApusStore *st = calloc(1, sizeof *st);
    st->n_layers = c.n_layers;
    st->n_main = c.n_layers - c.n_mtp;
    st->E = c.n_experts;
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    pthread_cond_init(&st->jq_cv, NULL);

    /* collect expert tensor refs */
    int need = c.n_layers * c.n_experts * 6;
    ApusExpertTensorRef *refs = malloc((size_t)need * sizeof *refs);
    int *have = calloc((size_t)c.n_layers * c.n_experts, sizeof(int));
    int n_refs = 0;
    for (size_t i = 0; i < json_obj_len(wm); i++) {
        const char *name = json_obj_key(wm, i);
        int L, e, w, sc, im;
        if (apus_parse_expert_name(name, &L, &e, &w, &sc, &im)) continue;
        /* M8: "mtp.K.*" maps to store layer n_main + K */
        if (im) {
            if (L >= c.n_mtp) continue;
            L += st->n_main;
        } else if (L >= st->n_main) {
            continue;   /* would collide with the MTP range: not an expert
                           layer we serve */
        }
        if (L >= c.n_layers || e >= c.n_experts) continue;
        ApusExpertTensorRef *r = &refs[n_refs++];
        r->layer = L; r->eid = e; r->w = w; r->is_scale = sc; r->is_mtp = im;
        snprintf(r->shard, sizeof r->shard, "%s",
                 json_str(json_obj_val(wm, i)));
        have[(size_t)L * c.n_experts + e]++;
    }
    json_free(idx);
    for (int i = 0; i < c.n_layers * c.n_experts; i++) {
        if (have[i] != 6) {
            snprintf(err, errcap,
                     "store: expert %d.%d has %d/6 tensors in weight_map",
                     i / c.n_experts, i % c.n_experts, have[i]);
            free(refs); free(have);
            apus_store_close(st);
            return NULL;
        }
    }
    free(have);

    int nocache = c.nocache > 0 ? 1
                : c.nocache < 0 ? 0
                : apus_env_int("APUS_NOCACHE", 1);

    st->recs = calloc((size_t)c.n_layers * c.n_experts, sizeof *st->recs);
    /* derive slabs: sort refs by (layer, eid, member), then each expert's
     * six refs are contiguous */
    qsort(refs, (size_t)n_refs, sizeof *refs, apus_ref_cmp);
    for (int i = 0; i < n_refs; i += 6) {
        if (apus_store_derive_slab(st, dir, &refs[i], nocache, err, errcap)) {
            free(refs);
            apus_store_close(st);
            return NULL;
        }
    }
    free(refs);

    /* budgets */
    if (!c.cache_bytes)
        c.cache_bytes = apus_env_mb("APUS_EXPERT_CACHE_MB", 4096) << 20;
    if (!c.pin_bytes)
        c.pin_bytes = apus_env_mb("APUS_PIN_MB", 512) << 20;
    if (!c.rss_budget_bytes)
        c.rss_budget_bytes = apus_env_mb("APUS_RSS_GUARD_MB", 26624) << 20;
    if (!c.io_threads)
        c.io_threads = apus_env_int("APUS_IO_THREADS", 4);
    st->boost = apus_env_int("APUS_STORE_BOOST", 1);
    st->rss_budget = c.rss_budget_bytes;
    if (c.usage_decay == 0.0) {
        const char *d = getenv("APUS_USAGE_DECAY");
        c.usage_decay = d ? atof(d) : 1.0;
        if (c.usage_decay <= 0.0 || c.usage_decay > 1.0) c.usage_decay = 1.0;
    }
    st->usage_decay = c.usage_decay;

    int spl = c.slots_per_layer;
    if (spl <= 0) {
        size_t per = c.cache_bytes / ((size_t)c.n_layers * st->slab_bytes);
        spl = per ? (int)per : 1;
    }
    if (spl > c.n_experts) spl = c.n_experts;
    int ppl = c.pins_per_layer;
    if (ppl < 0) ppl = 0;
    if (c.pins_per_layer == 0) {
        size_t per = c.pin_bytes / ((size_t)c.n_layers * st->slab_bytes);
        ppl = (int)per;
    }
    if (ppl > c.n_experts) ppl = c.n_experts;

    st->lc = calloc((size_t)c.n_layers, sizeof *st->lc);
    for (int l = 0; l < c.n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        lc->n_slots = spl;
        lc->slots = calloc((size_t)spl, sizeof *lc->slots);
        for (int i = 0; i < spl; i++) lc->slots[i].eid = -1;
        lc->n_pins = ppl;
        lc->pins = calloc((size_t)(ppl ? ppl : 1), sizeof *lc->pins);
        for (int i = 0; i < ppl; i++) lc->pins[i].eid = -1;
    }

    /* usage history (APUS_USAGE_PATH overrides the default location;
     * "" disables — keeps read-only model dirs pristine) */
    if (!c.usage_path) c.usage_path = getenv("APUS_USAGE_PATH");
    if (!c.usage_path) {
        snprintf(st->usage_path, sizeof st->usage_path, "%s/apus.usage",
                 model_dir);
        st->usage_enabled = 1;
    } else if (*c.usage_path) {
        snprintf(st->usage_path, sizeof st->usage_path, "%s", c.usage_path);
        st->usage_enabled = 1;
    }
    if (st->usage_enabled) apus_store_load_usage(st);

    /* I/O pool */
    if (c.io_threads > 0) {
        st->n_threads = c.io_threads;
        st->threads = calloc((size_t)st->n_threads, sizeof *st->threads);
        for (int i = 0; i < st->n_threads; i++)
            pthread_create(&st->threads[i], NULL, apus_store_worker, st);
    }
    /* M6c buffer-recycling free list; M9c: capacity overridable via
     * APUS_BUF_FREE (deeper prefill prefetch wants a bigger recycle pool;
     * each retained buffer is one slab, ~13.4 MB on the real model). */
    st->buf_free_cap = apus_env_int("APUS_BUF_FREE", APUS_BUF_FREE_MAX);
    if (st->buf_free_cap < 0) st->buf_free_cap = 0;
    if (st->buf_free_cap > 1024) st->buf_free_cap = 1024;
    st->buf_free = calloc((size_t)st->buf_free_cap, sizeof *st->buf_free);
    if (!st->buf_free) st->buf_free_cap = 0;
    return st;
}

void apus_store_close(ApusStore *st) {
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->stopping = 1;
    pthread_cond_broadcast(&st->jq_cv);
    pthread_mutex_unlock(&st->mu);
    for (int i = 0; i < st->n_threads; i++)
        pthread_join(st->threads[i], NULL);
    free(st->threads);
    free(st->jobs);
    if (st->lc) {
        for (int l = 0; l < st->n_layers; l++) {
            ApusLayerCache *lc = &st->lc[l];
            /* slot/pin/wait-slot buffers all trace back to
             * apus_store_buf_get → apus_aligned_alloc — they MUST go to
             * apus_aligned_free (Windows _aligned_malloc storage aborts
             * the heap in plain free(); POSIX never noticed because
             * posix_memalign storage is free()-legal). */
            for (int i = 0; i < lc->n_slots; i++) apus_aligned_free(lc->slots[i].buf);
            for (int i = 0; i < lc->n_pins; i++) apus_aligned_free(lc->pins[i].buf);
            for (int i = 0; i < lc->ws_n; i++) {
                apus_aligned_free(lc->ws[i]->buf);
                free(lc->ws[i]);
            }
            free(lc->slots);
            free(lc->pins);
            free(lc->ws);
        }
        free(st->lc);
    }
    for (int i = 0; i < st->shards_n; i++) {
        free(st->shards[i].name);
        apus_st_lazy_close(st->shards[i].lz);
    }
    free(st->shards);
    free(st->recs);
    for (int i = 0; i < st->buf_free_n; i++) apus_aligned_free(st->buf_free[i]);
    free(st->buf_free);
    pthread_mutex_destroy(&st->mu);
    pthread_cond_destroy(&st->cv);
    pthread_cond_destroy(&st->jq_cv);
    free(st);
}

/* --- stats / misc ------------------------------------------------------------------*/

void apus_store_stats(const ApusStore *st, ApusStoreStats *out) {
    pthread_mutex_lock(&((ApusStore *)st)->mu);
    *out = st->stats;
    pthread_mutex_unlock(&((ApusStore *)st)->mu);
}

size_t apus_store_slab_bytes(const ApusStore *st) { return st->slab_bytes; }

size_t apus_store_resident_bytes(ApusStore *st) {
    pthread_mutex_lock(&st->mu);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int i = 0; i < lc->n_slots; i++)
            if (lc->slots[i].buf) n += st->slab_bytes;
        for (int i = 0; i < lc->n_pins; i++)
            if (lc->pins[i].buf) n += st->slab_bytes;
        for (int i = 0; i < lc->ws_n; i++)
            if (lc->ws[i]->buf) n += st->slab_bytes;
    }
    pthread_mutex_unlock(&st->mu);
    return n;
}

/* --- moe hook trampolines ---------------------------------------------------------*/

static int apus_store_tr_resolve(void *ctx, int layer, int eid,
                                 ApusFp4W *w1, ApusFp4W *w2, ApusFp4W *w3) {
    return apus_store_resolve(ctx, layer, eid, w1, w2, w3);
}

static void apus_store_tr_hint(void *ctx, int layer, int eid) {
    /* MoE batch-union storm: these experts are resolved within the same
     * sublayer — demand class, boosted ahead of speculative pilot loads. */
    apus_store_hint_demand(ctx, layer, eid);
}

static void apus_store_tr_layer_end(void *ctx, int layer) {
    apus_store_layer_end(ctx, layer);
}

void apus_store_attach_moe(ApusStore *st, ApusMoeW *mw) {
    mw->hook_ctx = st;
    mw->hook_resolve = apus_store_tr_resolve;
    mw->hook_hint = apus_store_tr_hint;
    mw->hook_layer_end = apus_store_tr_layer_end;
}

/* --- introspection / test hooks -----------------------------------------------------*/

int apus_store_debug_layer(ApusStore *st, int layer,
                           int32_t *lru_eids, int n_lru,
                           int32_t *pin_eids, int n_pins) {
    if (!st || layer < 0 || layer >= st->n_layers) return -1;
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    if (lru_eids) {
        int n = n_lru < lc->n_slots ? n_lru : lc->n_slots;
        for (int i = 0; i < n; i++)
            lru_eids[i] = lc->slots[i].buf ? lc->slots[i].eid : -1;
    }
    if (pin_eids) {
        int n = n_pins < lc->n_pins ? n_pins : lc->n_pins;
        for (int i = 0; i < n; i++) pin_eids[i] = lc->pins[i].eid;
    }
    pthread_mutex_unlock(&st->mu);
    return 0;
}

void apus_store_debug_set_pre_claim(ApusStore *st,
    void (*fn)(ApusStore *st, int layer, int32_t eid, uint64_t gen)) {
    st->test_pre_claim = fn;
}

int apus_store_debug_stale_gen(ApusStore *st, int layer, int32_t eid) {
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    ApusSlot *w = apus_ws_find(lc, eid);
    if (!w || w->state != APUS_SLOT_LOADING) {
        pthread_mutex_unlock(&st->mu);
        return -1;
    }
    w->gen += 1000;                      /* no longer matches the job tag */
    pthread_mutex_unlock(&st->mu);
    return 0;
}

int apus_store_debug_present(ApusStore *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    /* present := resolve would not submit a fresh load: payload resident
     * or a load already in flight (a reserved-but-unloaded pin or an
     * RSS-dropped LRU slot does NOT count) */
    int present = apus_store_lookup(st, layer, eid, &slot, &kind) == 0
                  && (slot->buf != NULL || slot->state == APUS_SLOT_LOADING);
    pthread_mutex_unlock(&st->mu);
    return present;
}

int apus_store_debug_ready(ApusStore *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    int ready = apus_store_lookup(st, layer, eid, &slot, &kind) == 0
                && slot->state == APUS_SLOT_READY && slot->buf != NULL;
    pthread_mutex_unlock(&st->mu);
    return ready;
}

#endif /* APUS_CACHE_IMPLEMENTATION */
#endif /* APUS_CACHE_H */
