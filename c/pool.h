/*
 * c/pool.h — persistent pthread pool + per-thread scratch arena (M6c).
 * C11, libc + pthreads only. Header-only (static inline) so every
 * single-TU test binary links unchanged; the pool spawns lazily on
 * first apus_pool_run, never at load time.
 *
 * Thread-count independence (the m6a/m6b invariance contract):
 * apus_pool_run(n, fn, ctx) partitions the row range [0, n) into T
 * contiguous blocks, one per lane (lane 0 is the calling thread). Each
 * output row is computed entirely by one thread with the same per-row
 * code as the sequential path, and lanes write disjoint rows, so the
 * results are BITWISE identical for any thread count (including
 * APUS_THREADS=1). Callers must keep each output row's FP32
 * accumulation inside that row (no cross-row reductions); that is what
 * makes the partitioning irrelevant to the numerics.
 *
 * Thread count: APUS_THREADS env (>=1, clamped to 32); default =
 * performance-core count (hw.perflevel0.physicalcpu on macOS,
 * sysconf(_SC_NPROCESSORS_ONLN) elsewhere).
 *
 * Scratch arena: apus_scratch_* is a grow-only thread-local segmented
 * bump allocator for the decode hot path (replaces per-call malloc/free
 * in the linear/expert/attention paths). Usage is strictly LIFO: take a
 * mark at function entry, reset to it at exit. Growth appends a NEW
 * segment instead of reallocating, so pointers handed out earlier in
 * the same scope stay valid. Pool workers never touch the arena
 * (kernels take explicit buffers) and the M6b pilot consumer thread has
 * its own TLS copy, so no locking is needed.
 */
#ifndef APUS_POOL_H
#define APUS_POOL_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pthread.h>

#include "compat.h"     /* apus_ncpu, apus_aligned_alloc/free (M15) */

#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>     /* sysconf(_SC_NPROCESSORS_ONLN) — via compat.h */
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_POOL_MAX_THREADS 32

typedef struct {
    pthread_t       tid[APUS_POOL_MAX_THREADS - 1];
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;
    pthread_cond_t  cv_done;
    uint64_t        gen;        /* dispatch generation */
    int             started;    /* workers spawned */
    int             nthreads;   /* lanes configured (incl. caller) */
    int             nlane;      /* lanes used by the current generation */
    /* one job at a time (the decode forward is serial at dispatch level) */
    void          (*fn)(void *ctx, size_t i0, size_t i1);
    void           *ctx;
    size_t          lo[APUS_POOL_MAX_THREADS];
    size_t          hi[APUS_POOL_MAX_THREADS];
    int             pending;    /* worker lanes still running this gen */
} ApusPool;

static void *apus_pool_worker(void *arg);

static inline int apus_pool_default_threads(void) {
    int n = 0;
#ifdef __APPLE__
    size_t sz = sizeof n;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) || n <= 0) {
        sz = sizeof n;
        if (sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0) || n <= 0)
            n = 1;
    }
#else
    n = apus_ncpu();   /* sysconf on POSIX, GetSystemInfo on Windows (M15) */
#endif
    return n;
}

static inline ApusPool *apus_pool_get(void) {
    /* function-local statics in an inline function: one per TU. Every
     * engine/test binary in this project has a single implementation TU,
     * so there is exactly one pool per process. */
    static ApusPool pool;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;
    static int inited;
    pthread_mutex_lock(&init_mu);
    if (!inited) {
        int n = apus_pool_default_threads();
        const char *e = getenv("APUS_THREADS");
        if (e && *e) {
            int v = atoi(e);
            if (v >= 1) n = v;
        }
        if (n < 1) n = 1;
        if (n > APUS_POOL_MAX_THREADS) n = APUS_POOL_MAX_THREADS;
        pool.nthreads = n;
        pthread_mutex_init(&pool.mu, NULL);
        pthread_cond_init(&pool.cv_work, NULL);
        pthread_cond_init(&pool.cv_done, NULL);
        inited = 1;
    }
    pthread_mutex_unlock(&init_mu);
    return &pool;
}

static void *apus_pool_worker(void *arg) {
    ApusPool *p = apus_pool_get();
    int lane = (int)(intptr_t)arg;              /* 1..nthreads-1 */
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (p->gen == seen)
            pthread_cond_wait(&p->cv_work, &p->mu);
        seen = p->gen;
        int active = lane < p->nlane;
        void (*fn)(void *, size_t, size_t) = p->fn;
        void *ctx = p->ctx;
        size_t lo = p->lo[lane], hi = p->hi[lane];
        pthread_mutex_unlock(&p->mu);
        if (active && hi > lo) fn(ctx, lo, hi);
        pthread_mutex_lock(&p->mu);
        if (active && --p->pending == 0)
            pthread_cond_signal(&p->cv_done);
    }
    return NULL;
}

/* Number of lanes the next apus_pool_run would use (>= 1). */
static inline int apus_pool_threads(void) {
    return apus_pool_get()->nthreads;
}

/* Run fn(ctx, i0, i1) over a contiguous balanced partition of [0, n)
 * across the pool; blocks until all lanes finish. The calling thread
 * runs lane 0. Falls back to a plain call when nthreads == 1 (no
 * dispatch machinery, no overhead). */
static inline void apus_pool_run(size_t n,
                                 void (*fn)(void *ctx, size_t i0, size_t i1),
                                 void *ctx) {
    ApusPool *p = apus_pool_get();
    int T = p->nthreads;
    if (T <= 1 || n < 2) {
        fn(ctx, 0, n);
        return;
    }
    if ((size_t)T > n) T = (int)n;
    /* balanced contiguous partition: first rem lanes get base+1 rows */
    size_t base = n / (size_t)T, rem = n % (size_t)T, pos = 0;
    for (int i = 0; i < T; i++) {
        size_t cnt = base + ((size_t)i < rem ? 1 : 0);
        p->lo[i] = pos;
        p->hi[i] = pos + cnt;
        pos += cnt;
    }
    pthread_mutex_lock(&p->mu);
    if (!p->started) {
        for (int i = 1; i < p->nthreads; i++)
            pthread_create(&p->tid[i - 1], NULL, apus_pool_worker,
                           (void *)(intptr_t)i);
        p->started = 1;
    }
    p->fn = fn;
    p->ctx = ctx;
    p->nlane = T;
    p->pending = T - 1;
    p->gen++;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    if (p->hi[0] > p->lo[0]) fn(ctx, p->lo[0], p->hi[0]);
    pthread_mutex_lock(&p->mu);
    while (p->pending)
        pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

/* --- thread-local segmented scratch arena (LIFO) ---------------------------*/

#define APUS_SCRATCH_MAX_SEG 64

typedef struct {
    char  *seg[APUS_SCRATCH_MAX_SEG];
    size_t cap[APUS_SCRATCH_MAX_SEG];
    int    nseg;
    int    cur;         /* active segment */
    size_t off;         /* bump offset in the active segment */
} ApusScratch;

typedef struct {
    int    seg;
    size_t off;
} ApusScratchMark;

static inline ApusScratch *apus_scratch_get(void) {
    static _Thread_local ApusScratch sc;
    return &sc;
}

/* Bump-allocate `bytes` (64-byte aligned) from the calling thread's
 * arena. Growth appends a new segment (older allocations stay put).
 * Returns NULL only on OOM. */
static inline void *apus_scratch_alloc(size_t bytes) {
    ApusScratch *sc = apus_scratch_get();
    if (sc->nseg == 0) sc->cur = -1;    /* first call on this thread */
    size_t off = sc->cur >= 0 ? (sc->off + 63u) & ~(size_t)63u : 0;
    if (sc->cur < 0 || off + bytes > sc->cap[sc->cur]) {
        if (sc->cur + 1 >= APUS_SCRATCH_MAX_SEG) return NULL;   /* misuse */
        sc->cur++;
        if (sc->cur >= sc->nseg) {
            size_t cap = sc->cur > 0 ? 2 * sc->cap[sc->cur - 1]
                                     : (size_t)1 << 16;
            while (cap < bytes) cap *= 2;
            sc->seg[sc->cur] = apus_aligned_alloc(64, cap);
            if (!sc->seg[sc->cur]) { sc->cur--; return NULL; }
            sc->cap[sc->cur] = cap;
            sc->nseg = sc->cur + 1;
        } else if (sc->cap[sc->cur] < bytes) {
            /* Reusing a DEAD segment (LIFO: a reset dropped cur below nseg,
             * so nothing above the caller's mark is live) that is too small:
             * grow it in place. Without this check the bump offset ran past
             * the segment end — a heap overflow (m5 non-determinism /
             * segfault at APUS_THREADS=1). */
            size_t cap = sc->cap[sc->cur];
            while (cap < bytes) cap *= 2;
            apus_aligned_free(sc->seg[sc->cur]);   /* pairs with the alloc */
            sc->seg[sc->cur] = apus_aligned_alloc(64, cap);
            if (!sc->seg[sc->cur]) { sc->cap[sc->cur] = 0; sc->cur--; return NULL; }
            sc->cap[sc->cur] = cap;
        }
        sc->off = 0;
        off = 0;
    }
    sc->off = off + bytes;
    return sc->seg[sc->cur] + off;
}

static inline ApusScratchMark apus_scratch_mark(void) {
    ApusScratch *sc = apus_scratch_get();
    ApusScratchMark m = { sc->cur, sc->off };
    return m;
}

static inline void apus_scratch_reset(ApusScratchMark m) {
    ApusScratch *sc = apus_scratch_get();
    sc->cur = m.seg;
    sc->off = m.off;
}

#ifdef __cplusplus
}
#endif

#endif /* APUS_POOL_H */
