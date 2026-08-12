/*
 * c/st.h — minimal safetensors shard reader for the apus container format.
 * C11, libc only (uses c/json.h for header/index parsing).
 *
 * Format (tests/m1/stutil.py, tools/convert.py):
 *   8-byte little-endian header length N, then N bytes of JSON
 *   {"<name>": {"dtype": "...", "shape": [...], "data_offsets": [a, b]}},
 *   then the raw tensor payloads concatenated (offsets relative to the end
 *   of the JSON header).
 *
 * Shards are opened header-only; each tensor payload is materialized on
 * first resolve (at/find) into its own malloc'd buffer via pread, and stays
 * resident until apus_st_close. Tensor data is exposed through the same
 * zero-copy ApusStTensor.data pointer as before (stable until close — the
 * M7b Metal backend wraps these buffers no-copy). Only tensors that are
 * actually resolved occupy RAM, so expert-sized payloads never touched
 * through this API (tiered mode, c/cache.h owns expert reads) never enter
 * memory; the real 5.4 GB shards are never slurped whole.
 *
 * Dtypes resolved: I8, U8, F8_E8M0, F8_E4M3, BF16, F16, F32, I32, I64.
 * Typed helpers: apus_st_f32 (F32/BF16 -> f32, BF16 losslessly widened),
 * apus_st_i64. Raw-byte access via ApusStTensor.data for the quantized
 * formats (FP8 codes / UE8M0 scales / packed FP4 stay raw by design).
 *
 * Also provides ApusStSet: a directory with model.safetensors.index.json
 * mapping tensor name -> shard file (the real checkpoint layout), with
 * lazy per-shard loading.
 *
 * Usage: #define APUS_ST_IMPLEMENTATION in exactly one TU. That TU (or
 * another) must also define APUS_JSON_IMPLEMENTATION (c/json.h).
 */
#ifndef APUS_ST_H
#define APUS_ST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    APUS_ST_I8, APUS_ST_U8, APUS_ST_F8_E8M0, APUS_ST_F8_E4M3,
    APUS_ST_BF16, APUS_ST_F16, APUS_ST_F32, APUS_ST_I32, APUS_ST_I64,
    APUS_ST_OTHER
} ApusStDtype;

#define APUS_ST_MAX_NDIM 8

typedef struct {
    char        name[160];
    ApusStDtype dtype;
    int         ndim;
    int64_t     shape[APUS_ST_MAX_NDIM];
    const void *data;      /* payload buffer, materialized on first access */
    size_t      nbytes;    /* payload size in bytes */
} ApusStTensor;

typedef struct ApusSt ApusSt;

/* Header-only open; payloads are pread-materialized per tensor on first
 * at/find. Returns NULL on error (err filled if given). */
ApusSt *apus_st_open(const char *path, char *err, size_t errcap);
void    apus_st_close(ApusSt *st);

int                 apus_st_ntensors(const ApusSt *st);
const ApusStTensor *apus_st_at(const ApusSt *st, int i);
const ApusStTensor *apus_st_find(const ApusSt *st, const char *name);

size_t apus_st_nelem(const ApusStTensor *t);

/* Typed accessors. Return 0 on success, -1 on dtype/size mismatch. */
int apus_st_f32(const ApusStTensor *t, float *out, size_t n);   /* F32/BF16 */
int apus_st_i64(const ApusStTensor *t, int64_t *out, size_t n); /* I64 */

/* --- shard set via model.safetensors.index.json ---------------------------*/
typedef struct ApusStSet ApusStSet;

/* dir contains model.safetensors.index.json + shard files. */
ApusStSet *apus_st_set_open(const char *dir, char *err, size_t errcap);
void       apus_st_set_close(ApusStSet *set);

/* Resolve a tensor by full name (e.g. "layers.1.attn.wq_a.weight").
 * The returned view is valid until apus_st_set_close. */
const ApusStTensor *apus_st_set_get(ApusStSet *set, const char *name);

/* Quantized weight views (zero-copy into shard data). */
typedef struct { const uint8_t *codes, *scales; int64_t O, K; } ApusFp8W;
typedef struct { const uint8_t *packed, *scales; int64_t O, K; } ApusFp4W;

/* Resolve "name" -> name.weight (F8_E4M3 [O,K]) + name.scale (F8_E8M0
 * [O/128,K/128]) into *w. Returns 0 on success. */
int apus_st_fp8w(ApusStSet *set, const char *name, ApusFp8W *w);
/* Resolve "name" -> name.weight (I8 packed FP4 [O,K/2]) + name.scale
 * (F8_E8M0 [O,K/32]) into *w. Returns 0 on success. */
int apus_st_fp4w(ApusStSet *set, const char *name, ApusFp4W *w);

/* When deferred, apus_st_fp4w-style expert resolution is skipped by callers
 * that check apus_st_set_deferred (c/layer.h uses this in tiered mode: the
 * expert store c/cache.h owns expert reads; nothing expert-sized is slurped). */
void apus_st_set_defer_experts(ApusStSet *set, int on);
int  apus_st_set_deferred(const ApusStSet *set);

/* --- lazy shard reader (M6a): header-only open + pread at offset ----------
 * The slurp path above stays the default for fixtures/dense tensors; the
 * lazy path is for expert payloads: the shard file is opened (plus an
 * F_NOCACHE twin fd for streaming reads that must not churn the page
 * cache), only the JSON header is parsed, and tensor payloads are read on
 * demand by absolute file offset. mmap is deliberately NOT used (page-cache
 * pressure surprises); plain pread is the primary path. Buffers are
 * caller-owned; 4 KiB alignment is honored by callers that need it
 * (F_NOCACHE pread has no alignment requirement on macOS). */

typedef struct {
    char        name[160];
    ApusStDtype dtype;
    int         ndim;
    int64_t     shape[APUS_ST_MAX_NDIM];
    uint64_t    file_off;   /* absolute offset of the payload in the file */
    uint64_t    nbytes;
} ApusStLazyTensor;

typedef struct ApusStLazy ApusStLazy;

/* Header-only open. nocache != 0 also opens an F_NOCACHE twin fd used for
 * all reads (streaming hygiene); 0 reads through the normal cached fd. */
ApusStLazy *apus_st_lazy_open(const char *path, int nocache, char *err,
                              size_t errcap);
void        apus_st_lazy_close(ApusStLazy *lz);

int                     apus_st_lazy_ntensors(const ApusStLazy *lz);
const ApusStLazyTensor *apus_st_lazy_at(const ApusStLazy *lz, int i);
const ApusStLazyTensor *apus_st_lazy_find(const ApusStLazy *lz,
                                          const char *name);

/* Read nbytes at absolute file offset into dst (loops on short reads).
 * Thread-safe (positional I/O, no shared FILE*). Returns 0 on success. */
int apus_st_lazy_pread(ApusStLazy *lz, uint64_t file_off, void *dst,
                       size_t nbytes);

/* Instrumentation (atomic): number of apus_st_lazy_pread calls and bytes
 * served — tests assert the one-pread-per-expert coalescing invariant. */
uint64_t apus_st_lazy_read_count(const ApusStLazy *lz);
uint64_t apus_st_lazy_read_bytes(const ApusStLazy *lz);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_ST_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdatomic.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "compat.h"   /* M15: apus_sys_open_ro/fsize/pread shims */
#include "json.h"

struct ApusSt {
    int           fd;      /* cached fd, open until apus_st_close */
    ApusStTensor *tensors;
    uint64_t    *offs;    /* parallel to tensors: absolute payload offsets */
    int           n;
};

struct ApusStLazy {
    int              fd;        /* cached fd */
    int              fd_nc;     /* F_NOCACHE twin, -1 when not requested */
    ApusStLazyTensor *tensors;
    int              n;
    _Atomic uint64_t n_reads;
    _Atomic uint64_t n_bytes;
};

struct ApusStSet {
    char     dir[1024];
    int      defer_experts;   /* tiered mode: experts resolved via cache.h */
    /* flat tensor table across all shards, filled lazily shard by shard */
    struct { char *name; char *shard; } *map;
    int       map_n;
    ApusSt  **shards;      /* parallel to shard_names [shards_n] */
    char    **shard_names;
    int       shards_n, shards_cap;
    /* resolved cache: name -> (shard index, tensor) */
    struct { ApusSt *st; const ApusStTensor *t; char *name; } *cache;
    int cache_n, cache_cap;
};

static ApusStDtype apus_st_dtype_of(const char *s) {
    if (!strcmp(s, "I8")) return APUS_ST_I8;
    if (!strcmp(s, "U8")) return APUS_ST_U8;
    if (!strcmp(s, "F8_E8M0")) return APUS_ST_F8_E8M0;
    if (!strcmp(s, "F8_E4M3")) return APUS_ST_F8_E4M3;
    if (!strcmp(s, "BF16")) return APUS_ST_BF16;
    if (!strcmp(s, "F16")) return APUS_ST_F16;
    if (!strcmp(s, "F32")) return APUS_ST_F32;
    if (!strcmp(s, "I32")) return APUS_ST_I32;
    if (!strcmp(s, "I64")) return APUS_ST_I64;
    return APUS_ST_OTHER;
}

static void apus_st_err(char *err, size_t cap, const char *msg, const char *arg) {
    if (err && cap) snprintf(err, cap, msg, arg);
}

ApusSt *apus_st_open(const char *path, char *err, size_t errcap) {
    int fd = apus_sys_open_ro(path);
    if (fd < 0) { apus_st_err(err, errcap, "st: cannot open %s", path); return NULL; }
    int64_t fszi = apus_sys_fsize(fd);
    if (fszi < 0) {
        close(fd); apus_st_err(err, errcap, "st: stat failed %s", path);
        return NULL;
    }
    uint64_t fsz = (uint64_t)fszi;
    uint8_t h8[8];
    if (fsz < 8 || apus_sys_pread(fd, h8, 8, 0) != 8) {
        close(fd); apus_st_err(err, errcap, "st: %s too small", path);
        return NULL;
    }
    uint64_t hlen;
    memcpy(&hlen, h8, 8);                  /* little-endian host assumed (M1) */
    if (hlen > (1u << 28) || 8 + hlen > fsz) {
        close(fd); apus_st_err(err, errcap, "st: bad header in %s", path);
        return NULL;
    }
    char *hbuf = malloc((size_t)hlen);
    if (!hbuf || apus_sys_pread(fd, hbuf, (size_t)hlen, 8) != (int64_t)hlen) {
        free(hbuf); close(fd);
        apus_st_err(err, errcap, "st: header read failed %s", path);
        return NULL;
    }
    char jerr[128];
    JVal *hdr = json_parse(hbuf, (size_t)hlen, jerr, sizeof jerr);
    free(hbuf);
    if (!hdr || json_type(hdr) != J_OBJ) {
        close(fd); apus_st_err(err, errcap, "st: header JSON: %s", jerr);
        return NULL;
    }
    ApusSt *st = calloc(1, sizeof *st);
    st->fd = fd;
    size_t n = json_obj_len(hdr);
    st->tensors = calloc(n ? n : 1, sizeof *st->tensors);
    st->offs = calloc(n ? n : 1, sizeof *st->offs);
    const uint64_t data_base = 8 + hlen;
    for (size_t i = 0; i < n; i++) {
        const char *key = json_obj_key(hdr, i);
        if (!strcmp(key, "__metadata__")) continue;
        JVal *m = json_obj_val(hdr, i);
        ApusStTensor *t = &st->tensors[st->n];
        snprintf(t->name, sizeof t->name, "%s", key);
        JVal *dt = json_obj_get(m, "dtype");
        t->dtype = apus_st_dtype_of(dt ? json_str(dt) : "");
        JVal *sh = json_obj_get(m, "shape");
        t->ndim = (int)json_arr_len(sh);
        if (t->ndim > APUS_ST_MAX_NDIM) t->ndim = APUS_ST_MAX_NDIM;
        for (int d = 0; d < t->ndim; d++)
            t->shape[d] = (int64_t)json_num(json_arr_get(sh, d));
        JVal *off = json_obj_get(m, "data_offsets");
        uint64_t a = (uint64_t)json_num(json_arr_get(off, 0));
        uint64_t b = (uint64_t)json_num(json_arr_get(off, 1));
        if (b < a || data_base + b > fsz) {
            apus_st_err(err, errcap, "st: tensor past EOF in %s", path);
            json_free(hdr); apus_st_close(st); return NULL;
        }
        t->nbytes = (size_t)(b - a);
        st->offs[st->n] = data_base + a;
        st->n++;
    }
    json_free(hdr);
    return st;
}

void apus_st_close(ApusSt *st) {
    if (!st) return;
    for (int i = 0; i < st->n; i++) free((void *)st->tensors[i].data);
    if (st->fd >= 0) close(st->fd);
    free(st->offs);
    free(st->tensors);
    free(st);
}

int apus_st_ntensors(const ApusSt *st) { return st->n; }

/* Materialize tensor i's payload on first access (single-threaded contract,
 * like the ApusStSet resolve cache). Stable until apus_st_close. */
static const ApusStTensor *apus_st_materialize(const ApusSt *cst, int i) {
    ApusSt *st = (ApusSt *)cst;
    if (i < 0 || i >= st->n) return NULL;
    ApusStTensor *t = &st->tensors[i];
    if (!t->data) {
        uint8_t *buf = malloc(t->nbytes ? t->nbytes : 1);
        if (!buf) return NULL;
        size_t done = 0;
        while (done < t->nbytes) {
            int64_t r = apus_sys_pread(st->fd, buf + done, t->nbytes - done,
                                       (uint64_t)(st->offs[i] + done));
            if (r <= 0) { free(buf); return NULL; }
            done += (size_t)r;
        }
        t->data = buf;
    }
    return t;
}

const ApusStTensor *apus_st_at(const ApusSt *st, int i) {
    return apus_st_materialize(st, i);
}

const ApusStTensor *apus_st_find(const ApusSt *st, const char *name) {
    for (int i = 0; i < st->n; i++)
        if (!strcmp(st->tensors[i].name, name))
            return apus_st_materialize(st, i);
    return NULL;
}

size_t apus_st_nelem(const ApusStTensor *t) {
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= (size_t)t->shape[d];
    return n;
}

int apus_st_f32(const ApusStTensor *t, float *out, size_t n) {
    if (apus_st_nelem(t) != n) return -1;
    if (t->dtype == APUS_ST_F32) {
        memcpy(out, t->data, n * sizeof(float));
        return 0;
    }
    if (t->dtype == APUS_ST_BF16) {
        const uint16_t *b = t->data;
        for (size_t i = 0; i < n; i++) {
            uint32_t u = (uint32_t)b[i] << 16;
            memcpy(&out[i], &u, 4);
        }
        return 0;
    }
    return -1;
}

int apus_st_i64(const ApusStTensor *t, int64_t *out, size_t n) {
    if (t->dtype != APUS_ST_I64 || apus_st_nelem(t) != n) return -1;
    memcpy(out, t->data, n * sizeof(int64_t));
    return 0;
}

/* --- shard set ------------------------------------------------------------*/

ApusStSet *apus_st_set_open(const char *dir, char *err, size_t errcap) {
    char path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    char jerr[128];
    JVal *idx = json_parse_file(path, jerr, sizeof jerr);
    if (!idx) { apus_st_err(err, errcap, "stset: %s", jerr); return NULL; }
    JVal *wm = json_obj_get(idx, "weight_map");
    if (!wm || json_type(wm) != J_OBJ) {
        json_free(idx); apus_st_err(err, errcap, "stset: no weight_map in %s", path);
        return NULL;
    }
    ApusStSet *set = calloc(1, sizeof *set);
    snprintf(set->dir, sizeof set->dir, "%s", dir);
    set->map_n = (int)json_obj_len(wm);
    set->map = calloc((size_t)set->map_n, sizeof *set->map);
    for (int i = 0; i < set->map_n; i++) {
        set->map[i].name = strdup(json_obj_key(wm, i));
        set->map[i].shard = strdup(json_str(json_obj_val(wm, i)));
    }
    json_free(idx);
    return set;
}

void apus_st_set_close(ApusStSet *set) {
    if (!set) return;
    for (int i = 0; i < set->map_n; i++) {
        free(set->map[i].name);
        free(set->map[i].shard);
    }
    free(set->map);
    if (set->shards) {
        for (int i = 0; i < set->shards_n; i++) {
            if (set->shards[i]) apus_st_close(set->shards[i]);
            free(set->shard_names[i]);
        }
        free(set->shards);
        free(set->shard_names);
    }
    if (set->cache) {
        for (int i = 0; i < set->cache_n; i++) free(set->cache[i].name);
        free(set->cache);
    }
    free(set);
}

const ApusStTensor *apus_st_set_get(ApusStSet *set, const char *name) {
    for (int i = 0; i < set->cache_n; i++)
        if (!strcmp(set->cache[i].name, name)) return set->cache[i].t;
    const char *shard = NULL;
    for (int i = 0; i < set->map_n; i++)
        if (!strcmp(set->map[i].name, name)) { shard = set->map[i].shard; break; }
    if (!shard) return NULL;
    /* find or open the shard (shards are few; linear scan is fine) */
    ApusSt *st = NULL;
    if (!set->shards) {
        set->shards_cap = 8;
        set->shards = calloc((size_t)set->shards_cap, sizeof *set->shards);
        set->shard_names = calloc((size_t)set->shards_cap, sizeof(char *));
    }
    char **names = set->shard_names;
    for (int i = 0; i < set->shards_n; i++)
        if (!strcmp(names[i], shard)) { st = set->shards[i]; break; }
    if (!st) {
        if (set->shards_n == set->shards_cap) {
            set->shards_cap *= 2;
            set->shards = realloc(set->shards,
                                  (size_t)set->shards_cap * sizeof *set->shards);
            set->shard_names = realloc(set->shard_names,
                                       (size_t)set->shards_cap * sizeof(char *));
            names = set->shard_names;
        }
        char path[1200], err[128];
        snprintf(path, sizeof path, "%s/%s", set->dir, shard);
        st = apus_st_open(path, err, sizeof err);
        if (!st) return NULL;
        names[set->shards_n] = strdup(shard);
        set->shards[set->shards_n] = st;
        set->shards_n++;
    }
    const ApusStTensor *t = apus_st_find(st, name);
    if (!t) return NULL;
    if (set->cache_n == set->cache_cap) {
        set->cache_cap = set->cache_cap ? 2 * set->cache_cap : 64;
        set->cache = realloc(set->cache, (size_t)set->cache_cap * sizeof *set->cache);
    }
    set->cache[set->cache_n].name = strdup(name);
    set->cache[set->cache_n].st = st;
    set->cache[set->cache_n].t = t;
    set->cache_n++;
    return t;
}

int apus_st_fp8w(ApusStSet *set, const char *name, ApusFp8W *w) {
    char buf[192];
    snprintf(buf, sizeof buf, "%s.weight", name);
    const ApusStTensor *c = apus_st_set_get(set, buf);
    snprintf(buf, sizeof buf, "%s.scale", name);
    const ApusStTensor *s = apus_st_set_get(set, buf);
    if (!c || !s || c->dtype != APUS_ST_F8_E4M3 || s->dtype != APUS_ST_F8_E8M0
        || c->ndim != 2 || s->ndim != 2
        || s->shape[0] * 128 != c->shape[0] || s->shape[1] * 128 != c->shape[1])
        return -1;
    w->codes = c->data;
    w->scales = s->data;
    w->O = c->shape[0];
    w->K = c->shape[1];
    return 0;
}

int apus_st_fp4w(ApusStSet *set, const char *name, ApusFp4W *w) {
    char buf[192];
    snprintf(buf, sizeof buf, "%s.weight", name);
    const ApusStTensor *p = apus_st_set_get(set, buf);
    snprintf(buf, sizeof buf, "%s.scale", name);
    const ApusStTensor *s = apus_st_set_get(set, buf);
    if (!p || !s || p->dtype != APUS_ST_I8 || s->dtype != APUS_ST_F8_E8M0
        || p->ndim != 2 || s->ndim != 2
        || p->shape[0] != s->shape[0] || p->shape[1] * 2 != s->shape[1] * 32)
        return -1;
    w->packed = p->data;
    w->scales = s->data;
    w->O = p->shape[0];
    w->K = p->shape[1] * 2;
    return 0;
}

void apus_st_set_defer_experts(ApusStSet *set, int on) {
    set->defer_experts = on;
}

int apus_st_set_deferred(const ApusStSet *set) {
    return set->defer_experts;
}

/* --- lazy shard reader -----------------------------------------------------*/

ApusStLazy *apus_st_lazy_open(const char *path, int nocache, char *err,
                              size_t errcap) {
    int fd = apus_sys_open_ro(path);
    if (fd < 0) { apus_st_err(err, errcap, "stlazy: cannot open %s", path); return NULL; }
    uint8_t h8[8];
    if (apus_sys_pread(fd, h8, 8, 0) != 8) {
        close(fd); apus_st_err(err, errcap, "stlazy: %s too small", path);
        return NULL;
    }
    uint64_t hlen;
    memcpy(&hlen, h8, 8);                  /* little-endian host assumed (M1) */
    if (hlen > (1u << 28)) {               /* 256 MB header reserve sanity */
        close(fd); apus_st_err(err, errcap, "stlazy: bad header in %s", path);
        return NULL;
    }
    char *hbuf = malloc((size_t)hlen);
    if (!hbuf || apus_sys_pread(fd, hbuf, (size_t)hlen, 8) != (int64_t)hlen) {
        free(hbuf); close(fd);
        apus_st_err(err, errcap, "stlazy: header read failed %s", path);
        return NULL;
    }
    char jerr[128];
    JVal *hdr = json_parse(hbuf, (size_t)hlen, jerr, sizeof jerr);
    free(hbuf);
    if (!hdr || json_type(hdr) != J_OBJ) {
        close(fd); apus_st_err(err, errcap, "stlazy: header JSON: %s", jerr);
        return NULL;
    }
    ApusStLazy *lz = calloc(1, sizeof *lz);
    lz->fd = fd;
    lz->fd_nc = -1;
    if (nocache) {
        lz->fd_nc = apus_sys_open_ro(path);
#ifdef __APPLE__
        if (lz->fd_nc >= 0 && fcntl(lz->fd_nc, F_NOCACHE, 1)) {
#else
        if (lz->fd_nc >= 0) {            /* no F_NOCACHE equivalent; cached */
#endif
            close(lz->fd_nc);
            lz->fd_nc = -1;
        }
    }
    uint64_t data_base = 8 + hlen;
    size_t n = json_obj_len(hdr);
    lz->tensors = calloc(n ? n : 1, sizeof *lz->tensors);
    for (size_t i = 0; i < n; i++) {
        const char *key = json_obj_key(hdr, i);
        if (!strcmp(key, "__metadata__")) continue;
        JVal *m = json_obj_val(hdr, i);
        ApusStLazyTensor *t = &lz->tensors[lz->n];
        snprintf(t->name, sizeof t->name, "%s", key);
        JVal *dt = json_obj_get(m, "dtype");
        t->dtype = apus_st_dtype_of(dt ? json_str(dt) : "");
        JVal *sh = json_obj_get(m, "shape");
        t->ndim = (int)json_arr_len(sh);
        if (t->ndim > APUS_ST_MAX_NDIM) t->ndim = APUS_ST_MAX_NDIM;
        for (int d = 0; d < t->ndim; d++)
            t->shape[d] = (int64_t)json_num(json_arr_get(sh, d));
        JVal *off = json_obj_get(m, "data_offsets");
        uint64_t a = (uint64_t)json_num(json_arr_get(off, 0));
        uint64_t b = (uint64_t)json_num(json_arr_get(off, 1));
        t->file_off = data_base + a;
        t->nbytes = b - a;
        lz->n++;
    }
    json_free(hdr);
    return lz;
}

void apus_st_lazy_close(ApusStLazy *lz) {
    if (!lz) return;
    if (lz->fd >= 0) close(lz->fd);
    if (lz->fd_nc >= 0) close(lz->fd_nc);
    free(lz->tensors);
    free(lz);
}

int apus_st_lazy_ntensors(const ApusStLazy *lz) { return lz->n; }

const ApusStLazyTensor *apus_st_lazy_at(const ApusStLazy *lz, int i) {
    return (i >= 0 && i < lz->n) ? &lz->tensors[i] : NULL;
}

const ApusStLazyTensor *apus_st_lazy_find(const ApusStLazy *lz,
                                          const char *name) {
    for (int i = 0; i < lz->n; i++)
        if (!strcmp(lz->tensors[i].name, name)) return &lz->tensors[i];
    return NULL;
}

int apus_st_lazy_pread(ApusStLazy *lz, uint64_t file_off, void *dst,
                       size_t nbytes) {
    int fd = lz->fd_nc >= 0 ? lz->fd_nc : lz->fd;
    uint8_t *p = dst;
    size_t done = 0;
    while (done < nbytes) {
        int64_t r = apus_sys_pread(fd, p + done, nbytes - done,
                                   file_off + done);
        if (r < 0) return -1;
        if (r == 0) return -1;             /* short file */
        done += (size_t)r;
    }
    atomic_fetch_add_explicit(&lz->n_reads, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&lz->n_bytes, nbytes, memory_order_relaxed);
    return 0;
}

uint64_t apus_st_lazy_read_count(const ApusStLazy *lz) {
    return atomic_load_explicit(&lz->n_reads, memory_order_relaxed);
}

uint64_t apus_st_lazy_read_bytes(const ApusStLazy *lz) {
    return atomic_load_explicit(&lz->n_bytes, memory_order_relaxed);
}

#endif /* APUS_ST_IMPLEMENTATION */
#endif /* APUS_ST_H */
