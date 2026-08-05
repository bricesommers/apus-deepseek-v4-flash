/*
 * tests/m8/test_m8.c — M8 MTP speculative decoding verification driver.
 *
 * Loads the synthetic mini-model WITH an MTP block (tests/m8/fixtures;
 * regenerate with `make golden-m8`) and checks:
 *
 *   1. mtp_prefill — MTP true-pair replay: per-prefix MTP logits vs the
 *      oracle f32 goldens (m5 tolerancing: scale-relative error + argmax
 *      flips excused only at small golden top1-top2 gaps).
 *   2. mtp_chain — greedy draft chain (3 drafts) vs goldens.
 *   3. EQUIVALENCE (the hard gate): spec decode (depth 1/2/3) vs non-spec
 *      decode, greedy AND sampled (fixed seed) — emitted streams BITWISE
 *      identical. This is the point of speculative decoding: same output,
 *      batched verify.
 *   4. Rollback: FNV digest of the full model state after a spec run ==
 *      digest after decoding exactly the accepted tokens non-speculatively
 *      (rejected drafts must leave no trace: pos, SWA window rings,
 *      compressor carries, compressed entries).
 *   5. Forced draft patterns (draft_override): truth-oracle drafts (full
 *      accept + bonus), garbage drafts (all reject), mixed (partial) —
 *      streams and state digests still bitwise == non-spec, accept stats
 *      match the pattern.
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
#define APUS_MTP_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
#include "mtp.h"

#define FIX "tests/m8/fixtures"

/* --- npy reader (same as tests/m5/test_full.c) ----------------------------*/

typedef struct {
    char descr[8];
    int ndim;
    int64_t shape[8];
    unsigned char *data;
    size_t nbytes;
} Npy;

static int npy_load(const char *path, Npy *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);
    if (sz < 10 || memcmp(buf, "\x93NUMPY", 6)) { free(buf); return -1; }
    int major = buf[6];
    size_t hlen, off;
    if (major == 1) { hlen = buf[8] | (buf[9] << 8); off = 10; }
    else { hlen = buf[8] | (buf[9] << 8) | (buf[10] << 16) | ((size_t)buf[11] << 24); off = 12; }
    char *hdr = malloc(hlen + 1);
    memcpy(hdr, buf + off, hlen);
    hdr[hlen] = 0;
    memset(n, 0, sizeof *n);
    char *d = strstr(hdr, "'descr'");
    if (!d) { free(hdr); free(buf); return -1; }
    d = strchr(d, '\'');
    d = strchr(d + 1, '\'');
    d = strchr(d + 1, '\'');
    char *e = strchr(d + 1, '\'');
    size_t dl = (size_t)(e - d - 1) < 7 ? (size_t)(e - d - 1) : 7;
    memcpy(n->descr, d + 1, dl);
    char *sh = strstr(hdr, "'shape'");
    sh = strchr(sh, '(') + 1;
    while (*sh && *sh != ')') {
        while (*sh == ' ' || *sh == ',') sh++;
        if (*sh == ')') break;
        n->shape[n->ndim++] = strtoll(sh, &sh, 10);
    }
    n->nbytes = (size_t)sz - off - hlen;
    n->data = malloc(n->nbytes);
    memcpy(n->data, buf + off + hlen, n->nbytes);
    free(hdr);
    free(buf);
    return 0;
}

static void npy_free(Npy *n) { free(n->data); n->data = NULL; }

static size_t npy_nelem(const Npy *n) {
    size_t c = 1;
    for (int i = 0; i < n->ndim; i++) c *= (size_t)n->shape[i];
    return c;
}

static float *load_f32(const char *path, size_t *nelem) {
    Npy n;
    if (npy_load(path, &n)) return NULL;
    size_t c = npy_nelem(&n);
    float *out = malloc(c * sizeof(float));
    int ok = 1;
    if (!strcmp(n.descr, "<f4")) memcpy(out, n.data, c * sizeof(float));
    else if (!strcmp(n.descr, "<f8")) {
        for (size_t i = 0; i < c; i++) out[i] = (float)((double *)n.data)[i];
    } else ok = 0;
    npy_free(&n);
    if (!ok) { free(out); return NULL; }
    if (nelem) *nelem = c;
    return out;
}

static int64_t *load_i64(const char *path, size_t *nelem) {
    Npy n;
    if (npy_load(path, &n)) return NULL;
    size_t c = npy_nelem(&n);
    int64_t *out = malloc(c * sizeof(int64_t));
    int ok = 1;
    if (!strcmp(n.descr, "<i8")) memcpy(out, n.data, c * sizeof(int64_t));
    else if (!strcmp(n.descr, "<i4")) {
        for (size_t i = 0; i < c; i++) out[i] = ((int32_t *)n.data)[i];
    } else ok = 0;
    npy_free(&n);
    if (!ok) { free(out); return NULL; }
    if (nelem) *nelem = c;
    return out;
}

/* --- checks ----------------------------------------------------------------*/

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

/* rel error + argmax flips with a golden-gap excuse policy (m5): a flip is
 * excused iff the golden top1-top2 gap <= gap_tol. Returns unexcused
 * flips; *rel_out gets max|diff|/max|golden|. */
static size_t cmp_logits_gap(const float *g, const float *a, size_t rows,
                             size_t V, double gap_tol, double *rel_out,
                             size_t *flips_out) {
    double mx = 0.0, sc = 0.0;
    for (size_t i = 0; i < rows * V; i++) {
        double d = fabs((double)g[i] - (double)a[i]);
        if (d > mx) mx = d;
        double ag = fabs((double)g[i]);
        if (ag > sc) sc = ag;
    }
    *rel_out = sc > 0 ? mx / sc : 0.0;
    size_t flips = 0, unexcused = 0;
    for (size_t t = 0; t < rows; t++) {
        const float *gr = g + t * V, *ar = a + t * V;
        size_t bg = 0, ba = 0;
        float g1 = -1e30f, g2 = -1e30f;
        for (size_t i = 0; i < V; i++) {
            if (gr[i] > gr[bg]) bg = i;
            if (ar[i] > ar[ba]) ba = i;
            if (gr[i] > g1) { g2 = g1; g1 = gr[i]; }
            else if (gr[i] > g2) g2 = gr[i];
        }
        if (bg != ba) {
            flips++;
            if ((double)(g1 - g2) > gap_tol)
                unexcused++;
        }
    }
    if (flips_out) *flips_out = flips;
    return unexcused;
}

/* --- state digest (rollback check) -----------------------------------------*/

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t comp_digest(const ApusCompS *c, uint64_t h) {
    if (!c->kv) return h;
    size_t cb = (size_t)c->rows * c->cols * sizeof(float);
    h = fnv1a(&c->nb, sizeof c->nb, h);
    h = fnv1a(c->kv, cb, h);
    h = fnv1a(c->sc, cb, h);
    h = fnv1a(c->cache, (size_t)c->nb * c->d * sizeof(float), h);
    return h;
}

static uint64_t state_digest(const ApusModel *m, ApusModelState *st) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a(&st->pos, sizeof st->pos, h);
    for (int i = 0; i < m->n_layers; i++) {
        ApusAttnS *as = &st->layers[i].attn;
        size_t wb = (size_t)m->layers[i].acfg.window
                    * m->layers[i].acfg.head_dim * sizeof(float);
        h = fnv1a(&as->pos, sizeof as->pos, h);
        h = fnv1a(as->win, wb, h);
        h = comp_digest(&as->comp, h);
        h = comp_digest(&as->idx_comp, h);
    }
    return h;
}

/* --- drivers ------------------------------------------------------------------*/

static ApusModel g_m;
static ApusMtp g_mt;
static int g_V;

/* Non-speculative reference: prefill + `steps` sample/forward iterations.
 * out gets the emitted tokens; the state ends fed through n+steps-1. */
static void nonspec_run(ApusModelState *st, const int64_t *ids, int n,
                        int steps, float temp, float top_p, uint64_t seed,
                        int *out) {
    float *logits = malloc((size_t)g_V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)g_V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    apus_model_forward(&g_m, st, ids, n, logits, 0);
    for (int k = 0; k < steps; k++) {
        int t = apus_sample(logits, (size_t)g_V, temp, top_p, &rng, scratch);
        out[k] = t;
        int64_t nx = t;
        apus_model_forward(&g_m, st, &nx, 1, logits, 0);
    }
    free(logits);
    free(scratch);
}

typedef struct {
    const int64_t *truth;   /* full true sequence (prompt + stream) */
    int V;
    int mode;               /* 0 = truth, 1 = garbage, 2 = partial (d2 ok,
                               d3+ wrong) */
} DraftCtx;

static void draft_hook(void *v, int64_t q, int64_t *drafts, int n) {
    DraftCtx *dc = v;
    for (int i = 0; i < n; i++) {
        int t = (int)dc->truth[q + i];
        if (dc->mode == 1 || (dc->mode == 2 && i >= 2))
            t = (t + 1) % dc->V;    /* != truth[q+i], so always rejected */
        drafts[i] = t;
    }
}

/* Speculative run. use_mtp: the real MTP head drafts; otherwise dc hooks
 * forced drafts. Returns via out/emitted (stream may overshoot target to
 * step granularity; *emitted is the full emitted count). */
static void spec_run(ApusModelState *st, ApusMtpState *mst,
                     const int64_t *ids, int n, int target, int depth,
                     float temp, float top_p, uint64_t seed,
                     int use_mtp, DraftCtx *dc, int *out, int *emitted,
                     uint64_t *acc, uint64_t *off) {
    void *scratch = malloc(apus_sample_scratch_size((size_t)g_V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    ApusSpec sp;
    apus_spec_init(&sp, &g_m, st, use_mtp ? &g_mt : NULL,
                   use_mtp ? mst : NULL, depth, temp, top_p, &rng, scratch);
    if (!use_mtp) {
        sp.draft_override = draft_hook;
        sp.draft_ctx = dc;
    }
    apus_spec_prefill(&sp, ids, n);
    int buf[64], n_out = 0;
    while ((int)sp.emitted < target) {
        int ne = apus_spec_step(&sp, buf, 64);
        if (ne <= 0) break;
        for (int i = 0; i < ne; i++) out[n_out++] = buf[i];
    }
    *emitted = n_out;
    if (acc) *acc = sp.accepted;
    if (off) *off = sp.offered;
    apus_spec_free(&sp);
    free(scratch);
}

/* --- golden tests --------------------------------------------------------------*/

static void test_mtp_prefill(double rel_bound, double gap_tol) {
    size_t n_ids, n_l;
    int64_t *ids = load_i64(FIX "/golden/mtp_prefill/input_ids.npy", &n_ids);
    float *gold = load_f32(FIX "/golden/mtp_prefill/logits_f32.npy", &n_l);
    CHECK(ids && gold, "mtp_prefill goldens load");
    if (!ids || !gold) return;
    int s = (int)n_ids;
    size_t hcdim = (size_t)g_m.cfg.hc_mult * g_m.cfg.dim;
    float *h_all = malloc((size_t)s * hcdim * sizeof(float));
    float *logits = malloc((size_t)g_V * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward_h(&g_m, &st, ids, s, logits, 0, h_all);
    /* per-prefix replay: MTP logits row t == golden row t (the engine's
     * batched replay is chunk-invariant, so prefix replays match it) */
    float *got = malloc((size_t)s * g_V * sizeof(float));
    for (int t = 1; t <= s; t++) {
        ApusMtpState mst;
        apus_mtp_state_init(&mst, &g_mt);
        apus_mtp_forward(&g_mt, &mst, h_all, ids, t, 0, logits, NULL);
        memcpy(got + (size_t)(t - 1) * g_V, logits,
               (size_t)g_V * sizeof(float));
        apus_mtp_state_free(&mst, &g_mt);
    }
    double rel;
    size_t flips;
    size_t unexcused = cmp_logits_gap(gold, got, (size_t)s, (size_t)g_V,
                                      gap_tol, &rel, &flips);
    printf("  mtp_prefill: rel %.3g (bound %.3g), argmax flips %zu/%d"
           " (unexcused %zu)\n", rel, rel_bound, flips, s, unexcused);
    CHECK(rel <= rel_bound, "mtp_prefill rel %.3g > %.3g", rel, rel_bound);
    CHECK(unexcused == 0, "mtp_prefill %zu unexcused argmax flips", unexcused);
    apus_model_state_free(&st, &g_m);
    free(h_all);
    free(logits);
    free(got);
    free(ids);
    free(gold);
}

static void test_mtp_chain(double rel_bound, double gap_tol) {
    size_t n_ids, n_dr, n_l;
    int64_t *ids = load_i64(FIX "/golden/mtp_chain/prompt_ids.npy", &n_ids);
    int64_t *drafts_g = load_i64(FIX "/golden/mtp_chain/drafts_f32.npy", &n_dr);
    float *lg_g = load_f32(FIX "/golden/mtp_chain/logits_f32.npy", &n_l);
    CHECK(ids && drafts_g && lg_g, "mtp_chain goldens load");
    if (!ids || !drafts_g || !lg_g) return;
    int s = (int)n_ids, nd = (int)n_dr;
    size_t hcdim = (size_t)g_m.cfg.hc_mult * g_m.cfg.dim;
    float *h_all = malloc((size_t)s * hcdim * sizeof(float));
    float *logits = malloc((size_t)g_V * sizeof(float));
    float *hmtp = malloc(hcdim * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward_h(&g_m, &st, ids, s, logits, 0, h_all);
    ApusMtpState mst;
    apus_mtp_state_init(&mst, &g_mt);
    apus_mtp_forward(&g_mt, &mst, h_all, ids, s, 0, logits, hmtp);
    float *got_l = malloc((size_t)nd * g_V * sizeof(float));
    int64_t got_d[8];
    for (int i = 0; i < nd; i++) {
        got_d[i] = apus_sample_argmax(logits, (size_t)g_V);
        memcpy(got_l + (size_t)i * g_V, logits, (size_t)g_V * sizeof(float));
        if (i + 1 < nd)
            apus_mtp_forward(&g_mt, &mst, hmtp, &got_d[i], 1,
                             (int64_t)s + i, logits, hmtp);
    }
    double rel;
    size_t flips;
    size_t unexcused = cmp_logits_gap(lg_g, got_l, (size_t)nd, (size_t)g_V,
                                      gap_tol, &rel, &flips);
    int dmatch = 1;
    for (int i = 0; i < nd; i++)
        if (got_d[i] != drafts_g[i]) dmatch = 0;
    printf("  mtp_chain: rel %.3g (bound %.3g), logit flips %zu/%d"
           " (unexcused %zu), drafts %s\n",
           rel, rel_bound, flips, nd, unexcused,
           dmatch ? "match" : "DIVERGE (near-tie class)");
    CHECK(rel <= rel_bound, "mtp_chain rel %.3g > %.3g", rel, rel_bound);
    CHECK(unexcused == 0, "mtp_chain %zu unexcused argmax flips", unexcused);
    apus_mtp_state_free(&mst, &g_mt);
    apus_model_state_free(&st, &g_m);
    free(h_all);
    free(logits);
    free(hmtp);
    free(got_l);
    free(ids);
    free(drafts_g);
    free(lg_g);
}

/* --- equivalence + rollback + forced drafts -------------------------------------*/

#define EQ_TARGET 32
#define EQ_TRUTH_STEPS 40

static void test_equivalence(const int64_t *pids, int n, float temp,
                             float top_p, uint64_t seed, const char *tag) {
    /* reference: non-spec */
    int ref[EQ_TRUTH_STEPS];
    ApusModelState st0;
    apus_model_state_init(&st0, &g_m);
    nonspec_run(&st0, pids, n, EQ_TRUTH_STEPS, temp, top_p, seed, ref);
    /* spec at depths 1/2/3 with the real MTP head */
    for (int depth = 1; depth <= 3; depth++) {
        ApusModelState st;
        ApusMtpState mst;
        apus_model_state_init(&st, &g_m);
        apus_mtp_state_init(&mst, &g_mt);
        int out[128];
        int emitted = 0;
        uint64_t acc = 0, off = 0;
        spec_run(&st, &mst, pids, n, EQ_TARGET, depth, temp, top_p, seed,
                 1, NULL, out, &emitted, &acc, &off);
        int same = emitted >= EQ_TARGET;
        for (int i = 0; i < EQ_TARGET && same; i++)
            if (out[i] != ref[i]) same = 0;
        CHECK(same, "%s: spec depth %d stream != non-spec", tag, depth);
        /* rollback: state after the spec run == state after decoding
         * exactly `emitted` tokens non-speculatively (rejected drafts
         * must leave no trace) */
        int ref2[128];
        ApusModelState st_ref;
        apus_model_state_init(&st_ref, &g_m);
        nonspec_run(&st_ref, pids, n, emitted, temp, top_p, seed, ref2);
        uint64_t d_spec = state_digest(&g_m, &st);
        uint64_t d_ref = state_digest(&g_m, &st_ref);
        CHECK(d_spec == d_ref,
              "%s: spec depth %d state digest %llx != non-spec %llx",
              tag, depth, (unsigned long long)d_spec,
              (unsigned long long)d_ref);
        int same2 = 1;
        for (int i = 0; i < emitted; i++)
            if (out[i] != ref2[i]) same2 = 0;
        CHECK(same2, "%s: spec depth %d full stream != non-spec", tag, depth);
        printf("  %s depth %d: emitted %d, accept %llu/%llu, stream %s,"
               " state %s\n", tag, depth, emitted,
               (unsigned long long)acc, (unsigned long long)off,
               same ? "BITWISE" : "DIFFERS",
               d_spec == d_ref ? "BITWISE" : "DIFFERS");
        apus_mtp_state_free(&mst, &g_mt);
        apus_model_state_free(&st, &g_m);
        apus_model_state_free(&st_ref, &g_m);
    }
    apus_model_state_free(&st0, &g_m);
}

static void test_forced(const int64_t *pids, int n) {
    /* truth = prompt + non-spec greedy stream */
    int stream[EQ_TRUTH_STEPS];
    ApusModelState st0;
    apus_model_state_init(&st0, &g_m);
    nonspec_run(&st0, pids, n, EQ_TRUTH_STEPS, 0.0f, 1.0f, 0, stream);
    apus_model_state_free(&st0, &g_m);
    int64_t truth[128];
    for (int i = 0; i < n; i++) truth[i] = pids[i];
    for (int i = 0; i < EQ_TRUTH_STEPS; i++) truth[n + i] = stream[i];

    static const struct { int mode, depth; const char *name; } pats[] = {
        { 0, 2, "truth-oracle d2 (full accept+bonus)" },
        { 0, 3, "truth-oracle d3 (full accept+bonus)" },
        { 1, 2, "garbage d2 (all reject)" },
        { 1, 3, "garbage d3 (all reject)" },
        { 2, 3, "mixed d3 (partial accept)" },
    };
    for (size_t p = 0; p < sizeof pats / sizeof pats[0]; p++) {
        DraftCtx dc = { truth, g_V, pats[p].mode };
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        int out[128];
        int emitted = 0;
        uint64_t acc = 0, off = 0;
        spec_run(&st, NULL, pids, n, EQ_TARGET, pats[p].depth, 0.0f, 1.0f,
                 0, 0, &dc, out, &emitted, &acc, &off);
        int same = emitted >= EQ_TARGET;
        for (int i = 0; i < EQ_TARGET && same; i++)
            if (out[i] != stream[i]) same = 0;
        CHECK(same, "forced %s: stream != non-spec", pats[p].name);
        /* state rollback digest */
        int ref2[128];
        ApusModelState st_ref;
        apus_model_state_init(&st_ref, &g_m);
        nonspec_run(&st_ref, pids, n, emitted, 0.0f, 1.0f, 0, ref2);
        uint64_t d_spec = state_digest(&g_m, &st);
        uint64_t d_ref = state_digest(&g_m, &st_ref);
        CHECK(d_spec == d_ref, "forced %s: state digest differs",
              pats[p].name);
        /* accept stats match the pattern */
        if (pats[p].mode == 0) CHECK(acc == off && off > 0,
              "forced %s: expected 100%% accept, got %llu/%llu",
              pats[p].name, (unsigned long long)acc, (unsigned long long)off);
        if (pats[p].mode == 1) CHECK(acc == 0 && off > 0,
              "forced %s: expected 0%% accept, got %llu/%llu",
              pats[p].name, (unsigned long long)acc, (unsigned long long)off);
        if (pats[p].mode == 2) CHECK(acc > 0 && acc < off,
              "forced %s: expected partial accept, got %llu/%llu",
              pats[p].name, (unsigned long long)acc, (unsigned long long)off);
        printf("  forced %s: emitted %d, accept %llu/%llu, stream %s,"
               " state %s\n", pats[p].name, emitted,
               (unsigned long long)acc, (unsigned long long)off,
               same ? "BITWISE" : "DIFFERS",
               d_spec == d_ref ? "BITWISE" : "DIFFERS");
        apus_model_state_free(&st, &g_m);
        apus_model_state_free(&st_ref, &g_m);
    }
}

int main(void) {
    printf("test_m8: M8 MTP speculative decoding verification\n");
    char err[256];
    if (apus_model_load(&g_m, FIX, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        return 1;
    }
    g_V = g_m.cfg.vocab_size;
    printf("  model: %d layers, dim %d, vocab %d, n_mtp %d (ratio %d)\n",
           g_m.n_layers, g_m.cfg.dim, g_V, g_m.n_mtp, g_m.mtp_ratio);
    CHECK(g_m.n_mtp == 1, "fixture model must declare one MTP block");
    if (apus_mtp_load(&g_mt, &g_m, 0, err, sizeof err)) {
        fprintf(stderr, "mtp load: %s\n", err);
        return 1;
    }

    /* golden bounds: measured values recorded in tests/m8/README.md
     * (O2: rel 0.184 / 0.0396; O1/ubsan: 0.2 / 0.184 — the chain's first
     * row IS the replay logits, so its bound must cover the prefill
     * divergence class at both optimization levels). C-vs-f32 sits well
     * inside the oracle's own f32-vs-f64 envelope (mtp_prefill self-check
     * rel 0.31, flip 21% — the random-weight near-tie cascade class). */
    test_mtp_prefill(2.5e-1, 0.5);
    test_mtp_chain(2.5e-1, 0.5);

    size_t n_p;
    int64_t *pids = load_i64(FIX "/golden/spec_episode/prompt_ids.npy", &n_p);
    CHECK(pids != NULL, "spec_episode prompt load");
    if (pids) {
        test_equivalence(pids, (int)n_p, 0.0f, 1.0f, 0, "greedy");
        test_equivalence(pids, (int)n_p, 0.8f, 0.95f, 12345, "sampled");
        test_forced(pids, (int)n_p);
        free(pids);
    }

    apus_mtp_free(&g_mt);
    apus_model_free(&g_m);
    printf("test_m8: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
