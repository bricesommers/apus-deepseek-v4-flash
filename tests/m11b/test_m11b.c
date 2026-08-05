/*
 * tests/m11b/test_m11b.c — M11b DSpark speculative decoding verification.
 *
 * Loads the M11a mini-model WITH 3 DSpark stages (tests/m11a/fixtures;
 * regenerate with `make golden-m11a`) and checks, under the m5/m8 margin
 * policy (tests/m11a/README.md "Tolerancing guidance for M11b"):
 *
 *   1. join_prefill_len{24,140} — per-position main_hidden (target-layer
 *      hc-means), stage-0 main_x, and the 3 stage KV rings after prefill
 *      vs the oracle f32 goldens (rel bounds; rings at the bf16 code-step
 *      tolerance, the m4b chunk-noise class).
 *   2. draft_rounds_len24 (3 teacher-forced rounds) + draft_round_len140
 *      (ring wrap, full-window topk branch) — drafts, logits_base/final,
 *      markov_bias/embed, conf_hidden, confidence, per-stage block outputs
 *      vs goldens. Argmax flips excused only at golden margin <= 0.5.
 *   3. EQUIVALENCE (the hard gate): DSpark spec decode vs non-spec decode,
 *      greedy AND seeded-sampled — emitted streams BITWISE identical, and
 *      the rollback state digest (canonical oracle_dspark.state_digest
 *      layout) spec == non-spec. Free-running C-vs-oracle streams are
 *      reported but are NOT a gate (random-weight near-tie cascade, same
 *      policy as m8).
 *   4. Forced draft patterns (draft_override): all_accept (accepted [5],
 *      bonus every round), all_reject ([0], full rollback), mixed ([2] on
 *      the len140 prompt — rings wrap mid-episode). Exact per-round accept
 *      counts, streams + digests == non-spec.
 *   5. Stage-ring catch-up (D13): after the forced_mixed run, every stage
 *      ring slot holds the newest true position's KV (bitwise for decode
 *      positions; bf16 code-step tolerance for prompt positions).
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
#define APUS_DSPARK_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
#include "mtp.h"
#include "dspark.h"

#define FIX "tests/m11a/fixtures"
#define GOLD FIX "/golden"

/* --- npy reader (same as tests/m8/test_m8.c) -----------------------------*/

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

/* max|diff| / max|golden| over n elements */
static double rel_err(const float *g, const float *a, size_t n) {
    double mx = 0.0, sc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)g[i] - (double)a[i]);
        if (d > mx) mx = d;
        double ag = fabs((double)g[i]);
        if (ag > sc) sc = ag;
    }
    return sc > 0 ? mx / sc : 0.0;
}

/* rel error + argmax flips with a golden-gap excuse policy (m5/m8): a flip
 * is excused iff the golden top1-top2 gap <= gap_tol. Returns unexcused
 * flips; *rel_out gets max|diff|/max|golden|. */
static size_t cmp_logits_gap(const float *g, const float *a, size_t rows,
                             size_t V, double gap_tol, double *rel_out,
                             size_t *flips_out) {
    *rel_out = rel_err(g, a, rows * V);
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

/* bf16 code-step elementwise tolerance (the m4b chunk-noise class; same
 * rule as tests/m11a/check_oracle.py's stage-window catch-up check):
 * |diff| <= 2 * 0.0078125 * max(|exp|, 0.5). Returns the worst ratio. */
static double codestep_ratio(const float *exp, const float *got, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double tol = 2 * 0.0078125 * fmax(fabs((double)exp[i]), 0.5);
        double r = fabs((double)exp[i] - (double)got[i]) / tol;
        if (r > worst) worst = r;
    }
    return worst;
}

/* --- canonical state digest (oracle_dspark.state_digest layout) -------------
 * FNV-1a 64 over: per main layer (pos i64, win, [comp cache-used, kv, sc,
 * [idx cache-used, kv, sc]]), then per DSpark stage (pos i64, win). */

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t comp_digest(const ApusCompS *c, uint64_t h) {
    if (!c->kv) return h;
    size_t cb = (size_t)c->rows * c->cols * sizeof(float);
    h = fnv1a(c->cache, (size_t)c->nb * c->d * sizeof(float), h);
    h = fnv1a(c->kv, cb, h);
    h = fnv1a(c->sc, cb, h);
    return h;
}

static uint64_t digest_main(const ApusModel *m, ApusModelState *st,
                            uint64_t h) {
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

static uint64_t digest_stages(const ApusDspark *ds, ApusDsparkState *dst,
                              uint64_t h) {
    for (int k = 0; k < ds->n_stages; k++) {
        ApusAttnS *as = &dst->stages[k].attn;
        size_t wb = (size_t)ds->stages[k].acfg.window
                    * ds->stages[k].acfg.head_dim * sizeof(float);
        h = fnv1a(&as->pos, sizeof as->pos, h);
        h = fnv1a(as->win, wb, h);
    }
    return h;
}

/* --- drivers ------------------------------------------------------------------*/

static ApusModel g_m;
static ApusDspark g_ds;
static int g_V;
static int g_B;         /* block_size (5) */
static int g_dim, g_hc, g_mhd, g_rank;

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
    int mode;               /* 0 = all accept, 1 = all reject, 2 = mixed
                               (draft 3 of every 5 corrupted) */
} DraftCtx;

static void draft_hook(void *v, int64_t f, int64_t anchor, int64_t *drafts,
                       int n) {
    DraftCtx *dc = v;
    (void)anchor;
    for (int j = 0; j < n; j++) {
        int t = (int)dc->truth[f + 2 + j];
        if (dc->mode == 1 || (dc->mode == 2 && j == 2))
            t = (t + 1) % dc->V;    /* != truth, so rejected at j */
        drafts[j] = t;
    }
}

/* --- 1. join_prefill ----------------------------------------------------------*/

static void test_join_prefill(const char *tag, double mh_bound,
                              double mx_bound) {
    char path[256];
    size_t n_ids, n_mh, n_mx;
    snprintf(path, sizeof path, GOLD "/%s/input_ids.npy", tag);
    int64_t *ids = load_i64(path, &n_ids);
    snprintf(path, sizeof path, GOLD "/%s/main_hidden_f32.npy", tag);
    float *mh_g = load_f32(path, &n_mh);
    snprintf(path, sizeof path, GOLD "/%s/main_x_f32.npy", tag);
    float *mx_g = load_f32(path, &n_mx);
    CHECK(ids && mh_g && mx_g, "%s goldens load", tag);
    if (!ids || !mh_g || !mx_g) goto out;
    {
        int s = (int)n_ids;
        float *logits = malloc((size_t)g_V * sizeof(float));
        float *mh = malloc((size_t)s * g_mhd * sizeof(float));
        float *mx = malloc((size_t)s * g_dim * sizeof(float));
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        apus_model_forward_mh(&g_m, &st, ids, s, logits, 0, mh);
        double rel = rel_err(mh_g, mh, (size_t)s * g_mhd);
        printf("  %s main_hidden: rel %.3g (bound %.3g)\n", tag, rel,
               mh_bound);
        CHECK(rel <= mh_bound, "%s main_hidden rel %.3g > %.3g", tag, rel,
              mh_bound);
        apus_dspark_stage0_project(&g_ds, mh, s, mx);
        rel = rel_err(mx_g, mx, (size_t)s * g_dim);
        printf("  %s main_x: rel %.3g (bound %.3g)\n", tag, rel, mx_bound);
        CHECK(rel <= mx_bound, "%s main_x rel %.3g > %.3g", tag, rel,
              mx_bound);
        ApusDsparkState dst;
        apus_dspark_state_init(&dst, &g_ds);
        apus_dspark_prefill(&g_ds, &dst, mx, s);
        int d = g_ds.stages[0].acfg.head_dim;
        int win = g_ds.stages[0].acfg.window;
        for (int k = 0; k < g_ds.n_stages; k++) {
            snprintf(path, sizeof path, GOLD "/%s/stage%d_win_f32.npy", tag,
                     k);
            float *wg = load_f32(path, NULL);
            CHECK(wg != NULL, "%s stage%d win load", tag, k);
            if (!wg) continue;
            /* C-vs-oracle: rel bound (the ring rows inherit the main_x
             * cascade through wkv/rope/QAT; the bf16 code-step rule is
             * for C-INTERNAL recompute consistency — see the catch-up
             * check, which is bitwise) */
            double rel = rel_err(wg, dst.stages[k].attn.win,
                                 (size_t)win * d);
            printf("  %s stage%d ring: rel %.3g (bound 0.50)\n", tag, k,
                   rel);
            CHECK(rel <= 0.50, "%s stage%d ring rel %.3g > 0.50", tag, k,
                  rel);
            free(wg);
        }
        apus_dspark_state_free(&dst, &g_ds);
        apus_model_state_free(&st, &g_m);
        free(logits);
        free(mh);
        free(mx);
    }
out:
    free(ids);
    free(mh_g);
    free(mx_g);
}

/* --- 2. draft rounds ------------------------------------------------------------*/

/* Compare one teacher-forced draft round against its golden dir.
 * Rows past the first draft flip follow a different sampled chain (the
 * markov bias embeds the PRECEDING token), so bias/final/embed/confidence
 * are compared only over the matching prefix (rows 0..first_diverge).
 * Bounds are calibrated to the oracle's OWN f32-vs-f64 envelope
 * (tests/m11a/check_oracle.py divergence report; tests/m11b/README.md). */
static void check_draft_round(const char *dir, int64_t anchor, int64_t f,
                              const float *mh_last) {
    char path[256];
    int B = g_B, V = g_V, rank = g_rank;
    size_t hcdim = (size_t)g_hc * g_dim;
    ApusDsparkDbg dbg;
    memset(&dbg, 0, sizeof dbg);
    float *logits_base = malloc((size_t)B * V * sizeof(float));
    float *markov_bias = malloc((size_t)B * V * sizeof(float));
    float *logits_final = malloc((size_t)B * V * sizeof(float));
    float *markov_embed = malloc((size_t)B * rank * sizeof(float));
    float *conf_hidden = malloc((size_t)B * g_dim * sizeof(float));
    float *confidence = malloc((size_t)B * sizeof(float));
    float *stage_h[APUS_DSPARK_MAX_STAGES] = { 0 };
    for (int k = 0; k < g_ds.n_stages; k++)
        stage_h[k] = malloc((size_t)B * hcdim * sizeof(float));
    dbg.logits_base = logits_base;
    dbg.markov_bias = markov_bias;
    dbg.logits_final = logits_final;
    dbg.markov_embed = markov_embed;
    dbg.conf_hidden = conf_hidden;
    dbg.confidence = confidence;
    for (int k = 0; k < g_ds.n_stages; k++) dbg.stage_h[k] = stage_h[k];

    extern ApusDsparkState *g_round_dst;   /* set by the caller */
    float *mx = malloc((size_t)g_dim * sizeof(float));
    apus_dspark_stage0_project(&g_ds, mh_last, 1, mx);
    int64_t *drafts = malloc((size_t)B * sizeof(int64_t));
    ApusRng drng;
    apus_rng_seed(&drng, 4242);      /* greedy rounds: no draft draws */
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    apus_dspark_draft_round(&g_ds, g_round_dst, anchor, mx, f, 0.0f, &drng,
                            scratch, drafts, &dbg);
    free(scratch);
    free(mx);

    /* the goldens (len140 carries only a subset — each is optional) */
    snprintf(path, sizeof path, "%s/drafts_f32.npy", dir);
    int64_t *dr_g = load_i64(path, NULL);
    snprintf(path, sizeof path, "%s/margins_f32.npy", dir);
    float *mg_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/logits_base_f32.npy", dir);
    float *lb_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/markov_bias_f32.npy", dir);
    float *mb_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/logits_final_f32.npy", dir);
    float *lf_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/markov_embed_f32.npy", dir);
    float *me_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/conf_hidden_f32.npy", dir);
    float *ch_g = load_f32(path, NULL);
    snprintf(path, sizeof path, "%s/confidence_f32.npy", dir);
    float *cf_g = load_f32(path, NULL);
    CHECK(dr_g && lf_g, "%s goldens load (drafts/logits_final)", dir);
    if (!(dr_g && lf_g)) goto out;

    /* drafts: mismatches excused only in the near-tie class — golden draw
     * margin (or golden top1-top2 gap when margins are not dumped) <=
     * 1.0, OR the C row's own gap <= 0.75. Justification: the oracle's
     * own f32-vs-f64 modes flip at gaps up to 0.71 (m11a README), and
     * the O1 (ubsan) codegen class flips one 0.934-gap draw (measured;
     * the logits_final f32-vs-f64 self-envelope is maxabs 5.0 — a 1.0
     * gap is deep inside this random-weight cascade class). Draft values
     * only affect the acceptance RATE, never the emitted stream. */
    int first_div = B, mism = 0;
    for (int i = 0; i < B; i++) {
        if (drafts[i] == dr_g[i]) continue;
        mism++;
        double mg;
        if (mg_g) {
            mg = (double)mg_g[i];
        } else {
            const float *gr = lf_g + (size_t)i * V;
            float g1 = -1e30f, g2 = -1e30f;
            for (int v = 0; v < V; v++) {
                if (gr[v] > g1) { g2 = g1; g1 = gr[v]; }
                else if (gr[v] > g2) g2 = gr[v];
            }
            mg = (double)(g1 - g2);
        }
        /* C row's own top1-top2 gap (the final logits after the bias) */
        const float *cr = logits_final + (size_t)i * V;
        float c1 = -1e30f, c2 = -1e30f;
        for (int v = 0; v < V; v++) {
            if (cr[v] > c1) { c2 = c1; c1 = cr[v]; }
            else if (cr[v] > c2) c2 = cr[v];
        }
        double cg = (double)(c1 - c2);
        CHECK(mg <= 1.0 || cg <= 0.75,
              "%s draft %d mismatch outside the near-tie class (golden"
              " margin %.3g, C gap %.3g)", dir, i, mg, cg);
        if (first_div == B) first_div = i;
    }
    int mrows = first_div < B ? first_div + 1 : B;   /* matching-context rows */
    printf("  %s drafts: %d/%d mismatch%s (context rows %d/%d)\n", dir,
           mism, B, mism ? " (near-tie class)" : "", mrows, B);
    CHECK(mrows >= 1, "%s: no matching-context rows", dir);

    double rel;
    if (lb_g) {
        /* logits_base: all rows (context-independent of the sampled
         * chain); oracle self-envelope rel 0.33, bound 0.40; argmax flips
         * excused at golden gap <= 0.75 (oracle self-flips to 0.71) */
        size_t flips;
        size_t unx = cmp_logits_gap(lb_g, logits_base, (size_t)B, (size_t)V,
                                    0.75, &rel, &flips);
        printf("  %s logits_base: rel %.3g (bound 0.40), flips %zu/%d"
               " (unexcused %zu)\n", dir, rel, flips, B, unx);
        CHECK(rel <= 0.40 && unx == 0, "%s logits_base rel %.3g unx %zu",
              dir, rel, unx);
    }
    /* matching-context rows only (oracle self-envelope in parentheses) */
    rel = rel_err(lf_g, logits_final, (size_t)mrows * V);
    printf("  %s logits_final[%d]: rel %.3g (bound 1.0; self 0.97)\n", dir,
           mrows, rel);
    CHECK(rel <= 1.0, "%s logits_final rel %.3g > 1.0", dir, rel);
    if (mb_g) {
        rel = rel_err(mb_g, markov_bias, (size_t)mrows * V);
        printf("  %s markov_bias[%d]: rel %.3g (bound 1.3; self 1.25 — "
               "near-cancellation array)\n", dir, mrows, rel);
        CHECK(rel <= 1.3, "%s markov_bias rel %.3g > 1.3", dir, rel);
    }
    if (ch_g) {
        rel = rel_err(ch_g, conf_hidden, (size_t)B * g_dim);
        printf("  %s conf_hidden: rel %.3g (bound 0.40; self 0.36)\n", dir,
               rel);
        CHECK(rel <= 0.40, "%s conf_hidden rel %.3g > 0.40", dir, rel);
    }
    if (me_g) {
        rel = rel_err(me_g, markov_embed, (size_t)mrows * rank);
        printf("  %s markov_embed[%d]: rel %.3g\n", dir, mrows, rel);
        CHECK(rel <= 1e-6, "%s markov_embed rel %.3g (exact lookup rows)",
              dir, rel);
    }
    if (cf_g) {
        rel = rel_err(cf_g, confidence, (size_t)mrows);
        /* M12b: draft_round_len140 is near-tie-chaotic — the deterministic
         * oracle's f32 goldens for this case differ between the macOS and
         * Linux REALIZATIONS of the residual (transcendental/reduction)
         * ulp noise by rel 1.36 (draft-token flips cascade into the
         * confidence head), and the x86 scalar C path lands at rel 1.85
         * vs the ARM/NEON 0.59. Per-platform bound for THIS case, same
         * rule as the M12a-1 FP*_GEMM_ANCHOR anchors; all other cases
         * keep 0.9 (measured 0.15-0.59 on both platforms). */
        double cf_bound = 0.9;
#if !defined(__ARM_NEON)
        if (strstr(dir, "draft_round_len140")) cf_bound = 2.0;
#endif
        printf("  %s confidence[%d]: rel %.3g (bound %.3g; self 0.84)\n",
               dir, mrows, rel, cf_bound);
        CHECK(rel <= cf_bound, "%s confidence rel %.3g > %.3g", dir, rel,
              cf_bound);
    }
    for (int k = 0; k < g_ds.n_stages; k++) {
        snprintf(path, sizeof path, "%s/stage%d_h_f32.npy", dir, k);
        float *sh_g = load_f32(path, NULL);
        CHECK(sh_g != NULL, "%s stage%d_h load", dir, k);
        if (!sh_g) continue;
        rel = rel_err(sh_g, stage_h[k], (size_t)B * hcdim);
        /* oracle self-envelope: stage0 0.25, stage2 0.48 (stage1 between) */
        double bound = k == 0 ? 0.30 : k == g_ds.n_stages - 1 ? 0.55 : 0.40;
        printf("  %s stage%d_h: rel %.3g (bound %.2f)\n", dir, k, rel,
               bound);
        CHECK(rel <= bound, "%s stage%d_h rel %.3g > %.2f", dir, k, rel,
              bound);
        free(sh_g);
    }
out:
    free(dr_g); free(mg_g); free(lb_g); free(mb_g); free(lf_g); free(me_g);
    free(ch_g); free(cf_g);
    free(drafts);
    free(logits_base); free(markov_bias); free(logits_final);
    free(markov_embed); free(conf_hidden); free(confidence);
    for (int k = 0; k < g_ds.n_stages; k++) free(stage_h[k]);
}

ApusDsparkState *g_round_dst;

/* len24: 3 teacher-forced rounds with truth advancement between them. */
static void test_draft_rounds_len24(void) {
    const char *base = GOLD "/draft_rounds_len24";
    char path[256];
    size_t n_ids;
    snprintf(path, sizeof path, "%s/round0/prompt_ids.npy", base);
    int64_t *ids = load_i64(path, &n_ids);
    CHECK(ids != NULL, "draft_rounds_len24 prompt load");
    if (!ids) return;
    int s = (int)n_ids;
    float *logits = malloc((size_t)g_V * sizeof(float));
    float *mh = malloc((size_t)g_mhd * sizeof(float));
    float *mh_all = malloc((size_t)s * g_mhd * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward_mh(&g_m, &st, ids, s, logits, 0, mh_all);
    ApusDsparkState dst;
    apus_dspark_state_init(&dst, &g_ds);
    {
        float *mx_all = malloc((size_t)s * g_dim * sizeof(float));
        apus_dspark_stage0_project(&g_ds, mh_all, s, mx_all);
        apus_dspark_prefill(&g_ds, &dst, mx_all, s);
        free(mx_all);
    }
    g_round_dst = &dst;
    int64_t f = s - 1;
    float *mh_last = malloc((size_t)g_mhd * sizeof(float));
    memcpy(mh_last, mh_all + (size_t)(s - 1) * g_mhd,
           (size_t)g_mhd * sizeof(float));
    for (int r = 0; r < 3; r++) {
        snprintf(path, sizeof path, "%s/round%d", base, r);
        char dir[256];
        snprintf(dir, sizeof dir, "%s", path);
        snprintf(path, sizeof path, "%s/anchor_f32.npy", dir);
        int64_t *anchor_p = load_i64(path, NULL);
        snprintf(path, sizeof path, "%s/start_pos_f32.npy", dir);
        int64_t *sp_p = load_i64(path, NULL);
        CHECK(anchor_p && sp_p, "%s anchor/start_pos load", dir);
        if (!anchor_p || !sp_p) { free(anchor_p); free(sp_p); continue; }
        CHECK(*sp_p == f, "%s start_pos %lld != expected %lld", dir,
              (long long)*sp_p, (long long)f);
        check_draft_round(dir, *anchor_p, f, mh_last);
        /* advance truth: feed the GOLDEN anchor through the main model
         * (teacher-forced — near-tie flips cannot cascade) */
        int64_t tok = *anchor_p;
        apus_model_forward_mh(&g_m, &st, &tok, 1, logits, 0, mh);
        memcpy(mh_last, mh, (size_t)g_mhd * sizeof(float));
        f++;
        free(anchor_p);
        free(sp_p);
    }
    free(mh_last);
    apus_dspark_state_free(&dst, &g_ds);
    apus_model_state_free(&st, &g_m);
    free(logits);
    free(mh);
    free(mh_all);
    free(ids);
    g_round_dst = NULL;
}

static void test_draft_round_len140(void) {
    const char *dir = GOLD "/draft_round_len140";
    char path[256];
    size_t n_ids;
    snprintf(path, sizeof path, "%s/prompt_ids.npy", dir);
    int64_t *ids = load_i64(path, &n_ids);
    snprintf(path, sizeof path, "%s/anchor_f32.npy", dir);
    int64_t *anchor_p = load_i64(path, NULL);
    snprintf(path, sizeof path, "%s/start_pos_f32.npy", dir);
    int64_t *sp_p = load_i64(path, NULL);
    CHECK(ids && anchor_p && sp_p, "draft_round_len140 goldens load");
    if (!(ids && anchor_p && sp_p)) goto out;
    {
        int s = (int)n_ids;
        CHECK(*sp_p == s - 1, "draft_round_len140 start_pos %lld != %d",
              (long long)*sp_p, s - 1);
        float *logits = malloc((size_t)g_V * sizeof(float));
        float *mh_all = malloc((size_t)s * g_mhd * sizeof(float));
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        apus_model_forward_mh(&g_m, &st, ids, s, logits, 0, mh_all);
        ApusDsparkState dst;
        apus_dspark_state_init(&dst, &g_ds);
        float *mx_all = malloc((size_t)s * g_dim * sizeof(float));
        apus_dspark_stage0_project(&g_ds, mh_all, s, mx_all);
        apus_dspark_prefill(&g_ds, &dst, mx_all, s);
        free(mx_all);
        g_round_dst = &dst;
        check_draft_round(dir, *anchor_p, *sp_p,
                          mh_all + (size_t)(s - 1) * g_mhd);
        g_round_dst = NULL;
        apus_dspark_state_free(&dst, &g_ds);
        apus_model_state_free(&st, &g_m);
        free(logits);
        free(mh_all);
    }
out:
    free(ids);
    free(anchor_p);
    free(sp_p);
}

/* --- 3. spec episodes (THE HARD GATE) ------------------------------------------*/

/* Run a DSpark spec episode until >= target tokens emitted. Returns the
 * full emitted count (the last round is not truncated); out gets the
 * stream (cap >= target + B + 1). */
static int spec_run(ApusModelState *st, ApusDsparkState *dst,
                    const int64_t *ids, int n, int target, float temp,
                    float top_p, uint64_t seed, DraftCtx *dc, int *out,
                    uint64_t *acc, uint64_t *off, uint64_t *bonus,
                    int max_rounds) {
    void *scratch = malloc(apus_sample_scratch_size((size_t)g_V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    ApusDspec sp;
    apus_dspec_init(&sp, &g_m, st, &g_ds, dst, temp, top_p, &rng,
                    seed + 0x9E3779B97F4A7C15ull, scratch);
    if (dc) {
        sp.draft_override = draft_hook;
        sp.draft_ctx = dc;
    }
    apus_dspec_prefill(&sp, ids, n);
    int n_out = 0, rounds = 0;
    int buf[16];
    while ((int)sp.emitted < target && rounds < max_rounds) {
        int ne = apus_dspec_step(&sp, buf, 16);
        if (ne <= 0) break;
        for (int i = 0; i < ne; i++) out[n_out++] = buf[i];
        rounds++;
    }
    if (acc) *acc = sp.accepted;
    if (off) *off = sp.offered;
    if (bonus) *bonus = sp.bonus_rounds;
    apus_dspec_free(&sp);
    free(scratch);
    return n_out;
}

static void test_episode(const char *tag, float temp, float top_p,
                         uint64_t seed, int target, int max_rounds) {
    char path[256];
    size_t n_ids, n_tok;
    snprintf(path, sizeof path, GOLD "/%s/prompt_ids.npy", tag);
    int64_t *ids = load_i64(path, &n_ids);
    snprintf(path, sizeof path, GOLD "/%s/tokens_spec_f32.npy", tag);
    int64_t *tok_g = load_i64(path, &n_tok);
    CHECK(ids && tok_g, "%s goldens load", tag);
    if (!(ids && tok_g)) { free(ids); free(tok_g); return; }
    /* the gate runs LONGER than the golden episode (the near-tie cascade
     * makes natural acceptance low, so the golden's 8 rounds cover few
     * tokens; the bitwise gate needs more) */
    int cap = target + g_B + 2;
    int *out = malloc((size_t)cap * sizeof(int));
    int *ref = malloc((size_t)cap * sizeof(int));

    ApusModelState st;
    ApusDsparkState dst;
    apus_model_state_init(&st, &g_m);
    apus_dspark_state_init(&dst, &g_ds);
    uint64_t acc = 0, off = 0, bonus = 0;
    int emitted = spec_run(&st, &dst, ids, (int)n_ids, target, temp, top_p,
                           seed, NULL, out, &acc, &off, &bonus, max_rounds);

    /* the hard gate: C-spec stream BITWISE == C-non-spec stream of the
     * same length, and the main-state digest after rollback == non-spec */
    ApusModelState st_ref;
    apus_model_state_init(&st_ref, &g_m);
    nonspec_run(&st_ref, ids, (int)n_ids, emitted, temp, top_p, seed, ref);
    int same = 1;
    for (int i = 0; i < emitted; i++)
        if (out[i] != ref[i]) same = 0;
    CHECK(same, "%s: spec stream != non-spec (%d tokens)", tag, emitted);
    uint64_t d_spec = digest_main(&g_m, &st, 14695981039346656037ull);
    uint64_t d_ref = digest_main(&g_m, &st_ref, 14695981039346656037ull);
    CHECK(d_spec == d_ref,
          "%s: spec state digest %016llx != non-spec %016llx", tag,
          (unsigned long long)d_spec, (unsigned long long)d_ref);
    /* full state (main + stage rings) in the canonical
     * oracle_dspark.state_digest layout — deterministic across thread
     * counts (diffed by the Makefile); the VALUES cannot equal the numpy
     * goldens (different f32 numerics class — documented, not a gate) */
    uint64_t d_full = digest_stages(&g_ds, &dst, d_spec);
    /* informational: free-running C stream vs the oracle golden prefix
     * (NOT a gate — random-weight near-tie cascade, m8 policy) */
    int gsame = 1;
    for (int i = 0; i < (int)n_tok && i < emitted && gsame; i++)
        if (out[i] != (int)tok_g[i]) gsame = 0;
    printf("  %s: emitted %d (target %d), accept %llu/%llu (%.1f%%), "
           "bonus %llu, stream vs C-nonspec %s, digest %s, vs oracle %s,"
           " fulldigest %016llx\n",
           tag, emitted, target,
           (unsigned long long)acc, (unsigned long long)off,
           off ? 100.0 * (double)acc / (double)off : 0.0,
           (unsigned long long)bonus,
           same ? "BITWISE" : "DIFFERS",
           d_spec == d_ref ? "BITWISE" : "DIFFERS",
           gsame ? "match" : "diverges (near-tie class, not a gate)",
           (unsigned long long)d_full);
    apus_model_state_free(&st_ref, &g_m);
    apus_dspark_state_free(&dst, &g_ds);
    apus_model_state_free(&st, &g_m);
    free(out);
    free(ref);
    free(ids);
    free(tok_g);
}

/* --- 4. forced patterns ---------------------------------------------------------*/

#define FORCED_ROUNDS 6

/* Returns the spec end-state for the catch-up test (caller frees). */
static void test_forced(const char *tag, int mode, int expect,
                        int expect_bonus, int do_catchup) {
    char path[256];
    size_t n_ids, n_tok;
    snprintf(path, sizeof path, GOLD "/%s/prompt_ids.npy", tag);
    int64_t *ids = load_i64(path, &n_ids);
    snprintf(path, sizeof path, GOLD "/%s/tokens_spec_f32.npy", tag);
    int64_t *tok_g = load_i64(path, &n_tok);
    CHECK(ids && tok_g, "%s goldens load", tag);
    if (!(ids && tok_g)) { free(ids); free(tok_g); return; }
    int plen = (int)n_ids;
    /* truth = prompt + the C non-spec greedy stream (long enough for
     * FORCED_ROUNDS full-accept rounds + margin) */
    int n_truth = FORCED_ROUNDS * (g_B + 2) + 8;
    int *stream = malloc((size_t)n_truth * sizeof(int));
    {
        ApusModelState st0;
        apus_model_state_init(&st0, &g_m);
        nonspec_run(&st0, ids, plen, n_truth, 0.0f, 1.0f, 0, stream);
        apus_model_state_free(&st0, &g_m);
    }
    int64_t *truth = malloc(((size_t)plen + n_truth) * sizeof(int64_t));
    for (int i = 0; i < plen; i++) truth[i] = ids[i];
    for (int i = 0; i < n_truth; i++) truth[plen + i] = stream[i];

    DraftCtx dc = { truth, g_V, mode };
    ApusModelState st;
    ApusDsparkState dst;
    apus_model_state_init(&st, &g_m);
    apus_dspark_state_init(&dst, &g_ds);
    /* run exactly FORCED_ROUNDS rounds, checking per-round emission */
    void *scratch = malloc(apus_sample_scratch_size((size_t)g_V));
    ApusRng rng;
    apus_rng_seed(&rng, 0);
    ApusDspec sp;
    apus_dspec_init(&sp, &g_m, &st, &g_ds, &dst, 0.0f, 1.0f, &rng,
                    0x9E3779B97F4A7C15ull, scratch);
    sp.draft_override = draft_hook;
    sp.draft_ctx = &dc;
    apus_dspec_prefill(&sp, ids, plen);
    int *out = malloc(((size_t)FORCED_ROUNDS * (g_B + 1)) * sizeof(int));
    int n_out = 0, counts_ok = 1, bonus_ok = 1;
    uint64_t bonus0 = sp.bonus_rounds;
    for (int r = 0; r < FORCED_ROUNDS; r++) {
        int buf[16];
        int ne = apus_dspec_step(&sp, buf, 16);
        if (ne != 1 + expect) counts_ok = 0;
        uint64_t b = sp.bonus_rounds - bonus0;
        if ((int)b != (r + 1) * expect_bonus) bonus_ok = 0;
        for (int i = 0; i < ne; i++) out[n_out++] = buf[i];
    }
    CHECK(counts_ok, "%s: per-round emitted != 1 + accepted(%d)", tag,
          expect);
    CHECK(bonus_ok, "%s: bonus flags != %d per round", tag, expect_bonus);
    CHECK(sp.accepted == (uint64_t)(expect * FORCED_ROUNDS),
          "%s: accepted %llu != %d", tag,
          (unsigned long long)sp.accepted, expect * FORCED_ROUNDS);
    /* stream + digest vs the C non-spec run of the same length */
    int *ref = malloc((size_t)n_out * sizeof(int));
    ApusModelState st_ref;
    apus_model_state_init(&st_ref, &g_m);
    nonspec_run(&st_ref, ids, plen, n_out, 0.0f, 1.0f, 0, ref);
    int same = 1;
    for (int i = 0; i < n_out; i++)
        if (out[i] != ref[i]) same = 0;
    CHECK(same, "%s: forced stream != non-spec", tag);
    uint64_t d_spec = digest_main(&g_m, &st, 14695981039346656037ull);
    uint64_t d_ref = digest_main(&g_m, &st_ref, 14695981039346656037ull);
    CHECK(d_spec == d_ref, "%s: forced state digest differs", tag);
    /* informational: accept counts vs the golden manifest */
    printf("  %s: emitted %d in %d rounds, accepted %d/round, stream %s,"
           " digest %s\n", tag, n_out, FORCED_ROUNDS, expect,
           same ? "BITWISE" : "DIFFERS",
           d_spec == d_ref ? "BITWISE" : "DIFFERS");

    /* 5. stage-ring catch-up (D13): after the run, every stage ring slot
     * holds the newest TRUE position's KV. */
    if (do_catchup) {
        int64_t f_final = sp.f;
        int win = g_ds.stages[0].acfg.window;
        int d = g_ds.stages[0].acfg.head_dim;
        /* recompute the true per-position main_hidden rows: batched
         * prefill rows, then per-token decode rows for the fed truth */
        float *rows = malloc(((size_t)f_final + 1) * g_mhd * sizeof(float));
        float *logits = malloc((size_t)g_V * sizeof(float));
        ApusModelState st2;
        apus_model_state_init(&st2, &g_m);
        apus_model_forward_mh(&g_m, &st2, ids, plen, logits, 0, rows);
        for (int64_t p = plen; p <= f_final; p++) {
            int64_t tok = truth[p];
            apus_model_forward_mh(&g_m, &st2, &tok, 1, logits, 0,
                                  rows + (size_t)p * g_mhd);
        }
        float *mx = malloc((size_t)g_dim * sizeof(float));
        float *kv = malloc((size_t)d * sizeof(float));
        int n_bit = 0, n_tol = 0, bad = 0;
        double worst = 0.0;
        for (int64_t p = 0; p <= f_final; p++) {
            if (p + win <= f_final) continue;  /* slot overwritten by p+win */
            apus_dspark_stage0_project(&g_ds, rows + (size_t)p * g_mhd, 1,
                                       mx);
            for (int k = 0; k < g_ds.n_stages; k++) {
                apus_dspark_stage_kv(&g_ds, k, mx, p, kv);
                const float *got = dst.stages[k].attn.win
                                   + (size_t)(p % win) * d;
                if (!memcmp(got, kv, (size_t)d * sizeof(float))) {
                    n_bit++;
                    continue;
                }
                double r = codestep_ratio(kv, got, (size_t)d);
                if (r > worst) worst = r;
                if (p < plen && r <= 1.0) n_tol++;
                else bad++;
            }
        }
        printf("  %s catch-up: slots newest-true bitwise %d, tolerated %d,"
               " bad %d (worst %.3g)\n", tag, n_bit, n_tol, bad, worst);
        CHECK(bad == 0, "%s: %d stage ring slots stale/wrong", tag, bad);
        free(logits);
        free(rows);
        free(mx);
        free(kv);
        apus_model_state_free(&st2, &g_m);
    }
    apus_dspec_free(&sp);
    free(scratch);
    apus_model_state_free(&st_ref, &g_m);
    apus_dspark_state_free(&dst, &g_ds);
    apus_model_state_free(&st, &g_m);
    free(out);
    free(ref);
    free(stream);
    free(truth);
    free(ids);
    free(tok_g);
}

int main(void) {
    printf("test_m11b: M11b DSpark speculative decoding verification\n");
    char err[256];
    if (apus_model_load(&g_m, FIX, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        return 1;
    }
    g_V = g_m.cfg.vocab_size;
    g_dim = g_m.cfg.dim;
    g_hc = g_m.cfg.hc_mult;
    CHECK(g_m.dspark, "fixture model must declare DSpark config");
    CHECK(g_m.dsc.block_size == 5 && g_m.dsc.n_stages == 3
          && g_m.dsc.n_targets == 3 && g_m.dsc.targets[0] == 2
          && g_m.dsc.targets[1] == 3 && g_m.dsc.targets[2] == 4
          && g_m.dsc.markov_rank == 64 && g_m.dsc.noise_id == 511,
          "fixture DSpark config parse (block %d, stages %d, targets %d,"
          " rank %d, noise %d)", g_m.dsc.block_size, g_m.dsc.n_stages,
          g_m.dsc.n_targets, g_m.dsc.markov_rank, g_m.dsc.noise_id);
    CHECK(g_m.n_mtp == 0, "DSpark model must not set classic n_mtp");
    if (apus_dspark_load(&g_ds, &g_m, err, sizeof err)) {
        fprintf(stderr, "dspark load: %s\n", err);
        return 1;
    }
    g_B = g_ds.cfg.block_size;
    g_mhd = g_ds.cfg.n_targets * g_dim;
    g_rank = g_ds.cfg.markov_rank;
    printf("  model: %d layers, dim %d, vocab %d; dspark stages %d,"
           " block %d, targets [%d,%d,%d], rank %d\n",
           g_m.n_layers, g_dim, g_V, g_ds.n_stages, g_B,
           g_ds.cfg.targets[0], g_ds.cfg.targets[1], g_ds.cfg.targets[2],
           g_rank);

    /* golden bounds: measured values recorded in tests/m11b/README.md.
     * C-vs-f32 sits inside the oracle's own f32-vs-f64 envelope (the
     * random-weight near-tie cascade; m11a README "Tolerancing"). */
    printf("== join_prefill ==\n");
    test_join_prefill("join_prefill_len24", 0.35, 0.35);
    test_join_prefill("join_prefill_len140", 0.35, 0.35);
    printf("== draft rounds ==\n");
    test_draft_rounds_len24();
    test_draft_round_len140();
    printf("== equivalence (hard gate: bitwise) ==\n");
    test_episode("spec_episode_greedy", 0.0f, 1.0f, 0, 48, 64);
    test_episode("spec_episode_sampled", 0.8f, 1.0f, 12345, 48, 64);
    printf("== forced draft patterns ==\n");
    test_forced("forced_all_accept", 0, 5, 1, 0);
    test_forced("forced_all_reject", 1, 0, 0, 0);
    test_forced("forced_mixed", 2, 2, 0, 1);

    apus_dspark_free(&g_ds);
    apus_model_free(&g_m);
    printf("test_m11b: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
