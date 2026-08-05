/*
 * tests/m7b/test_model.c — M7b model-level gate: the full m5 synthetic
 * mini-model forward through the Metal backend.
 *
 *   A. Metal vs the m5 oracle goldens — the same battery and tolerances as
 *      tests/m5/test_full.c (prefill_len6/200, decode_from64, greedy_from24,
 *      sampled_from24), proving the m5 contract holds with dense ops on GPU.
 *   B. CPU vs Metal in the same process (THE gate): identical model, hooks
 *      toggled. Teacher-forced greedy on greedy_from24: token streams must
 *      be IDENTICAL; a flip is excused only when the CPU-logit top1-top2 gap
 *      is <= 0.5 (the m5 near-tie policy). Prefill_len200 per-position
 *      argmax identity with the same policy; logit divergence measured and
 *      documented (FP32 summation-order noise cascading through BF16
 *      rounding points — the same class as the m5 C-vs-f32 budget).
 *   C. Metal-path determinism (repeated prefill bitwise) and chunk
 *      invariance (prefill(200) == prefill(120) + 80 decodes, bitwise).
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "sample.h"
#include "backend_metal.h"

#define FIX "tests/m5/fixtures"

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
    if (!strcmp(n.descr, "<f4")) memcpy(out, n.data, c * sizeof(float));
    else if (!strcmp(n.descr, "<f8")) {
        for (size_t i = 0; i < c; i++) out[i] = (float)((double *)n.data)[i];
    } else { free(out); out = NULL; }
    if (nelem) *nelem = c;
    npy_free(&n);
    return out;
}

static int64_t *load_i64(const char *path, size_t *nelem) {
    Npy n;
    if (npy_load(path, &n)) return NULL;
    size_t c = npy_nelem(&n);
    int64_t *out = malloc(c * sizeof(int64_t));
    if (!strcmp(n.descr, "<i8")) memcpy(out, n.data, c * sizeof(int64_t));
    else if (!strcmp(n.descr, "<i4")) {
        for (size_t i = 0; i < c; i++) out[i] = ((int32_t *)n.data)[i];
    } else { free(out); out = NULL; }
    if (nelem) *nelem = c;
    npy_free(&n);
    return out;
}

static double *load_f64(const char *path, size_t *nelem) {
    Npy n;
    if (npy_load(path, &n)) return NULL;
    size_t c = npy_nelem(&n);
    double *out = malloc(c * sizeof(double));
    if (!strcmp(n.descr, "<f8")) memcpy(out, n.data, c * sizeof(double));
    else if (!strcmp(n.descr, "<f4")) {
        for (size_t i = 0; i < c; i++) out[i] = ((float *)n.data)[i];
    } else { free(out); out = NULL; }
    if (nelem) *nelem = c;
    npy_free(&n);
    return out;
}

/* --- compare/report helpers (same conventions as tests/m5) -----------------*/

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

typedef struct { double maxabs, scale; size_t n, nargmax_bad, rows; } CmpRes;

static CmpRes cmp_logits(const float *g, const float *a, size_t rows, size_t V) {
    CmpRes r = {0.0, 0.0, rows * V, 0, rows};
    for (size_t i = 0; i < rows * V; i++) {
        double d = fabs((double)g[i] - (double)a[i]);
        if (d > r.maxabs) r.maxabs = d;
        double ag = fabs((double)g[i]);
        if (ag > r.scale) r.scale = ag;
    }
    for (size_t t = 0; t < rows; t++) {
        size_t bg = 0, ba = 0;
        for (size_t i = 1; i < V; i++) {
            if (g[t * V + i] > g[t * V + bg]) bg = i;
            if (a[t * V + i] > a[t * V + ba]) ba = i;
        }
        if (bg != ba) r.nargmax_bad++;
    }
    return r;
}

static void report_logits(const char *stage, CmpRes r, double rel_tol,
                          double argmax_tol) {
    double rel = r.maxabs / (r.scale > 1e-30 ? r.scale : 1e-30);
    double af = r.rows ? (double)r.nargmax_bad / (double)r.rows : 0.0;
    int ok = rel <= rel_tol && af <= argmax_tol;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s rel=%.3e maxabs=%.3e argmaxflip=%.2f%%\n",
           ok ? "ok  " : "FAIL", stage, rel, r.maxabs, 100.0 * af);
    if (!ok)
        printf("       (tol rel=%.1e argmax=%.2f%%)\n", rel_tol,
               100.0 * argmax_tol);
}

/* --- globals: one model, Metal enabled once, hooks toggled CPU<->Metal ----*/

static ApusModel g_m;
static int g_V;
static ApusBackendHooks g_metal_hooks;

static void hooks_off(void) {
    memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
}
static void hooks_on(void) {
    apus_backend_hooks = g_metal_hooks;
}

/* prefill with logits_all, fresh state */
static void run_prefill(const int64_t *ids, int s, float *logits) {
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward(&g_m, &st, ids, s, logits, 1);
    apus_model_state_free(&st, &g_m);
}

/* --- A. Metal vs oracle goldens (m5 battery, m5 tolerances) ----------------*/

static void test_prefill(const char *name, double rel_tol, double argmax_tol) {
    char path[256];
    size_t n_ids, n_g;
    snprintf(path, sizeof path, FIX "/golden/%s/input_ids.npy", name);
    int64_t *ids = load_i64(path, &n_ids);
    snprintf(path, sizeof path, FIX "/golden/%s/logits_f32.npy", name);
    float *g32 = load_f32(path, &n_g);
    CHECK(ids && g32, "%s: fixture load", name);
    if (!ids || !g32) { free(ids); free(g32); return; }
    int s = (int)n_ids;
    float *logits = malloc((size_t)s * g_V * sizeof(float));
    hooks_on();
    run_prefill(ids, s, logits);
    hooks_off();
    char stage[128];
    snprintf(stage, sizeof stage, "[metal] %s logits vs f32", name);
    report_logits(stage, cmp_logits(g32, logits, (size_t)s, (size_t)g_V),
                  rel_tol, argmax_tol);
    free(logits);
    free(ids);
    free(g32);
}

static void test_decode_chain(double rel_tol, double argmax_tol) {
    size_t n_p, n_d;
    int64_t *pids = load_i64(FIX "/golden/decode_from64/prompt_ids.npy", &n_p);
    int64_t *dids = load_i64(FIX "/golden/decode_from64/decode_ids.npy", &n_d);
    float *g32 = load_f32(FIX "/golden/decode_from64/logits_f32.npy", NULL);
    CHECK(pids && dids && g32, "decode_from64: fixture load");
    if (!pids || !dids || !g32) { free(pids); free(dids); free(g32); return; }
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    float *lg = malloc((size_t)n_p * g_V * sizeof(float));
    hooks_on();
    apus_model_forward(&g_m, &st, pids, (int)n_p, lg, 0);
    free(lg);
    float *logits = malloc((size_t)g_V * sizeof(float));
    float *all = malloc(n_d * g_V * sizeof(float));
    for (size_t k = 0; k < n_d; k++) {
        apus_model_forward(&g_m, &st, &dids[k], 1, logits, 0);
        memcpy(all + k * g_V, logits, (size_t)g_V * sizeof(float));
    }
    hooks_off();
    report_logits("[metal] decode_from64 logits vs f32",
                  cmp_logits(g32, all, n_d, (size_t)g_V), rel_tol, argmax_tol);
    apus_model_state_free(&st, &g_m);
    free(all);
    free(logits);
    free(pids);
    free(dids);
    free(g32);
}

static void test_greedy_oracle(double gap_tol, double rel_tol,
                               double argmax_tol) {
    size_t n_p, n_t;
    int64_t *pids = load_i64(FIX "/golden/greedy_from24/prompt_ids.npy", &n_p);
    int64_t *gtok = load_i64(FIX "/golden/greedy_from24/tokens.npy", &n_t);
    float *gap = load_f32(FIX "/golden/greedy_from24/gap.npy", NULL);
    float *gl = load_f32(FIX "/golden/greedy_from24/logits_f32.npy", NULL);
    CHECK(pids && gtok && gap && gl, "greedy_from24: fixture load");
    if (!pids || !gtok || !gap || !gl) {
        free(pids); free(gtok); free(gap); free(gl);
        return;
    }
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    float *logits = malloc((size_t)g_V * sizeof(float));
    float *all = malloc(n_t * g_V * sizeof(float));
    hooks_on();
    apus_model_forward(&g_m, &st, pids, (int)n_p, logits, 0);
    int flips = 0, excused = 0;
    double max_flip_gap = 0;
    for (size_t k = 0; k < n_t; k++) {
        if (k) {
            int64_t prev = gtok[k - 1];
            apus_model_forward(&g_m, &st, &prev, 1, logits, 0);
        }
        memcpy(all + k * g_V, logits, (size_t)g_V * sizeof(float));
        int tok = apus_sample_argmax(logits, (size_t)g_V);
        if ((int64_t)tok != gtok[k]) {
            flips++;
            if (gap[k] > max_flip_gap) max_flip_gap = gap[k];
            if (gap[k] <= gap_tol) excused++;
            printf("       [metal] greedy flip at step %zu: C=%lld golden=%lld "
                   "gap=%.3e (%s)\n", k, (long long)tok, (long long)gtok[k],
                   gap[k], gap[k] <= gap_tol ? "near-tie, excused"
                                             : "NOT excused");
        }
    }
    hooks_off();
    report_logits("[metal] greedy_from24 teacher-forced logits",
                  cmp_logits(gl, all, n_t, (size_t)g_V), rel_tol, argmax_tol);
    int ok = flips == excused;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s flips=%d (excused=%d, max flip gap=%.3e)\n",
           ok ? "ok  " : "FAIL", "[metal] greedy_from24 vs oracle tokens",
           flips, excused, max_flip_gap);
    apus_model_state_free(&st, &g_m);
    free(logits);
    free(all);
    free(pids);
    free(gtok);
    free(gap);
    free(gl);
}

static void test_sampled_oracle(double temp, double top_p, double margin_tol,
                                double rel_tol, double argmax_tol) {
    size_t n_p, n_t, n_u;
    int64_t *pids = load_i64(FIX "/golden/sampled_from24/prompt_ids.npy", &n_p);
    int64_t *gtok = load_i64(FIX "/golden/sampled_from24/tokens.npy", &n_t);
    double *uni = load_f64(FIX "/golden/sampled_from24/uniforms.npy", &n_u);
    double *margin = load_f64(FIX "/golden/sampled_from24/margin.npy", NULL);
    float *gl = load_f32(FIX "/golden/sampled_from24/logits_f32.npy", NULL);
    CHECK(pids && gtok && uni && margin && gl && n_u == n_t,
          "sampled_from24: fixture load");
    if (!pids || !gtok || !uni || !margin || !gl) {
        free(pids); free(gtok); free(uni); free(margin); free(gl);
        return;
    }
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    float *logits = malloc((size_t)g_V * sizeof(float));
    float *all = malloc(n_t * g_V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)g_V));
    hooks_on();
    apus_model_forward(&g_m, &st, pids, (int)n_p, logits, 0);
    int flips = 0, excused = 0;
    double max_flip_margin = 0;
    for (size_t k = 0; k < n_t; k++) {
        if (k) {
            int64_t prev = gtok[k - 1];
            apus_model_forward(&g_m, &st, &prev, 1, logits, 0);
        }
        memcpy(all + k * g_V, logits, (size_t)g_V * sizeof(float));
        int tok = apus_sample_logits_u(logits, (size_t)g_V, (float)temp,
                                       (float)top_p, uni[k], scratch);
        if ((int64_t)tok != gtok[k]) {
            flips++;
            if (margin[k] > max_flip_margin) max_flip_margin = margin[k];
            if (margin[k] <= margin_tol) excused++;
            printf("       [metal] sampled flip at step %zu: C=%lld "
                   "golden=%lld margin=%.3e (%s)\n", k, (long long)tok,
                   (long long)gtok[k], margin[k],
                   margin[k] <= margin_tol ? "near-boundary, excused"
                                           : "NOT excused");
        }
    }
    hooks_off();
    report_logits("[metal] sampled_from24 teacher-forced logits",
                  cmp_logits(gl, all, n_t, (size_t)g_V), rel_tol, argmax_tol);
    int ok = flips == excused;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s flips=%d (excused=%d, max flip margin=%.3e)\n",
           ok ? "ok  " : "FAIL", "[metal] sampled_from24 vs oracle tokens",
           flips, excused, max_flip_margin);
    apus_model_state_free(&st, &g_m);
    free(scratch);
    free(logits);
    free(all);
    free(pids);
    free(gtok);
    free(uni);
    free(margin);
    free(gl);
}

/* --- B. CPU vs Metal (THE gate) --------------------------------------------*/

/* teacher-forced decode over greedy_from24 with both backends; per-step
 * logits compared, token identity with the m5 near-tie policy (gap from the
 * CPU logits, tol 0.5) */
static void test_cpu_vs_metal_greedy(double gap_tol, double rel_tol) {
    size_t n_p, n_t;
    int64_t *pids = load_i64(FIX "/golden/greedy_from24/prompt_ids.npy", &n_p);
    int64_t *gtok = load_i64(FIX "/golden/greedy_from24/tokens.npy", &n_t);
    CHECK(pids && gtok, "cpu-vs-metal: fixture load");
    if (!pids || !gtok) { free(pids); free(gtok); return; }

    float *lc = malloc(n_t * g_V * sizeof(float));
    float *lm = malloc(n_t * g_V * sizeof(float));
    for (int backend = 0; backend < 2; backend++) {
        float *out = backend ? lm : lc;
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        float *logits = malloc((size_t)g_V * sizeof(float));
        if (backend) hooks_on();
        apus_model_forward(&g_m, &st, pids, (int)n_p, logits, 0);
        for (size_t k = 0; k < n_t; k++) {
            if (k) {
                int64_t prev = gtok[k - 1];
                apus_model_forward(&g_m, &st, &prev, 1, logits, 0);
            }
            memcpy(out + k * g_V, logits, (size_t)g_V * sizeof(float));
        }
        hooks_off();
        free(logits);
        apus_model_state_free(&st, &g_m);
    }

    int flips = 0, excused = 0, identical = 1;
    double max_flip_gap = 0;
    for (size_t k = 0; k < n_t; k++) {
        int tc = apus_sample_argmax(lc + k * g_V, (size_t)g_V);
        int tm = apus_sample_argmax(lm + k * g_V, (size_t)g_V);
        if (tc != tm) {
            identical = 0;
            flips++;
            /* gap from the CPU logits (top1 - top2) */
            float t1 = -INFINITY, t2 = -INFINITY;
            for (int i = 0; i < g_V; i++) {
                float v = lc[k * g_V + i];
                if (v > t1) { t2 = t1; t1 = v; }
                else if (v > t2) t2 = v;
            }
            double gap = (double)(t1 - t2);
            if (gap > max_flip_gap) max_flip_gap = gap;
            if (gap <= gap_tol) excused++;
            printf("       cpu-vs-metal flip at step %zu: cpu=%d metal=%d "
                   "cpu-gap=%.3e (%s)\n", k, tc, tm, gap,
                   gap <= gap_tol ? "near-tie, excused" : "NOT excused");
        }
    }
    int ok = flips == excused;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s flips=%d/%zu (excused=%d, max flip gap=%.3e)%s\n",
           ok ? "ok  " : "FAIL", "GATE greedy tokens CPU == Metal",
           flips, n_t, excused, max_flip_gap,
           identical ? " — IDENTICAL" : "");
    report_logits("cpu-vs-metal teacher-forced logits",
                  cmp_logits(lc, lm, n_t, (size_t)g_V), rel_tol, 1.0);
    free(lc);
    free(lm);
    free(pids);
    free(gtok);
}

/* prefill_len200: per-position argmax identity + logit divergence */
static void test_cpu_vs_metal_prefill(double gap_tol, double rel_tol) {
    size_t n_ids;
    int64_t *ids = load_i64(FIX "/golden/prefill_len200/input_ids.npy", &n_ids);
    CHECK(ids, "cpu-vs-metal prefill: fixture load");
    if (!ids) return;
    int s = (int)n_ids;
    float *lc = malloc((size_t)s * g_V * sizeof(float));
    float *lm = malloc((size_t)s * g_V * sizeof(float));
    hooks_off();
    run_prefill(ids, s, lc);
    hooks_on();
    run_prefill(ids, s, lm);
    hooks_off();
    int flips = 0, excused = 0, identical = 1;
    double max_flip_gap = 0;
    for (int t = 0; t < s; t++) {
        int tc = apus_sample_argmax(lc + (size_t)t * g_V, (size_t)g_V);
        int tm = apus_sample_argmax(lm + (size_t)t * g_V, (size_t)g_V);
        if (tc != tm) {
            identical = 0;
            flips++;
            float t1 = -INFINITY, t2 = -INFINITY;
            for (int i = 0; i < g_V; i++) {
                float v = lc[(size_t)t * g_V + i];
                if (v > t1) { t2 = t1; t1 = v; }
                else if (v > t2) t2 = v;
            }
            double gap = (double)(t1 - t2);
            if (gap > max_flip_gap) max_flip_gap = gap;
            if (gap <= gap_tol) excused++;
            printf("       cpu-vs-metal prefill flip pos %d: cpu=%d metal=%d "
                   "cpu-gap=%.3e (%s)\n", t, tc, tm, gap,
                   gap <= gap_tol ? "near-tie, excused" : "NOT excused");
        }
    }
    int ok = flips == excused;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s flips=%d/%d (excused=%d, max flip gap=%.3e)%s\n",
           ok ? "ok  " : "FAIL", "GATE prefill argmax CPU == Metal",
           flips, s, excused, max_flip_gap, identical ? " — IDENTICAL" : "");
    report_logits("cpu-vs-metal prefill_len200 logits",
                  cmp_logits(lc, lm, (size_t)s, (size_t)g_V), rel_tol, 1.0);
    free(lc);
    free(lm);
    free(ids);
}

/* --- C. Metal determinism + chunk invariance --------------------------------*/

static void test_metal_determinism(void) {
    size_t n_ids;
    int64_t *ids = load_i64(FIX "/golden/prefill_len200/input_ids.npy", &n_ids);
    CHECK(ids, "metal determinism: fixture load");
    if (!ids) return;
    float *a = malloc(n_ids * g_V * sizeof(float));
    float *b = malloc(n_ids * g_V * sizeof(float));
    hooks_on();
    run_prefill(ids, (int)n_ids, a);
    run_prefill(ids, (int)n_ids, b);
    hooks_off();
    int ok = memcmp(a, b, n_ids * g_V * sizeof(float)) == 0;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s\n", ok ? "ok  " : "FAIL",
           "[metal] determinism prefill_len200 bitwise");
    free(a);
    free(b);
    free(ids);
}

static void test_metal_chunk_invariance(void) {
    size_t n_ids;
    int64_t *ids = load_i64(FIX "/golden/prefill_len200/input_ids.npy", &n_ids);
    CHECK(ids, "metal chunkinv: fixture load");
    if (!ids) return;
    int s = (int)n_ids, split = 120;
    float *one = malloc((size_t)s * g_V * sizeof(float));
    float *two = malloc((size_t)s * g_V * sizeof(float));
    hooks_on();
    run_prefill(ids, s, one);
    {
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        apus_model_forward(&g_m, &st, ids, split, two, 1);
        float *lg = malloc((size_t)g_V * sizeof(float));
        for (int t = split; t < s; t++) {
            apus_model_forward(&g_m, &st, &ids[t], 1, lg, 0);
            memcpy(two + (size_t)t * g_V, lg, (size_t)g_V * sizeof(float));
        }
        free(lg);
        apus_model_state_free(&st, &g_m);
    }
    hooks_off();
    size_t nbit = 0;
    double maxabs = 0;
    for (size_t i = 0; i < (size_t)s * g_V; i++) {
        uint32_t ua, ub;
        memcpy(&ua, &one[i], 4); memcpy(&ub, &two[i], 4);
        if (ua != ub) nbit++;
        double d = fabs((double)one[i] - (double)two[i]);
        if (d > maxabs) maxabs = d;
    }
    int ok = nbit == 0;
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] %-46s bitdiff=%zu/%zu maxabs=%.3e\n",
           ok ? "ok  " : "FAIL", "[metal] chunkinv 200@120 (want bitwise)",
           nbit, (size_t)s * g_V, maxabs);
    free(one);
    free(two);
    free(ids);
}

int main(void) {
    printf("test_model: M7b Metal backend model-level gate\n");
    char err[256];
    if (apus_metal_enable(err, sizeof err)) {
        printf("  Metal unavailable (%s) — skipping (not a failure on "
               "non-GPU hosts)\n", err);
        return 0;
    }
    g_metal_hooks = apus_backend_hooks;
    hooks_off();   /* CPU default; each test toggles explicitly */
    if (apus_model_load(&g_m, FIX, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        return 1;
    }
    g_V = g_m.cfg.vocab_size;
    printf("  model: %d layers, dim %d, vocab %d — Metal enabled\n",
           g_m.n_layers, g_m.cfg.dim, g_V);

    printf(" -- A. Metal vs oracle goldens (m5 battery, m5 tolerances)\n");
    test_prefill("prefill_len6", 5e-2, 0.02);
    test_prefill("prefill_len200", 3.5e-1, 0.15);
    test_decode_chain(5e-1, 0.30);
    test_greedy_oracle(0.5, 3.5e-1, 0.15);
    test_sampled_oracle(0.8, 0.95, 1e-2, 2.5e-1, 0.08);

    printf(" -- B. CPU vs Metal (the M7b gate: token identity)\n");
    test_cpu_vs_metal_greedy(0.5, 3.5e-1);
    test_cpu_vs_metal_prefill(0.5, 3.5e-1);

    printf(" -- C. Metal-path determinism + chunk invariance\n");
    test_metal_determinism();
    test_metal_chunk_invariance();

    printf("  wrapped %.1f MB zero-copy, uploaded %.1f MB, %llu dispatches\n",
           (double)apus_metal_bytes_wrapped() / 1048576.0,
           (double)apus_metal_bytes_uploaded() / 1048576.0,
           (unsigned long long)apus_metal_dispatches());
    apus_metal_disable();
    apus_model_free(&g_m);
    printf("test_model: %ld checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
