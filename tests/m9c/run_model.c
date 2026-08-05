/*
 * tests/m9c/run_model.c — M9c instrumented real-model driver. Same run
 * path as c/apus.c (`run` mode: tiered store + pilot), but prints the full
 * M9c scheduling stats: store waits/wait_ns (just-in-time resolve stalls)
 * and pilot prefill union-lookahead counters. Used for the M9c
 * measurements only; not a quality gate (the gates are test_m9c + the
 * unchanged m2..m9b suites).
 *
 *   run_model --model DIR [--prompt "text" | --ids "1,2,3"]
 *             [--max-tokens N] [--seed S] [--temp T]
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
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#define APUS_PILOT_IMPLEMENTATION
#define APUS_MTP_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "model.h"
#include "sample.h"
#include "tok.h"
#include "encoding.h"
#include "cache.h"
#include "pilot.h"
#include "mtp.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int64_t *parse_ids(const char *s, int *n_out) {
    int cap = 16, n = 0;
    int64_t *ids = malloc((size_t)cap * sizeof(int64_t));
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end;
        long long v = strtoll(p, &end, 10);
        if (end == p) { free(ids); return NULL; }
        p = end;
        if (n == cap) { cap *= 2; ids = realloc(ids, (size_t)cap * sizeof *ids); }
        ids[n++] = v;
    }
    *n_out = n;
    return ids;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *prompt = NULL, *ids_str = NULL;
    int max_tokens = 1;
    uint64_t seed = 1;
    float temp = 0.0f, top_p = 1.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    if (!model_dir || (!prompt && !ids_str)) {
        fprintf(stderr, "need --model and (--prompt|--ids)\n");
        return 2;
    }

    char err[256];
    ApusModel m;
    double t0 = now_s();
    if (apus_model_load_ex(&m, model_dir, 1, err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    ApusStoreCfg scfg = {
        .n_layers = m.n_layers,
        .n_experts = m.cfg.n_routed_experts,
    };
    ApusStore *store = apus_store_open(model_dir, &scfg, err, sizeof err);
    if (!store) { fprintf(stderr, "store: %s\n", err); return 1; }
    for (int i = 0; i < m.n_layers; i++)
        apus_store_attach_moe(store, &m.layers[i].mw);
    ApusPilotCfg pcfg = {
        .store = store,
        .n_layers = m.n_layers,
        .n_experts = m.cfg.n_routed_experts,
        .topk = m.cfg.n_activated_experts,
        .dim = m.cfg.dim,
        .hc_mult = m.cfg.hc_mult,
        .sinkhorn_iters = m.cfg.hc_sinkhorn_iters,
        .vocab = m.cfg.vocab_size,
        .norm_eps = m.cfg.norm_eps,
        .hc_eps = m.cfg.hc_eps,
        .enabled = apus_env_int("APUS_PILOT", 1),
        .pilot_k = apus_env_int("APUS_PILOT_K", 8),
        .hash_prefetch = apus_env_int("APUS_PILOT_HASH", 1),
        .prefill_last_only = 1,
        .ring_entries = (size_t)apus_env_int("APUS_PILOT_RING", 4096),
    };
    ApusPilot *pilot = apus_pilot_create(&pcfg);
    apus_pilot_attach_model(pilot, &m);
    apus_pilot_start(pilot);
    fprintf(stderr, "run_model: loaded (%d layers%s) in %.2fs\n", m.n_layers,
            pilot->cfg.prefill_k > 0 ? ", prefill-union ON" : "",
            now_s() - t0);

    /* input ids */
    Tok *tok = NULL;
    int64_t *ids = NULL;
    int n_ids = 0;
    if (ids_str) {
        ids = parse_ids(ids_str, &n_ids);
    } else {
        char tpath[1024];
        snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
        tok = tok_load(tpath);
        if (!tok) { fprintf(stderr, "tok load failed\n"); return 1; }
        JVal *msgs = json_new_arr();
        JVal *msg = json_new_obj();
        json_obj_set(msg, "role", json_new_str("user"));
        json_obj_set(msg, "content", json_new_str(prompt));
        json_arr_push(msgs, msg);
        DsEncOpts opts = DS_ENC_OPTS_DEFAULT;
        size_t n = 0;
        uint32_t *u32 = ds_encode_ids(tok, msgs, &opts, &n);
        json_free(msgs);
        ids = malloc(n * sizeof(int64_t));
        for (size_t i = 0; i < n; i++) ids[i] = u32[i];
        n_ids = (int)n;
        free(u32);
    }

    ApusModelState st;
    apus_model_state_init(&st, &m);
    int V = m.cfg.vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);

    apus_pilot_prefetch_hash(pilot, ids, n_ids, 0);
    t0 = now_s();
    apus_model_forward(&m, &st, ids, n_ids, logits, 0);
    double t_prefill = now_s() - t0;

    int n_gen = 0;
    double t_gen0 = now_s();
    for (int step = 0; step < max_tokens; step++) {
        int tok_id = apus_sample(logits, (size_t)V, temp, top_p, &rng, scratch);
        if (tok) {
            size_t len = 0;
            char *text = tok_decode(tok, (const uint32_t[]){(uint32_t)tok_id}, 1, &len);
            if (text) { fwrite(text, 1, len, stdout); free(text); }
            fflush(stdout);
        } else {
            printf("%d ", tok_id);
            fflush(stdout);
        }
        n_gen++;
        if (tok_id == m.eos_id) break;
        int64_t next = tok_id;
        apus_pilot_prefetch_hash(pilot, &next, 1, st.pos);
        apus_model_forward(&m, &st, &next, 1, logits, 0);
    }
    double t_gen = now_s() - t_gen0;
    printf("\n");

    ApusStoreStats ss;
    apus_store_stats(store, &ss);
    ApusPilotStats ps;
    apus_pilot_stats(pilot, &ps);
    fprintf(stderr,
            "run_model: store: %llu hits %llu misses (%llu preads, %.1f MB; "
            "%llu hint %llu demand), %llu evictions, %llu rss drops, "
            "waits %llu (%.2fs blocked), pread %.2fs (duty %.0f%%)\n",
            (unsigned long long)ss.hits, (unsigned long long)ss.misses,
            (unsigned long long)ss.preads, (double)ss.bytes_read / 1048576.0,
            (unsigned long long)ss.hint_loads,
            (unsigned long long)ss.demand_loads,
            (unsigned long long)ss.evictions,
            (unsigned long long)ss.rss_drops,
            (unsigned long long)ss.waits,
            (double)ss.wait_ns / 1e9,
            (double)ss.pread_ns / 1e9,
            100.0 * (double)ss.pread_ns / 1e9 / (t_prefill + t_gen)
                / (double)apus_env_int("APUS_IO_THREADS", 4));
    fprintf(stderr,
            "run_model: pilot: %llu predictions (prefill %llu tokens, "
            "%llu union hints), %llu ring hints (%llu hash), recall %llu/%llu"
            ", d2 %llu preds rescue %llu/%llu\n",
            (unsigned long long)ps.predictions,
            (unsigned long long)ps.prefill_predictions,
            (unsigned long long)ps.prefill_hints,
            (unsigned long long)ps.hints_enqueued,
            (unsigned long long)ps.hash_hints,
            (unsigned long long)ps.actual_hits,
            (unsigned long long)ps.actual_experts,
            (unsigned long long)ps.d2_predictions,
            (unsigned long long)ps.d2_rescue,
            (unsigned long long)ps.d2_missed);
    fprintf(stderr, "run_model: prefill %d tok in %.2fs (%.1f tok/s); "
            "decode %d tok in %.2fs (%.2f tok/s)\n",
            n_ids, t_prefill, n_ids / t_prefill,
            n_gen, t_gen, n_gen > 0 ? n_gen / t_gen : 0.0);

    free(logits);
    free(scratch);
    free(ids);
    if (tok) tok_free(tok);
    apus_model_state_free(&st, &m);
    apus_store_save_usage(store);
    apus_pilot_destroy(pilot);
    apus_store_close(store);
    apus_model_free(&m);
    return 0;
}
