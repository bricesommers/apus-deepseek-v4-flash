/*
 * c/apus.c — apus engine driver / CLI (M5). Full forward pass over a
 * DeepSeek-V4-Flash checkpoint (or the synthetic tests/m5 fixture model):
 * embed -> N blocks -> hc_head -> norm -> head -> logits -> sampling.
 *
 *   apus run --model DIR [--prompt "text" | --ids "1,2,3"]
 *            [--max-tokens N] [--seed S] [--temp T] [--top-p P] [--greedy]
 *
 * --prompt tokenizes via DIR/tokenizer.json and renders the dsv4 chat
 * format (c/encoding.h); --ids feeds raw token ids (used for the synthetic
 * fixture model, whose 512-row vocab has no tokenizer). Sampling defaults:
 * temp 1.0, top_p 1.0 (reference/generation_config.json); --greedy or
 * --temp 0 selects argmax. The RNG is splitmix64 seeded by --seed
 * (c/sample.h) — runs with the same arguments are bit-identical.
 *
 * All implementation TUs live here (single-binary build).
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
#define APUS_DSPARK_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _WIN32
#include <io.h>     /* _setmode (M15) */
#include <fcntl.h>  /* _O_BINARY */
#endif

#include "model.h"
#include "sample.h"
#include "tok.h"
#include "encoding.h"
#include "cache.h"
#include "pilot.h"
#include "mtp.h"
#include "dspark.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(FILE *f) {
    fprintf(f,
        "apus run --model DIR [--prompt \"text\" | --ids \"1,2,3\"]\n"
        "         [--max-tokens N] [--seed S] [--temp T] [--top-p P]\n"
        "         [--greedy] [--quiet] [--tiered] [--measure-locality FILE]\n"
        "         [--metal]\n"
        "apus serve --model DIR [--tiered] [--metal]\n"
        "         NDJSON request/response protocol on stdin/stdout (M7a;\n"
        "         see tests/m7a/README.md). Driven by tools/server.py.\n"
        "  --tiered: experts-on-demand via the M6a expert store (also\n"
        "         APUS_TIERED=1; budgets APUS_EXPERT_CACHE_MB/APUS_PIN_MB/\n"
        "         APUS_RSS_GUARD_MB/APUS_IO_THREADS)\n"
        "  --metal: Metal GPU backend for the dense compute (M7b; also\n"
        "         APUS_METAL=1; weight-buffer budget APUS_METAL_DENSE_MB,\n"
        "         default 8192). Fail-soft: unsupported ops and missing GPU\n"
        "         fall back to the CPU kernels; CPU is the default.\n"
        "  pilot (M6b, tiered only, APUS_PILOT default 1): router-lookahead\n"
        "         prefetch; knobs APUS_PILOT_K (8), APUS_PILOT_RING (4096),\n"
        "         APUS_PILOT_HASH (1), APUS_PILOT_DUMP (NDJSON P/A sets)\n"
        "  --measure-locality FILE: dump per-token chosen/predicted expert\n"
        "         sets (NDJSON) for tools/measure_router_locality.py\n"
        "  --spec [run mode]: speculative decoding (M8 classic MTP /\n"
        "         M11b DSpark; also APUS_SPEC=1), default OFF. On 0731\n"
        "         DSpark models the draft depth is fixed at the config's\n"
        "         dspark_block_size (5); on classic-MTP models it is\n"
        "         --spec-k D / APUS_SPEC_K (default 2 = 1 speculative\n"
        "         token per verify batch). Emitted tokens are bitwise\n"
        "         identical to non-speculative decoding for the same\n"
        "         seed/args.\n");
}

/* M6b measure-locality dump: per layer, per token: the ACTUAL chosen
 * expert set (router_idx, == tid2eid rows for hash layers) and the pilot's
 * predicted top-N set for layer L+1 from the post-attention hidden state
 * (same apus_pilot_predict the runtime pilot uses). */
#define APUS_MEASURE_N 24
typedef struct {
    FILE *f;
    ApusPilot *pilot;
    int topk;
} MeasureDump;

static void measure_cb(void *ctx, int layer, int64_t start_pos,
                       const int64_t *ids, int s,
                       const int32_t *router_idx, const float *post_attn_h) {
    MeasureDump *md = ctx;
    ApusPilot *p = md->pilot;
    int hc_dim = p ? p->cfg.hc_mult * p->cfg.dim : 0;
    int pn = p && p->cfg.n_experts < APUS_MEASURE_N ? p->cfg.n_experts
                                                    : APUS_MEASURE_N;
    for (int t = 0; t < s; t++) {
        fprintf(md->f, "{\"type\":\"A\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                (long long)(start_pos + t), layer);
        for (int j = 0; j < md->topk; j++)
            fprintf(md->f, "%s%d", j ? "," : "", router_idx[(size_t)t * md->topk + j]);
        fprintf(md->f, "]}\n");
        if (p && layer + 1 < p->cfg.n_layers) {
            int32_t idx[APUS_MEASURE_N];
            if (apus_pilot_predict(p, layer + 1,
                                   post_attn_h + (size_t)t * hc_dim,
                                   idx, pn) == 0) {
                fprintf(md->f,
                        "{\"type\":\"P\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                        (long long)(start_pos + t), layer + 1);
                for (int j = 0; j < pn; j++)
                    fprintf(md->f, "%s%d", j ? "," : "", idx[j]);
                fprintf(md->f, "]}\n");
            }
        }
    }
    (void)ids;
}

/* parse "1,2,3" into a malloc'd int64 array */
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
        if (n == cap) { cap *= 2; ids = realloc(ids, (size_t)cap * sizeof(int64_t)); }
        ids[n++] = v;
    }
    *n_out = n;
    return ids;
}

/* M7b: enable the Metal backend when --metal / APUS_METAL=1. Fail-soft:
 * on any error the hooks stay NULL and the engine runs the CPU kernels. */
static void metal_maybe_enable(int want, int quiet) {
    if (!want) return;
    char merr[256];
    if (apus_metal_enable(merr, sizeof merr)) {
        fprintf(stderr, "apus: metal backend unavailable (%s) — CPU fallback\n",
                merr);
    } else if (!quiet) {
        fprintf(stderr, "apus: metal backend enabled (dense offload)\n");
    }
}

/* ---- engine context shared by `run` and `serve` (M7a) ------------------ */

/* M8: ApusSpec pre-batch hook -> M6b hash-layer prefetch (hint-only). */
static void spec_pre_batch(void *ctx, const int64_t *ids, int n,
                           int64_t pos) {
    apus_pilot_prefetch_hash((ApusPilot *)ctx, ids, n, pos);
}

typedef struct {
    ApusModel m;
    ApusStore *store;   /* M6a expert store (tiered only) */
    ApusPilot *pilot;   /* M6b prefetch pilot (tiered, or measure mode) */
    int tiered;
    ApusMtp mtp;        /* M8 MTP draft head (classic-MTP spec only) */
    ApusMtpState mst;
    int has_mtp;
    ApusDspark ds;      /* M11b DSpark stages (0731 spec only) */
    ApusDsparkState dst;
    int has_dspark;
} Engine;

/* Load model + (tiered) expert store + (optional) pilot. pilot_wanted:
 * create the pilot; pilot_enabled: its prefetch is active (measure mode
 * creates a disabled pilot for apus_pilot_predict only). spec: also load
 * the draft machinery — the M8 classic MTP head (V4-Flash preview) or the
 * M11b DSpark stages (0731) — and give the store ownership of their
 * experts (store layers n_main + K). */
static int engine_init(Engine *e, const char *model_dir, int tiered,
                       int pilot_wanted, int pilot_enabled, int spec,
                       char *err, size_t errcap) {
    memset(e, 0, sizeof *e);
    e->tiered = tiered;
    if (apus_model_load_ex(&e->m, model_dir, tiered, err, errcap)) return -1;
    /* 0731 DSpark detection: config dspark_* keys / the fixture "dspark"
     * object (the parser sets m.dspark and m.dsc). */
    int dspark = e->m.dspark;
    if (spec && dspark && e->m.dsc.n_stages <= 0) {
        snprintf(err, errcap, "0731 DSpark markers present but the config "
                 "has no complete dspark_* section");
        return -1;
    }
    if (spec && !dspark && e->m.n_mtp == 0) {
        snprintf(err, errcap, "model has no MTP block or DSpark stages "
                 "(--spec unavailable)");
        return -1;
    }
    int n_spec = spec ? (dspark ? e->m.dsc.n_stages : e->m.n_mtp) : 0;

    /* M6a expert store: demand-loaded expert slabs through the bounded
     * cache; attaches the resolve hook to every layer's MoE. */
    if (tiered) {
        ApusStoreCfg scfg = {
            .n_layers = e->m.n_layers + n_spec,
            .n_experts = e->m.cfg.n_routed_experts,
            .n_mtp = n_spec,
        };
        e->store = apus_store_open(model_dir, &scfg, err, errcap);
        if (!e->store) return -1;
        for (int i = 0; i < e->m.n_layers; i++)
            apus_store_attach_moe(e->store, &e->m.layers[i].mw);
    }

    /* the draft machinery (after the store so its experts can attach) */
    if (spec && dspark) {
        /* M11b: DSpark stages (mtp.{0..K} full blocks + stage glue) */
        if (apus_dspark_load(&e->ds, &e->m, err, errcap)) return -1;
        if (e->store)
            for (int k = 0; k < e->ds.n_stages; k++)
                apus_store_attach_moe(e->store, &e->ds.stages[k].mw);
        apus_dspark_state_init(&e->dst, &e->ds);
        e->has_dspark = 1;
    } else if (spec) {
        /* M8: classic MTP draft head */
        if (!apus_st_set_get(e->m.set, "mtp.0.e_proj.weight")) {
            snprintf(err, errcap, "model has no classic MTP tensors "
                     "(--spec unavailable)");
            return -1;
        }
        if (apus_mtp_load(&e->mtp, &e->m, 0, err, errcap)) return -1;
        if (e->store) apus_store_attach_moe(e->store, &e->mtp.block.mw);
        apus_mtp_state_init(&e->mst, &e->mtp);
        e->has_mtp = 1;
    }

    /* M6b pilot: router-lookahead prefetch. Attach AFTER the store so the
     * pilot can wrap the store's MoE hint hooks for recall accounting. */
    if (pilot_wanted) {
        ApusPilotCfg pcfg = {
            .store = e->store,
            .n_layers = e->m.n_layers,
            .n_experts = e->m.cfg.n_routed_experts,
            .topk = e->m.cfg.n_activated_experts,
            .dim = e->m.cfg.dim,
            .hc_mult = e->m.cfg.hc_mult,
            .sinkhorn_iters = e->m.cfg.hc_sinkhorn_iters,
            .vocab = e->m.cfg.vocab_size,
            .norm_eps = e->m.cfg.norm_eps,
            .hc_eps = e->m.cfg.hc_eps,
            .enabled = pilot_enabled,
            .pilot_k = apus_env_int("APUS_PILOT_K", 8),
            .hash_prefetch = apus_env_int("APUS_PILOT_HASH", 1),
            .prefill_last_only = 1,
            .ring_entries = (size_t)apus_env_int("APUS_PILOT_RING", 4096),
            .dump_path = getenv("APUS_PILOT_DUMP"),
        };
        e->pilot = apus_pilot_create(&pcfg);
        if (e->pilot) {
            apus_pilot_attach_model(e->pilot, &e->m);
            apus_pilot_start(e->pilot);
        }
    }
    return 0;
}

static void engine_destroy(Engine *e) {
    if (e->store) apus_store_save_usage(e->store);
    if (e->pilot) apus_pilot_destroy(e->pilot);   /* joins the pilot thread first */
    if (e->has_mtp) {
        apus_mtp_state_free(&e->mst, &e->mtp);
        apus_mtp_free(&e->mtp);
    }
    if (e->has_dspark) {
        apus_dspark_state_free(&e->dst, &e->ds);
        apus_dspark_free(&e->ds);
    }
    if (e->store) apus_store_close(e->store);
    apus_model_free(&e->m);
}

/* ================= M7a serve mode =================
 *
 * NDJSON protocol on stdin/stdout: one JSON object per line in each
 * direction (justification: the gateway tools/server.py spawns this
 * process and owns all networking; stdio keeps the engine libc-only — no
 * sockets in C — and matches the colibri gateway-drives-engine split).
 * The process stays alive across requests; the model loads once. Every
 * request gets a FRESH ApusModelState (KV is per-request; conversation
 * state is the gateway's job — multi-turn context is re-prefilled; KV
 * reuse across turns is a later optimization).
 *
 * Request (client -> engine):
 *   {"id": <any>, "cmd": "encode",
 *    "messages": [...], "tools": [...]|null,
 *    "thinking": true|false, "reasoning_effort": "low"|"high"|"max"|null}
 *   {"id": <any>, "cmd": "generate",
 *    "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 *    "tools": [...]|null, "thinking": bool, "reasoning_effort": str|null,
 *    "max_tokens": int, "temperature": float, "top_p": float,
 *    "seed": uint, "stop": [str, ...]}
 * "tools" is attached to the first system/developer message (like the
 * reference encoding tests); with no such message an empty system carrier
 * is synthesized. "text" is tokenized verbatim (no chat template, no BOS).
 *
 * Events (engine -> client):
 *   {"id","type":"encoded","text","ids"}      (encode; ids need tokenizer)
 *   {"id","type":"prompt","prompt_tokens"}    (generate, first)
 *   {"id","type":"token","token_id","text"}   (per generated token; EOS is
 *                                             never emitted; text needs a
 *                                             tokenizer)
 *   {"id","type":"done","finish_reason","prompt_tokens",
 *    "completion_tokens","text"}              finish_reason: "stop" (EOS)
 *                                             | "length" | "stop_string"
 *   {"id","type":"error","message"}           (request failed; loop lives)
 * Stop strings are checked against the assembled decoded text; a match
 * truncates the text at the match start (a partial last token piece is
 * emitted if it precedes the match) and finishes "stop_string".
 */

static char *serve_read_line(void) {
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap);
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (c == '\n') break;
        if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(buf); return NULL; }
    buf[n] = 0;
    return buf;
}

static JVal *serve_resp(const JVal *id, const char *type) {
    JVal *o = json_new_obj();
    json_obj_set(o, "id", id ? json_clone(id) : json_new_null());
    json_obj_set(o, "type", json_new_str(type));
    return o;
}

static void serve_send(JVal *resp) {
    char *s = json_dumps(resp);
    fputs(s, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    free(s);
    json_free(resp);
}

static void serve_error(const JVal *id, const char *msg) {
    JVal *o = serve_resp(id, "error");
    json_obj_set(o, "message", json_new_str(msg ? msg : "error"));
    serve_send(o);
}

static void serve_token(const JVal *id, int token_id,
                        const char *text, size_t len) {
    JVal *o = serve_resp(id, "token");
    json_obj_set(o, "token_id", json_new_int(token_id));
    if (text) json_obj_set(o, "text", json_new_strn(text, len));
    serve_send(o);
}

/* Request tools -> a message's "tools" field (encoding.h renders tools
 * only from system/developer messages, as in the reference). */
static JVal *serve_attach_tools(JVal *messages, const JVal *tools) {
    if (!tools || !json_truthy(tools)) return messages;
    size_t n = json_arr_len(messages);
    for (size_t i = 0; i < n; i++) {
        JVal *m = json_arr_get(messages, i);
        JVal *rv = (m && json_type(m) == J_OBJ) ? json_obj_get(m, "role")
                                                : NULL;
        const char *role = json_type(rv) == J_STR ? json_str(rv) : "";
        if (!strcmp(role, "system") || !strcmp(role, "developer")) {
            json_obj_set(m, "tools", json_clone(tools));
            return messages;
        }
    }
    /* no system/developer message: synthesize an empty system carrier */
    JVal *out = json_new_arr();
    JVal *sys = json_new_obj();
    json_obj_set(sys, "role", json_new_str("system"));
    json_obj_set(sys, "content", json_new_str(""));
    json_obj_set(sys, "tools", json_clone(tools));
    json_arr_push(out, sys);
    for (size_t i = 0; i < n; i++)
        json_arr_push(out, json_clone(json_arr_get(messages, i)));
    json_free(messages);
    return out;
}

static void serve_enc_opts(const JVal *req, DsEncOpts *opts) {
    *opts = (DsEncOpts)DS_ENC_OPTS_DEFAULT;
    JVal *th = json_obj_get((JVal *)req, "thinking");
    if (th && json_type(th) == J_BOOL && !json_bool(th))
        opts->thinking_mode = "chat";
    JVal *re = json_obj_get((JVal *)req, "reasoning_effort");
    if (re && json_type(re) == J_STR) opts->reasoning_effort = json_str(re);
}

/* Encode request messages -> rendered prompt string (NULL on error,
 * message reported by the caller via ds_last_error()). */
static char *serve_render(const JVal *req, const JVal *id) {
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (!jmsgs || json_type(jmsgs) != J_ARR) {
        serve_error(id, "messages must be a JSON array");
        return NULL;
    }
    JVal *msgs = serve_attach_tools(json_clone(jmsgs),
                                    json_obj_get((JVal *)req, "tools"));
    DsEncOpts opts;
    serve_enc_opts(req, &opts);
    char *prompt = ds_encode_messages(msgs, &opts);
    json_free(msgs);
    if (!prompt) serve_error(id, ds_last_error());
    return prompt;
}

static void serve_cmd_encode(Tok *tok, const JVal *req, const JVal *id) {
    char *prompt = serve_render(req, id);
    if (!prompt) return;
    JVal *o = serve_resp(id, "encoded");
    json_obj_set(o, "text", json_new_str(prompt));
    if (tok) {
        size_t n = 0;
        uint32_t *ids = tok_encode_str(tok, prompt, 1, &n);
        JVal *arr = json_new_arr();
        for (size_t i = 0; i < n; i++)
            json_arr_push(arr, json_new_int((long long)ids[i]));
        json_obj_set(o, "ids", arr);
        free(ids);
    }
    free(prompt);
    serve_send(o);
}

static void serve_cmd_generate(Engine *e, Tok *tok,
                               const JVal *req, const JVal *id) {
    /* ---- prompt ids: ids | text | messages ---- */
    int64_t *ids = NULL;
    size_t n_ids = 0;
    JVal *jids = json_obj_get((JVal *)req, "ids");
    JVal *jtext = json_obj_get((JVal *)req, "text");
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (jids && json_type(jids) == J_ARR) {
        n_ids = json_arr_len(jids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) {
            JVal *v = json_arr_get(jids, i);
            if (!v || json_type(v) != J_NUM) {
                serve_error(id, "ids must be an array of numbers");
                free(ids);
                return;
            }
            ids[i] = (int64_t)json_num(v);
        }
    } else if (jtext && json_type(jtext) == J_STR) {
        if (!tok) { serve_error(id, "no tokenizer in model dir"); return; }
        uint32_t *u = tok_encode_str(tok, json_str(jtext), 1, &n_ids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else if (jmsgs) {
        if (!tok) { serve_error(id, "no tokenizer in model dir"); return; }
        char *prompt = serve_render(req, id);
        if (!prompt) return;
        uint32_t *u = tok_encode_str(tok, prompt, 1, &n_ids);
        free(prompt);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else {
        serve_error(id, "generate needs messages, text, or ids");
        return;
    }
    if (n_ids == 0) { serve_error(id, "empty prompt"); free(ids); return; }

    /* ---- sampling params ---- */
    int max_tokens = 32;
    double temperature = 1.0, top_p = 1.0;
    uint64_t seed = 0;
    JVal *v;
    if ((v = json_obj_get((JVal *)req, "max_tokens")) && json_type(v) == J_NUM)
        max_tokens = (int)json_num(v);
    if (max_tokens < 0) max_tokens = 0;
    if ((v = json_obj_get((JVal *)req, "temperature")) && json_type(v) == J_NUM)
        temperature = json_num(v);
    if ((v = json_obj_get((JVal *)req, "top_p")) && json_type(v) == J_NUM)
        top_p = json_num(v);
    if ((v = json_obj_get((JVal *)req, "seed")) && json_type(v) == J_NUM)
        seed = (uint64_t)json_num(v);
    const char *stops[16];
    int n_stops = 0;
    JVal *jstop = json_obj_get((JVal *)req, "stop");
    if (jstop && json_type(jstop) == J_ARR) {
        size_t ns = json_arr_len(jstop);
        for (size_t i = 0; i < ns && n_stops < 16; i++) {
            JVal *s = json_arr_get(jstop, i);
            if (s && json_type(s) == J_STR && json_strlen(s))
                stops[n_stops++] = json_str(s);
        }
    }

    /* clamp to the model's position tables */
    if ((int64_t)n_ids >= e->m.cfg.max_pos) {
        serve_error(id, "prompt too long for model max_pos");
        free(ids);
        return;
    }
    if ((int64_t)n_ids + max_tokens > e->m.cfg.max_pos)
        max_tokens = (int)(e->m.cfg.max_pos - (int64_t)n_ids);

    JVal *o = serve_resp(id, "prompt");
    json_obj_set(o, "prompt_tokens", json_new_int((long long)n_ids));
    serve_send(o);

    ApusModelState st;
    apus_model_state_init(&st, &e->m);
    int V = e->m.cfg.vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);

    if (e->pilot) apus_pilot_prefetch_hash(e->pilot, ids, (int)n_ids, 0);
    apus_model_forward(&e->m, &st, ids, (int)n_ids, logits, 0);

    SBuf text;
    sb_init(&text);
    int completion = 0;
    const char *finish = "length";
    for (int step = 0; step < max_tokens; step++) {
        int t = apus_sample(logits, (size_t)V, (float)temperature,
                            (float)top_p, &rng, scratch);
        if (t == e->m.eos_id) { finish = "stop"; break; }
        completion++;
        size_t plen = 0;
        char *piece = tok
            ? tok_decode(tok, (const uint32_t[]){ (uint32_t)t }, 1, &plen)
            : NULL;
        size_t oldn = text.n;
        if (piece) sb_write(&text, piece, plen);
        /* stop strings: earliest match over the assembled text */
        long mpos = -1;
        if (n_stops) {
            sb_reserve(&text, 0);
            text.p[text.n] = '\0';
            for (int k = 0; k < n_stops; k++) {
                char *hit = strstr(text.p, stops[k]);
                if (hit && (mpos < 0 || hit - text.p < mpos))
                    mpos = hit - text.p;
            }
        }
        if (mpos >= 0) {
            if ((size_t)mpos > oldn)
                serve_token(id, t, text.p + oldn, (size_t)mpos - oldn);
            text.n = (size_t)mpos;
            finish = "stop_string";
            free(piece);
            break;
        }
        serve_token(id, t, piece, plen);
        free(piece);
        int64_t next = t;
        if (e->pilot) apus_pilot_prefetch_hash(e->pilot, &next, 1, st.pos);
        apus_model_forward(&e->m, &st, &next, 1, logits, 0);
    }

    JVal *d = serve_resp(id, "done");
    json_obj_set(d, "finish_reason", json_new_str(finish));
    json_obj_set(d, "prompt_tokens", json_new_int((long long)n_ids));
    json_obj_set(d, "completion_tokens", json_new_int(completion));
    json_obj_set(d, "text", json_new_strn(text.p ? text.p : "", text.n));
    serve_send(d);

    sb_free(&text);
    free(logits);
    free(scratch);
    free(ids);
    apus_model_state_free(&st, &e->m);
}

static int serve_main(int argc, char **argv) {
    const char *model_dir = NULL;
    int tiered = apus_env_int("APUS_TIERED", 0);
    int metal = apus_env_int("APUS_METAL", 0);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--tiered")) tiered = 1;
        else if (!strcmp(argv[i], "--metal")) metal = 1;
        else { usage(stderr); return 2; }
    }
    if (!model_dir) { usage(stderr); return 2; }
    metal_maybe_enable(metal, 0);

    char err[256];
    Engine e;
    double t0 = now_s();
    int pilot_on = tiered && apus_env_int("APUS_PILOT", 1);
    if (engine_init(&e, model_dir, tiered, pilot_on, pilot_on, 0,
                    err, sizeof err)) {
        fprintf(stderr, "apus serve: %s\n", err);
        return 1;
    }
    fprintf(stderr,
            "apus serve: loaded %s (%d layers, dim %d, vocab %d%s) in %.2fs\n",
            model_dir, e.m.n_layers, e.m.cfg.dim, e.m.cfg.vocab_size,
            tiered ? ", tiered" : "", now_s() - t0);

    Tok *tok = NULL;
    char tpath[1024];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
    if (access(tpath, R_OK) == 0) {
        tok = tok_load(tpath);
        if (!tok) {
            fprintf(stderr, "apus serve: cannot load %s\n", tpath);
            engine_destroy(&e);
            return 1;
        }
    } else {
        fprintf(stderr, "apus serve: no %s (ids-only requests)\n", tpath);
    }

    char *line;
    while ((line = serve_read_line())) {
        char perr[128];
        JVal *req = json_parse(line, strlen(line), perr, sizeof perr);
        free(line);
        if (!req || json_type(req) != J_OBJ) {
            serve_error(NULL, "bad request line (not a JSON object)");
            json_free(req);
            continue;
        }
        JVal *id = json_obj_get(req, "id");
        JVal *cmdv = json_obj_get(req, "cmd");
        const char *cmd = json_type(cmdv) == J_STR ? json_str(cmdv) : NULL;
        if (cmd && !strcmp(cmd, "encode")) serve_cmd_encode(tok, req, id);
        else if (cmd && !strcmp(cmd, "generate"))
            serve_cmd_generate(&e, tok, req, id);
        else serve_error(id, "unknown cmd (want encode|generate)");
        json_free(req);
    }

    if (tok) tok_free(tok);
    engine_destroy(&e);
    return 0;
}

/* ================= run mode (CLI) ================= */

static int run_main(int argc, char **argv) {
    const char *model_dir = NULL, *prompt = NULL, *ids_str = NULL;
    const char *measure_path = NULL;
    int max_tokens = 32, quiet = 0;
    int dump_margins = apus_env_int("APUS_DUMP_MARGINS", 0);
    int tiered = apus_env_int("APUS_TIERED", 0);
    int metal = apus_env_int("APUS_METAL", 0);
    int spec = apus_env_int("APUS_SPEC", 0);
    int spec_k = apus_env_int("APUS_SPEC_K", 2);
    uint64_t seed = 0;
    float temp = 1.0f, top_p = 1.0f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) temp = 0.0f;
        else if (!strcmp(argv[i], "--tiered")) tiered = 1;
        else if (!strcmp(argv[i], "--metal")) metal = 1;
        else if (!strcmp(argv[i], "--spec")) spec = 1;
        else if (!strcmp(argv[i], "--spec-k") && i + 1 < argc) spec_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--measure-locality") && i + 1 < argc) measure_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--dump-margins")) dump_margins = 1;
        else { usage(stderr); return 2; }
    }
    if (!model_dir || (!prompt && !ids_str)) { usage(stderr); return 2; }
    if (spec && measure_path) {
        fprintf(stderr, "apus: --spec and --measure-locality are mutually exclusive\n");
        return 2;
    }
    if (spec_k < 1) spec_k = 1;
    if (spec_k > 63) spec_k = 63;
    metal_maybe_enable(metal, quiet);

    char err[256];
    double t0 = now_s();
    int pilot_on = tiered && apus_env_int("APUS_PILOT", 1);
    Engine e;
    if (engine_init(&e, model_dir, tiered, pilot_on || measure_path != NULL,
                    pilot_on, spec, err, sizeof err)) {
        fprintf(stderr, "apus: %s\n", err);
        return 1;
    }
    ApusModel *mp = &e.m;
    ApusStore *store = e.store;
    ApusPilot *pilot = e.pilot;
    ApusModel m;  /* read-only alias of e.m (owned/freed by engine_destroy) */
    memcpy(&m, mp, sizeof m);
    if (!quiet)
        fprintf(stderr, "apus: loaded %s (%d layers, dim %d, vocab %d%s) in %.2fs\n",
                model_dir, m.n_layers, m.cfg.dim, m.cfg.vocab_size,
                tiered ? ", tiered" : "", now_s() - t0);

    /* input ids */
    Tok *tok = NULL;
    int64_t *ids = NULL;
    int n_ids = 0;
    if (ids_str) {
        ids = parse_ids(ids_str, &n_ids);
        if (!ids || n_ids == 0) { fprintf(stderr, "apus: bad --ids\n"); return 2; }
    } else {
        char tpath[1024];
        snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
        tok = tok_load(tpath);
        if (!tok) { fprintf(stderr, "apus: cannot load %s\n", tpath); return 1; }
        JVal *msgs = json_new_arr();
        JVal *msg = json_new_obj();
        json_obj_set(msg, "role", json_new_str("user"));
        json_obj_set(msg, "content", json_new_str(prompt));
        json_arr_push(msgs, msg);
        DsEncOpts opts = DS_ENC_OPTS_DEFAULT;
        size_t n = 0;
        uint32_t *u32 = ds_encode_ids(tok, msgs, &opts, &n);
        json_free(msgs);
        if (!u32) {
            fprintf(stderr, "apus: encoding failed: %s\n", ds_last_error());
            return 1;
        }
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

    /* M6b hash-layer prefetch: layers 0-2 (tid2eid) are known exactly at
     * tokenization time — enqueue before the forward starts. */
    if (pilot) apus_pilot_prefetch_hash(pilot, ids, n_ids, 0);

    MeasureDump md;
    FILE *mf = NULL;
    if (measure_path) {
        mf = fopen(measure_path, "w");
        if (!mf) { fprintf(stderr, "apus: cannot open %s\n", measure_path); return 1; }
        md = (MeasureDump){ mf, pilot, m.cfg.n_activated_experts };
        /* self-contained dump: the prompt ids (generated ids follow as
         * "gen" lines in the decode loop) */
        fprintf(mf, "{\"type\":\"ids\",\"pos0\":0,\"ids\":[");
        for (int i = 0; i < n_ids; i++)
            fprintf(mf, "%s%lld", i ? "," : "", (long long)ids[i]);
        fprintf(mf, "]}\n");
    }

    /* prefill + decode */
    int n_gen = 0;
    double t_prefill, t_gen;
    if (spec && e.has_dspark) {
        /* M11b: DSpark draft/verify loop (c/dspark.h). Draft depth is the
         * config's dspark_block_size (0731: 5; --spec-k is ignored).
         * Emitted tokens are bitwise identical to the non-speculative
         * loop below for the same seed: every emitted token is the main
         * model's own apus_sample draw from its own (chunk-invariance-
         * bitwise) logits row, one main-stream uniform per token in
         * position order; drafts draw from a SEPARATE uniform stream and
         * confidence plays no part in acceptance (the M11a accept rule). */
        ApusDspec sp;
        apus_dspec_init(&sp, &m, &st, &e.ds, &e.dst, temp, top_p,
                        &rng, seed + 0x9E3779B97F4A7C15ull, scratch);
        if (pilot) {
            sp.pre_batch = spec_pre_batch;
            sp.pre_batch_ctx = pilot;
        }
        t0 = now_s();
        apus_dspec_prefill(&sp, ids, n_ids);
        t_prefill = now_s() - t0;
        double t_gen0 = now_s();
        int step_out[64], done = 0;
        while (n_gen < max_tokens && !done) {
            int ne = apus_dspec_step(&sp, step_out, 64);
            if (ne <= 0) break;
            for (int i = 0; i < ne && n_gen < max_tokens; i++) {
                int tok_id = step_out[i];
                if (quiet) {
                    printf("%d\n", tok_id);
                } else if (tok) {
                    size_t len = 0;
                    char *text = tok_decode(tok, (const uint32_t[]){(uint32_t)tok_id}, 1, &len);
                    if (text) { fwrite(text, 1, len, stdout); free(text); }
                    fflush(stdout);
                } else {
                    printf("%d ", tok_id);
                    fflush(stdout);
                }
                n_gen++;
                if (tok_id == m.eos_id) { done = 1; break; }
            }
        }
        t_gen = now_s() - t_gen0;
        if (!quiet)
            fprintf(stderr,
                    "apus: dspark spec: %llu emitted in %llu batches "
                    "(%.2f tok/batch, %llu re-fed), draft accept %llu/%llu"
                    " (%.1f%%), bonus rounds %llu/%llu\n",
                    (unsigned long long)sp.emitted,
                    (unsigned long long)sp.batches,
                    sp.batches ? (double)sp.emitted / (double)sp.batches : 0.0,
                    (unsigned long long)sp.refeed_tokens,
                    (unsigned long long)sp.accepted,
                    (unsigned long long)sp.offered,
                    sp.offered ? 100.0 * (double)sp.accepted / (double)sp.offered : 0.0,
                    (unsigned long long)sp.bonus_rounds,
                    (unsigned long long)sp.batches);
        apus_dspec_free(&sp);
    } else if (spec) {
        /* M8: MTP draft/verify loop (c/mtp.h). Emitted tokens are bitwise
         * identical to the non-speculative loop below for the same seed:
         * every emitted token is the main model's own apus_sample draw
         * from its own (chunk-invariance-bitwise) logits row, one RNG
         * uniform per token in position order; drafts consume no RNG. */
        ApusSpec sp;
        apus_spec_init(&sp, &m, &st, e.has_mtp ? &e.mtp : NULL,
                       e.has_mtp ? &e.mst : NULL, spec_k, temp, top_p,
                       &rng, scratch);
        if (pilot) {
            sp.pre_batch = spec_pre_batch;
            sp.pre_batch_ctx = pilot;
        }
        t0 = now_s();
        apus_spec_prefill(&sp, ids, n_ids);
        t_prefill = now_s() - t0;
        double t_gen0 = now_s();
        int step_out[64], done = 0;
        while (n_gen < max_tokens && !done) {
            int ne = apus_spec_step(&sp, step_out, 64);
            if (ne <= 0) break;
            for (int i = 0; i < ne && n_gen < max_tokens; i++) {
                int tok_id = step_out[i];
                if (quiet) {
                    printf("%d\n", tok_id);
                } else if (tok) {
                    size_t len = 0;
                    char *text = tok_decode(tok, (const uint32_t[]){(uint32_t)tok_id}, 1, &len);
                    if (text) { fwrite(text, 1, len, stdout); free(text); }
                    fflush(stdout);
                } else {
                    printf("%d ", tok_id);
                    fflush(stdout);
                }
                n_gen++;
                if (tok_id == m.eos_id) { done = 1; break; }
            }
        }
        t_gen = now_s() - t_gen0;
        if (!quiet)
            fprintf(stderr,
                    "apus: spec: %llu emitted in %llu batches "
                    "(%.2f tok/batch, %llu re-fed), draft accept %llu/%llu"
                    " (%.1f%%), d1 %llu/%llu\n",
                    (unsigned long long)sp.emitted,
                    (unsigned long long)sp.batches,
                    sp.batches ? (double)sp.emitted / (double)sp.batches : 0.0,
                    (unsigned long long)sp.refeed_tokens,
                    (unsigned long long)sp.accepted,
                    (unsigned long long)sp.offered,
                    sp.offered ? 100.0 * (double)sp.accepted / (double)sp.offered : 0.0,
                    (unsigned long long)sp.d1_hits,
                    (unsigned long long)sp.d1_offered);
        apus_spec_free(&sp);
    } else {
    /* prefill */
    t0 = now_s();
    if (mf)
        apus_model_forward_measure(&m, &st, ids, n_ids, logits, measure_cb, &md);
    else
        apus_model_forward(&m, &st, ids, n_ids, logits, 0);
    t_prefill = now_s() - t0;

    /* decode loop */
    double t_gen0 = now_s();
    for (int step = 0; step < max_tokens; step++) {
        if (dump_margins) {
            /* top1/top2 gap per step: near-tie adjudication for
             * cross-platform divergence (x86 vs ARM reorder class). */
            int i1 = 0, i2 = -1;
            float v1 = logits[0], v2 = -__builtin_inff();
            for (int j = 1; j < V; j++) {
                float v = logits[j];
                if (v > v1) { v2 = v1; i2 = i1; v1 = v; i1 = j; }
                else if (i2 < 0 || v > v2) { v2 = v; i2 = j; }
            }
            fprintf(stderr, "margin step=%d top1=%d top2=%d gap=%.6g\n",
                    step, i1, i2, v1 - v2);
        }
        int tok_id = apus_sample(logits, (size_t)V, temp, top_p, &rng, scratch);
        if (quiet) {
            printf("%d\n", tok_id);
        } else if (tok) {
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
        if (mf)
            fprintf(mf, "{\"type\":\"gen\",\"pos\":%lld,\"id\":%d}\n",
                    (long long)st.pos, tok_id);
        if (pilot) apus_pilot_prefetch_hash(pilot, &next, 1, st.pos);
        if (mf)
            apus_model_forward_measure(&m, &st, &next, 1, logits,
                                       measure_cb, &md);
        else
            apus_model_forward(&m, &st, &next, 1, logits, 0);
    }
    t_gen = now_s() - t_gen0;
    }
    if (mf) fclose(mf);
    if (pilot) {
        ApusPilotStats ps;
        apus_pilot_stats(pilot, &ps);
        if (!quiet)
            fprintf(stderr,
                    "apus: pilot: %llu predictions, %llu hints enqueued "
                    "(%llu hash, %llu dropped-full), %llu issued "
                    "(%llu dropped-stale), recall %llu/%llu\n",
                    (unsigned long long)ps.predictions,
                    (unsigned long long)ps.hints_enqueued,
                    (unsigned long long)ps.hash_hints,
                    (unsigned long long)ps.hints_dropped_full,
                    (unsigned long long)ps.hints_issued,
                    (unsigned long long)ps.hints_dropped_stale,
                    (unsigned long long)ps.actual_hits,
                    (unsigned long long)ps.actual_experts);
    }
    if (store) {
        apus_store_save_usage(store);
        ApusStoreStats ss;
        apus_store_stats(store, &ss);
        if (!quiet)
            fprintf(stderr,
                    "apus: expert store: %llu hits %llu misses "
                    "(%llu preads, %.1f MB; %llu hint-loads %llu demand-loads), "
                    "%llu evictions, %llu rss drops\n",
                    (unsigned long long)ss.hits, (unsigned long long)ss.misses,
                    (unsigned long long)ss.preads,
                    (double)ss.bytes_read / 1048576.0,
                    (unsigned long long)ss.hint_loads,
                    (unsigned long long)ss.demand_loads,
                    (unsigned long long)ss.evictions,
                    (unsigned long long)ss.rss_drops);
    }
    if (!quiet) {
        printf("\n");
        fprintf(stderr,
                "apus: prefill %d tok in %.2fs (%.1f tok/s); "
                "decode %d tok in %.2fs (%.2f tok/s)\n",
                n_ids, t_prefill, n_ids / t_prefill,
                n_gen, t_gen, n_gen > 0 ? n_gen / t_gen : 0.0);
    }

    free(logits);
    free(scratch);
    free(ids);
    if (tok) tok_free(tok);
    apus_model_state_free(&st, &m);
    engine_destroy(&e);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    /* M15: MSVCRT defaults stdin/stdout to text mode (\n -> \r\n), which
     * would corrupt the NDJSON serve protocol; force binary. */
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 2) { usage(stderr); return 2; }
    if (!strcmp(argv[1], "run")) return run_main(argc, argv);
    if (!strcmp(argv[1], "serve")) return serve_main(argc, argv);
    usage(stderr);
    return 2;
}
