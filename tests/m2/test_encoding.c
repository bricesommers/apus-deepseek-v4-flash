/*
 * tests/m2/test_encoding.c — DeepSeek-V4 message-format conformance.
 *
 * - The 4 reference conformance pairs (reference/encoding/tests/, byte-
 *   identical to reference-0731/encoding/tests/):
 *   build the message list from test_input_N.json (attaching tools to
 *   messages[0] for pair 1, like reference test_encoding_dsv4.py), run
 *   encoding.h, byte-compare with test_output_N.txt, and compare the
 *   token-id sequence (encoding.h + tok.h) against golden/enc_conf_N.ids.
 * - Extra generated cases golden/enc_extra_N.json -> enc_extra_N.txt (+ .ids)
 *   covering reasoning_effort low/high/max (0731 semantics: "high" = old
 *   preview "max" prefix, "max" = new "Beyond maximum" prefix, low/None =
 *   no prefix, chat-mode no-op), chat mode with reordered tool results,
 *   content_blocks/tool_result lists, response_format, wo_eos, developer
 *   role, task tokens, add_bos=false.
 * - Determinism: encode twice, byte-identical.
 * Tokenizer path: reference/tokenizer.json, overridable via APUS_TOK_JSON
 * (M10 tokenizer spot-check against reference-0731/tokenizer.json).
 * Run from the repository root.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#include "json.h"
#include "tok.h"
#include "encoding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[sz] = 0;
    *len = (size_t)sz;
    return buf;
}

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void check_ids(Tok *t, const JVal *messages, const JVal *context,
                      const DsEncOpts *opts,
                      const char *ids_path, const char *what) {
    size_t glen;
    unsigned char *g = (unsigned char *)read_file(ids_path, &glen);
    if (!g) { CHECK(0, "%s: missing %s", what, ids_path); return; }
    size_t n = 0;
    char *prompt = ds_encode_messages_ctx(messages, context, opts);
    uint32_t *ids = prompt ? tok_encode_str(t, prompt, 1, &n) : NULL;
    CHECK(ids != NULL || n == 0, "%s: encode failed: %s", what, ds_last_error());
    uint32_t gn = glen >= 4 ? rd_u32(g) : 0;
    int ok = ids && glen >= 4 && (size_t)gn * 4 + 4 == glen && gn == n;
    if (ok)
        for (uint32_t i = 0; i < gn; i++)
            if (rd_u32(g + 4 + i * 4) != ids[i]) { ok = 0; break; }
    CHECK(ok, "%s: token ids mismatch (golden %u, got %zu)", what, gn, n);
    /* ds_encode_ids convenience wrapper must agree (context-free path) */
    if (!context) {
        size_t n3 = 0;
        uint32_t *ids3 = ds_encode_ids(t, messages, opts, &n3);
        CHECK(ids3 && n3 == n && (n == 0 || memcmp(ids, ids3, n * 4) == 0),
              "%s: ds_encode_ids != tok_encode(prompt)", what);
        free(ids3);
    }
    free(ids);
    free(prompt);
    free(g);
}

/* Run one case: messages tree (owned by caller), opts, expected text file. */
static void run_case(Tok *t, JVal *messages, JVal *context, DsEncOpts *opts,
                     const char *expected_path, const char *ids_path,
                     const char *what) {
    size_t elen;
    char *expected = read_file(expected_path, &elen);
    CHECK(expected != NULL, "%s: missing %s", what, expected_path);

    char *p1 = ds_encode_messages_ctx(messages, context, opts);
    char *p2 = ds_encode_messages_ctx(messages, context, opts);
    CHECK(p1 != NULL, "%s: encode failed: %s", what, ds_last_error());
    CHECK(p1 && p2 && strcmp(p1, p2) == 0, "%s: non-deterministic encode", what);
    if (p1 && expected) {
        size_t plen = strlen(p1);
        if (!(plen == elen && memcmp(p1, expected, elen) == 0)) {
            /* locate first difference */
            size_t i = 0, m = plen < elen ? plen : elen;
            while (i < m && p1[i] == expected[i]) i++;
            CHECK(0, "%s: prompt mismatch (len %zu vs %zu) first diff at byte %zu: "
                  "got %.40s | want %.40s", what, plen, elen, i,
                  i < plen ? p1 + i : "", i < elen ? expected + i : "");
        } else {
            CHECK(1, "%s ok", what);
        }
    }
    if (ids_path) check_ids(t, messages, context, opts, ids_path, what);
    free(p1);
    free(p2);
    free(expected);
}

int main(void) {
    Tok *t = tok_load(getenv("APUS_TOK_JSON") ? getenv("APUS_TOK_JSON")
                                              : "reference/tokenizer.json");
    if (!t) { fprintf(stderr, "tok_load failed\n"); return 1; }

    /* ---- the 4 reference conformance pairs ---- */
    const char *modes[5] = { NULL, "thinking", "thinking", "thinking", "chat" };
    for (int i = 1; i <= 4; i++) {
        char ipath[256], opath[256], idspath[256], what[64];
        snprintf(ipath, sizeof ipath, "reference/encoding/tests/test_input_%d.json", i);
        snprintf(opath, sizeof opath, "reference/encoding/tests/test_output_%d.txt", i);
        snprintf(idspath, sizeof idspath, "tests/m2/golden/enc_conf_%d.ids", i);
        snprintf(what, sizeof what, "conformance pair %d", i);

        char err[256];
        JVal *root = json_parse_file(ipath, err, sizeof err);
        CHECK(root != NULL, "pair %d: %s", i, err);
        if (!root) continue;
        JVal *messages;
        if (json_type(root) == J_OBJ) {
            messages = json_clone(json_obj_get(root, "messages"));
            JVal *tools = json_obj_get(root, "tools");
            if (tools)
                json_obj_set(json_arr_get(messages, 0), "tools", json_clone(tools));
            json_free(root);
        } else {
            messages = root;
        }
        DsEncOpts opts = { modes[i], 1, 1, NULL };
        run_case(t, messages, NULL, &opts, opath, idspath, what);
        json_free(messages);
    }

    /* ---- extra generated cases ---- */
    for (int i = 1; ; i++) {
        char spath[256], tpath[256], idspath[256], what[64];
        snprintf(spath, sizeof spath, "tests/m2/golden/enc_extra_%d.json", i);
        snprintf(tpath, sizeof tpath, "tests/m2/golden/enc_extra_%d.txt", i);
        snprintf(idspath, sizeof idspath, "tests/m2/golden/enc_extra_%d.ids", i);
        snprintf(what, sizeof what, "extra case %d", i);

        char err[256];
        JVal *spec = json_parse_file(spath, err, sizeof err);
        if (!spec) break;
        JVal *messages = json_obj_get(spec, "messages");
        CHECK(messages && json_type(messages) == J_ARR, "extra %d: bad spec", i);
        DsEncOpts opts = DS_ENC_OPTS_DEFAULT;
        JVal *v;
        if ((v = json_obj_get(spec, "thinking_mode")) && json_type(v) == J_STR)
            opts.thinking_mode = json_str(v);
        if ((v = json_obj_get(spec, "drop_thinking")) && json_type(v) == J_BOOL)
            opts.drop_thinking = json_bool(v);
        if ((v = json_obj_get(spec, "add_default_bos_token")) && json_type(v) == J_BOOL)
            opts.add_bos = json_bool(v);
        if ((v = json_obj_get(spec, "reasoning_effort")) && json_type(v) == J_STR)
            opts.reasoning_effort = json_str(v);
        JVal *context = json_obj_get(spec, "context");
        run_case(t, messages, context, &opts, tpath, idspath, what);
        json_free(spec);
    }

    tok_free(t);
    printf("test_encoding: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
