/*
 * apus tok.h — BPE tokenizer for DeepSeek-V4-Flash, C11, libc only.
 *
 * Loads HF `tokenizers`-format tokenizer.json at runtime and replicates the
 * exact semantics of the reference model:
 *
 *   pre_tokenizer: Sequence[
 *     Split(\p{N}{1,3}, Isolated),             -- digit runs chopped to <=3
 *     Split([一-龥぀-ゟ゠-ヿ]+, Isolated),       -- CJK runs isolated
 *     Split(<gpt-style pattern>, Isolated),     -- see stage3 below
 *     ByteLevel(add_prefix_space=false, use_regex=false) ]
 *   model: BPE, byte-level alphabet (GPT-2 bytes_to_unicode), merges by rank
 *   added_tokens: 1283 (BOS/EOS/pad + 1280 more), matched leftmost-longest
 *   decoder: ByteLevel (alphabet chars -> bytes, others pass through UTF-8)
 *
 * The stage-3 pattern is:
 *   [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+   punct + ASCII letters
 *   | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+            opt prefix + letters/marks
 *   |  ?[\p{P}\p{S}]+[\r\n]*                          opt space + punct/symbols
 *   | \s*[\r\n]+                                      whitespace ending in newline
 *   | \s+(?!\S)                                       trailing whitespace
 *   | \s+                                             other whitespace
 * applied with leftmost match, alternation priority, greedy quantifiers
 * (PCRE-style, as the Rust regex crate does). Unicode classes come from
 * uni_tables.h, which is generated from the reference tokenizer's own
 * behavior (tests/m2/gen_uni_tables.py) — not from a Unicode database.
 *
 * Usage: #define APUS_TOK_IMPLEMENTATION in exactly one translation unit.
 */
#ifndef APUS_TOK_H
#define APUS_TOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tok Tok;

/* Load tokenizer.json. Returns NULL on error (message to stderr). */
Tok *tok_load(const char *path);
void tok_free(Tok *t);

/* Encode text (UTF-8 bytes, need not be NUL-terminated).
 * add_special=1: recognize added/special token strings in the input
 *                (HF default behavior).
 * add_special=0: treat special-token strings as plain text.
 * Returns malloc'd id array, *n_out = length (empty input -> NULL, 0). */
uint32_t *tok_encode(const Tok *t, const char *text, size_t len,
                     int add_special, size_t *n_out);
uint32_t *tok_encode_str(const Tok *t, const char *text,
                         int add_special, size_t *n_out);

/* Decode ids to exact bytes (malloc'd, NUL-terminated; *len_out = byte len).
 * Special/added tokens decode to their content strings. */
char *tok_decode(const Tok *t, const uint32_t *ids, size_t n, size_t *len_out);

/* Introspection */
int         tok_n_tokens(const Tok *t);
int32_t     tok_token_to_id(const Tok *t, const char *s);   /* -1 if unknown */
const char *tok_id_to_token(const Tok *t, uint32_t id);     /* NULL if bad id */

#ifdef __cplusplus
}
#endif

#endif /* APUS_TOK_H */

/* ================================================================== */
#if defined(APUS_TOK_IMPLEMENTATION) && !defined(APUS_TOK_IMPL_DONE)
#define APUS_TOK_IMPL_DONE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "json.h"
#include "uni_tables.h"

/* ---------------- unicode classes ---------------- */

static int cp_in_ranges(uint32_t cp, const uint32_t ranges[][2], uint32_t n) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid][0]) hi = mid;
        else if (cp > ranges[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int cls_N(uint32_t cp) {  /* \p{N} */
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return cp_in_ranges(cp, uni_n_ranges, UNI_N_NRANGES);
}
static int cls_LM(uint32_t cp) { /* \p{L} | \p{M} */
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return cp_in_ranges(cp, uni_lm_ranges, UNI_LM_NRANGES);
}
static int cls_PS(uint32_t cp) { /* \p{P} | \p{S} */
    if (cp < 0x80)
        return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') ||
               (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
    return cp_in_ranges(cp, uni_ps_ranges, UNI_PS_NRANGES);
}
static int cls_WS(uint32_t cp) { /* regex \s (White_Space) */
    if (cp < 0x80) return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20;
    return cp_in_ranges(cp, uni_ws_ranges, UNI_WS_NRANGES);
}
static int is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}
static int is_a1_punct(uint32_t cp) { /* the A1 ASCII punctuation set */
    if (cp >= 0x80) return 0;
    return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') ||
           (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
}
static int is_ascii_alpha(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
}

/* ---------------- byte-level alphabet ---------------- */

/* GPT-2 bytes_to_unicode: byte -> alphabet codepoint */
static uint32_t b2u_cp(int b) {
    if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
        return (uint32_t)b;
    /* others map to 256 + n in order */
    int n = 0;
    for (int x = 0; x < b; x++) {
        if (!((x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF)))
            n++;
    }
    return 256u + (uint32_t)n;
}

static int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* decode one UTF-8 char; returns cp (0x110000 on invalid byte) and length */
static uint32_t utf8_decode(const unsigned char *s, size_t len, size_t *adv) {
    if (!len) { *adv = 0; return 0x110000; }
    unsigned char c = s[0];
    if (c < 0x80) { *adv = 1; return c; }
    if (c >= 0xC2 && c < 0xE0 && len >= 2 && (s[1] & 0xC0) == 0x80) {
        *adv = 2;
        return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if (c >= 0xE0 && c < 0xF0 && len >= 3 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        uint32_t cp = ((uint32_t)(c & 0x0F) << 12) |
                      ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) { *adv = 3; return cp; }
    }
    if (c >= 0xF0 && c < 0xF5 && len >= 4 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp >= 0x10000 && cp <= 0x10FFFF) { *adv = 4; return cp; }
    }
    *adv = 1;
    return 0x110000; /* invalid byte: its own "other" char */
}

/* ---------------- hash maps ---------------- */

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* ---------------- Tok ---------------- */

#define MERGE_EMPTY UINT64_MAX

struct Tok {
    int n_ids;
    char **strs;        /* id -> token bytes */
    uint32_t *lens;

    /* string -> id open addressing (value = id+1, 0 = empty) */
    uint32_t *s2i;
    uint32_t s2i_mask;

    /* merge map: key (l<<32)|r -> value (result_id << 32) | rank */
    uint64_t *merge_k;
    uint64_t *merge_v;
    uint32_t merge_mask;

    uint32_t byte_id[256];  /* byte -> vocab id of its alphabet char */
    int16_t rev[0x144];     /* alphabet codepoint -> byte, -1 if none */

    /* added tokens, sorted by (first byte, -length) for leftmost-longest */
    int n_added;
    char **added;
    uint32_t *added_len;
    uint32_t *added_id;
    uint32_t added_start[256];  /* bucket start per first byte */
    uint32_t added_end[256];
};

static int32_t s2i_get(const Tok *t, const char *s, size_t n) {
    uint32_t i = (uint32_t)(fnv1a(s, n) & t->s2i_mask);
    while (t->s2i[i]) {
        uint32_t id = t->s2i[i] - 1;
        if (t->lens[id] == n && memcmp(t->strs[id], s, n) == 0) return (int32_t)id;
        i = (i + 1) & t->s2i_mask;
    }
    return -1;
}

static void s2i_put(Tok *t, uint32_t id) {
    uint32_t i = (uint32_t)(fnv1a(t->strs[id], t->lens[id]) & t->s2i_mask);
    while (t->s2i[i]) i = (i + 1) & t->s2i_mask;
    t->s2i[i] = id + 1;
}

static void merge_put(Tok *t, uint32_t l, uint32_t r, uint32_t result, uint32_t rank) {
    uint64_t key = ((uint64_t)l << 32) | r;
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull >> 32) & t->merge_mask);
    while (t->merge_k[i] != MERGE_EMPTY) i = (i + 1) & t->merge_mask;
    t->merge_k[i] = key;
    t->merge_v[i] = ((uint64_t)result << 32) | rank;
}

/* returns 1 if pair is mergeable; *result, *rank out */
static int merge_get(const Tok *t, uint32_t l, uint32_t r,
                     uint32_t *result, uint32_t *rank) {
    uint64_t key = ((uint64_t)l << 32) | r;
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull >> 32) & t->merge_mask);
    while (t->merge_k[i] != MERGE_EMPTY) {
        if (t->merge_k[i] == key) {
            *result = (uint32_t)(t->merge_v[i] >> 32);
            *rank = (uint32_t)(t->merge_v[i] & 0xFFFFFFFFu);
            return 1;
        }
        i = (i + 1) & t->merge_mask;
    }
    return 0;
}

/* ---------------- load ---------------- */

Tok *tok_load(const char *path) {
    char err[256];
    JVal *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "tok_load: %s\n", err);
        return NULL;
    }
    Tok *t = (Tok *)calloc(1, sizeof(Tok));

    JVal *model = json_obj_get(root, "model");
    JVal *vocab = model ? json_obj_get(model, "vocab") : NULL;
    JVal *merges = model ? json_obj_get(model, "merges") : NULL;
    JVal *added = json_obj_get(root, "added_tokens");
    if (!vocab || json_type(vocab) != J_OBJ || !merges || json_type(merges) != J_ARR) {
        fprintf(stderr, "tok_load: bad tokenizer.json structure\n");
        json_free(root);
        free(t);
        return NULL;
    }

    /* id space: max id over vocab and added tokens */
    int max_id = 0;
    for (size_t i = 0; i < json_obj_len(vocab); i++) {
        JVal *v = json_obj_val(vocab, i);
        if (json_type(v) == J_NUM && (int)json_num(v) > max_id) max_id = (int)json_num(v);
    }
    size_t n_added = added && json_type(added) == J_ARR ? json_arr_len(added) : 0;
    for (size_t i = 0; i < n_added; i++) {
        JVal *a = json_arr_get(added, i);
        JVal *id = a ? json_obj_get(a, "id") : NULL;
        if (id && json_type(id) == J_NUM && (int)json_num(id) > max_id)
            max_id = (int)json_num(id);
    }
    t->n_ids = max_id + 1;
    t->strs = (char **)calloc((size_t)t->n_ids, sizeof(char *));
    t->lens = (uint32_t *)calloc((size_t)t->n_ids, sizeof(uint32_t));

    for (size_t i = 0; i < json_obj_len(vocab); i++) {
        const char *key = json_obj_key(vocab, i);
        int id = (int)json_num(json_obj_val(vocab, i));
        if (id < 0 || id >= t->n_ids) continue;
        size_t n = strlen(key);
        t->strs[id] = (char *)malloc(n + 1);
        memcpy(t->strs[id], key, n + 1);
        t->lens[id] = (uint32_t)n;
    }
    for (size_t i = 0; i < n_added; i++) {
        JVal *a = json_arr_get(added, i);
        JVal *idv = json_obj_get(a, "id");
        JVal *cv = json_obj_get(a, "content");
        if (!idv || !cv || json_type(cv) != J_STR) continue;
        int id = (int)json_num(idv);
        if (id < 0 || id >= t->n_ids) continue;
        free(t->strs[id]);
        t->strs[id] = (char *)malloc(json_strlen(cv) + 1);
        memcpy(t->strs[id], json_str(cv), json_strlen(cv) + 1);
        t->lens[id] = (uint32_t)json_strlen(cv);
    }

    /* string -> id map */
    uint32_t cap = 1;
    while (cap < (uint32_t)t->n_ids * 2u) cap <<= 1;
    t->s2i = (uint32_t *)calloc(cap, sizeof(uint32_t));
    t->s2i_mask = cap - 1;
    for (int id = 0; id < t->n_ids; id++)
        if (t->strs[id]) s2i_put(t, (uint32_t)id);

    /* merge map */
    size_t n_merges = json_arr_len(merges);
    uint32_t mcap = 1;
    while (mcap < n_merges * 2u) mcap <<= 1;
    t->merge_k = (uint64_t *)malloc(mcap * sizeof(uint64_t));
    t->merge_v = (uint64_t *)calloc(mcap, sizeof(uint64_t));
    for (uint32_t i = 0; i < mcap; i++) t->merge_k[i] = MERGE_EMPTY;
    t->merge_mask = mcap - 1;

    char errm[128] = {0};
    for (size_t rank = 0; rank < n_merges; rank++) {
        JVal *mv = json_arr_get(merges, rank);
        if (json_type(mv) != J_STR) continue;
        const char *ms = json_str(mv);
        size_t mn = json_strlen(mv);
        const char *sp = memchr(ms, ' ', mn);
        if (!sp) {
            snprintf(errm, sizeof errm, "bad merge %zu", rank);
            break;
        }
        size_t ln = (size_t)(sp - ms), rn = mn - ln - 1;
        int32_t l = s2i_get(t, ms, ln);
        int32_t r = s2i_get(t, sp + 1, rn);
        /* merged token string = left + right */
        char buf[1024];
        char *cat = buf;
        if (ln + rn >= sizeof buf) cat = (char *)malloc(ln + rn + 1);
        memcpy(cat, ms, ln);
        memcpy(cat + ln, sp + 1, rn);
        int32_t res = s2i_get(t, cat, ln + rn);
        if (cat != buf) free(cat);
        if (l < 0 || r < 0 || res < 0) {
            snprintf(errm, sizeof errm, "merge %zu references unknown token", rank);
            break;
        }
        merge_put(t, (uint32_t)l, (uint32_t)r, (uint32_t)res, (uint32_t)rank);
    }
    if (errm[0]) {
        fprintf(stderr, "tok_load: %s\n", errm);
        tok_free(t);
        json_free(root);
        return NULL;
    }

    /* byte -> id table + reverse decode map */
    for (int i = 0; i < 0x144; i++) t->rev[i] = -1;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = b2u_cp(b);
        char enc[4];
        int en = utf8_encode(cp, enc);
        int32_t id = s2i_get(t, enc, (size_t)en);
        if (id < 0) {
            fprintf(stderr, "tok_load: byte char 0x%02x missing from vocab\n", b);
            tok_free(t);
            json_free(root);
            return NULL;
        }
        t->byte_id[b] = (uint32_t)id;
        if (cp < 0x144) t->rev[cp] = (int16_t)b;
    }

    /* added tokens sorted by (first byte, -length) */
    t->n_added = (int)n_added;
    t->added = (char **)calloc(n_added ? n_added : 1, sizeof(char *));
    t->added_len = (uint32_t *)calloc(n_added ? n_added : 1, sizeof(uint32_t));
    t->added_id = (uint32_t *)calloc(n_added ? n_added : 1, sizeof(uint32_t));
    /* insertion sort into buckets is overkill; collect then qsort with idx */
    {
        typedef struct { const char *s; uint32_t n, id; } AC;
        AC *tmp = (AC *)malloc((n_added ? n_added : 1) * sizeof(AC));
        size_t m = 0;
        for (size_t i = 0; i < n_added; i++) {
            JVal *a = json_arr_get(added, i);
            JVal *cv = json_obj_get(a, "content");
            JVal *idv = json_obj_get(a, "id");
            if (!cv || !idv || json_type(cv) != J_STR || json_strlen(cv) == 0) continue;
            tmp[m].s = json_str(cv);
            tmp[m].n = (uint32_t)json_strlen(cv);
            tmp[m].id = (uint32_t)json_num(idv);
            m++;
        }
        /* simple insertion sort: (first byte asc, length desc) — n is ~1.3k */
        for (size_t i = 1; i < m; i++) {
            AC v = tmp[i];
            size_t j = i;
            while (j > 0) {
                AC u = tmp[j - 1];
                int cmp = (unsigned char)u.s[0] - (unsigned char)v.s[0];
                if (cmp > 0 || (cmp == 0 && u.n < v.n)) {
                    tmp[j] = u;
                    j--;
                } else break;
            }
            tmp[j] = v;
        }
        for (int b = 0; b < 256; b++) { t->added_start[b] = (uint32_t)m; t->added_end[b] = (uint32_t)m; }
        for (size_t i = 0; i < m; i++) {
            t->added[i] = (char *)malloc(tmp[i].n + 1);
            memcpy(t->added[i], tmp[i].s, tmp[i].n + 1);
            t->added_len[i] = tmp[i].n;
            t->added_id[i] = tmp[i].id;
            unsigned char b0 = (unsigned char)tmp[i].s[0];
            if (i < t->added_start[b0] || t->added_end[b0] == t->added_start[b0])
                t->added_start[b0] = (uint32_t)i;
            t->added_end[b0] = (uint32_t)i + 1;
        }
        /* fix starts for empty buckets: start[b] = first index > b bucket */
        for (int b = 255; b >= 0; b--) {
            if (t->added_end[b] == t->added_start[b]) {
                /* empty bucket: point at next non-empty bucket start */
                uint32_t s = (uint32_t)m;
                for (int b2 = b + 1; b2 < 256; b2++)
                    if (t->added_end[b2] > t->added_start[b2]) { s = t->added_start[b2]; break; }
                t->added_start[b] = s;
                t->added_end[b] = s;
            }
        }
        free(tmp);
        t->n_added = (int)m;
    }

    json_free(root);
    return t;
}

void tok_free(Tok *t) {
    if (!t) return;
    if (t->strs)
        for (int i = 0; i < t->n_ids; i++) free(t->strs[i]);
    free(t->strs);
    free(t->lens);
    free(t->s2i);
    free(t->merge_k);
    free(t->merge_v);
    if (t->added)
        for (int i = 0; i < t->n_added; i++) free(t->added[i]);
    free(t->added);
    free(t->added_len);
    free(t->added_id);
    free(t);
}

int tok_n_tokens(const Tok *t) { return t->n_ids; }

int32_t tok_token_to_id(const Tok *t, const char *s) {
    return s2i_get(t, s, strlen(s));
}

const char *tok_id_to_token(const Tok *t, uint32_t id) {
    return (int)id < t->n_ids ? t->strs[id] : NULL;
}

/* ---------------- pre-tokenizer ---------------- */

typedef struct {
    uint32_t *cps;   /* codepoints */
    size_t *off;     /* byte offset of each cp (len cps+1) */
    size_t n;
} U8Text;

static void u8_decode_all(const char *text, size_t len, U8Text *u) {
    u->cps = (uint32_t *)malloc((len + 1) * sizeof(uint32_t));
    u->off = (size_t *)malloc((len + 2) * sizeof(size_t));
    u->n = 0;
    size_t i = 0;
    while (i < len) {
        size_t adv;
        uint32_t cp = utf8_decode((const unsigned char *)text + i, len - i, &adv);
        u->off[u->n] = i;
        u->cps[u->n++] = cp;
        i += adv;
    }
    u->off[u->n] = len;
}

static void u8_free(U8Text *u) {
    free(u->cps);
    free(u->off);
    u->cps = NULL;
    u->off = NULL;
    u->n = 0;
}

/* piece = half-open cp index range */
typedef struct {
    uint32_t *a;   /* 2*k: start, end */
    size_t n;      /* number of pieces */
    size_t cap;
} PieceList;

static void pieces_init(PieceList *pl) { pl->a = NULL; pl->n = 0; pl->cap = 0; }
static void pieces_free(PieceList *pl) { free(pl->a); pl->a = NULL; pl->n = pl->cap = 0; }
static void pieces_push(PieceList *pl, uint32_t s, uint32_t e) {
    if (s >= e) return;
    if (pl->n == pl->cap) {
        pl->cap = pl->cap ? pl->cap * 2 : 32;
        pl->a = (uint32_t *)realloc(pl->a, pl->cap * 2 * sizeof(uint32_t));
    }
    pl->a[pl->n * 2] = s;
    pl->a[pl->n * 2 + 1] = e;
    pl->n++;
}

/* stage 1: Split(\p{N}{1,3}, Isolated) */
static void stage1(const U8Text *u, const PieceList *in, PieceList *out) {
    for (size_t p = 0; p < in->n; p++) {
        uint32_t s = in->a[2 * p], e = in->a[2 * p + 1];
        uint32_t i = s;
        while (i < e) {
            if (cls_N(u->cps[i])) {
                uint32_t run = i;
                while (run < e && cls_N(u->cps[run])) run++;
                for (uint32_t k = i; k < run; k += 3)
                    pieces_push(out, k, k + 3 < run ? k + 3 : run);
                i = run;
            } else {
                uint32_t run = i;
                while (run < e && !cls_N(u->cps[run])) run++;
                pieces_push(out, i, run);
                i = run;
            }
        }
    }
}

/* stage 2: Split([一-龥぀-ゟ゠-ヿ]+, Isolated) */
static void stage2(const U8Text *u, const PieceList *in, PieceList *out) {
    for (size_t p = 0; p < in->n; p++) {
        uint32_t s = in->a[2 * p], e = in->a[2 * p + 1];
        uint32_t i = s;
        while (i < e) {
            uint32_t run = i;
            if (is_cjk(u->cps[i])) {
                while (run < e && is_cjk(u->cps[run])) run++;
            } else {
                while (run < e && !is_cjk(u->cps[run])) run++;
            }
            pieces_push(out, i, run);
            i = run;
        }
    }
}

/* stage 3 matcher: at cp index i within [cs,ce), return match end (>i) or i.
 * Implements the six alternatives with leftmost-first priority. */
static uint32_t stage3_match(const uint32_t *cps, uint32_t i, uint32_t ce) {
    uint32_t cp = cps[i];

    /* A1: ASCII punct + [A-Za-z]+ */
    if (is_a1_punct(cp) && i + 1 < ce && is_ascii_alpha(cps[i + 1])) {
        uint32_t j = i + 1;
        while (j < ce && is_ascii_alpha(cps[j])) j++;
        return j;
    }
    /* A2: [^\r\n L P S]? [LM]+ */
    if (cls_LM(cp)) {
        uint32_t j = i;
        while (j < ce && cls_LM(cps[j])) j++;
        return j;
    }
    if (cp != 0x0D && cp != 0x0A && !cls_PS(cp)) {
        if (i + 1 < ce && cls_LM(cps[i + 1])) {
            uint32_t j = i + 1;
            while (j < ce && cls_LM(cps[j])) j++;
            return j;
        }
    }
    /* A3: ' '? [PS]+ [\r\n]* */
    {
        uint32_t k = i;
        int ok = 0;
        if (cp == ' ') {
            if (i + 1 < ce && cls_PS(cps[i + 1])) { k = i + 1; ok = 1; }
        } else if (cls_PS(cp)) {
            ok = 1;
        }
        if (ok) {
            uint32_t j = k;
            while (j < ce && cls_PS(cps[j])) j++;
            while (j < ce && (cps[j] == 0x0D || cps[j] == 0x0A)) j++;
            return j;
        }
    }
    /* A4/A5/A6: whitespace */
    if (cls_WS(cp)) {
        uint32_t r = i;
        while (r < ce && cls_WS(cps[r])) r++;
        /* A4: \s*[\r\n]+ — longest prefix of the run ending after a CR/LF */
        int32_t last_nl = -1;
        for (uint32_t k = i; k < r; k++)
            if (cps[k] == 0x0D || cps[k] == 0x0A) last_nl = (int32_t)k;
        if (last_nl >= 0) return (uint32_t)last_nl + 1;
        /* A5: \s+(?!\S) — whole run at end, else run minus one */
        if (r == ce) return r;
        if (r - i >= 2) return r - 1;
        /* A6: \s+ */
        return r;
    }
    return i;
}

static void stage3(const U8Text *u, const PieceList *in, PieceList *out) {
    for (size_t p = 0; p < in->n; p++) {
        uint32_t s = in->a[2 * p], e = in->a[2 * p + 1];
        uint32_t i = s, gap = s;
        while (i < e) {
            uint32_t end = stage3_match(u->cps, i, e);
            if (end > i) {
                if (i > gap) pieces_push(out, gap, i);
                pieces_push(out, i, end);
                i = end;
                gap = i;
            } else {
                i++;
            }
        }
        if (gap < e) pieces_push(out, gap, e);
    }
}

/* ---------------- BPE ---------------- */

static void bpe_piece(const Tok *t, const char *bytes, size_t n,
                      uint32_t **out, size_t *out_n, size_t *out_cap,
                      uint32_t *work) {
    if (!n) return;
    uint32_t m = 0;
    for (size_t i = 0; i < n; i++) work[m++] = t->byte_id[(unsigned char)bytes[i]];
    while (m > 1) {
        uint32_t best_rank = 0x1FFFFu, bl = 0, br = 0, bres = 0;
        int found = 0;
        for (uint32_t i = 0; i + 1 < m; i++) {
            uint32_t res, rank;
            if (merge_get(t, work[i], work[i + 1], &res, &rank) && rank < best_rank) {
                best_rank = rank;
                bl = work[i];
                br = work[i + 1];
                bres = res;
                found = 1;
            }
        }
        if (!found) break;
        uint32_t w = 0;
        for (uint32_t i = 0; i < m; i++) {
            if (i + 1 < m && work[i] == bl && work[i + 1] == br) {
                work[w++] = bres;
                i++;
            } else {
                work[w++] = work[i];
            }
        }
        m = w;
    }
    if (*out_n + m > *out_cap) {
        while (*out_n + m > *out_cap) *out_cap = *out_cap ? *out_cap * 2 : 64;
        *out = (uint32_t *)realloc(*out, *out_cap * sizeof(uint32_t));
    }
    memcpy(*out + *out_n, work, m * sizeof(uint32_t));
    *out_n += m;
}

/* ---------------- added token matching ---------------- */

static size_t added_match(const Tok *t, const char *text, size_t len, uint32_t *id_out) {
    if (!len) return 0;
    unsigned char b0 = (unsigned char)text[0];
    for (uint32_t i = t->added_start[b0]; i < t->added_end[b0]; i++) {
        if (t->added_len[i] <= len && memcmp(text, t->added[i], t->added_len[i]) == 0) {
            *id_out = t->added_id[i];
            return t->added_len[i];
        }
    }
    return 0;
}

/* ---------------- encode ---------------- */

static void encode_normal(const Tok *t, const char *text, size_t len,
                          uint32_t **out, size_t *out_n, size_t *out_cap) {
    if (!len) return;
    U8Text u;
    u8_decode_all(text, len, &u);

    PieceList p0, p1, p2, p3;
    pieces_init(&p0);
    pieces_init(&p1);
    pieces_init(&p2);
    pieces_init(&p3);
    pieces_push(&p0, 0, (uint32_t)u.n);
    stage1(&u, &p0, &p1);
    stage2(&u, &p1, &p2);
    stage3(&u, &p2, &p3);

    uint32_t *work = (uint32_t *)malloc((len + 1) * sizeof(uint32_t));
    for (size_t p = 0; p < p3.n; p++) {
        size_t bs = u.off[p3.a[2 * p]], be = u.off[p3.a[2 * p + 1]];
        bpe_piece(t, text + bs, be - bs, out, out_n, out_cap, work);
    }
    free(work);
    pieces_free(&p0);
    pieces_free(&p1);
    pieces_free(&p2);
    pieces_free(&p3);
    u8_free(&u);
}

uint32_t *tok_encode(const Tok *t, const char *text, size_t len,
                     int add_special, size_t *n_out) {
    uint32_t *out = NULL;
    size_t n = 0, cap = 0;
    size_t span = 0, pos = 0;
    while (pos < len) {
        uint32_t aid = 0;
        size_t m = add_special ? added_match(t, text + pos, len - pos, &aid) : 0;
        if (m) {
            encode_normal(t, text + span, pos - span, &out, &n, &cap);
            if (n + 1 > cap) {
                cap = cap ? cap * 2 : 64;
                out = (uint32_t *)realloc(out, cap * sizeof(uint32_t));
            }
            out[n++] = aid;
            pos += m;
            span = pos;
        } else {
            pos++;
        }
    }
    encode_normal(t, text + span, len - span, &out, &n, &cap);
    *n_out = n;
    return out;
}

uint32_t *tok_encode_str(const Tok *t, const char *text,
                         int add_special, size_t *n_out) {
    return tok_encode(t, text, strlen(text), add_special, n_out);
}

/* ---------------- decode ---------------- */

char *tok_decode(const Tok *t, const uint32_t *ids, size_t n, size_t *len_out) {
    size_t cap = 64, len = 0;
    char *out = (char *)malloc(cap);
    for (size_t i = 0; i < n; i++) {
        if ((int)ids[i] >= t->n_ids || !t->strs[ids[i]]) continue;
        const unsigned char *s = (const unsigned char *)t->strs[ids[i]];
        size_t sn = t->lens[ids[i]], j = 0;
        while (j < sn) {
            size_t adv;
            uint32_t cp = utf8_decode(s + j, sn - j, &adv);
            char bytes[4];
            size_t bn;
            if (cp < 0x144 && t->rev[cp] >= 0) {
                bytes[0] = (char)t->rev[cp];
                bn = 1;
            } else if (cp == 0x110000) {
                /* invalid byte in token string: pass through */
                bytes[0] = (char)s[j];
                bn = 1;
                adv = 1;
            } else {
                bn = (size_t)utf8_encode(cp, bytes);
            }
            if (len + bn + 1 > cap) {
                while (len + bn + 1 > cap) cap *= 2;
                out = (char *)realloc(out, cap);
            }
            memcpy(out + len, bytes, bn);
            len += bn;
            j += adv;
        }
    }
    out[len] = '\0';
    if (len_out) *len_out = len;
    return out;
}

#endif /* APUS_TOK_IMPL_DONE */
