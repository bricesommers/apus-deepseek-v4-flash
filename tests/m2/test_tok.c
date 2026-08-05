/*
 * tests/m2/test_tok.c — tokenizer conformance vs Python-generated goldens.
 *
 * For every case in golden/tok_manifest.txt:
 *   - encode input twice (determinism), diff ids against golden exactly
 *   - decode(golden ids), diff bytes against golden exactly
 *   - round-trip: decode(encode(x)) == x
 *   - nosplit variants: encode with special recognition off
 * golden/specials.bin: every added token encodes to its id and decodes back
 * golden/codepoints.bin (optional): per-codepoint probe strings, exact ids
 * Tokenizer path: reference/tokenizer.json, overridable via APUS_TOK_JSON
 * (M10 tokenizer spot-check against reference-0731/tokenizer.json).
 * Run from the repository root.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#include "json.h"
#include "tok.h"

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

static int cmp_ids(const uint32_t *a, size_t na, const unsigned char *gold, size_t glen,
                   const char *what) {
    if (glen < 4) { fprintf(stderr, "FAIL %s: bad golden\n", what); return 0; }
    uint32_t n = rd_u32(gold);
    if ((size_t)n * 4 + 4 != glen || n != na) {
        fprintf(stderr, "FAIL %s: id count golden=%u got=%zu\n", what, n, na);
        return 0;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t g = rd_u32(gold + 4 + i * 4);
        if (g != a[i]) {
            fprintf(stderr, "FAIL %s: id[%u] golden=%u got=%u\n", what, i, g, a[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    Tok *t = tok_load(getenv("APUS_TOK_JSON") ? getenv("APUS_TOK_JSON")
                                              : "reference/tokenizer.json");
    if (!t) { fprintf(stderr, "tok_load failed\n"); return 1; }
    CHECK(tok_n_tokens(t) == 129280, "n_tokens=%d", tok_n_tokens(t));

    /* ---- manifest cases ---- */
    size_t mlen;
    char *manifest = read_file("tests/m2/golden/tok_manifest.txt", &mlen);
    if (!manifest) { fprintf(stderr, "missing tok_manifest.txt (run gen_golden.py)\n"); return 1; }

    char *save = NULL;
    for (char *name = strtok_r(manifest, "\n", &save); name;
         name = strtok_r(NULL, "\n", &save)) {
        char path[512];
        size_t tlen, ilen, dlen;
        snprintf(path, sizeof path, "tests/m2/golden/%s.txt", name);
        char *txt = read_file(path, &tlen);
        snprintf(path, sizeof path, "tests/m2/golden/%s.ids", name);
        char *ids_g = read_file(path, &ilen);
        snprintf(path, sizeof path, "tests/m2/golden/%s.dec", name);
        char *dec_g = read_file(path, &dlen);
        CHECK(txt && ids_g && dec_g, "case %s: missing golden files", name);
        if (!txt || !ids_g || !dec_g) { free(txt); free(ids_g); free(dec_g); continue; }

        size_t n1 = 0, n2 = 0;
        uint32_t *i1 = tok_encode(t, txt, tlen, 1, &n1);
        uint32_t *i2 = tok_encode(t, txt, tlen, 1, &n2);
        CHECK(n1 == n2 && (n1 == 0 || memcmp(i1, i2, n1 * 4) == 0),
              "case %s: non-deterministic encode", name);
        char lbl[600];
        snprintf(lbl, sizeof lbl, "case %s encode", name);
        CHECK(cmp_ids(i1, n1, (unsigned char *)ids_g, ilen, lbl), "case %s ids", name);

        size_t out_len = 0;
        char *out = tok_decode(t, i1, n1, &out_len);
        CHECK(out_len == dlen && memcmp(out, dec_g, dlen) == 0,
              "case %s: decode mismatch (golden %zu got %zu)", name, dlen, out_len);
        /* round-trip stability */
        CHECK(out_len == tlen && memcmp(out, txt, tlen) == 0,
              "case %s: round-trip mismatch", name);

        /* nosplit variant */
        snprintf(path, sizeof path, "tests/m2/golden/%s.nosplit.ids", name);
        size_t nslen;
        char *ns = read_file(path, &nslen);
        if (ns) {
            size_t n3 = 0;
            uint32_t *i3 = tok_encode(t, txt, tlen, 0, &n3);
            snprintf(lbl, sizeof lbl, "case %s nosplit", name);
            CHECK(cmp_ids(i3, n3, (unsigned char *)ns, nslen, lbl),
                  "case %s nosplit ids", name);
            free(i3);
            free(ns);
        }
        free(txt); free(ids_g); free(dec_g); free(i1); free(i2); free(out);
    }
    free(manifest);

    /* ---- specials coverage ---- */
    {
        size_t slen;
        unsigned char *sb = (unsigned char *)read_file("tests/m2/golden/specials.bin", &slen);
        CHECK(sb != NULL, "missing specials.bin");
        if (sb) {
            unsigned char covered[(129280 + 7) / 8];
            memset(covered, 0, sizeof covered);
            size_t pos = 0, nrec = 0;
            while (pos + 12 <= slen) {
                uint32_t id = rd_u32(sb + pos);
                uint32_t tl = rd_u32(sb + pos + 4);
                uint32_t ni = rd_u32(sb + pos + 8);
                pos += 12;
                CHECK(pos + tl + ni * 4 <= slen, "specials: truncated record");
                if (pos + tl + ni * 4 > slen) break;
                const char *text = (const char *)(sb + pos);
                size_t n = 0;
                uint32_t *ids = tok_encode(t, text, tl, 1, &n);
                CHECK(n == ni && memcmp(ids, sb + pos + tl, ni * 4) == 0,
                      "specials: id %u encode mismatch (n %zu vs %u)", id, n, ni);
                CHECK(n == 1 && ni == 1 && ids[0] == id,
                      "specials: id %u does not encode to itself", id);
                uint32_t one[1] = { id };
                size_t dl = 0;
                char *dec = tok_decode(t, one, 1, &dl);
                CHECK(dl == tl && memcmp(dec, text, tl) == 0,
                      "specials: id %u decode mismatch", id);
                covered[id >> 3] |= (unsigned char)(1u << (id & 7));
                free(ids); free(dec);
                pos += tl + ni * 4;
                nrec++;
            }
            CHECK(pos == slen, "specials: trailing bytes");
            size_t n_cov = 0;
            for (size_t i = 0; i < 129280; i++)
                if (covered[i >> 3] & (1u << (i & 7))) n_cov++;
            CHECK(n_cov == nrec, "specials: %zu records but %zu covered", nrec, n_cov);
            CHECK(n_cov == 1283, "specials: only %zu/1283 added ids covered", n_cov);
            printf("specials: %zu added-token records, all covered\n", nrec);
            free(sb);
        }
    }

    /* ---- optional exhaustive codepoint probes ---- */
    {
        size_t clen;
        unsigned char *cb = (unsigned char *)read_file("tests/m2/golden/codepoints.bin", &clen);
        if (cb) {
            CHECK(clen >= 8 && rd_u32(cb) == 0xC0DE0001u, "codepoints: bad magic");
            uint32_t nrec = clen >= 8 ? rd_u32(cb + 4) : 0;
            size_t pos = 8, done = 0, bad = 0;
            while (pos + 8 <= clen) {
                uint32_t tl = rd_u32(cb + pos);
                uint32_t ni = rd_u32(cb + pos + 4);
                pos += 8;
                if (pos + tl + ni * 4 > clen) { bad++; break; }
                const char *text = (const char *)(cb + pos);
                size_t n = 0;
                uint32_t *ids = tok_encode(t, text, tl, 1, &n);
                if (!(n == ni && (ni == 0 || memcmp(ids, cb + pos + tl, ni * 4) == 0))) {
                    if (bad < 10) {
                        fprintf(stderr, "FAIL codepoints: text");
                        for (uint32_t k = 0; k < tl; k++)
                            fprintf(stderr, " %02x", (unsigned char)text[k]);
                        fprintf(stderr, " golden n=%u got n=%zu", ni, n);
                        if (n == ni)
                            for (uint32_t k = 0; k < n; k++)
                                if (ids[k] != rd_u32(cb + pos + tl + k * 4))
                                    fprintf(stderr, " first diff [%u]=%u vs %u",
                                            k, rd_u32(cb + pos + tl + k * 4), ids[k]);
                        fprintf(stderr, "\n");
                    }
                    bad++;
                }
                free(ids);
                pos += tl + ni * 4;
                done++;
            }
            CHECK(bad == 0, "codepoints: %zu/%u mismatches", bad, nrec);
            CHECK(done == nrec, "codepoints: %zu/%u records read", done, nrec);
            printf("codepoints: %u probe records, %zu mismatches\n", nrec, bad);
            free(cb);
        } else {
            printf("codepoints: skipped (no codepoints.bin; run gen_golden.py --exhaustive)\n");
        }
    }

    /* ---- edge cases ---- */
    {
        size_t n = 999;
        uint32_t *ids = tok_encode(t, "", 0, 1, &n);
        CHECK(n == 0, "empty input: %zu ids", n);
        free(ids);
        /* unpaired lone byte (invalid UTF-8) must not crash */
        size_t n2 = 0;
        uint32_t *ids2 = tok_encode(t, "\xff\xfe abc", 7, 1, &n2);
        CHECK(n2 > 0, "invalid utf8: no ids");
        free(ids2);
        size_t dl = 0;
        uint32_t some[3] = { 100, 200, 300 };
        char *d = tok_decode(t, some, 3, &dl);
        free(d);
    }

    tok_free(t);
    printf("test_tok: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
