/*
 * apus json.h — minimal JSON parser/serializer, C11, libc only.
 *
 * Sufficient for tokenizer.json (6.4 MB: strings with full unicode escapes,
 * numbers, arrays, objects, bools, null) and for message-list inputs.
 *
 * Serialization (json_dumps) is byte-compatible with Python's
 * json.dumps(obj, ensure_ascii=False): ", " / ": " separators, insertion
 * order preserved, Python float repr, Python string escaping.
 *
 * Usage: #define APUS_JSON_IMPLEMENTATION in exactly one translation unit.
 */
#ifndef APUS_JSON_H
#define APUS_JSON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JVal JVal;

/* ---------- growable byte buffer (shared with encoding.h) ---------- */
typedef struct {
    char  *p;
    size_t n;
    size_t cap;
} SBuf;

void   sb_init(SBuf *b);
void   sb_free(SBuf *b);
void   sb_reserve(SBuf *b, size_t extra);
void   sb_putc(SBuf *b, char c);
void   sb_write(SBuf *b, const void *data, size_t len);
void   sb_puts(SBuf *b, const char *s);          /* NUL-terminated */
char  *sb_steal(SBuf *b);                        /* NUL-terminated heap string */

/* ---------- parse / free ---------- */
JVal *json_parse(const char *text, size_t len, char *err, size_t errcap);
JVal *json_parse_file(const char *path, char *err, size_t errcap);
void  json_free(JVal *v);
JVal *json_clone(const JVal *v);

/* ---------- accessors ---------- */
JType       json_type(const JVal *v);
int         json_bool(const JVal *v);            /* J_BOOL */
double      json_num(const JVal *v);             /* J_NUM */
const char *json_str(const JVal *v);             /* J_STR: UTF-8, NUL-terminated */
size_t      json_strlen(const JVal *v);
size_t      json_arr_len(const JVal *v);
JVal       *json_arr_get(const JVal *v, size_t i);
JVal       *json_obj_get(const JVal *v, const char *key);   /* NULL if absent */
JVal       *json_obj_getn(const JVal *v, const char *key, size_t klen);
size_t      json_obj_len(const JVal *v);
const char *json_obj_key(const JVal *v, size_t i);
JVal       *json_obj_val(const JVal *v, size_t i);
int         json_truthy(const JVal *v);          /* Python truthiness */

/* ---------- constructors / mutation (take ownership of children) ---------- */
JVal *json_new_null(void);
JVal *json_new_bool(int b);
JVal *json_new_int(long long x);
JVal *json_new_strn(const char *s, size_t len);
JVal *json_new_str(const char *s);
JVal *json_new_arr(void);
JVal *json_new_obj(void);
void  json_arr_push(JVal *arr, JVal *v);
void  json_arr_replace(JVal *arr, size_t i, JVal *v);   /* frees old item */
void  json_obj_setn(JVal *obj, const char *key, size_t klen, JVal *v);
void  json_obj_set(JVal *obj, const char *key, JVal *v);
void  json_obj_del(JVal *obj, const char *key);

/* ---------- serialize: Python json.dumps(ensure_ascii=False) ---------- */
char *json_dumps(const JVal *v);                 /* malloc'd */
void  json_dumps_sb(SBuf *b, const JVal *v);
/* Python repr() of a double, into buf (used for J_NUM non-integer values) */
void  json_py_float_repr(double d, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* APUS_JSON_H */

/* ================================================================== */
#if defined(APUS_JSON_IMPLEMENTATION) && !defined(APUS_JSON_IMPL_DONE)
#define APUS_JSON_IMPL_DONE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---------- SBuf ---------- */
void sb_init(SBuf *b) { b->p = NULL; b->n = 0; b->cap = 0; }

void sb_free(SBuf *b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

void sb_reserve(SBuf *b, size_t extra) {
    if (b->n + extra + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 64;
        while (nc < b->n + extra + 1) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) { fprintf(stderr, "apus: out of memory\n"); exit(1); }
        b->p = np;
        b->cap = nc;
    }
}

void sb_putc(SBuf *b, char c) {
    sb_reserve(b, 1);
    b->p[b->n++] = c;
}

void sb_write(SBuf *b, const void *data, size_t len) {
    if (!len) return;
    sb_reserve(b, len);
    memcpy(b->p + b->n, data, len);
    b->n += len;
}

void sb_puts(SBuf *b, const char *s) { sb_write(b, s, strlen(s)); }

char *sb_steal(SBuf *b) {
    sb_reserve(b, 0);
    b->p[b->n] = '\0';
    char *r = b->p;
    b->p = NULL; b->n = b->cap = 0;
    return r;
}

/* ---------- JVal ---------- */
struct JVal {
    JType type;
    union {
        int boolean;                     /* J_BOOL */
        struct { char *raw; double d; } num;   /* J_NUM: raw text + value */
        struct { char *p; size_t n; } str;     /* J_STR */
        struct { JVal **items; size_t n, cap; } arr;
        struct { char **keys; JVal **vals; size_t n, cap; } obj;
    } u;
};

static JVal *jv_new(JType t) {
    JVal *v = (JVal *)calloc(1, sizeof(JVal));
    if (!v) { fprintf(stderr, "apus: out of memory\n"); exit(1); }
    v->type = t;
    return v;
}

JVal *json_new_null(void) { return jv_new(J_NULL); }

JVal *json_new_bool(int b) { JVal *v = jv_new(J_BOOL); v->u.boolean = !!b; return v; }

JVal *json_new_int(long long x) {
    JVal *v = jv_new(J_NUM);
    char tmp[32];
    snprintf(tmp, sizeof tmp, "%lld", x);
    v->u.num.raw = strdup(tmp);
    v->u.num.d = (double)x;
    return v;
}

JVal *json_new_strn(const char *s, size_t len) {
    JVal *v = jv_new(J_STR);
    v->u.str.p = (char *)malloc(len + 1);
    memcpy(v->u.str.p, s, len);
    v->u.str.p[len] = '\0';
    v->u.str.n = len;
    return v;
}

JVal *json_new_str(const char *s) { return json_new_strn(s, strlen(s)); }

JVal *json_new_arr(void) { return jv_new(J_ARR); }
JVal *json_new_obj(void) { return jv_new(J_OBJ); }

void json_arr_push(JVal *arr, JVal *v) {
    if (arr->u.arr.n == arr->u.arr.cap) {
        size_t nc = arr->u.arr.cap ? arr->u.arr.cap * 2 : 8;
        arr->u.arr.items = (JVal **)realloc(arr->u.arr.items, nc * sizeof(JVal *));
        arr->u.arr.cap = nc;
    }
    arr->u.arr.items[arr->u.arr.n++] = v;
}

void json_arr_replace(JVal *arr, size_t i, JVal *v) {
    if (i >= arr->u.arr.n) { json_free(v); return; }
    json_free(arr->u.arr.items[i]);
    arr->u.arr.items[i] = v;
}

/* Lookup scans from the end: with duplicate keys the last value wins,
 * matching Python json.loads -> dict semantics. */
static size_t obj_find(const JVal *obj, const char *key, size_t klen) {
    for (size_t i = obj->u.obj.n; i-- > 0;)
        if (strlen(obj->u.obj.keys[i]) == klen && memcmp(obj->u.obj.keys[i], key, klen) == 0)
            return i;
    return (size_t)-1;
}

/* append without duplicate-key check (parser fast path) */
static void obj_append(JVal *obj, char *key, JVal *v) {
    if (obj->u.obj.n == obj->u.obj.cap) {
        size_t nc = obj->u.obj.cap ? obj->u.obj.cap * 2 : 8;
        obj->u.obj.keys = (char **)realloc(obj->u.obj.keys, nc * sizeof(char *));
        obj->u.obj.vals = (JVal **)realloc(obj->u.obj.vals, nc * sizeof(JVal *));
        obj->u.obj.cap = nc;
    }
    obj->u.obj.keys[obj->u.obj.n] = key;
    obj->u.obj.vals[obj->u.obj.n] = v;
    obj->u.obj.n++;
}

void json_obj_setn(JVal *obj, const char *key, size_t klen, JVal *v) {
    size_t i = obj_find(obj, key, klen);
    if (i != (size_t)-1) {
        json_free(obj->u.obj.vals[i]);
        obj->u.obj.vals[i] = v;
        return;
    }
    char *k = (char *)malloc(klen + 1);
    memcpy(k, key, klen);
    k[klen] = '\0';
    obj_append(obj, k, v);
}

void json_obj_set(JVal *obj, const char *key, JVal *v) {
    json_obj_setn(obj, key, strlen(key), v);
}

void json_obj_del(JVal *obj, const char *key) {
    size_t i = obj_find(obj, key, strlen(key));
    if (i == (size_t)-1) return;
    free(obj->u.obj.keys[i]);
    json_free(obj->u.obj.vals[i]);
    memmove(obj->u.obj.keys + i, obj->u.obj.keys + i + 1,
            (obj->u.obj.n - i - 1) * sizeof(char *));
    memmove(obj->u.obj.vals + i, obj->u.obj.vals + i + 1,
            (obj->u.obj.n - i - 1) * sizeof(JVal *));
    obj->u.obj.n--;
}

void json_free(JVal *v) {
    if (!v) return;
    switch (v->type) {
    case J_NUM: free(v->u.num.raw); break;
    case J_STR: free(v->u.str.p); break;
    case J_ARR:
        for (size_t i = 0; i < v->u.arr.n; i++) json_free(v->u.arr.items[i]);
        free(v->u.arr.items);
        break;
    case J_OBJ:
        for (size_t i = 0; i < v->u.obj.n; i++) {
            free(v->u.obj.keys[i]);
            json_free(v->u.obj.vals[i]);
        }
        free(v->u.obj.keys);
        free(v->u.obj.vals);
        break;
    default: break;
    }
    free(v);
}

JVal *json_clone(const JVal *v) {
    if (!v) return NULL;
    switch (v->type) {
    case J_NULL: return json_new_null();
    case J_BOOL: return json_new_bool(v->u.boolean);
    case J_NUM: {
        JVal *r = jv_new(J_NUM);
        r->u.num.raw = strdup(v->u.num.raw);
        r->u.num.d = v->u.num.d;
        return r;
    }
    case J_STR: return json_new_strn(v->u.str.p, v->u.str.n);
    case J_ARR: {
        JVal *r = json_new_arr();
        for (size_t i = 0; i < v->u.arr.n; i++)
            json_arr_push(r, json_clone(v->u.arr.items[i]));
        return r;
    }
    case J_OBJ: {
        JVal *r = json_new_obj();
        for (size_t i = 0; i < v->u.obj.n; i++)
            json_obj_set(r, v->u.obj.keys[i], json_clone(v->u.obj.vals[i]));
        return r;
    }
    }
    return NULL;
}

/* ---------- accessors ---------- */
JType json_type(const JVal *v) { return v ? v->type : J_NULL; }
int json_bool(const JVal *v) { return v && v->type == J_BOOL ? v->u.boolean : 0; }
double json_num(const JVal *v) { return v && v->type == J_NUM ? v->u.num.d : 0.0; }
const char *json_str(const JVal *v) { return v && v->type == J_STR ? v->u.str.p : NULL; }
size_t json_strlen(const JVal *v) { return v && v->type == J_STR ? v->u.str.n : 0; }
size_t json_arr_len(const JVal *v) { return v && v->type == J_ARR ? v->u.arr.n : 0; }
JVal *json_arr_get(const JVal *v, size_t i) {
    return (v && v->type == J_ARR && i < v->u.arr.n) ? v->u.arr.items[i] : NULL;
}
JVal *json_obj_getn(const JVal *v, const char *key, size_t klen) {
    if (!v || v->type != J_OBJ) return NULL;
    size_t i = obj_find(v, key, klen);
    return i == (size_t)-1 ? NULL : v->u.obj.vals[i];
}
JVal *json_obj_get(const JVal *v, const char *key) {
    return json_obj_getn(v, key, strlen(key));
}
size_t json_obj_len(const JVal *v) { return v && v->type == J_OBJ ? v->u.obj.n : 0; }
const char *json_obj_key(const JVal *v, size_t i) {
    return (v && v->type == J_OBJ && i < v->u.obj.n) ? v->u.obj.keys[i] : NULL;
}
JVal *json_obj_val(const JVal *v, size_t i) {
    return (v && v->type == J_OBJ && i < v->u.obj.n) ? v->u.obj.vals[i] : NULL;
}

int json_truthy(const JVal *v) {
    if (!v) return 0;
    switch (v->type) {
    case J_NULL: return 0;
    case J_BOOL: return v->u.boolean;
    case J_NUM: return v->u.num.d != 0.0;
    case J_STR: return v->u.str.n != 0;
    case J_ARR: return v->u.arr.n != 0;
    case J_OBJ: return v->u.obj.n != 0;
    }
    return 0;
}

/* ---------- parser ---------- */
typedef struct {
    const char *p;
    const char *end;
    const char *start;
    char *err;
    size_t errcap;
    int depth;
} P;

#define JSON_MAX_DEPTH 512

static void p_err(P *p, const char *msg) {
    if (p->err && p->errcap)
        snprintf(p->err, p->errcap, "json parse error at byte %ld: %s",
                 (long)(p->p - p->start), msg);
}

static void p_skip_ws(P *p) {
    while (p->p < p->end) {
        char c = *p->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->p++;
        else break;
    }
}

static JVal *p_value(P *p);

static int hex4(P *p, unsigned *out) {
    if (p->end - p->p < 4) return -1;
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *p->p++;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static void utf8_emit(SBuf *b, unsigned cp) {
    if (cp < 0x80) {
        sb_putc(b, (char)cp);
    } else if (cp < 0x800) {
        sb_putc(b, (char)(0xC0 | (cp >> 6)));
        sb_putc(b, (char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        sb_putc(b, (char)(0xE0 | (cp >> 12)));
        sb_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F)));
        sb_putc(b, (char)(0x80 | (cp & 0x3F)));
    } else {
        sb_putc(b, (char)(0xF0 | (cp >> 18)));
        sb_putc(b, (char)(0x80 | ((cp >> 12) & 0x3F)));
        sb_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F)));
        sb_putc(b, (char)(0x80 | (cp & 0x3F)));
    }
}

static JVal *p_string(P *p) {
    /* *p->p == '"' */
    p->p++;
    SBuf b;
    sb_init(&b);
    while (p->p < p->end) {
        unsigned char c = (unsigned char)*p->p;
        if (c == '"') {
            p->p++;
            JVal *v = jv_new(J_STR);
            v->u.str.n = b.n;
            v->u.str.p = sb_steal(&b);
            return v;
        }
        if (c == '\\') {
            p->p++;
            if (p->p >= p->end) break;
            char e = *p->p++;
            switch (e) {
            case '"': sb_putc(&b, '"'); break;
            case '\\': sb_putc(&b, '\\'); break;
            case '/': sb_putc(&b, '/'); break;
            case 'b': sb_putc(&b, '\b'); break;
            case 'f': sb_putc(&b, '\f'); break;
            case 'n': sb_putc(&b, '\n'); break;
            case 'r': sb_putc(&b, '\r'); break;
            case 't': sb_putc(&b, '\t'); break;
            case 'u': {
                unsigned cp;
                if (hex4(p, &cp) != 0) { sb_free(&b); p_err(p, "bad \\u escape"); return NULL; }
                if (cp >= 0xD800 && cp <= 0xDBFF && p->end - p->p >= 6 &&
                    p->p[0] == '\\' && p->p[1] == 'u') {
                    unsigned lo;
                    const char *save = p->p;
                    p->p += 2;
                    if (hex4(p, &lo) == 0 && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else {
                        p->p = save; /* lone high surrogate: emit as-is (WTF-8) */
                    }
                }
                utf8_emit(&b, cp);
                break;
            }
            default:
                sb_free(&b);
                p_err(p, "bad escape");
                return NULL;
            }
        } else if (c < 0x20) {
            sb_free(&b);
            p_err(p, "raw control char in string");
            return NULL;
        } else {
            sb_putc(&b, (char)c);
            p->p++;
        }
    }
    sb_free(&b);
    p_err(p, "unterminated string");
    return NULL;
}

static JVal *p_number(P *p) {
    const char *start = p->p;
    if (p->p < p->end && *p->p == '-') p->p++;
    while (p->p < p->end && *p->p >= '0' && *p->p <= '9') p->p++;
    if (p->p < p->end && *p->p == '.') {
        p->p++;
        while (p->p < p->end && *p->p >= '0' && *p->p <= '9') p->p++;
    }
    if (p->p < p->end && (*p->p == 'e' || *p->p == 'E')) {
        p->p++;
        if (p->p < p->end && (*p->p == '+' || *p->p == '-')) p->p++;
        while (p->p < p->end && *p->p >= '0' && *p->p <= '9') p->p++;
    }
    size_t n = (size_t)(p->p - start);
    if (n == 0) { p_err(p, "bad number"); return NULL; }
    JVal *v = jv_new(J_NUM);
    v->u.num.raw = (char *)malloc(n + 1);
    memcpy(v->u.num.raw, start, n);
    v->u.num.raw[n] = '\0';
    v->u.num.d = strtod(v->u.num.raw, NULL);
    return v;
}

static JVal *p_array(P *p) {
    p->p++; /* '[' */
    if (++p->depth > JSON_MAX_DEPTH) { p_err(p, "too deep"); return NULL; }
    JVal *arr = json_new_arr();
    p_skip_ws(p);
    if (p->p < p->end && *p->p == ']') { p->p++; p->depth--; return arr; }
    for (;;) {
        p_skip_ws(p);
        JVal *v = p_value(p);
        if (!v) { json_free(arr); p->depth--; return NULL; }
        json_arr_push(arr, v);
        p_skip_ws(p);
        if (p->p >= p->end) break;
        if (*p->p == ',') { p->p++; continue; }
        if (*p->p == ']') { p->p++; p->depth--; return arr; }
        break;
    }
    json_free(arr);
    p->depth--;
    p_err(p, "unterminated array");
    return NULL;
}

static JVal *p_object(P *p) {
    p->p++; /* '{' */
    if (++p->depth > JSON_MAX_DEPTH) { p_err(p, "too deep"); return NULL; }
    JVal *obj = json_new_obj();
    p_skip_ws(p);
    if (p->p < p->end && *p->p == '}') { p->p++; p->depth--; return obj; }
    for (;;) {
        p_skip_ws(p);
        if (p->p >= p->end || *p->p != '"') {
            json_free(obj); p->depth--; p_err(p, "expected object key"); return NULL;
        }
        JVal *k = p_string(p);
        if (!k) { json_free(obj); p->depth--; return NULL; }
        p_skip_ws(p);
        if (p->p >= p->end || *p->p != ':') {
            json_free(k); json_free(obj); p->depth--; p_err(p, "expected ':'"); return NULL;
        }
        p->p++;
        p_skip_ws(p);
        JVal *v = p_value(p);
        if (!v) { json_free(k); json_free(obj); p->depth--; return NULL; }
        /* Fast append; duplicate keys are all kept (lookups return the last,
         * matching Python). Dedup would be O(n^2) on the 128k-key vocab. */
        {
            JVal *kv = k;
            obj_append(obj, kv->u.str.p, v);
            kv->u.str.p = NULL;
            json_free(kv);
        }
        p_skip_ws(p);
        if (p->p >= p->end) break;
        if (*p->p == ',') { p->p++; continue; }
        if (*p->p == '}') { p->p++; p->depth--; return obj; }
        break;
    }
    json_free(obj);
    p->depth--;
    p_err(p, "unterminated object");
    return NULL;
}

static JVal *p_value(P *p) {
    p_skip_ws(p);
    if (p->p >= p->end) { p_err(p, "unexpected end"); return NULL; }
    char c = *p->p;
    if (c == '"') return p_string(p);
    if (c == '{') return p_object(p);
    if (c == '[') return p_array(p);
    if (c == 't' && p->end - p->p >= 4 && memcmp(p->p, "true", 4) == 0) {
        p->p += 4; return json_new_bool(1);
    }
    if (c == 'f' && p->end - p->p >= 5 && memcmp(p->p, "false", 5) == 0) {
        p->p += 5; return json_new_bool(0);
    }
    if (c == 'n' && p->end - p->p >= 4 && memcmp(p->p, "null", 4) == 0) {
        p->p += 4; return json_new_null();
    }
    if (c == '-' || (c >= '0' && c <= '9')) return p_number(p);
    p_err(p, "unexpected character");
    return NULL;
}

JVal *json_parse(const char *text, size_t len, char *err, size_t errcap) {
    P p = { text, text + len, text, err, errcap, 0 };
    if (err && errcap) err[0] = '\0';
    JVal *v = p_value(&p);
    if (!v) {
        if (err && errcap && err[0] == '\0')
            snprintf(err, errcap, "json parse error");
        return NULL;
    }
    p_skip_ws(&p);
    if (p.p != p.end) {
        json_free(v);
        if (err && errcap)
            snprintf(err, errcap, "json parse error: trailing data at byte %ld",
                     (long)(p.p - text));
        return NULL;
    }
    return v;
}

JVal *json_parse_file(const char *path, char *err, size_t errcap) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err && errcap) snprintf(err, errcap, "cannot open %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); if (err && errcap) snprintf(err, errcap, "cannot stat %s", path); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        free(buf);
        if (err && errcap) snprintf(err, errcap, "cannot read %s", path);
        return NULL;
    }
    JVal *v = json_parse(buf, (size_t)sz, err, errcap);
    free(buf);
    return v;
}

/* ---------- Python-compatible serialization ---------- */

void json_py_float_repr(double d, char *buf, size_t cap) {
    if (isnan(d)) { snprintf(buf, cap, "NaN"); return; }
    if (isinf(d)) { snprintf(buf, cap, d < 0 ? "-Infinity" : "Infinity"); return; }
    if (d == 0.0) { snprintf(buf, cap, signbit(d) ? "-0.0" : "0.0"); return; }

    /* shortest round-trip digits */
    char tmp[40];
    int prec;
    for (prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*e", prec - 1, d);
        if (strtod(tmp, NULL) == d) break;
    }
    /* tmp = [-]d[.ddd]e[+-]xx ; extract digits and exponent x (value = m * 10^x) */
    char digits[20];
    int nd = 0;
    long x = 0;
    int neg = 0;
    const char *s = tmp;
    if (*s == '-') { neg = 1; s++; }
    while (*s && *s != 'e' && *s != 'E') {
        if (*s != '.') digits[nd++] = *s;
        s++;
    }
    if (*s == 'e' || *s == 'E') x = strtol(s + 1, NULL, 10);
    digits[nd] = '\0';
    long decpt = x + 1; /* number of digits left of the decimal point in fixed form */

    SBuf b;
    sb_init(&b);
    if (neg) sb_putc(&b, '-');
    if (decpt <= -4 || decpt > 16) {
        /* exponential: d[.ddd]e±XX (exponent at least 2 digits) */
        sb_putc(&b, digits[0]);
        if (nd > 1) {
            sb_putc(&b, '.');
            sb_write(&b, digits + 1, (size_t)(nd - 1));
        }
        char ebuf[24];
        snprintf(ebuf, sizeof ebuf, "e%c%02ld", x < 0 ? '-' : '+', x < 0 ? -x : x);
        sb_puts(&b, ebuf);
    } else if (decpt <= 0) {
        sb_puts(&b, "0.");
        for (long i = 0; i < -decpt; i++) sb_putc(&b, '0');
        sb_write(&b, digits, (size_t)nd);
    } else if (decpt >= nd) {
        sb_write(&b, digits, (size_t)nd);
        for (long i = 0; i < decpt - nd; i++) sb_putc(&b, '0');
        sb_puts(&b, ".0");
    } else {
        sb_write(&b, digits, (size_t)decpt);
        sb_putc(&b, '.');
        sb_write(&b, digits + decpt, (size_t)(nd - decpt));
    }
    char *r = sb_steal(&b);
    snprintf(buf, cap, "%s", r);
    free(r);
}

static void dumps_str(SBuf *b, const char *s, size_t n) {
    sb_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': sb_puts(b, "\\\""); break;
        case '\\': sb_puts(b, "\\\\"); break;
        case '\b': sb_puts(b, "\\b"); break;
        case '\f': sb_puts(b, "\\f"); break;
        case '\n': sb_puts(b, "\\n"); break;
        case '\r': sb_puts(b, "\\r"); break;
        case '\t': sb_puts(b, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof tmp, "\\u%04x", c);
                sb_puts(b, tmp);
            } else {
                sb_putc(b, (char)c);
            }
        }
    }
    sb_putc(b, '"');
}

static int raw_is_int(const char *raw) {
    /* ^-?\d+$ -> integer: print raw verbatim (Python ints are arbitrary
     * precision); the only non-canonical valid form is "-0" -> "0". */
    const char *s = raw;
    if (*s == '-') s++;
    if (!*s) return 0;
    for (; *s; s++)
        if (*s < '0' || *s > '9') return 0;
    return 1;
}

void json_dumps_sb(SBuf *b, const JVal *v) {
    if (!v) { sb_puts(b, "null"); return; }
    switch (v->type) {
    case J_NULL: sb_puts(b, "null"); break;
    case J_BOOL: sb_puts(b, v->u.boolean ? "true" : "false"); break;
    case J_NUM:
        if (raw_is_int(v->u.num.raw)) {
            if (strcmp(v->u.num.raw, "-0") == 0) sb_putc(b, '0');
            else sb_puts(b, v->u.num.raw);
        } else {
            char tmp[64];
            json_py_float_repr(v->u.num.d, tmp, sizeof tmp);
            sb_puts(b, tmp);
        }
        break;
    case J_STR: dumps_str(b, v->u.str.p, v->u.str.n); break;
    case J_ARR:
        sb_putc(b, '[');
        for (size_t i = 0; i < v->u.arr.n; i++) {
            if (i) sb_puts(b, ", ");
            json_dumps_sb(b, v->u.arr.items[i]);
        }
        sb_putc(b, ']');
        break;
    case J_OBJ:
        sb_putc(b, '{');
        for (size_t i = 0; i < v->u.obj.n; i++) {
            if (i) sb_puts(b, ", ");
            dumps_str(b, v->u.obj.keys[i], strlen(v->u.obj.keys[i]));
            sb_puts(b, ": ");
            json_dumps_sb(b, v->u.obj.vals[i]);
        }
        sb_putc(b, '}');
        break;
    }
}

char *json_dumps(const JVal *v) {
    SBuf b;
    sb_init(&b);
    json_dumps_sb(&b, v);
    return sb_steal(&b);
}

#endif /* APUS_JSON_IMPL_DONE */
