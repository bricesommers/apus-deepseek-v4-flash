/*
 * c/sample.h — sampling for the apus engine: greedy argmax, temperature +
 * top-p (nucleus) sampling, and a deterministic RNG. C11, libc only.
 *
 * Contract (pinned by tests/m5 against the numpy oracle, tools/oracle.py
 * probs_from_logits/top_p_draw):
 *   - argmax: lowest index wins exact ties (numpy argmax semantics).
 *   - temperature: z = logits/temp in FP32, max-subtracted softmax, FP32.
 *   - top-p: sort probabilities descending, stable on exact ties (lower
 *     index first); keep token i iff cumsum_before_i <= top_p (the HF
 *     shift rule; always keeps the top token); renormalize kept mass.
 *   - draw: smallest j with CDF[j] > u, u in [0,1); if rounding leaves no
 *     such j, the last kept token. Given the same probabilities and the
 *     same u, the numpy oracle and this header pick the same token.
 *
 * RNG: splitmix64 — deterministic, explicit seed (CLI --seed), one f64
 * uniform in [0,1) per sampled token. Reference sampling defaults are
 * temperature 1.0, top_p 1.0 (reference/generation_config.json);
 * temp <= 0 means greedy.
 *
 * Usage: #define APUS_SAMPLE_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_SAMPLE_H
#define APUS_SAMPLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Greedy: index of the max logit, lowest index on exact ties. */
int apus_sample_argmax(const float *logits, size_t n);

/* Scratch sizing for the calls below. top_p needs n int32 (the sorted
 * order; probabilities are read, never copied); logits_u additionally
 * needs n floats (the softmax probs). */
size_t apus_sample_scratch_size(size_t n);    /* n*(sizeof(float)+sizeof(int32_t)) */
size_t apus_top_p_scratch_size(size_t n);     /* n*sizeof(int32_t) */

/* Top-p nucleus sample from probabilities with an explicit uniform
 * u in [0,1). probs need not sum exactly to 1 (renormalized on the fly).
 * scratch: apus_top_p_scratch_size(n) bytes, or NULL for
 * n <= APUS_SAMPLE_STACK_MAX (stack buffer). */
#define APUS_SAMPLE_STACK_MAX 8192
int apus_sample_top_p_u(const float *probs, size_t n, float top_p, double u,
                        void *scratch);

/* Softmax(logits/temp) then top-p draw with u. temp <= 0 -> argmax
 * (top_p and u ignored). scratch: apus_sample_scratch_size(n) or NULL for
 * small n. */
int apus_sample_logits_u(const float *logits, size_t n, float temp,
                         float top_p, double u, void *scratch);

/* splitmix64 stream. */
typedef struct { uint64_t s; } ApusRng;
void     apus_rng_seed(ApusRng *r, uint64_t seed);
uint64_t apus_rng_next(ApusRng *r);
double   apus_rng_uniform(ApusRng *r);        /* [0,1), 53-bit */

/* Full pipeline with the internal RNG: temp <= 0 -> greedy. */
int apus_sample(const float *logits, size_t n, float temp, float top_p,
                ApusRng *rng, void *scratch);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_SAMPLE_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>

int apus_sample_argmax(const float *logits, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;   /* strict >: ties -> lowest */
    return (int)best;
}

size_t apus_sample_scratch_size(size_t n) {
    return n * (sizeof(float) + sizeof(int32_t));
}

size_t apus_top_p_scratch_size(size_t n) {
    return n * sizeof(int32_t);
}

int apus_sample_top_p_u(const float *probs, size_t n, float top_p, double u,
                        void *scratch) {
    int32_t sbuf[APUS_SAMPLE_STACK_MAX];
    int32_t *ord = scratch ? (int32_t *)scratch : sbuf;
    for (size_t i = 0; i < n; i++) ord[i] = (int32_t)i;
    /* sort ord by prob descending; exact FP32 ties break by lower index
     * (== numpy stable argsort). Shell sort: O(n^1.5) worst, no globals. */
    for (size_t gap = n / 2; gap; gap /= 2)
        for (size_t i = gap; i < n; i++) {
            int32_t v = ord[i];
            size_t j = i;
            while (j >= gap) {
                int32_t w = ord[j - gap];
                if (probs[w] > probs[v] || (probs[w] == probs[v] && w < v))
                    break;
                ord[j] = w;
                j -= gap;
            }
            ord[j] = v;
        }
    /* nucleus: keep token i iff cumsum_before_i <= top_p (HF shift rule);
     * always keeps at least the top token */
    double cum = 0.0, kept = 0.0;
    size_t nk = 0;
    for (size_t i = 0; i < n; i++) {
        if (cum <= (double)top_p) {
            cum += probs[ord[i]];
            kept = cum;
            nk = i + 1;
        } else {
            break;
        }
    }
    if (nk == 0) { nk = 1; kept = probs[ord[0]]; }
    /* renormalized CDF draw: smallest j with CDF[j] > u */
    double c = 0.0;
    for (size_t i = 0; i < nk; i++) {
        c += (double)probs[ord[i]] / kept;
        if (u < c) return ord[i];
    }
    return ord[nk - 1];
}

int apus_sample_logits_u(const float *logits, size_t n, float temp,
                         float top_p, double u, void *scratch) {
    if (temp <= 0.0f) return apus_sample_argmax(logits, n);
    float psbuf[APUS_SAMPLE_STACK_MAX];
    int32_t isbuf[APUS_SAMPLE_STACK_MAX];
    float *p;
    void *oscratch;
    if (scratch) {
        p = (float *)scratch;
        oscratch = (int32_t *)(p + n);
    } else {
        p = psbuf;
        oscratch = isbuf;
    }
    float mx = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        p[i] = expf((logits[i] - mx) / temp);
        sum += p[i];
    }
    for (size_t i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
    return apus_sample_top_p_u(p, n, top_p, u, oscratch);
}

void apus_rng_seed(ApusRng *r, uint64_t seed) { r->s = seed; }

uint64_t apus_rng_next(ApusRng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double apus_rng_uniform(ApusRng *r) {
    return (double)(apus_rng_next(r) >> 11) * 0x1.0p-53;
}

int apus_sample(const float *logits, size_t n, float temp, float top_p,
                ApusRng *rng, void *scratch) {
    if (temp <= 0.0f) return apus_sample_argmax(logits, n);
    return apus_sample_logits_u(logits, n, temp, top_p,
                                apus_rng_uniform(rng), scratch);
}

#endif /* APUS_SAMPLE_IMPLEMENTATION */
#endif /* APUS_SAMPLE_H */
