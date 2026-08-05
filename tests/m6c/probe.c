/* probe: per-layer h digests for a prefill, printed for T=1 vs T=8 diff */
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "pool.h"

static uint64_t dig(const float *p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, p + i, 4);
        h ^= u;
        h *= 0x100000001B3ull;
    }
    return h;
}

int main(void) {
    ApusModel m;
    char err[256];
    if (apus_model_load(&m, "tests/m5/fixtures", err, sizeof err)) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    int dim = m.cfg.dim, hc = m.cfg.hc_mult, V = m.cfg.vocab_size;
    int64_t ids[64];
    for (int i = 0; i < 64; i++) ids[i] = (i * 37 + 11) % V;
    ApusModelState st;
    apus_model_state_init(&st, &m);
    float *h = malloc((size_t)64 * hc * dim * sizeof(float));
    float *logits = malloc((size_t)V * sizeof(float));
    for (int t = 0; t < 64; t++)
        apus_model_embed(&m, ids[t], h + (size_t)t * hc * dim);
    for (int i = 0; i < m.n_layers; i++) {
        apus_block_forward(&m.layers[i], &st.layers[i], h, ids, 64, 0, NULL);
        printf("layer %d h digest %016llx\n", i,
               (unsigned long long)dig(h, (size_t)64 * hc * dim));
    }
    (void)logits;
    apus_model_state_free(&st, &m);
    apus_model_free(&m);
    free(h);
    free(logits);
    return 0;
}
