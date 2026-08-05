/*
 * tests/m7b/bench_metal.c — M7b performance comparison (informational, no
 * pass/fail): real-shape dense GEMVs CPU-NEON vs Metal, and decode tok/s on
 * the m5 mini-model CPU vs Metal.
 *
 * Reported per op: ms/call and effective weight-streaming bandwidth (GB/s).
 * Decode GEMVs are memory-bound: the weight bytes are read once per call,
 * so GB/s is the honest metric. On the mini-model the Metal path is slower
 * (per-dispatch overhead dominates at 256-dim toy shapes) — expected; the
 * backend exists for the real 160 GB container's shapes.
 *
 * Run from the repository root: make bench-m7b
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
#include <time.h>

#include "fp8.h"
#include "attn.h"
#include "model.h"
#include "sample.h"
#include "backend_metal.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t ru(void) {
    uint64_t z = (rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rf(void) {
    return ((double)(ru() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}

/* FP8 GEMV (decode, M=1) CPU vs Metal on one shape. */
static void bench_fp8(int O, int K) {
    size_t nb = apus_fp8_blocks((size_t)K), nbo = ((size_t)O + 127) / 128;
    ApusFp8W w;
    uint8_t *wc = malloc((size_t)O * K), *wsc = malloc(nbo * nb);
    for (long i = 0; i < (long)O * K; i++) {
        uint8_t b = (uint8_t)(ru() >> 56);
        if ((b & 0x7F) == 0x7F) b ^= 1;
        wc[i] = b;
    }
    for (size_t i = 0; i < nbo * nb; i++) wsc[i] = (uint8_t)(120 + ru() % 8);
    w.codes = wc; w.scales = wsc; w.O = O; w.K = K;
    float *x = malloc((size_t)K * sizeof(float));
    float *out = malloc((size_t)O * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = rf();

    /* warmup + calibration */
    apus_fp8_linear(&w, x, out, 1, K, O);
    apus_metal_fp8_linear(&w, x, out, 1, K, O);
    double t0 = now_s();
    apus_fp8_linear(&w, x, out, 1, K, O);
    double est = now_s() - t0;
    int iters = est > 0.05 ? 5 : est > 0.005 ? 20 : 50;

    t0 = now_s();
    for (int i = 0; i < iters; i++) apus_fp8_linear(&w, x, out, 1, K, O);
    double tcpu = (now_s() - t0) / iters;
    t0 = now_s();
    for (int i = 0; i < iters; i++) apus_metal_fp8_linear(&w, x, out, 1, K, O);
    double tgpu = (now_s() - t0) / iters;
    double mb = (double)O * K / 1048576.0;
    printf("  fp8 gemv %6d x %-6d (%6.1f MB)  cpu %8.3f ms (%6.1f GB/s)  "
           "metal %8.3f ms (%6.1f GB/s)  x%.2f\n",
           O, K, mb, tcpu * 1e3, mb / 1024.0 / tcpu,
           tgpu * 1e3, mb / 1024.0 / tgpu, tcpu / tgpu);
    free(x); free(out);   /* weights kept (pointer-cache invariant) */
}

/* BF16 LM-head GEMV CPU vs Metal. */
static void bench_head(int64_t O, int64_t K) {
    uint16_t *w = malloc((size_t)O * K * 2);
    for (int64_t i = 0; i < O * K; i++) {
        float f = rf();
        uint32_t u;
        memcpy(&u, &f, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        w[i] = (uint16_t)(u >> 16);
    }
    float *x = malloc((size_t)K * sizeof(float));
    float *out = malloc((size_t)O * sizeof(float));
    for (int64_t i = 0; i < K; i++) x[i] = rf();
    ApusStTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = APUS_ST_BF16;
    t.ndim = 2;
    t.shape[0] = O; t.shape[1] = K;
    t.data = w;
    t.nbytes = (size_t)O * K * 2;

    apus_head_gemv(&t, x, out, O, K);
    apus_metal_head_gemv_bf16(w, x, out, O, K);
    double t0 = now_s();
    apus_head_gemv(&t, x, out, O, K);
    double est = now_s() - t0;
    int iters = est > 0.05 ? 5 : 20;
    t0 = now_s();
    for (int i = 0; i < iters; i++) apus_head_gemv(&t, x, out, O, K);
    double tcpu = (now_s() - t0) / iters;
    t0 = now_s();
    for (int i = 0; i < iters; i++)
        apus_metal_head_gemv_bf16(w, x, out, O, K);
    double tgpu = (now_s() - t0) / iters;
    double mb = (double)O * K * 2 / 1048576.0;
    printf("  bf16 head %6lld x %-6lld (%6.1f MB)  cpu %8.3f ms (%6.1f GB/s)  "
           "metal %8.3f ms (%6.1f GB/s)  x%.2f\n",
           (long long)O, (long long)K, mb, tcpu * 1e3, mb / 1024.0 / tcpu,
           tgpu * 1e3, mb / 1024.0 / tgpu, tcpu / tgpu);
    free(x); free(out);
}

/* FP32 GEMV (router gate semantics, no rounding) CPU vs Metal. */
static void bench_f32(int O, int K) {
    float *w = malloc((size_t)O * K * sizeof(float));
    float *x = malloc((size_t)K * sizeof(float));
    float *out = malloc((size_t)O * sizeof(float));
    for (long i = 0; i < (long)O * K; i++) w[i] = apus_bf16_round(rf());
    for (int i = 0; i < K; i++) x[i] = rf();
    apus_f32_linear(w, x, out, 1, K, O);
    apus_metal_f32_linear(w, x, out, 1, K, O, 0);
    double t0 = now_s();
    for (int i = 0; i < 20; i++) apus_f32_linear(w, x, out, 1, K, O);
    double tcpu = (now_s() - t0) / 20;
    t0 = now_s();
    for (int i = 0; i < 20; i++) apus_metal_f32_linear(w, x, out, 1, K, O, 0);
    double tgpu = (now_s() - t0) / 20;
    double mb = (double)O * K * 4 / 1048576.0;
    printf("  f32 gemv %6d x %-6d (%6.1f MB)  cpu %8.3f ms (%6.1f GB/s)  "
           "metal %8.3f ms (%6.1f GB/s)  x%.2f\n",
           O, K, mb, tcpu * 1e3, mb / 1024.0 / tcpu,
           tgpu * 1e3, mb / 1024.0 / tgpu, tcpu / tgpu);
    free(x); free(out);
}

/* decode tok/s on the m5 fixture, CPU vs Metal */
static void bench_toks(const char *fixdir) {
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, fixdir, err, sizeof err)) {
        fprintf(stderr, "bench: model load: %s\n", err);
        return;
    }
    int V = m.cfg.vocab_size;
    int64_t ids[24];
    for (int i = 0; i < 24; i++) ids[i] = (int64_t)(ru() % (uint64_t)V);
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    ApusRng rng;
    ApusBackendHooks saved = apus_backend_hooks;   /* Metal hooks */

    double res[2];
    for (int backend = 0; backend < 2; backend++) {
        if (backend) apus_backend_hooks = saved;
        else memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
        ApusModelState st;
        apus_model_state_init(&st, &m);
        apus_rng_seed(&rng, 1234);
        apus_model_forward(&m, &st, ids, 24, logits, 0);
        double t0 = now_s();
        int n = 0;
        for (int step = 0; step < 48; step++) {
            int tok = apus_sample(logits, (size_t)V, 0.0f, 1.0f, &rng, scratch);
            int64_t next = tok;
            apus_model_forward(&m, &st, &next, 1, logits, 0);
            n++;
        }
        res[backend] = n / (now_s() - t0);
        apus_model_state_free(&st, &m);
    }
    memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
    printf("  m5 mini-model greedy decode:  cpu %.1f tok/s   metal %.1f tok/s"
           "  (x%.2f; dispatch-bound at toy shapes)\n",
           res[0], res[1], res[0] / res[1]);
    free(logits);
    free(scratch);
    apus_model_free(&m);
}

int main(void) {
    printf("bench_metal: M7b CPU-NEON vs Metal (informational)\n");
    char err[256];
    if (apus_metal_enable(err, sizeof err)) {
        printf("  Metal unavailable (%s) — nothing to bench\n", err);
        return 0;
    }
    /* bench helpers call the CPU/Metal entry points directly (not through
     * the hooks), so hook state is irrelevant until bench_toks */
    printf(" -- real dense FP8 GEMV shapes (wq_a, wq_b, wkv, wo_b, shared)\n");
    bench_fp8(1024, 4096);    /* wq_a */
    bench_fp8(32768, 1024);   /* wq_b */
    bench_fp8(512, 4096);     /* wkv */
    bench_fp8(4096, 8192);    /* wo_b */
    bench_fp8(2048, 4096);    /* shared w1/w3 */
    bench_fp8(4096, 2048);    /* shared w2 */
    bench_fp8(8192, 1024);    /* indexer wq_b */
    printf(" -- BF16 LM head (129280 x 4096 = 1.0 GB)\n");
    bench_head(129280, 4096);
    printf(" -- FP32 router gate (256 x 4096)\n");
    bench_f32(256, 4096);
    printf(" -- mini-model tok/s\n");
    bench_toks("tests/m5/fixtures");
    printf("  zero-copy wrapped %.1f MB, uploaded %.1f MB\n",
           (double)apus_metal_bytes_wrapped() / 1048576.0,
           (double)apus_metal_bytes_uploaded() / 1048576.0);
    apus_metal_disable();
    return 0;
}
