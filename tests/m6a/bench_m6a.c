/*
 * tests/m6a/bench_m6a.c — M6a measurements (informational, no assertions):
 *
 *   1. NVMe random-read benchmark at the REAL expert slab size
 *      (13,369,344 B = 12.75 MiB), F_NOCACHE pread vs cached pread, on this
 *      Mac's SSD. Reports GB/s and the implied cold-decode tok/s floor at
 *      258 experts/token (43 layers x 6 top-k, 3.45 GB/token all-miss).
 *   2. Fixture forward hit-rate vs tok/s curve: greedy decode on the M6a
 *      mini-model with cache sizes 64/32/16/8/4/2 slots per layer.
 *
 * Run from the repository root. The benchmark file lives in TMPDIR (not in
 * the repo).
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
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "model.h"
#include "sample.h"
#include "cache.h"

#define FIX "tests/m6a/fixtures"

#define REAL_SLAB 13369344u     /* bytes per real expert (12.75 MiB) */
#define N_BLOCKS 153            /* 153 x 12.75 MiB ~= 1.91 GiB bench file */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void bench_nvme(void) {
    printf("== NVMe random reads at %u B (real expert slab) ==\n", REAL_SLAB);
    char path[1024];
    const char *tmp = getenv("TMPDIR");
    snprintf(path, sizeof path, "%s/apus_m6a_iobench.bin",
             tmp ? tmp : "/tmp");

    /* create the bench file if missing or wrong size */
    struct stat st;
    int need = stat(path, &st) || st.st_size != (off_t)((uint64_t)N_BLOCKS * REAL_SLAB);
    if (need) {
        printf("  creating %s (%d blocks, %.2f GiB)...\n", path, N_BLOCKS,
               (double)N_BLOCKS * REAL_SLAB / (1u << 30));
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { perror("open"); return; }
        void *buf = malloc(REAL_SLAB);
        memset(buf, 0x5a, REAL_SLAB);
        for (int i = 0; i < N_BLOCKS; i++)
            if (write(fd, buf, REAL_SLAB) != REAL_SLAB) {
                perror("write");
                free(buf);
                close(fd);
                return;
            }
        free(buf);
        fsync(fd);
        close(fd);
    }

    /* random permutation of blocks (LCG, fixed seed) */
    int *perm = malloc(N_BLOCKS * sizeof(int));
    for (int i = 0; i < N_BLOCKS; i++) perm[i] = i;
    uint64_t rng = 0x9e3779b97f4a7c15ull;
    for (int i = N_BLOCKS - 1; i > 0; i--) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        int j = (int)((rng >> 33) % (uint64_t)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }

    void *buf;
    if (posix_memalign(&buf, 4096, REAL_SLAB)) { free(perm); return; }

    for (int mode = 0; mode < 2; mode++) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open"); break; }
        if (mode == 0) fcntl(fd, F_NOCACHE, 1);
        double t0 = now_s();
        for (int i = 0; i < N_BLOCKS; i++) {
            uint64_t off = (uint64_t)perm[i] * REAL_SLAB;
            uint8_t *p = buf;
            size_t done = 0;
            while (done < REAL_SLAB) {
                ssize_t r = pread(fd, p + done, REAL_SLAB - done,
                                  (off_t)(off + done));
                if (r <= 0) { perror("pread"); break; }
                done += (size_t)r;
            }
        }
        double dt = now_s() - t0;
        double gb = (double)N_BLOCKS * REAL_SLAB / (1u << 30);
        double gbps = gb / dt;
        printf("  %-24s %d reads in %.2fs -> %.2f GB/s  "
               "(cold tok/s floor at 3.45 GB/token: %.2f)\n",
               mode == 0 ? "F_NOCACHE pread" : "cached pread",
               N_BLOCKS, dt, gbps, gbps / 3.45);
        close(fd);
    }
    free(buf);
    free(perm);
    printf("  implied tok/s = GB/s / (3.45 GB x miss-rate); e.g. at 2 GB/s: "
           "m=100%% -> 0.58, m=50%% -> 1.16, m=25%% -> 2.32 tok/s\n");
}

/* --- fixture hit-rate vs tok/s -------------------------------------------*/

#define N_PROMPT 8
#define N_GEN 24
static const int64_t PROMPT[N_PROMPT] = {3, 41, 7, 200, 511, 0, 128, 65};

static void bench_forward(int slots) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, FIX, 1, err, sizeof err)) {
        fprintf(stderr, "model load: %s\n", err);
        exit(1);
    }
    ApusStoreCfg c = {0};
    c.n_layers = m.n_layers;
    c.n_experts = m.cfg.n_routed_experts;
    c.slots_per_layer = slots;
    c.pins_per_layer = 0;
    c.rss_budget_bytes = 1ull << 40;
    c.io_threads = 4;
    c.usage_path = "";
    ApusStore *st = apus_store_open(FIX, &c, err, sizeof err);
    if (!st) { fprintf(stderr, "store: %s\n", err); exit(1); }
    for (int i = 0; i < m.n_layers; i++)
        apus_store_attach_moe(st, &m.layers[i].mw);

    int V = m.cfg.vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    ApusModelState stt;
    apus_model_state_init(&stt, &m);
    double t0 = now_s();
    apus_model_forward(&m, &stt, PROMPT, N_PROMPT, logits, 0);
    double tp = now_s() - t0;
    t0 = now_s();
    for (int t = 0; t < N_GEN; t++) {
        int tok = apus_sample_argmax(logits, (size_t)V);
        if (t + 1 < N_GEN) {
            int64_t next = tok;
            apus_model_forward(&m, &stt, &next, 1, logits, 0);
        }
    }
    double td = now_s() - t0;
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    double hr = (double)ss.hits / (double)(ss.hits + ss.misses);
    printf("  slots/layer %3d  hit-rate %5.1f%%  prefill %6.0f tok/s  "
           "decode %6.0f tok/s  (preads %llu)\n",
           slots, 100.0 * hr, N_PROMPT / tp, N_GEN / td,
           (unsigned long long)ss.preads);
    free(logits);
    apus_model_state_free(&stt, &m);
    apus_store_close(st);
    apus_model_free(&m);
}

int main(int argc, char **argv) {
    int skip_nvme = argc > 1 && !strcmp(argv[1], "--skip-nvme");
    if (!skip_nvme) bench_nvme();
    printf("== fixture forward: hit-rate vs tok/s (mini-model, "
           "informational) ==\n");
    int sizes[] = {64, 32, 16, 8, 4, 2};
    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++)
        bench_forward(sizes[i]);
    return 0;
}
