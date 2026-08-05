/*
 * c/compat.h — platform shims (M6a), colibri compat.h pattern adapted for
 * the Apple M1 target: process RSS measurement (mach task_info), F_NOCACHE
 * uncached-read fds (the macOS answer to O_DIRECT / posix_fadvise(DONTNEED)
 * for streaming weight reads that must not evict the page cache), and env
 * parsing helpers for the APUS_* tuning knobs.
 *
 * Linux port (M12a-1) shim list:
 *   - apus_rss_bytes: /proc/self/statm (current RSS in pages), matching the
 *     mach resident_size semantics; getrusage ru_maxrss (peak RSS) remains
 *     the last-resort fallback on non-Linux non-Apple platforms.
 *   - apus_fd_nocache: no Linux equivalent is engaged (O_DIRECT is NOT
 *     trivially safe — it imposes buffer/offset/length alignment the slab
 *     pread path does not currently guarantee), so it returns -1 and reads
 *     stay page-cached. Cache-pressure hygiene is left to the kernel's
 *     reclaim heuristics; revisit with posix_fadvise(NT_DOREUSE) or aligned
 *     O_DIRECT buffers when the Linux port is tuned (M12a-2+).
 *   - apus_fadvise_dontneed: posix_fadvise(POSIX_FADV_DONTNEED) when the
 *     caller passes a real fd (fd >= 0); a deliberate no-op for fd < 0
 *     (c/cache.h passes -1 where macOS F_NOCACHE already covers it).
 *
 * C11, libc only. Usage: #define APUS_COMPAT_IMPLEMENTATION in one TU.
 */
#ifndef APUS_COMPAT_H
#define APUS_COMPAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Process resident size in bytes (mach task_info TASK_BASIC_INFO_64 on
 * macOS, getrusage ru_maxrss fallback elsewhere). 0 on failure. */
uint64_t apus_rss_bytes(void);

/* Mark an fd for uncached reads (F_NOCACHE): streaming reads through it
 * bypass/Do-not-populate the buffer cache. Returns 0 on success, -1 if the
 * platform has no equivalent (reads still work, just cached). */
int apus_fd_nocache(int fd);

/* Best-effort "don't need these pages" after a streaming read. macOS has no
 * posix_fadvise; F_NOCACHE on the reading fd already keeps the pages out of
 * the cache, so this is a deliberate no-op there (kept for the Linux port). */
void apus_fadvise_dontneed(int fd, uint64_t off, uint64_t len);

/* Env helpers: APUS_*_MB in mebibytes, plain ints otherwise. */
size_t apus_env_mb(const char *name, size_t def_mb);
int    apus_env_int(const char *name, int def);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_COMPAT_IMPLEMENTATION

#include <stdlib.h>

#ifdef __APPLE__
#include <fcntl.h>
#include <mach/mach.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#ifdef __linux__
#include <stdio.h>      /* /proc/self/statm */
#include <fcntl.h>      /* posix_fadvise */
#endif
#endif

uint64_t apus_rss_bytes(void) {
#ifdef __APPLE__
    task_basic_info_64_data_t info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_64_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO_64,
                  (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return (uint64_t)info.resident_size;
#elif defined(__linux__)
    /* Current RSS (pages) from /proc/self/statm — the mach resident_size
     * analogue (getrusage ru_maxrss is the PEAK, not the current RSS, so it
     * would mis-fire the c/cache.h RSS guard after any historical spike). */
    FILE *f = fopen("/proc/self/statm", "r");
    if (f) {
        unsigned long total = 0, resident = 0;
        int ok = fscanf(f, "%lu %lu", &total, &resident);
        fclose(f);
        if (ok == 2)
            return (uint64_t)resident * 4096u;   /* PAGE_SIZE on x86_64 */
    }
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru)) return 0;
    return (uint64_t)ru.ru_maxrss * 1024;   /* Linux: KiB; macOS: bytes */
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru)) return 0;
    return (uint64_t)ru.ru_maxrss * 1024;   /* Linux: KiB; macOS: bytes */
#endif
}

int apus_fd_nocache(int fd) {
#ifdef __APPLE__
    return fcntl(fd, F_NOCACHE, 1);
#else
    /* Linux: no uncached-read fd mode engaged (see the header shim list —
     * O_DIRECT's alignment rules make it non-trivial here). -1 = "cached",
     * which every caller already handles. */
    (void)fd;
    return -1;
#endif
}

void apus_fadvise_dontneed(int fd, uint64_t off, uint64_t len) {
#if defined(__APPLE__)
    (void)fd; (void)off; (void)len;
    /* macOS: F_NOCACHE on the streaming fd covers the hygiene (see header). */
#elif defined(__linux__)
    /* Real hygiene shim: drop the streaming-read pages from the page cache.
     * fd < 0 is a deliberate no-op (c/cache.h's F_NOCACHE-covered call site). */
    if (fd >= 0)
        (void)posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
#else
    (void)fd; (void)off; (void)len;
#endif
}

size_t apus_env_mb(const char *name, size_t def_mb) {
    const char *v = getenv(name);
    if (!v || !*v) return def_mb;
    char *end = NULL;
    unsigned long long mb = strtoull(v, &end, 10);
    if (end == v) return def_mb;
    return (size_t)mb;
}

int apus_env_int(const char *name, int def) {
    const char *v = getenv(name);
    if (!v || !*v) return def;
    return (int)strtol(v, NULL, 10);
}

#endif /* APUS_COMPAT_IMPLEMENTATION */
#endif /* APUS_COMPAT_H */
