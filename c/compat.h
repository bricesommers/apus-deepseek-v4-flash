/*
 * c/compat.h — platform shims (M6a), colibri compat.h pattern adapted for
 * the Apple M1 target: process RSS measurement (mach task_info), F_NOCACHE
 * uncached-read fds (the macOS answer to O_DIRECT / posix_fadvise(DONTNEED)
 * for streaming weight reads that must not evict the page cache), and env
 * parsing helpers for the APUS_* tuning knobs.
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
#endif

uint64_t apus_rss_bytes(void) {
#ifdef __APPLE__
    task_basic_info_64_data_t info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_64_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO_64,
                  (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return (uint64_t)info.resident_size;
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
    (void)fd;
    return -1;
#endif
}

void apus_fadvise_dontneed(int fd, uint64_t off, uint64_t len) {
    (void)fd; (void)off; (void)len;
    /* macOS: F_NOCACHE on the streaming fd covers the hygiene (see header). */
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
