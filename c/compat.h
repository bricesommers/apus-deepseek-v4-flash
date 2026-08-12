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

/* --- M15: Windows port shims (static inline: no link dependencies) --------
 * The engine's POSIX surface is small and funnels through these wrappers:
 * thread-safe positioned reads (the expert store preads the same fd from
 * several I/O workers), 64-bit file size, paired aligned alloc/free
 * (_aligned_malloc storage must NOT pass through free()), and the online
 * CPU count. Non-Windows builds map 1:1 onto the POSIX calls. */
#ifdef _WIN32
#include <io.h>
#include <malloc.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <windows.h>
#include <psapi.h>
#else
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

static inline int apus_ncpu(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    return c > 0 ? (int)c : 1;
#endif
}

static inline void *apus_aligned_alloc(size_t align, size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, n)) return NULL;
    return p;
#endif
}

static inline void apus_aligned_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Read-only open suitable for apus_sys_pread. Windows: an overlapped-
 * capable handle (required for thread-safe positioned reads) wrapped in a
 * CRT fd in binary mode. */
static inline int apus_sys_open_ro(const char *path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING,
                           FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                           NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) { CloseHandle(h); return -1; }
    return fd;
#else
    return open(path, O_RDONLY);
#endif
}

/* 64-bit-safe file size. */
static inline int64_t apus_sys_fsize(int fd) {
#ifdef _WIN32
    struct _stati64 sb;
    if (_fstati64(fd, &sb)) return -1;
    return (int64_t)sb.st_size;
#else
    struct stat sb;
    if (fstat(fd, &sb)) return -1;
    return (int64_t)sb.st_size;
#endif
}

/* fsync for the usage-history crash-safe write (c/cache.h). */
static inline int apus_sys_fsync(FILE *f) {
#ifdef _WIN32
    return _commit(_fileno(f));
#else
    return fsync(fileno(f));
#endif
}

/* pread semantics: positioned read that does not disturb the file pointer
 * and is safe to call concurrently on the same fd. Windows: ReadFile with
 * a per-call OVERLAPPED on the overlapped handle from apus_sys_open_ro,
 * waited on synchronously; chunks capped at 1 GiB (DWORD count). Returns
 * the byte count (short at EOF), -1 on error. */
static inline int64_t apus_sys_pread(int fd, void *buf, size_t n,
                                     uint64_t off) {
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    uint64_t done = 0;
    while (done < n) {
        DWORD want = (n - done > (size_t)1 << 30)
                     ? (DWORD)((size_t)1 << 30) : (DWORD)(n - done);
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        uint64_t pos = off + done;
        ov.Offset = (DWORD)(pos & 0xffffffffu);
        ov.OffsetHigh = (DWORD)((pos >> 32) & 0xffffffffu);
        DWORD got = 0;
        if (!ReadFile(h, (char *)buf + done, want, &got, &ov)) {
            DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) return done ? (int64_t)done : -1;
            if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
                if (GetLastError() == ERROR_HANDLE_EOF) break;
                return done ? (int64_t)done : -1;
            }
        }
        if (got == 0) break;
        done += got;
    }
    return (int64_t)done;
#else
    return (int64_t)pread(fd, buf, n, (off_t)off);
#endif
}

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
#include <sys/time.h>
#ifndef _WIN32
#include <sys/resource.h>   /* getrusage fallback (POSIX only) */
#endif
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
#elif defined(_WIN32)
    /* Current working set — the mach resident_size / /proc statm analogue
     * (psapi; the RSS guard in c/cache.h needs CURRENT, not peak). */
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (uint64_t)pmc.WorkingSetSize;
    return 0;
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
