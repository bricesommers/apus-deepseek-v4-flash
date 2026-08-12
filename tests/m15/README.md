# tests/m15 — Windows port (MinGW-w64)

Milestone M15: build and run apus on **Windows 10/11 x86_64**, completing
the platform matrix (macOS/ARM + Metal, Linux/x86_64, Windows/x86_64).
Toolchain: **MinGW-w64 gcc** (MSYS2 UCRT64 on CI; MSVC is not supported —
the engine uses C11 + GNU extensions). There is no new test binary in this
directory: **the gate is the entire portable CI battery running green on
windows-latest**, plus this README's port notes.

## The POSIX surface and how it was shimmed

The engine's non-libc surface was small and already funneled:

| POSIX usage | where | Windows shim (all in `c/compat.h` unless noted) |
|---|---|---|
| pthreads (compute pool, I/O pool, pilot) | `pool.h`, `cache.h`, `pilot.h` | **winpthreads** (ships with MinGW-w64 gcc) — zero call-site changes |
| `pread` (thread-safe positioned reads: the expert store preads one fd from several I/O workers) | `st.h` | `apus_sys_pread`: `CreateFileA` with `FILE_FLAG_OVERLAPPED` wrapped in a CRT fd (`_open_osfhandle`, `_O_BINARY`), then `ReadFile` + per-call `OVERLAPPED` waited via `GetOverlappedResult`; 1 GiB chunks (DWORD count), `ERROR_HANDLE_EOF` → short read |
| `open(O_RDONLY)` / `fstat` | `st.h` | `apus_sys_open_ro` (above) / `apus_sys_fsize` (`_fstati64` — shards are >4 GB, 32-bit `st_size` would overflow) |
| `posix_memalign` / `aligned_alloc` | `cache.h` slabs, `pool.h` scratch | `apus_aligned_alloc`/`apus_aligned_free` → `_aligned_malloc`/`_aligned_free` (must pair — `free()` on `_aligned_malloc` storage is UB) |
| `sysconf(_SC_NPROCESSORS_ONLN)` | `pool.h` | `apus_ncpu` → `GetSystemInfo` |
| RSS (the cache RSS guard needs CURRENT, not peak) | `compat.h` | `GetProcessMemoryInfo().WorkingSetSize` (psapi; `-lpsapi`) |
| `fsync` (usage-history crash-safe write) | `cache.h` | `apus_sys_fsync` → `_commit(_fileno(f))` |
| text-mode stdio (`\n`→`\r\n` would corrupt the NDJSON serve protocol) | `apus.c` | `_setmode(stdin/stdout, _O_BINARY)` in `main()` |
| `F_NOCACHE` / `posix_fadvise` | — | already no-op'd off-Apple (M12a-1); unchanged on Windows |
| BLAS dispatch | `blas.h` | `APUS_BLAS == 0` off-Apple by construction; Windows GEMMs take the `x86.h` AVX2 paths |

Makefile: `OS=Windows_NT` branch — MinGW gcc, `-std=gnu11` (strict
`-std=c11` defines `__STRICT_ANSI__`, under which MinGW hides `strdup`,
`clock_gettime` et al.; gnu11 exposes them; FP semantics unchanged,
`-ffp-contract=off` stays pinned), `-lpsapi`, no `_GNU_SOURCE`. MinGW gcc
appends `.exe` to outputs; MSYS2 bash resolves extension-less invocations,
so no Makefile binary paths changed.

## Verification

- Local cross-compile on the dev Mac (`x86_64-w64-mingw32-gcc`, Homebrew
  mingw-w64): engine + representative test binaries compile `-Wall -Wextra`
  (warnings only where x86 already had them: NEON-gated helpers unused).
- macOS + Linux batteries after the port edits: green, digests identical
  to pre-M15 (the shims are 1:1 wrappers on POSIX).
- Windows CI: the full battery runs on windows-latest (MSYS2 UCRT64
  gcc) in the `windows` job of `.github/workflows/ci.yml`, with fixtures
  regenerated on-runner via a native Windows Python.

## Numerics note (expected, documented)

Windows/x86_64 runs the M12a-2 AVX2 kernels, which are bitwise-identical
to the scalar anchors **within** the platform. Cross-platform bit-identity
does not hold for this model (the M12c finding: 43 layers amplify last-ulp
libm/ordering noise into near-tie token flips; UCRT `expf` differs from
glibc/dyld in the last ulp). The T=1/4/8 thread digests and the fixture
tolerances are the Windows gates — fixtures regenerate on the Windows
runner itself, so they are self-consistent by construction.

## Running on Windows (users)

Everything happens in an MSYS2 UCRT64 shell
(`pacman -S mingw-w64-ucrt-x86_64-gcc make`, then the README quickstart
commands work unchanged): `make apus`, `python tools/download.py ...`,
`./bin/apus run --model weights/apus-0731 --tiered --prompt ...`. The
Python tools (`server.py`, `chat.py`) are cross-platform. Metal is
macOS-only; `--metal` is absent on Windows. `ubsan-*` targets are
POSIX-only (MinGW has no sanitizer runtimes).

**Locale note:** Windows Python defaults to the cp1252 text encoding. The
user-facing tools pin UTF-8 explicitly, but if you run the test battery
(the fixture oracles read UTF-8 JSON), set `PYTHONUTF8=1` first — the CI
windows job does exactly this.
