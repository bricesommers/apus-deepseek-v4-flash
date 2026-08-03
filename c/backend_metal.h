/*
 * c/backend_metal.h — optional Metal backend for the dense compute of
 * DeepSeek-V4-Flash (M7b): FP8-E4M3 blockwise dense GEMV/GEMM, BF16 LM-head
 * GEMV, FP32 gate/wo_a linears and RMSNorm as FP32 Metal shaders, zero-copy
 * unified-memory buffers. Pure C11 interface (the implementation lives in
 * c/backend_metal.mm, Objective-C++, only in the `metal=1` build).
 *
 * Backend interface design (the "runtime/backend interface" of M7b):
 * a FUNCTION-POINTER TABLE, `apus_backend_hooks` (defined once in the
 * APUS_ATTN_IMPLEMENTATION TU — attn.h is linked into every engine binary),
 * all-NULL by default = the existing CPU kernels. apus_metal_enable() (strong
 * definition in c/backend_metal.mm; weak stub in the attn.h TU so the plain
 * CPU binary links and behaves exactly as before) initializes Metal and
 * fills the table. Every call site (c/attn.h, c/moe.h, c/model.h) tries the
 * hook first and falls back to the CPU kernel when the hook pointer is NULL
 * or the call returns nonzero (unsupported shape / budget exhausted / no
 * GPU) — per-op fail-soft, never a crash.
 *
 * Numerics contract: FP32 compute only. The shaders reproduce c/fp8.h's
 * semantics step for step: BF16 RNE round of the activation input, per-128
 * act quant (amax floor 1e-4, 2^ceil(log2(amax/448)) via the FP32 bit trick,
 * clamp +-448, RNE to E4M3), per-128-block FP32 dot, scale product FIRST
 * (sa*sb, one rounding), total += dot*sc, BF16-rounded output for the
 * linear entry points. No FP16/BF16 arithmetic anywhere in shader math.
 * Only the FP32 summation order differs from the CPU kernels (measured in
 * tests/m7b, same tolerance class as the scalar-vs-NEON precedent of
 * tests/m3/m4a). Shaders are compiled with fast-math DISABLED so no
 * mul+add contraction or reassociation changes the rounding sequence.
 *
 * Buffers: resident dense weights (FP8 codes/scales in the st.h shard slabs,
 * BF16 head shard view) are wrapped as MTLBuffers on first use, keyed by
 * CPU pointer. Zero-copy (newBufferWithBytesNoCopy over the vm_region the
 * pointer lives in, unified memory) whenever the region covers the
 * page-rounded range; otherwise one upload copy. APUS_METAL_DENSE_MB
 * (default 8192) caps wrapped+uploaded weight bytes; past the cap the hook
 * returns "unsupported" and the op runs on the CPU.
 *
 * Weight-pointer invariant: the cache assumes weight pointers are STABLE
 * and their contents immutable for the lifetime of the backend (true for
 * the engine: shard slabs and layer arrays live until apus_model_free).
 * A freed-and-reused address would alias a stale buffer. The cache is
 * dropped by apus_metal_disable(); disable before freeing model weights.
 */
#ifndef APUS_BACKEND_METAL_H
#define APUS_BACKEND_METAL_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"

#ifdef __cplusplus
extern "C" {
#endif

/* f32_linear hook flags: BF16-RNE rounding of the input elements / output. */
#define APUS_HOOK_R_IN  1
#define APUS_HOOK_R_OUT 2
/* M6c: w points to BF16 BITS (u16), not f32 (wo_a storage). The backend
 * widens bf16->f32 (exact) once and caches the widened copy. */
#define APUS_HOOK_W_BF16 4

/* Backend hook table. All-NULL = CPU kernels (default). A hook returns 0
 * when it produced the result, nonzero = unsupported (caller runs the CPU
 * kernel). All shapes match the CPU helpers they replace:
 *  - fp8_linear:  attn.h apus_fp8_linear  (x [M,K] f32 -> BF16-rounded out)
 *  - f32_linear:  w [O,K] f32 (or BF16 bits with APUS_HOOK_W_BF16), x [M,K],
 *                 out [M,O]; flags APUS_HOOK_R_* | APUS_HOOK_W_BF16
 *                 (moe.h router gate: flags 0; attn.h apus_bf16_linear:
 *                 R_IN|R_OUT; attn.h wo_a groups: R_OUT|W_BF16)
 *  - head_gemv:   model.h apus_head_gemv (BF16 or F32 shard view, FP32
 *                 accumulate, unrounded out [O]) */
typedef struct {
    int (*fp8_linear)(const ApusFp8W *w, const float *x, float *out,
                      int M, int K, int O);
    int (*f32_linear)(const float *w, const float *x, float *out,
                      int M, int K, int O, int flags);
    int (*head_gemv)(const ApusStTensor *head, const float *x, float *out,
                     int64_t O, int64_t K);
} ApusBackendHooks;

extern ApusBackendHooks apus_backend_hooks;

/* Enable the Metal backend: initialize the device/queue/pipelines and fill
 * apus_backend_hooks. Returns 0 on success; nonzero (err filled) when Metal
 * is unavailable or the backend was not compiled in — the hooks stay NULL
 * and the engine runs on the CPU kernels (fail-soft). Idempotent. */
int  apus_metal_enable(char *err, size_t errcap);
/* Detach the hooks (CPU kernels again) and release backend state. */
void apus_metal_disable(void);
int  apus_metal_is_enabled(void);

/* Instrumentation: resident weight bytes wrapped zero-copy / uploaded by
 * copy, GPU dispatches submitted, weight-buffer cache hits. */
uint64_t apus_metal_bytes_wrapped(void);
uint64_t apus_metal_bytes_uploaded(void);
uint64_t apus_metal_dispatches(void);

/* Direct entry points (kernel-level tests; same semantics as the hooks,
 * no hook-table indirection, Metal initialized lazily). Return 0 on
 * success, 1 when Metal/unsupported. */
int apus_metal_fp8_linear(const ApusFp8W *w, const float *x, float *out,
                          int M, int K, int O);
int apus_metal_f32_linear(const float *w, const float *x, float *out,
                          int M, int K, int O, int flags);
int apus_metal_head_gemv_bf16(const void *w_bf16, const float *x, float *out,
                              int64_t O, int64_t K);
/* Raw FP8 GEMM on pre-quantized activations (unrounded FP32 out) — mirrors
 * apus_fp8_gemm_neon. */
int apus_metal_fp8_gemm(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as, float *out,
                        int M, int O, int K);
/* Per-128 act quant (BF16-round inside, same as apus_fp8_linear's front
 * end). codes [M,K], scales [M, ceil(K/128)]. */
int apus_metal_fp8_act_quant(const float *x, int M, int K,
                             uint8_t *codes, float *scales);
/* RMSNorm, BF16-rounded output, same formula as apus_rms_norm. */
int apus_metal_rmsnorm(const float *x, const float *w, float eps, float *y,
                       int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* APUS_BACKEND_METAL_H */
