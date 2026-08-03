/*
 * c/backend_metal.mm — Metal backend for the apus dense compute (M7b).
 * Objective-C++ (clang++); the only file in the project that needs the
 * Metal/Foundation frameworks. Built only into bin/apus_metal and the
 * tests/m7b binaries (make metal=1 / test-m7b); the plain CPU binary never
 * links it and is behaviorally untouched.
 *
 * What runs on the GPU (all FP32 shader math, fast-math DISABLED — no
 * contraction, no reassociation, IEEE division/sqrt):
 *   - FP8-E4M3 dense linears (c/fp8.h semantics): BF16 input round, per-128
 *     act quant (amax floor 1e-4, bit-trick pow2 scale, clamp +-448, RNE),
 *     per-block FP32 dot, scale product first (sa*sb), total += dot*sc,
 *     BF16-rounded output. Two-pass (block dots, then fold); since M9a the
 *     fp8_dot shader accumulates in the exact c/fp8.h NEON canonical order
 *     (4 independent float4 accumulators, fixed combine), so GPU block dots
 *     are BIT-IDENTICAL to the CPU NEON kernel.
 *   - BF16 LM-head GEMV (FP32 accumulate, unrounded out; one thread per
 *     output row, sequential K — the CPU scalar summation order).
 *   - FP32 linears (router gate: no rounding; apus_bf16_linear: BF16-round
 *     input+output; wo_a group slices: BF16-round output only).
 *   - RMSNorm (exposed + kernel-tested; NOT wired into the forward path —
 *     per-op dispatch overhead exceeds the CPU cost at these sizes; kept
 *     for a future fused-layer milestone).
 * Everything else (FP4 experts, mHC, compressors, indexer math, sampling)
 * stays on the CPU this milestone.
 *
 * Buffers: weight tensors (FP8 codes/scales in the st.h shard slabs, BF16
 * head view, f32 gate/wo_a arrays) are wrapped as MTLBuffers on first use,
 * keyed by CPU pointer. Zero-copy newBufferWithBytesNoCopy over the
 * vm_region the pointer lives in (unified memory — GPU and CPU share the
 * same pages) whenever the region covers the page-rounded range; otherwise
 * a single upload copy. APUS_METAL_DENSE_MB (default 8192) caps
 * wrapped+uploaded weight bytes; past the cap hooks return "unsupported"
 * and the op falls back to the CPU kernel (per-op fail-soft). No copies of
 * resident weights in the common case: ~6 GB FP8 + ~1 GB BF16 head stay in
 * the same pages the CPU loader allocated.
 *
 * Synchronous execution: one command buffer per op call, commit +
 * waitUntilCompleted. The engine is single-threaded; no locks.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unordered_map>

#include "backend_metal.h"

/* ========================================================================
 * Shader source (compiled at apus_metal_enable with fastMathEnabled = NO).
 * Every kernel mirrors its CPU counterpart step for step; see c/fp8.h,
 * c/fp4.h (apus_e4m3_quant_f32 / apus_ue8m0_f32 / apus_bf16_round),
 * c/attn.h (apus_rms_norm, apus_bf16_linear) for the normative order.
 * ====================================================================== */

static NSString *const kShaderSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

/* apus_bf16_round: RNE to BF16, kept in an FP32 container; NaN untouched. */
inline float bf16r(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    return as_type<float>(u);
}

/* apus_ue8m0_f32: byte 0 -> 2^-127 (subnormal), 1..254 -> 2^(b-127),
 * 255 -> +inf (bit pattern (uint)b << 23). */
inline float ue8m0(uint b) {
    uint bits = (b == 0u) ? 0x00400000u : (b << 23);
    return as_type<float>(bits);
}

/* apus_e4m3_dequant_f32, exact (NaN codes decode as +-480, as in fp4.h). */
inline float e4m3_deq(uint c) {
    uint sign = (c & 0x80u) << 24;
    uint e = (c >> 3) & 0xFu;
    uint m = c & 7u;
    uint nrm = (((e + 120u) << 23) | (m << 20)) | sign;   /* (1+m/8)*2^(e-7) */
    float sub = (float)m * 0x1p-9f;                      /* m*2^-9, exact */
    uint subbits = as_type<uint>(sub) | sign;
    return as_type<float>(e == 0u ? subbits : nrm);
}

inline uint rne_shift(uint m, int sh) {   /* fp4.h apus_rne_shift */
    uint hbit = 1u << (sh - 1);
    uint mask = (1u << sh) - 1u;
    uint r = m & mask;
    uint v = m >> sh;
    if (r > hbit || (r == hbit && (v & 1u))) v++;
    return v;
}

/* apus_e4m3_quant_f32: clamp +-448, RNE to E4M3. */
inline uchar e4m3_quant(float x) {
    x = clamp(x, -448.0f, 448.0f);
    uint u = as_type<uint>(x);
    uint sign = (u >> 31) << 7;
    uint a = u & 0x7fffffffu;
    if (a == 0u) return (uchar)sign;
    int e = (int)(a >> 23) - 127;
    uint m23 = (a & 0x007fffffu) | 0x00800000u;
    uint code;
    if (e >= -6) {
        uint v = rne_shift(m23, 20);          /* 8..16 */
        if (v == 16u) { v = 8u; e++; }
        if (e > 8) code = 0x7Eu;              /* saturate to +-448 */
        else code = ((uint)(e + 7) << 3) | (v & 7u);
    } else if (e >= -10) {
        code = rne_shift(m23, 14 - e);        /* 0..8 */
    } else {
        code = 0u;
    }
    return (uchar)(sign | code);
}

/* Per-128-along-K activation quantization (fp4.h apus_fp4_act_quant_scalar,
 * on the BF16-rounded input like apus_fp8_linear). One 128-thread
 * threadgroup per (row m, block kb). */
kernel void act_quant(device const float *x [[buffer(0)]],
                      device uchar *codes [[buffer(1)]],
                      device float *scales [[buffer(2)]],
                      constant uint &K [[buffer(3)]],
                      constant uint &nb [[buffer(4)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    uint m = tgid / nb;
    uint kb = tgid % nb;
    uint i = kb * 128u + tid;
    bool valid = i < K;
    float v = valid ? bf16r(x[(ulong)m * K + i]) : 0.0f;
    float am = simd_max(fabs(v));
    threadgroup float red[4];               /* 128 threads = 4 simdgroups */
    if (lane == 0u) red[sg] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = max(max(red[0], red[1]), max(red[2], red[3]));
    if (amax < 1e-4f) amax = 1e-4f;
    float p = amax * (1.0f / 448.0f);       /* reference bit trick */
    uint pb = as_type<uint>(p);
    int e = (int)((pb >> 23) & 0xFFu) - 127 + ((pb & 0x007fffffu) != 0u ? 1 : 0);
    float s = as_type<float>((uint)(e + 127) << 23);
    if (tid == 0u) scales[(ulong)m * nb + kb] = s;
    if (valid) codes[(ulong)m * K + i] = e4m3_quant(v / s);
}

/* Pass 1 of the FP8 GEMM: block dots. One thread per (m, o, kb): the
 * 128-wide block dot in FP32, in the M9a canonical CPU NEON order
 * (c/fp8.h apus_fp8_dot128_neon) so GPU and CPU block dots are BIT-IDENTICAL:
 * four independent float4 accumulators, acc[j] lane l taking elements
 * 4j+l of every 16-element chunk in chunk order; combined lane-wise
 * ((a0+a1)+(a2+a3)), horizontally ((s.x+s.y)+(s.z+s.w)); a partial trailing
 * chunk (< 16) is added scalar AFTER the reduction.
 * woff = byte offset of the tensor inside buffer 0 (page-rounded wraps). */
kernel void fp8_dot(device const uchar *w [[buffer(0)]],    /* [O,K] codes */
                    device const uchar *ac [[buffer(1)]],   /* [M,K] codes */
                    device float *dot [[buffer(2)]],        /* [M,O,nb] */
                    constant uint &K [[buffer(3)]],
                    constant uint &O [[buffer(4)]],
                    constant uint &nb [[buffer(5)]],
                    constant ulong &woff [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
    uint kb = gid % nb;
    uint o = (gid / nb) % O;
    uint m = gid / (nb * O);
    uint lo = kb * 128u;
    uint hi = min(lo + 128u, K);
    ulong wb = woff + (ulong)o * K;
    ulong ab = (ulong)m * K;
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    float4 a2 = float4(0.0f), a3 = float4(0.0f);
    uint i = lo;
    for (; i + 16u <= hi; i += 16u) {
        float4 w0 = float4(e4m3_deq(w[wb+i]),      e4m3_deq(w[wb+i+1u]),
                           e4m3_deq(w[wb+i+2u]),   e4m3_deq(w[wb+i+3u]));
        float4 w1 = float4(e4m3_deq(w[wb+i+4u]),   e4m3_deq(w[wb+i+5u]),
                           e4m3_deq(w[wb+i+6u]),   e4m3_deq(w[wb+i+7u]));
        float4 w2 = float4(e4m3_deq(w[wb+i+8u]),   e4m3_deq(w[wb+i+9u]),
                           e4m3_deq(w[wb+i+10u]),  e4m3_deq(w[wb+i+11u]));
        float4 w3 = float4(e4m3_deq(w[wb+i+12u]),  e4m3_deq(w[wb+i+13u]),
                           e4m3_deq(w[wb+i+14u]),  e4m3_deq(w[wb+i+15u]));
        float4 x0 = float4(e4m3_deq(ac[ab+i]),     e4m3_deq(ac[ab+i+1u]),
                           e4m3_deq(ac[ab+i+2u]),  e4m3_deq(ac[ab+i+3u]));
        float4 x1 = float4(e4m3_deq(ac[ab+i+4u]),  e4m3_deq(ac[ab+i+5u]),
                           e4m3_deq(ac[ab+i+6u]),  e4m3_deq(ac[ab+i+7u]));
        float4 x2 = float4(e4m3_deq(ac[ab+i+8u]),  e4m3_deq(ac[ab+i+9u]),
                           e4m3_deq(ac[ab+i+10u]), e4m3_deq(ac[ab+i+11u]));
        float4 x3 = float4(e4m3_deq(ac[ab+i+12u]), e4m3_deq(ac[ab+i+13u]),
                           e4m3_deq(ac[ab+i+14u]), e4m3_deq(ac[ab+i+15u]));
        a0 = fma(w0, x0, a0);
        a1 = fma(w1, x1, a1);
        a2 = fma(w2, x2, a2);
        a3 = fma(w3, x3, a3);
    }
    float4 s = (a0 + a1) + (a2 + a3);
    float d = (s.x + s.y) + (s.z + s.w);
    for (; i < hi; i++)   /* partial trailing chunk (< 16), after the reduce */
        d = fma(e4m3_deq(w[wb + i]), e4m3_deq(ac[ab + i]), d);
    dot[((ulong)m * O + o) * nb + kb] = d;
}

/* Pass 2: scale-product-first fold (kernel.py:242-249 order, c/fp8.h):
 * sc = sa*sb (one rounding), then the block fold. The CPU kernel's
 * `total += dot * sc` compiles to a single fmadd (clang FP contraction —
 * verified in the binary), so the shader accumulates with fma() to stay
 * BIT-IDENTICAL to the CPU NEON path (M9a). */
kernel void fp8_fold(device const float *dot [[buffer(0)]],
                     device const uchar *ws8 [[buffer(1)]], /* [O/128,nb] */
                     device const float *as [[buffer(2)]],  /* [M,nb] */
                     device float *out [[buffer(3)]],       /* [M,O] */
                     constant uint &O [[buffer(4)]],
                     constant uint &nb [[buffer(5)]],
                     constant uint &round_out [[buffer(6)]],
                     constant ulong &wsoff [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    uint o = gid % O;
    uint m = gid / O;
    ulong db = ((ulong)m * O + o) * nb;
    ulong sb = wsoff + (ulong)(o / 128u) * nb;
    float total = 0.0f;
    for (uint kb = 0u; kb < nb; kb++) {
        float sc = as[(ulong)m * nb + kb] * ue8m0(ws8[sb + kb]);
        total = fma(dot[db + kb], sc, total);
    }
    out[(ulong)m * O + o] = round_out ? bf16r(total) : total;
}

/* FP32 matmul, one thread per output (sequential K = CPU order).
 * flags bit0 (APUS_HOOK_R_IN): BF16-round input elements;
 * flags bit1 (APUS_HOOK_R_OUT): BF16-round the output. */
kernel void f32_gemm(device const uchar *w8 [[buffer(0)]],  /* [O,K] f32 */
                     device const float *x [[buffer(1)]],   /* [M,K] */
                     device float *out [[buffer(2)]],       /* [M,O] */
                     constant uint &K [[buffer(3)]],
                     constant uint &O [[buffer(4)]],
                     constant uint &flags [[buffer(5)]],
                     constant ulong &woff [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    device const float *w = (device const float *)(w8 + woff);
    uint o = gid % O;
    uint m = gid / O;
    ulong xb = (ulong)m * K, wb = (ulong)o * K;
    float acc = 0.0f;
    for (uint k = 0u; k < K; k++) {
        float xv = (flags & 1u) ? bf16r(x[xb + k]) : x[xb + k];
        acc = fma(xv, w[wb + k], acc);
    }
    out[(ulong)m * O + o] = (flags & 2u) ? bf16r(acc) : acc;
}

/* BF16 weight GEMV (LM head), FP32 accumulate, unrounded out; one thread
 * per output row, sequential K (CPU scalar order). */
kernel void bf16_gemv(device const uchar *w8 [[buffer(0)]], /* [O,K] bf16 */
                      device const float *x [[buffer(1)]],  /* [K] */
                      device float *out [[buffer(2)]],      /* [O] */
                      constant uint &K [[buffer(3)]],
                      constant ulong &woff [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    device const ushort *w = (device const ushort *)(w8 + woff);
    ulong wb = (ulong)gid * K;
    float acc = 0.0f;
    for (uint k = 0u; k < K; k++) {
        float wv = as_type<float>((uint)w[wb + k] << 16);
        acc = fma(wv, x[k], acc);
    }
    out[gid] = acc;
}

/* apus_rms_norm: y = bf16r(w * (x * rsqrt(mean(x^2)+eps))). One 1024-thread
 * threadgroup per row (the sum order differs from the scalar CPU loop —
 * measured in tests/m7b, same class as scalar-vs-NEON). */
kernel void rmsnorm(device const float *x [[buffer(0)]],
                    device const uchar *w8 [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant uint &n [[buffer(3)]],
                    constant float &eps [[buffer(4)]],
                    constant ulong &woff [[buffer(5)]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    device const float *w = (device const float *)(w8 + woff);
    float ss = 0.0f;
    for (uint i = tid; i < n; i += 1024u) {
        float v = x[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    threadgroup float red[32];              /* 1024 threads = 32 simdgroups */
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0u; j < 32u; j++) total += red[j];
    float inv = 1.0f / precise::sqrt(total / (float)n + eps);
    for (uint i = tid; i < n; i += 1024u)
        y[i] = bf16r(w[i] * (x[i] * inv));
}
)MSL";

/* ========================================================================
 * Context
 * ====================================================================== */

struct BufEnt {
    id<MTLBuffer> buf;
    size_t bytes;
    size_t off;     /* tensor's byte offset inside buf (page-rounded wraps) */
    int nocopy;
};

struct MtlCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLComputePipelineState> pso_actquant = nil;
    id<MTLComputePipelineState> pso_fp8dot = nil;
    id<MTLComputePipelineState> pso_fp8fold = nil;
    id<MTLComputePipelineState> pso_f32 = nil;
    id<MTLComputePipelineState> pso_bf16 = nil;
    id<MTLComputePipelineState> pso_rms = nil;
    std::unordered_map<const void *, BufEnt> wcache;
    std::unordered_map<const void *, float *> widen_cache;  /* bf16->f32 (M6c) */
    uint64_t budget = 0;
    uint64_t bytes_wrapped = 0;   /* zero-copy weight views */
    uint64_t bytes_uploaded = 0;  /* copied weight buffers */
    uint64_t dispatches = 0;
    /* staging (grown on demand) */
    id<MTLBuffer> xbuf = nil;   size_t xcap = 0;    /* f32 activations */
    id<MTLBuffer> cbuf = nil;   size_t ccap = 0;    /* u8 act codes */
    id<MTLBuffer> asbuf = nil;  size_t ascap = 0;   /* f32 act scales */
    id<MTLBuffer> dotbuf = nil; size_t dotcap = 0;  /* f32 block dots */
    id<MTLBuffer> obuf = nil;   size_t ocap = 0;    /* f32 outputs */
};

static MtlCtx *g_ctx = NULL;

static id<MTLBuffer> stage_buf(MtlCtx *c, id<MTLBuffer> &b, size_t &cap,
                               size_t need) {
    if (b && cap >= need) return b;
    b = [c->dev newBufferWithLength:need options:MTLResourceStorageModeShared];
    cap = b ? need : 0;
    return b;
}

/* Zero-copy wrap of an existing CPU allocation: verify the page-rounded
 * range [lo, hi) is covered by (possibly several) contiguous mapped
 * vm_regions — large mallocs are split into region chunks by the allocator
 * — then wrap it (page-aligned base, page-multiple length, no deallocator:
 * the engine owns the memory for the process lifetime). The tensor starts
 * *off bytes into the returned buffer. nil when not possible. */
static id<MTLBuffer> wrap_nocopy(MtlCtx *c, const void *ptr, size_t len,
                                 size_t *off) {
    uintptr_t lo = (uintptr_t)ptr & ~(uintptr_t)4095;
    uintptr_t hi = ((uintptr_t)ptr + len + 4095) & ~(uintptr_t)4095;
    if (hi <= lo) return nil;
    uintptr_t cur = lo;
    while (cur < hi) {
        vm_address_t q = (vm_address_t)cur;
        vm_size_t rsize = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        if (vm_region_64(mach_task_self(), &q, &rsize,
                         VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
                         &count, &obj) != KERN_SUCCESS)
            return nil;
        if ((uintptr_t)q > cur || (uintptr_t)q + rsize <= cur)
            return nil;   /* gap / unexpected layout */
        if (!(info.protection & VM_PROT_READ)) return nil;
        cur = (uintptr_t)q + rsize;
    }
    *off = (uintptr_t)ptr - lo;
    return [c->dev newBufferWithBytesNoCopy:(void *)lo
                                     length:(NSUInteger)(hi - lo)
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

/* Weight buffer for CPU range [ptr, ptr+len): cached; zero-copy when
 * possible (tensor then sits *off_out bytes into the buffer), upload copy
 * otherwise (offset 0); nil when the budget is exhausted. */
static id<MTLBuffer> weight_buf(MtlCtx *c, const void *ptr, size_t len,
                                size_t *off_out) {
    auto it = c->wcache.find(ptr);
    if (it != c->wcache.end()) {
        *off_out = it->second.off;
        return it->second.buf;
    }
    if (c->bytes_wrapped + c->bytes_uploaded + len > c->budget) return nil;
    size_t off = 0;
    id<MTLBuffer> b = wrap_nocopy(c, ptr, len, &off);
    BufEnt e;
    e.buf = b;
    e.bytes = len;
    e.off = off;
    if (b) {
        e.nocopy = 1;
        c->bytes_wrapped += len;
    } else {
        b = [c->dev newBufferWithLength:len
                                options:MTLResourceStorageModeShared];
        if (!b) return nil;
        memcpy([b contents], ptr, len);
        e.buf = b;
        e.off = 0;
        e.nocopy = 0;
        c->bytes_uploaded += len;
    }
    c->wcache.emplace(ptr, e);
    *off_out = e.off;
    return b;
}

/* One synchronous dispatch: encode with `encode`, commit, wait. Returns 0
 * on success, 1 on GPU error (fail-soft: caller falls back to CPU). */
typedef void (^EncodeFn)(id<MTLComputeCommandEncoder> enc);
static int run_op(MtlCtx *c, EncodeFn encode) {
    id<MTLCommandBuffer> cb = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    c->dispatches++;
    return [cb status] == MTLCommandBufferStatusError ? 1 : 0;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc, NSUInteger total,
                        NSUInteger tg) {
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

/* ========================================================================
 * Ops (all return 0 = done, 1 = unsupported -> CPU fallback)
 * ====================================================================== */

#define APUS_MTL_CHUNK 8   /* activation rows per dispatch (staging bound) */

static int mtl_fp8(MtlCtx *c, const uint8_t *w, const uint8_t *ws,
                   const float *x, const uint8_t *acodes, const float *as,
                   float *out, int M, int K, int O, int round_out) {
    if (!c || M <= 0 || K <= 0 || O <= 0) return 1;
    size_t nb = ((size_t)K + 127) / 128;
    size_t nbo = ((size_t)O + 127) / 128;
    size_t woff = 0, wsoff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, (size_t)O * (size_t)K, &woff);
    id<MTLBuffer> sb = weight_buf(c, ws, nbo * nb, &wsoff);
    if (!wb || !sb) return 1;
    for (int m0 = 0; m0 < M; m0 += APUS_MTL_CHUNK) {
        uint mc = (uint)(M - m0 < APUS_MTL_CHUNK ? M - m0 : APUS_MTL_CHUNK);
        id<MTLBuffer> cb = stage_buf(c, c->cbuf, c->ccap, (size_t)mc * K);
        id<MTLBuffer> ab = stage_buf(c, c->asbuf, c->ascap,
                                     (size_t)mc * nb * sizeof(float));
        id<MTLBuffer> db = stage_buf(c, c->dotbuf, c->dotcap,
                                     (size_t)mc * O * nb * sizeof(float));
        id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap,
                                     (size_t)mc * O * sizeof(float));
        if (!cb || !ab || !db || !ob) return 1;
        if (x) {   /* full linear: quantize the (BF16-rounded) activation */
            id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                         (size_t)mc * K * sizeof(float));
            if (!xb) return 1;
            memcpy([xb contents], x + (size_t)m0 * K,
                   (size_t)mc * K * sizeof(float));
            uint Ku = (uint)K, nb32 = (uint)nb;
            int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:c->pso_actquant];
                [enc setBuffer:xb offset:0 atIndex:0];
                [enc setBuffer:cb offset:0 atIndex:1];
                [enc setBuffer:ab offset:0 atIndex:2];
                [enc setBytes:&Ku length:4 atIndex:3];
                [enc setBytes:&nb32 length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(mc * nb32, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            });
            if (rc) return 1;
        } else {   /* raw GEMM: caller supplies codes + scales */
            memcpy([cb contents], acodes + (size_t)m0 * K, (size_t)mc * K);
            memcpy([ab contents], as + (size_t)m0 * nb,
                   (size_t)mc * nb * sizeof(float));
        }
        uint Ku = (uint)K, Ou = (uint)O, nb32 = (uint)nb, rd = (uint)round_out;
        uint64_t woff64 = woff, wsoff64 = wsoff;
        int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_fp8dot];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:cb offset:0 atIndex:1];
            [enc setBuffer:db offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc setBytes:&Ou length:4 atIndex:4];
            [enc setBytes:&nb32 length:4 atIndex:5];
            [enc setBytes:&woff64 length:8 atIndex:6];
            dispatch_1d(enc, (NSUInteger)mc * Ou * nb32, 128);
            [enc setComputePipelineState:c->pso_fp8fold];
            [enc setBuffer:db offset:0 atIndex:0];
            [enc setBuffer:sb offset:0 atIndex:1];
            [enc setBuffer:ab offset:0 atIndex:2];
            [enc setBuffer:ob offset:0 atIndex:3];
            [enc setBytes:&Ou length:4 atIndex:4];
            [enc setBytes:&nb32 length:4 atIndex:5];
            [enc setBytes:&rd length:4 atIndex:6];
            [enc setBytes:&wsoff64 length:8 atIndex:7];
            dispatch_1d(enc, (NSUInteger)mc * Ou, 128);
        });
        if (rc) return 1;
        memcpy(out + (size_t)m0 * O, [ob contents],
               (size_t)mc * O * sizeof(float));
    }
    return 0;
}

static int mtl_f32(MtlCtx *c, const float *w, const float *x, float *out,
                   int M, int K, int O, int flags) {
    if (!c || M <= 0 || K <= 0 || O <= 0) return 1;
    /* M6c: BF16-bit weights (wo_a) — widen to f32 once, cache the copy
     * (exact widening; same shader and numerics as the f32 path). */
    if (flags & APUS_HOOK_W_BF16) {
        auto it = c->widen_cache.find((const void *)w);
        if (it == c->widen_cache.end()) {
            size_t n = (size_t)O * K;
            float *cpy = (float *)malloc(n * sizeof(float));
            if (!cpy) return 1;
            const uint16_t *b = (const uint16_t *)w;
            for (size_t i = 0; i < n; i++) {
                uint32_t u = (uint32_t)b[i] << 16;
                memcpy(&cpy[i], &u, 4);
            }
            it = c->widen_cache.emplace((const void *)w, cpy).first;
        }
        w = it->second;
    }
    size_t woff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, (size_t)O * K * sizeof(float), &woff);
    if (!wb) return 1;
    for (int m0 = 0; m0 < M; m0 += APUS_MTL_CHUNK) {
        uint mc = (uint)(M - m0 < APUS_MTL_CHUNK ? M - m0 : APUS_MTL_CHUNK);
        id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                     (size_t)mc * K * sizeof(float));
        id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap,
                                     (size_t)mc * O * sizeof(float));
        if (!xb || !ob) return 1;
        memcpy([xb contents], x + (size_t)m0 * K,
               (size_t)mc * K * sizeof(float));
        uint Ku = (uint)K, Ou = (uint)O,
             fl = (uint)(flags & (APUS_HOOK_R_IN | APUS_HOOK_R_OUT));
        uint64_t woff64 = woff;
        int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_f32];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:ob offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc setBytes:&Ou length:4 atIndex:4];
            [enc setBytes:&fl length:4 atIndex:5];
            [enc setBytes:&woff64 length:8 atIndex:6];
            dispatch_1d(enc, (NSUInteger)mc * Ou, 128);
        });
        if (rc) return 1;
        memcpy(out + (size_t)m0 * O, [ob contents],
               (size_t)mc * O * sizeof(float));
    }
    return 0;
}

static int mtl_head(MtlCtx *c, const ApusStTensor *head, const float *x,
                    float *out, int64_t O, int64_t K) {
    if (!c || O <= 0 || K <= 0 || !head) return 1;
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                 (size_t)K * sizeof(float));
    id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap,
                                 (size_t)O * sizeof(float));
    if (!xb || !ob) return 1;
    memcpy([xb contents], x, (size_t)K * sizeof(float));
    uint Ku = (uint)K;
    if (head->dtype == APUS_ST_BF16) {
        size_t woff = 0;
        id<MTLBuffer> wb = weight_buf(c, head->data,
                                      (size_t)O * K * sizeof(uint16_t),
                                      &woff);
        if (!wb) return 1;
        uint64_t woff64 = woff;
        int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_bf16];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:ob offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc setBytes:&woff64 length:8 atIndex:4];
            dispatch_1d(enc, (NSUInteger)O, 128);
        });
        if (rc) return 1;
    } else if (head->dtype == APUS_ST_F32) {
        size_t woff = 0;
        id<MTLBuffer> wb = weight_buf(c, head->data,
                                      (size_t)O * K * sizeof(float), &woff);
        if (!wb) return 1;
        uint Ou = (uint)O, fl = 0;
        uint64_t woff64 = woff;
        int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_f32];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:ob offset:0 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc setBytes:&Ou length:4 atIndex:4];
            [enc setBytes:&fl length:4 atIndex:5];
            [enc setBytes:&woff64 length:8 atIndex:6];
            dispatch_1d(enc, (NSUInteger)O, 128);
        });
        if (rc) return 1;
    } else {
        return 1;
    }
    memcpy(out, [ob contents], (size_t)O * sizeof(float));
    return 0;
}

static int mtl_rmsnorm(MtlCtx *c, const float *x, const float *w, float eps,
                       float *y, int64_t n) {
    if (!c || n <= 0) return 1;
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                 (size_t)n * sizeof(float));
    id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap,
                                 (size_t)n * sizeof(float));
    size_t woff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, (size_t)n * sizeof(float), &woff);
    if (!xb || !ob || !wb) return 1;
    memcpy([xb contents], x, (size_t)n * sizeof(float));
    uint nu = (uint)n;
    float ep = eps;
    uint64_t woff64 = woff;
    int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:c->pso_rms];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&nu length:4 atIndex:3];
        [enc setBytes:&ep length:4 atIndex:4];
        [enc setBytes:&woff64 length:8 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    });
    if (rc) return 1;
    memcpy(y, [ob contents], (size_t)n * sizeof(float));
    return 0;
}

/* ========================================================================
 * Public C API (strong definitions; weak stubs live in the attn.h TU)
 * ====================================================================== */

static int mtl_init(char *err, size_t errcap) {
    if (g_ctx) return 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            snprintf(err, errcap, "no Metal device");
            return -1;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        /* FP32 IEEE semantics: no fast-math contraction/reassociation.
         * (mathMode supersedes fastMathEnabled on macOS 15+.) */
        if ([opts respondsToSelector:@selector(setMathMode:)])
            opts.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *nserr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kShaderSrc
                                               options:opts
                                                 error:&nserr];
        if (!lib) {
            snprintf(err, errcap, "shader compile: %s",
                     nserr ? [[nserr localizedDescription] UTF8String]
                           : "unknown");
            return -1;
        }
        MtlCtx *c = new MtlCtx();
        c->dev = dev;
        c->q = [dev newCommandQueue];
        if (!c->q) { snprintf(err, errcap, "no command queue"); delete c; return -1; }
        struct { const char *name; id<MTLComputePipelineState> *slot; } t[] = {
            { "act_quant", &c->pso_actquant },
            { "fp8_dot",   &c->pso_fp8dot },
            { "fp8_fold",  &c->pso_fp8fold },
            { "f32_gemm",  &c->pso_f32 },
            { "bf16_gemv", &c->pso_bf16 },
            { "rmsnorm",   &c->pso_rms },
        };
        for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:t[i].name]];
            *t[i].slot = fn ? [dev newComputePipelineStateWithFunction:fn
                                                                 error:&nserr]
                            : nil;
            if (!*t[i].slot) {
                snprintf(err, errcap, "pipeline %s failed", t[i].name);
                delete c;
                return -1;
            }
        }
        long mb = 8192;
        const char *e = getenv("APUS_METAL_DENSE_MB");
        if (e && *e) mb = atol(e);
        if (mb < 0) mb = 0;
        c->budget = (uint64_t)mb << 20;
        g_ctx = c;
    }
    return 0;
}

/* --- hook trampolines --- */
static int hook_fp8_linear(const ApusFp8W *w, const float *x, float *out,
                           int M, int K, int O) {
    return mtl_fp8(g_ctx, w->codes, w->scales, x, NULL, NULL, out,
                   M, K, O, 1);
}
static int hook_f32_linear(const float *w, const float *x, float *out,
                           int M, int K, int O, int flags) {
    return mtl_f32(g_ctx, w, x, out, M, K, O, flags);
}
static int hook_head_gemv(const ApusStTensor *head, const float *x,
                          float *out, int64_t O, int64_t K) {
    return mtl_head(g_ctx, head, x, out, O, K);
}

int apus_metal_enable(char *err, size_t errcap) {
    char e2[256];
    if (!err) { err = e2; errcap = sizeof e2; }
    if (mtl_init(err, errcap)) return -1;
    apus_backend_hooks.fp8_linear = hook_fp8_linear;
    apus_backend_hooks.f32_linear = hook_f32_linear;
    apus_backend_hooks.head_gemv = hook_head_gemv;
    return 0;
}

void apus_metal_disable(void) {
    memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
    MtlCtx *c = g_ctx;
    g_ctx = NULL;
    if (c) delete c;   /* ARC off: ObjC members leak harmlessly at exit */
}

int apus_metal_is_enabled(void) {
    return g_ctx && apus_backend_hooks.fp8_linear;
}

uint64_t apus_metal_bytes_wrapped(void) { return g_ctx ? g_ctx->bytes_wrapped : 0; }
uint64_t apus_metal_bytes_uploaded(void) { return g_ctx ? g_ctx->bytes_uploaded : 0; }
uint64_t apus_metal_dispatches(void) { return g_ctx ? g_ctx->dispatches : 0; }

/* --- direct entry points (kernel-level tests; lazy init, no hook fill) --- */

int apus_metal_fp8_linear(const ApusFp8W *w, const float *x, float *out,
                          int M, int K, int O) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_fp8(g_ctx, w->codes, w->scales, x, NULL, NULL, out,
                   M, K, O, 1);
}

int apus_metal_f32_linear(const float *w, const float *x, float *out,
                          int M, int K, int O, int flags) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_f32(g_ctx, w, x, out, M, K, O, flags);
}

int apus_metal_head_gemv_bf16(const void *w_bf16, const float *x, float *out,
                              int64_t O, int64_t K) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    ApusStTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = APUS_ST_BF16;
    t.data = w_bf16;
    return mtl_head(g_ctx, &t, x, out, O, K);
}

int apus_metal_fp8_gemm(const uint8_t *w, const uint8_t *ws,
                        const uint8_t *acodes, const float *as, float *out,
                        int M, int O, int K) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_fp8(g_ctx, w, ws, NULL, acodes, as, out, M, K, O, 0);
}

int apus_metal_fp8_act_quant(const float *x, int M, int K,
                             uint8_t *codes, float *scales) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    MtlCtx *c = g_ctx;
    if (!c || M <= 0 || K <= 0) return 1;
    size_t nb = ((size_t)K + 127) / 128;
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                 (size_t)M * K * sizeof(float));
    id<MTLBuffer> cb = stage_buf(c, c->cbuf, c->ccap, (size_t)M * K);
    id<MTLBuffer> ab = stage_buf(c, c->asbuf, c->ascap,
                                 (size_t)M * nb * sizeof(float));
    if (!xb || !cb || !ab) return 1;
    memcpy([xb contents], x, (size_t)M * K * sizeof(float));
    uint Ku = (uint)K, nb32 = (uint)nb;
    int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:c->pso_actquant];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:cb offset:0 atIndex:1];
        [enc setBuffer:ab offset:0 atIndex:2];
        [enc setBytes:&Ku length:4 atIndex:3];
        [enc setBytes:&nb32 length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M * nb32, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
    if (rc) return 1;
    memcpy(codes, [cb contents], (size_t)M * K);
    memcpy(scales, [ab contents], (size_t)M * nb * sizeof(float));
    return 0;
}

int apus_metal_rmsnorm(const float *x, const float *w, float eps, float *y,
                       int64_t n) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_rmsnorm(g_ctx, x, w, eps, y, n);
}
