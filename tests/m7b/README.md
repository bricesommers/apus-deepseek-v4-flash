# tests/m7b — Metal backend: dense compute on the Apple GPU (optional)

Milestone M7b: an **optional** Metal backend (`c/backend_metal.mm` +
`c/backend_metal.h`) that offloads the dense compute of the forward pass —
attention FP8 linears (wq_a/wq_b/wkv/wo_a/wo_b, indexer.wq_b), the shared
expert (w1/w2/w3), the router gate matmul, and the BF16 LM head — to the
Apple GPU, so the CPU is freer for expert streaming. FP32 compute only,
zero-copy unified-memory buffers, per-op fail-soft to the CPU kernels.

```
make metal=1 apus    # builds bin/apus_metal (== make bin/apus_metal)
make test-m7b        # kernel + model gates + m7a server suite on the metal binary
make ubsan-m7b       # same, UBSan on the C side
make bench-m7b       # CPU-vs-Metal microbench + tok/s (informational)
bin/apus_metal run --model DIR --metal ...   # or APUS_METAL=1
```

**CPU stays the default and is untouched behaviorally**: `bin/apus` never
links the backend; every hook defaults to NULL; with no `--metal` flag the
`bin/apus_metal` binary runs the identical CPU kernels. All prior suites
(m2…m7a) green, including under UBSan.

## Backend interface (the wiring design)

A **function-pointer table**, `ApusBackendHooks apus_backend_hooks`
(`c/backend_metal.h`), defined once in the `APUS_ATTN_IMPLEMENTATION` TU
(attn.h is linked into every engine binary) and all-NULL by default = CPU
kernels. `apus_metal_enable()` — strong definition in `c/backend_metal.mm`,
**weak stub** in the attn.h TU so the plain CPU binary links and prints a
clean "not compiled in" fallback message — initializes the
device/queue/pipelines and fills the table. Call sites try the hook first
and run the CPU kernel when the pointer is NULL or the call returns
nonzero:

| hook | call site | semantics |
|---|---|---|
| `fp8_linear` | `c/attn.h` `apus_fp8_linear` (wq_a/wq_b/wkv/wo_b, indexer.wq_b, shared-expert w1/w2/w3 via `c/moe.h`) | BF16-round input, per-128 act quant, blockwise FP8 GEMM, BF16-round output |
| `f32_linear` | `c/moe.h` `apus_router_score` gate matmul (flags 0); `c/attn.h` `apus_bf16_linear` (R_IN\|R_OUT); `c/attn.h` wo_a group slices (R_OUT\|W_BF16) | FP32 matmul with optional BF16 RNE rounding of input/output; W_BF16 (M6c) = weights passed as BF16 bits, widened exactly once and cached |
| `head_gemv` | `c/model.h` `apus_head_gemv` | BF16 (or F32) shard-view GEMV, FP32 accumulate, unrounded logits |

Everything else — FP4 routed experts, mHC/Sinkhorn, compressors, indexer
score math, RoPE, sampling — stays on the CPU this milestone. RMSNorm has a
shader (kernel-tested below) but is **not wired**: at these sizes the
~30–50 µs per-dispatch overhead exceeds the CPU cost; it is staged for a
future fused-layer milestone (fused norm+matmul per the ARCHITECTURE §7
"full-layer command buffers" idea).

## Numerics contract (FP32 only, exact rounding sequence)

Shaders are compiled with **fast math disabled** (`MTLMathModeSafe`): no
mul+add contraction, no reassociation, IEEE division/sqrt (`precise::`).
All math is FP32; the only FP16/BF16-width operations are the same
quantize/dequantize/round simulations the CPU kernels perform (bit-exact
ports, no FP16 arithmetic anywhere). The FP8 path reproduces `c/fp8.h`
step for step:

- act quant (one 128-thread threadgroup per block): BF16 RNE input round,
  amax floored at 1e-4, scale = 2^ceil(log2(amax/448)) via the FP32 bit
  trick, clamp ±448, RNE to E4M3 — **bitwise identical** to
  `apus_fp4_act_quant_scalar` (verified below);
- GEMM as two passes: per-128-block FP32 dots, then the fold `sc = sa*sb`
  (scale product FIRST), `total += dot*sc` — the kernel.py:242-249 order.
  **Since M9a** the `fp8_dot` shader accumulates in the exact CPU NEON
  canonical order (c/fp8.h: four independent accumulators per row, fixed
  combine) and `fp8_fold` mirrors the CPU's contracted fmadd fold, so the
  raw GEMM outputs are **bit-identical** to the CPU NEON kernel (asserted
  in the battery below);
- BF16-round output for the linear entry points.

Historically the only divergence source vs the CPU was FP32 summation order
(GPU sequential+fma vs NEON 4-lane) — the same class as the scalar-vs-NEON
precedent documented in tests/m3/m4a; M9a eliminated it for the FP8 path by
giving both sides the same fixed order.

## Buffer management (zero-copy)

Weight tensors are wrapped as `MTLBuffer`s **on first use**, keyed by CPU
pointer: `newBufferWithBytesNoCopy` over the page-rounded range, validated
by walking the (possibly several, for >128 MB mallocs) contiguous
`vm_region`s covering it — unified memory, so GPU and CPU share the same
pages and resident dense weights (~6 GB FP8 + ~1 GB BF16 head on the real
container) are **not duplicated**. If a range can't be wrapped, one upload
copy is made instead. `APUS_METAL_DENSE_MB` (default 8192) caps
wrapped+uploaded weight bytes; past the cap the hook returns "unsupported"
and the op silently runs on the CPU. Activations/outputs use small grow-on-
demand shared staging buffers; prefill rows are chunked 8 at a time to keep
the dot staging bounded. One command buffer per op, synchronous
`waitUntilCompleted` (the engine is single-threaded).

**Stable-pointer invariant**: the pointer-keyed cache assumes weights are
never freed while the backend is enabled (true for the engine: shard slabs
and layer arrays live until `apus_model_free`; `apus_metal_disable()` drops
the cache). A freed-then-reused address would alias a stale buffer — this
bit once in the kernel tests themselves (documented in test_kernels.c).

## Measured results (MacBook Pro M1 Pro, 32 GB)

### Kernel-level (`make test-m7b` → test_kernels, 117 checks, 0 failures)

- **act quant**: 21,504 codes + all scales **bitwise** vs CPU (incl. zero
  rows → amax floor, tiny/outlier rows, partial K blocks 64/448).
- **FP8 GEMM battery** (real dense shapes 1024×4096 … 8192×1024, mini-model
  shapes, odd/partial shapes, M ∈ {1,2,3,5,9}) vs in-test FP64 truth with
  the m4a `esc` metric: worst **err/esc = 1.31e-07** (bound 2e-5).
  **Since M9a: GPU-vs-CPU-NEON is BITWISE on every output of the battery
  (asserted)** — the shader uses the CPU's canonical accumulation order.
  Edge cases: zero weights + scale byte 254 → exact zeros;
  byte 0 (2⁻¹²⁷ subnormal) maxdiff 1.6e-39; byte 255 (+inf) → NaN parity
  with the CPU.
- **Full fp8 linear** (BF16-rounded outputs): **24,580/24,580 bitwise** vs
  the CPU composition (FP32-order noise never crossed a BF16 RNE boundary
  in the battery).
- **f32 linears**: bf16-linear and wo_a variants bitwise; router gate
  (unrounded) worst 4.3e-4 abs on a 9.4e2 scale (~5e-7 rel — fma
  contraction class). **BF16 head GEMV**: worst 2.0e-4 on a 6.7e2 scale
  (~3e-7 rel). **RMSNorm**: bitwise at n ≤ 512, worst 6.1e-5 at n = 4096
  (parallel reduction order).
- Zero-copy engaged everywhere: 64.3 MB wrapped, **0 MB uploaded** (kernel
  tests); 1,110 MB wrapped incl. a 1 GB head tensor (bench).

### Model-level gate (test_model, 22 checks, 0 failures)

- **A. Metal vs the m5 oracle goldens** (same battery + tolerances as
  tests/m5): prefill_len6 rel 2.8e-2, prefill_len200 rel 3.0e-1,
  decode_from64 rel 4.0e-1 — all inside the m5 bounds; greedy flips 3/24,
  the *same three* near-ties as the CPU path (gaps 0.013/0.067/0.207);
  sampled flips 19/24 all near-boundary (margin ≤ 5.2e-3). The m5 contract
  holds unchanged through the backend.
- **B. THE GATE — CPU vs Metal, same process**:
  **greedy teacher-forced token stream IDENTICAL, 0/24 flips.**
  Since M9a (GPU FP8 GEMM bit-identical to CPU): prefill_len200 per-position
  argmax **0/200 flips — IDENTICAL** (pre-M9a: 8/200 flips, all at CPU-logit
  top1−top2 gaps ≤ 5.9e-2 ≪ 0.5, the m5 near-tie class).
  Logit divergence (measured): teacher-forced rel 3.2e-7, prefill rel
  2.8e-7 maxabs 1.2e-6 (post-M9a — down from rel 2.2e-2 / 2.9e-1, the old
  summation-order cascade class of tests/m5; the FP8 dense path no longer
  contributes any divergence).
- **C.** Metal-path determinism bitwise (repeated prefill), chunk
  invariance bitwise (prefill(200) == prefill(120)+80 decodes,
  0/102,400 diffs).

### Server-level

The full **m7a server suite passes against the Metal binary** (34/34 tests,
`APUS_BIN=bin/apus_metal APUS_METAL=1`), both O2 and UBSan — serving is
unchanged (encode/generate/SSE/tool calls/stop strings/seeds).

### Performance (informational, `make bench-m7b`)

Decode GEMV (M=1), effective weight-streaming bandwidth:

| op | shape | CPU-NEON | Metal | speedup |
|---|---|---|---|---|
| fp8 wq_a | 1024×4096 | 1.57 ms (2.5 GB/s) | 1.66 ms (2.3 GB/s) | x0.94 |
| fp8 wq_b | 32768×1024 | 4.32 ms (7.2 GB/s) | 4.29 ms (7.3 GB/s) | x1.01 |
| fp8 wkv | 512×4096 | 0.68 ms (2.9 GB/s) | 0.66 ms (2.9 GB/s) | x1.02 |
| fp8 wo_b | 4096×8192 | 6.11 ms (5.1 GB/s) | 6.10 ms (5.1 GB/s) | x1.00 |
| fp8 shared w1/w3 | 2048×4096 | 1.62 ms (4.8 GB/s) | 1.65 ms (4.7 GB/s) | x0.98 |
| fp8 shared w2 | 4096×2048 | 1.40 ms (5.6 GB/s) | 1.39 ms (5.6 GB/s) | x1.01 |
| fp8 indexer wq_b | 8192×1024 | 1.29 ms (6.1 GB/s) | 1.28 ms (6.1 GB/s) | x1.01 |
| bf16 head | 129280×4096 (1 GB) | 69.7 ms (14.2 GB/s) | 69.8 ms (14.1 GB/s) | x1.00 |
| f32 router gate | 256×4096 | 1.04 ms (3.8 GB/s) | 0.90 ms (4.4 GB/s) | x1.15 |

m5 mini-model greedy decode: **CPU 604 tok/s, Metal 59 tok/s** (x10
*slower* — ~30–50 µs dispatch overhead × ~35 hooked ops/token dominates at
256-dim toy shapes; expected, and the reason RMSNorm is not wired).

Reading of the numbers: on unified memory both sides stream from the same
DRAM, and the CPU baseline is single-threaded, so the GPU is at **parity**
on GEMV shapes — the win this milestone is architectural (dense compute
off the CPU, freeing it for expert resolve/disk I/O per ARCHITECTURE §7),
not raw speed. The current shaders are deliberately naive (scalar byte
loads, one thread per 128-block); measured headroom to ~14 GB/s DRAM-bound
(and beyond with `uint4` vectorized code paths, simdgroup-per-row
reductions, and a fused single-pass dot+fold) is real but left for a future
milestone — numerics correctness came first.

### Fail-soft behavior (all covered by tests)

- No Metal device / shader compile failure → `apus_metal_enable` returns
  nonzero, hooks stay NULL, engine runs CPU (CLI prints the reason). The
  CPU binary with `APUS_METAL=1` prints "metal backend not compiled in" and
  continues on CPU.
- Weight-buffer budget exhausted (`APUS_METAL_DENSE_MB=1` verified on the
  CLI and in test_kernels) → per-op "unsupported" → CPU kernel for that op;
  the hooked fallback is **bitwise identical** to the pure CPU path
  (asserted).
- GPU command-buffer error → the op returns 1 → CPU fallback for that op.
- `apus_metal_disable()` clears the hooks and releases the backend.

## What a Metal-expert-GEMV milestone (GPU FP4 kernels) would need

- **FP4-E2M1 dequant + GEMV shaders** mirroring `c/fp4.h`: 16-entry LUT,
  per-32 UE8M0 block scales (exponent-add semantics), act quant per-128
  E4M3 (already ported here), FP32 accumulation with the FP4 scale order
  `(dot*sa)*sb` — note it differs from the FP8 product-first order
  (tests/m4a README warns against unifying them).
- **Buffer model**: expert slabs are transient (LRU-resident in the M6a
  store), which breaks the stable-pointer invariant — the backend needs
  generation-tagged buffer entries keyed by (slab id, generation) or an
  explicit `apus_metal_weight_forget(ptr)` invalidation call from the
  store's eviction path; zero-copy wraps remain valid while the slab lives.
- **Batching**: per-expert GEMV dispatches (~13 MB weights) would pay the
  ~30–50 µs overhead per expert per token (258/token); a batched
  multi-expert dispatch (one command buffer per layer's top-k) is required
  for the offload to pay off, plus overlap with the I/O pool.
- The two-pass dot+fold structure, staging, and hook plumbing built here
  carry over directly (add an `fp4_linear` hook to `ApusFp4W`).

## Remaining gaps / future work

- RMSNorm shader tested but unwired (dispatch overhead; needs fused
  norm+matmul to pay off — the ARCHITECTURE §7 full-layer fusion).
- Router *scoring* (sqrtsoftplus) stays on CPU; only the gate matmul is
  offloaded.
- Shader perf: scalar-load kernels at DRAM parity, not yet using
  vectorized loads / simdgroup reductions / fused single pass.
- MTP-block FP8 linears (e_proj/h_proj) route through `apus_fp8_linear` and
  would work through the same hook, but M8 hasn't landed yet — untested.
- Real-container smoke test (weights/apus/) still pending per risk R1;
  everything here is verified on the synthetic fixtures.
