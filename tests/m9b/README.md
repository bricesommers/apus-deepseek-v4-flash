# tests/m9b — Accelerate (AMX) BLAS for batch-M GEMMs + MoE prefill batching

Milestone M9b: attack the prefill side of the roofline. Two independent
changes, both behind hard quality gates:

1. **MoE prefill batching (c/moe.h)** — the pre-9b MoE forward ran every
   expert at M=1 per token (s·topk expert forwards per layer, each streaming
   its 12.6 MB weight slab from DRAM/SSD). Prefill now groups the (t,j)
   routings by expert and runs each expert's three linears ONCE at
   M=count, and runs the shared expert once at M=s. **Bitwise identical**
   to the per-token path by construction: the M9a fp4/fp8 GEMM row bodies
   are M-independent bitwise, act quant and SwiGLU are per-row elementwise,
   and the per-token weighted accumulation keeps the exact (t,j) order and
   per-element expression (`y[t] += wt*eo`, via an eo_all staging buffer).
   Proven bitwise in this test (hash and non-hash routing) and by the
   unmodified m4c/m5 chunk-invariance and m6a/m6b cross-config bitwise
   gates.
2. **BLAS dispatch for large batch M (c/blas.h + dispatch in
   apus_fp4_gemm_mt / apus_fp8_gemm_mt)** — calls with
   M >= APUS_BLAS_M_MIN (256) route to `cblas_sgemm` from Apple's
   Accelerate.framework (on M1 this uses the AMX matrix coprocessor);
   smaller M stays on the M9a NEON kernels. Decode (M=1) is untouched.

```
make test-m9b    # 25 checks; digest diffed across APUS_THREADS=1/4/8
make ubsan-m9b   # same under -fsanitize=undefined
make bench-m9b   # NEON vs NEON-mt vs BLAS at the real shapes, M=2..512
```

## Build notes

Accelerate is a macOS SYSTEM framework (ships with the OS, like libSystem);
the Makefile links `-framework Accelerate` on Darwin for all targets
(`ifeq ($(shell uname),Darwin)` guard; `c/blas.h` compiles to a no-op
stub elsewhere). The LP64 cblas interface is SDK-deprecated since macOS
13.3 in favor of the ILP64-capable headers; the symbols remain the stable
system ABI (deprecation warning silenced with a pragma in c/blas.h).

The BLAS path pins `VECLIB_MAXIMUM_THREADS=1` on first use
(`apus_blas_available()`), so every sgemm runs on a single vecLib thread.
Parallelism comes from the engine's own pool: the weight matrix is split
into a FIXED grid of ≤8 MB row tiles (multiple of 128 rows; the grid
depends only on (O, K)), tiles are distributed over the c/pool.h lanes,
and each lane dequantizes its tiles into its own TLS scratch and calls
sgemm per tile. Every output element is therefore produced by an sgemm of
the same shape in every thread configuration — bitwise thread-count
independent by construction (asserted by the T=1/4/8 digest diff, which
also came out identical to the pre-tiling single-call variant). AMX is
per-core, so N lanes drive N AMX units.

## Numerics contract (exact scale folding; reorder-only deltas)

All scale factors in both kernels are UE8M0 powers of two, and scaling by
a power of two is EXACT in FP32 (barring over/underflow, which real
checkpoint scales never approach) and distributes exactly over a dot
product's partial sums. The BLAS path therefore folds the scales into the
dequantized operands and runs ONE full-K sgemm per weight tile:

- fp8: `a'[m,i] = e4m3(acode)·sa[m,i/128]`, `w'[o,i] = e4m3(w)·sb[o/128,i/128]`
  → `Σ a'·w'` has the identical VALUE as the contract's
  `Σ_kb dot_kb·(sa·sb)` (product-first), because both the scale product
  and every fold are exact pow2 operations.
- fp4: `a'[m,i] = e4m3(acode)·sa[m,i/128]`, `w'[o,i] = LUT·sb[o,i/32]`
  (the kernel's codes×2 with the exact 0.5 folded into sb) → identical
  VALUE to `Σ_kb (dot32·sa)·sb`.

The ONLY difference vs the NEON/scalar paths is the FP32 summation order
inside the dots — the project's accepted reorder tolerance class
(tests/m3, m9a). Measured here against FP64 truth: worst err/esc
4.5e-7 (fp8) / 6.5e-7 (fp4), bound 2e-5 (the m7b battery's bound).
Act-quant inputs and the BF16-rounding boundaries are outside the GEMM
and untouched. Degenerate scale byte 255 (inf) can diverge from the NEON
path (inf would fold into weights before the dot); it never occurs in the
checkpoint, and the BLAS path is only exercised on real-scale data.

Weight dequant+fold streams each weight tile once per call into a ≤8 MB
FP32 scratch tile (per-lane TLS scratch arena, heap fallback), then sgemm
(NoTrans, Trans) per tile; activations are folded in place in the
caller's M·K scratch. Dequant cost is O(O·K), amortized M×.

## Dispatch cutoff: why 256

The measured NEON-mt/BLAS crossover is far below 256 (see bench table:
BLAS wins from M≈16-32 on the big shapes), but 256 is the smallest value
that keeps EVERY pre-M9b suite on its pinned NEON path: the largest
batched M in m4c (250), m5/m7b (200), m6a/m6b (8), m8 (k+1) is 250.
Those suites carry BITWISE gates that compare batched-M forwards against
M=1 decode (chunk invariance, spec/non-spec streams, CPU-vs-Metal
identity); a reorder-class prefill delta below M=256 would break them,
and the mission requires them green UNLOOSENED. The cost: prompts of
256 tokens or less get no BLAS attention/shared-expert GEMMs (they still
get the MoE batching win, which is bitwise). This is a gate-driven
cutoff, not a perf-driven one; relaxing it needs the m5/m7b/m8
chunk-invariance gates re-negotiated.

## Measured numbers (microbench, best-of-3, quiet machine)

Machine: MacBook Pro M1 Pro 32 GB. `neon` = M9a single-thread kernel,
`mt` = M9a kernel on the 8-lane pool (the engine's pre-9b path for every
GEMM call; measured with APUS_NO_BLAS=1 so the dispatch stays out),
`blas` = the M9b pool-tiled Accelerate path. GF = GFLOP/s.

| shape | M | neon | mt (8T NEON) | blas (AMX tiles) | vs mt |
|---|---|---|---|---|---|
| fp8 wq_b 32768x1024 | 2 | 14.6 | 94.9 | 17.4 | 0.2x |
| fp4 ex_w1w3 2048x4096 | 2 | 37.0 | 167.4 | 23.0 | 0.1x |
| fp8 wq_b | 4 | 25.9 | 159.6 | 34.9 | 0.2x |
| fp4 ex_w1w3 | 4 | 43.1 | 180.9 | 47.0 | 0.3x |
| fp8 wq_b | 8 | 28.1 | 197.8 | 70.6 | 0.4x |
| fp4 ex_w1w3 | 8 | 42.3 | 204.5 | 94.0 | 0.5x |
| fp8 wq_b | 32 | 30.8 | 205.0 | 322.1 | **1.6x** |
| fp4 ex_w1w3 | 32 | 42.5 | 233.8 | 338.0 | **1.4x** |
| fp8 wq_b | 128 | 30.8 | 218.4 | 913.1 | **4.2x** |
| fp8 sh_w1w3 2048x4096 | 128 | 30.9 | 197.0 | 581.7 | **3.0x** |
| fp8 sh_w2 4096x2048 | 128 | 31.0 | 204.7 | 643.0 | **3.1x** |
| fp4 ex_w1w3 | 128 | 42.0 | 259.4 | 598.5 | **2.3x** |
| fp4 ex_w2 4096x2048 | 128 | 42.6 | 274.9 | 630.9 | **2.3x** |
| fp8 wq_b | 256 | 30.6 | 218.4 | 1007.7 | **4.6x** |
| fp8 sh_w1w3 | 256 | 30.7 | 201.4 | 828.7 | **4.1x** |
| fp4 ex_w1w3 | 256 | 41.6 | 244.9 | 861.7 | **3.5x** |
| fp8 wq_b | 512 | 30.6 | 220.1 | 1351.7 | **6.1x** |
| fp8 sh_w1w3 | 512 | 30.6 | 198.7 | 1098.0 | **5.5x** |
| fp8 sh_w2 | 512 | 30.9 | 211.6 | 1219.1 | **5.8x** |
| fp4 ex_w1w3 | 512 | 42.0 | 265.6 | 1109.2 | **4.2x** |
| fp4 ex_w2 | 512 | 42.4 | 281.1 | 1235.0 | **4.4x** |

Takeaways:

- **M=2..8 (decode + M8 spec-verify): BLAS loses 2-5x.** sgemm call +
  packing overhead dominates; the M9a NEON GEMV/small-M path (DRAM-bound)
  stays. The M8 spec-verify batched forward (M=k+1 <= 4) therefore keeps
  the NEON kernels — measured, not assumed — and the dispatch cutoff
  (256) keeps it there.
- Crossover vs the 8-thread NEON pool is ~M=16-32.
- At M=512 the tiled BLAS path reaches **1.1-1.35 TFLOP/s FP32** (8 AMX
  lanes) vs ~200-280 for the NEON pool: 4.2-6.1x. Pool tiling added
  ~1.6x over a single sgemm call (which measured ~840 GF/s at M=512
  before tiling).
- fp4 vs fp8 at the same shape: same GF/s within noise — the dequant
  fold is fully hidden behind sgemm at these M.

## Real-model before/after

Command (filler prompts of exact length via --ids, 1 decode token to
isolate prefill; same machine, same tiered store, runs interleaved with
the baseline window, load avg ~3):
`./bin/apus run --model weights/apus --tiered --ids <N x 100> --max-tokens 1 --seed 1 --temp 0`

| prompt | before (M9a) | after (M9b) | speedup |
|---|---|---|---|
| 32 tok | 15.48 s (2.1 tok/s) | 12.60 s (2.5 tok/s) | 1.23x |
| 128 tok | 50.75 s (2.5 tok/s) | 40.76 s (3.1 tok/s) | 1.24x |
| 512 tok | 196.72 s (2.6 tok/s) | 123.82 s (4.1 tok/s) | **1.59x** |

Quality gate: `./bin/apus run --model weights/apus --tiered --prompt
"The capital of France is" --max-tokens 24 --seed 1 --temp 0` produces
the M9a tokens EXACTLY (`We need to respond to the user's query: "The
capital of France is". This is a straightforward factual question.`).
Expected by construction (the 9-token prompt keeps every GEMM at M<=54
< 256, i.e. the NEON path, and the MoE batching is bitwise); verified.

Decode regression: the M=1 path is bit-identical by construction
(untouched gemv/NEON kernels, untouched s==1 MoE path, dispatch at
M>=256 only). bench-m9a after the change: fp8 wq_b GEMV 4.44 ms,
fp4 ex_w1w3 GEMV 582 us — same kernels, same or better than the M9a
numbers (quieter machine). Smoke-run decode rate 0.39 tok/s sits inside
the load-dependent 0.44-0.76 tok/s window documented in tests/m9a.

**Why "only" 1.6x at 512:** the tiered expert store, not compute, is now
the prefill wall. The 512-token prefill streams 42.8 GB of expert slabs
(3358 preads, ~12.7 MB each) both before AND after — the M6b batch-union
hint already coalesced the reads; M9b removed the per-token compute and
weight re-streaming on top (hits 27-51 vs misses ~3000: the cache can
hold almost nothing across layers). Compute no longer dominates:
attention + shared-expert GEMMs at M=512 run at ~1.1-1.35 TF/s (BLAS)
and routed experts at M~12 on the NEON pool. The next prefill win is
expert-I/O (fewer bytes or deeper prefetch/compute overlap), not GEMM.

## Honest notes

- The dispatch cutoff (256) is gate-driven, not perf-driven: BLAS already
  wins ~1.4-1.6x at M=32, but every pre-M9b bitwise gate that compares a
  batched-M forward against M=1 decode (m4c/m5 chunk invariance, m6c
  mt==NEON, m7b GPU==CPU, m8 spec streams) sits at M<=250. Prompts of
  <=256 tokens therefore get the MoE-batching win only; their attention
  and shared-expert GEMMs stay NEON. Re-negotiating those gates to
  tolerance-based comparisons would unlock another ~1.5-4x for 32..256
  token prompts.
- The MoE batching changes store-internal accounting: experts are
  resolved once per unique expert per layer during prefill (was: once
  per (t,j)). Slab bytes, outputs, and all cross-config bitwise gates
  are unchanged (m6a/m6b invariance green); the m6b pilot test's pinned
  ask count was updated to the new (computed) expectation, its 100%
  coverage / 0-demand-loads assertions untouched.
- Routed-expert GEMMs during prefill run at M=count (avg s*topk/E ~ 12
  at s=512) — below the BLAS cutoff, so they stay on the NEON pool
  kernels. At M~12 the NEON-mt path (~200 GF/s) still beats BLAS
  (measured 0.4-0.5x), so this is the right per-shape choice.
- Degenerate UE8M0 scale byte 255 (inf) would fold inf into the
  dequantized weights (the NEON path multiplies the finite dot by inf
  instead); NaN/inf patterns can differ there. Byte 255 never occurs in
  the checkpoint and the m3/m4a edge coverage pins the dequant-level
  semantics, which are unchanged.
- Decode (M=1) is bit-identical to M9a by construction: the gemv/gemm
  NEON kernels and the s==1 MoE path are untouched, and the dispatch
  only engages at M>=256.

## Gate inventory (all green)

- test-m9b: 25 checks; digest `b398ab7989152252` identical across
  APUS_THREADS=1/4/8 AND identical UBSan-vs-O2 (BLAS path is
  single-vecLib-thread with a fixed tile grid).
- m2 (5176+79), m3 (15503), m4a (143+111), m4c (1217), m5 (19): 0
  failures, tolerances untouched; m5 chunkinv 200@120 bitdiff=0/102400,
  determinism bitwise; m4c chunkinv/determinism bitwise.
- m6a (85+5), m6b (47+7+recall): cross-config bitwise preserved.
  m6b test_pilot's pinned resolve-count expectation was UPDATED
  (per-(t,j) -> per-unique-expert asks, computed from tid2eid); the
  100%-coverage and 0-demand-loads assertions are unchanged.
- m6c: 514 checks, digest `8bc96af4504a5161` — byte-identical to the M9a
  value (small-M paths untouched); T=1/4/8 diffed.
- m7a server: 34/34. m7b: 117+22 checks, GPU==CPU bitwise preserved
  (prefill argmax 0/200 IDENTICAL, chunkinv bitwise) with ZERO Metal
  changes — the cutoff keeps every m7b shape on the NEON path.
- m8: 41 checks, spec/non-spec streams bitwise (verify batches M<=4 stay
  NEON — measured right too: BLAS loses 2-5x at M=2..8).
- m9a: 164 checks, digest `5f7755151c61a979` unchanged.
- UBSan: m2,m3,m4a,m4c,m5,m6a,m6b,m6c,m7b,m8,m9a,m9b all green.
- Real-model 24-token smoke: IDENTICAL tokens to the M9a build.
