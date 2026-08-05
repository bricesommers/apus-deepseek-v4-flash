# tests/m9d — prefill compute utilization: dense/wo_a BLAS dispatch + row pooling

Milestone M9d: attack the prefill compute side of the M9c profile (57%
cvwait idle; `f32_linear_rows_scalar` ~12–14%, `fp4_gemm_neon` ~8–10%,
`woa` ~4–6.6%, `sparse_attn` ~4.8–6%). Two change classes:

1. **BLAS dispatch for the pinned scalar dense linears and wo_a at
   M >= 256** (c/blas.h, c/attn.h) — the same gate-safe cutoff pattern as
   M9b: every pre-M9d bitwise gate runs at M<=250 and keeps its pinned
   path; only real prefill batches >= 256 tokens engage Accelerate (AMX).
   FP32 summation-order reorder ONLY (the accepted tolerance class,
   tests/m3/m9a/m9b) — these dots carry no scales, so there is nothing
   else to preserve.
2. **Pooling the serial per-row loops** (c/attn.h, c/layer.h, c/moe.h) —
   indexer q-prep/scores/top-k, compressor pooling + finish, q/kv
   norm+rope+QAT, hc pre/post, act quant, SwiGLU, MoE weighted
   accumulation. Each row's per-element op sequence is EXACTLY the serial
   loop's (rows are independent), so results are bitwise identical for any
   thread count by the pool.h contract. Pure lane-utilization work: zero
   numerics change, proven by every pre-M9d digest staying byte-identical.

```
make test-m9d    # 30 checks; digest diffed across APUS_THREADS=1/4/8
make ubsan-m9d   # same under -fsanitize=undefined
make bench-m9d   # pre-M9d pooled rows vs M9d BLAS at the real shapes
```

## What the M9c profile said (premise check, this machine)

sample(1) of a 512-token prefill on the M9c build (12 s window, 8 pool
lanes + 4 I/O threads + pilot):

| frame | share of thread time |
|---|---|
| `__psynch_cvwait` (idle) | ~57% |
| `apus_f32_linear_rows_scalar` | ~14.5% |
| `apus_fp4_gemm_neon_rows` | ~8.2% |
| `apus_sparse_attn_head` | ~6.0% |
| `pread` (I/O workers) | ~4.4% |
| `apus_woa_rows` | ~4.1% |
| `apus_mhc_prepost_scalar` | ~1.1% |
| `apus_indexer` (serial scores/topk) | ~0.6% |

Matches tests/m9c/README.md. The f32/bf16 linears (compressor wkv/wgate,
router gate, idx_wproj, pilot mixes) were pool-parallel but LATENCY-bound
scalar dots: ~16.6 GF/s aggregate on 8 lanes (bench below). The
main-thread stack showed the serial indexer scores loop (~1 GFLOP per
indexer layer: s x nb x ih x idm MACs) and the per-token hc/mixes loops
running while all pool lanes sat in cvwait.

## 1. BLAS dispatch (reorder class, gate-safe cutoff)

- `apus_f32_linear` / `apus_bf16_linear` (c/attn.h): at
  M >= APUS_BLAS_M_MIN (256) route to `apus_f32_gemm_blas` (c/blas.h) —
  pool-tiled single-vecLib-thread sgemm over the FIXED (O,K)-only tile
  grid (the M9b design; weights are already FP32 in RAM and stream
  directly, no dequant). bf16_linear keeps its input pre-round and
  per-element output round. Below the cutoff the pinned scalar-order row
  path is untouched (load-bearing for m4c/m5/m6c/m7b/m8, all at M<=250,
  and for decode at M=1).
- Grouped wo_a o-proj (c/attn.h): at s >= 256 route to
  `apus_woa_gemm_blas` — ONE pool dispatch over G x the fixed (ol,sub)
  tile grid (all 8 lanes stay busy across groups); BF16 weights widened
  EXACTLY to FP32 per tile (the same bit shift as the NEON kernel),
  sgemm per group tile, then the same per-element BF16 output round.
  Engages only where no pre-M9d gate can see it; the Metal hook still
  takes precedence when enabled.
- The M9c prefill-union pilot mixes (`apus_f32_linear` in c/pilot.h) ride
  the same dispatch. The pilot only chooses WHICH slabs to prefetch —
  outputs are invariant to it — and the m9c ON/OFF/eager bitwise gate
  stays green (same build, deterministic path).

Measured here (test_m9d): f32 BLAS vs FP64 truth worst err/esc 1.6e-07;
wo_a BLAS vs FP64 worst 1.1e-07 (bound 2e-5). vs the pinned paths the
delta is the reorder class: f32 BLAS came out BITWISE equal to the scalar
rows on the tested shapes (single-thread vecLib accumulates
k-sequentially per output there); wo_a shows ~1e-4 of outputs flipping by
exactly ONE bf16 ulp at rounding boundaries — the same class the M6c
wo_a NEON notes document.

## 2. Row pooling (bitwise, zero numerics change)

Serial loops that ran on the main thread while the pool idled, now
`apus_pool_run` over their independent rows (same row body; a small
threshold keeps tiny jobs inline — thresholds affect speed only, never
values):

- attn.h: fp8/fp4 act-quant rows (M>=8), indexer q-prep (s*ih rows),
  indexer scores (s rows — the big one, ~1 GFLOP/layer serial before),
  indexer top-k (s rows), compressor prefill pooling (nb entries),
  compressor finish (nb entries), q per-head norm+rope (s*h rows),
  window-kv norm/rope/QAT (s rows), qa/kva RMSNorm (s rows).
- layer.h: hc_pre / hc_post token loops (s>=4), attn/ffn RMSNorm (s>=8).
- moe.h: SwiGLU element ranges (n>=16384), routed-expert weighted
  accumulation (s>=8 tokens, exact (t,j) order per element).

sparse_attn was deliberately NOT touched: its per-head q*k dot and P*V
orders are pinned against decode (the 24-token smoke is bitwise), and it
is already pooled over s*h rows.

## Measured before/after (microbench, best-of-3, APUS_THREADS=8)

`make bench-m9d` on this Mac:

| path | shape | before | after | speedup |
|---|---|---|---|---|
| f32 linear (compressor wkv/wgate) | M=512 O=1024 K=4096 | 259.3 ms (16.6 GF/s) | 3.46 ms (1241 GF/s) | 74.9x |
| f32 linear (router gate) | M=512 O=256 K=4096 | 64.5 ms | 0.72 ms (1500 GF/s) | 90.1x |
| f32 linear (idx_wproj) | M=512 O=64 K=4096 | 16.1 ms | 0.21 ms | 76.3x |
| f32 linear (compressor r128) | M=512 O=512 K=4096 | 130.3 ms | 1.51 ms | 86.4x |
| f32 linear | M=256 O=1024 K=4096 | 129.8 ms | 1.87 ms | 69.5x |
| wo_a grouped | M=512 G=8 ol=1024 sub=4096 | 196.7 ms (175 GF/s) | 38.9 ms (882 GF/s) | 5.1x |
| wo_a grouped | M=256 G=8 ol=1024 sub=4096 | 87.9 ms | 26.1 ms | 3.4x |

## Real-model before/after

Same-window interleaved A/B (tests/m9d/finals.sh), median of 3; BEFORE =
the M9c build (/tmp/run_model_before), AFTER = M9d. Machine load was
high (loadavg ~9-12, shared desktop) during the batch, so absolute walls
sit above the quiet-window M9c numbers; the interleave keeps the RATIO
meaningful. Real-prompt runs used a 615-token English text (temp 0).

| metric | M9c (median) | M9d (median) | speedup |
|---|---|---|---|
| prefill 32 wall | 16.35 s (15.05/16.35/16.49) | 13.68 s (12.30/13.68/14.09) | 1.20x |
| prefill 128 wall | 39.30 s (38.28/39.30/41.34) | 36.65 s (32.27/36.65/44.36) | 1.07x |
| prefill 512 wall | 131.44 s (127.10/131.44/134.37) | 59.47 s (53.86/59.47/75.40) | **2.21x** |
| prefill 615 (real text) | 188.58 s (3.3 tok/s) | 119.24 s (5.2 tok/s) | 1.58x |
| decode 24 wall | 56.33 s (60.49/49.12/56.33) | 57.22 s (58.40/57.22/54.36) | parity (1.0x) |
| decode tok/s | 0.43 | 0.42 | unchanged (bit-identical path) |
| 615-tok prompt, 16 greedy tokens | `We need to answer the user's query. The user provided a long text about` | IDENTICAL | quality gate |
| 24-token smoke (9-tok prompt) | baseline string | IDENTICAL in every run | quality gate |

Prefill tok/s (median walls): 32: 2.0 -> 2.3; 128: 3.3 -> 3.5;
512: 3.9 -> 8.6. Quieter-window single runs on the M9d build reached
9.5 tok/s at 512 (53.86 s). The 32/128 gains come from row pooling only
(the BLAS dispatch engages at M>=256); the 512 gain stacks BLAS + pooling.
Decode is on the M=1 pinned paths in both builds — bit-identical outputs,
parity wall (median spread ~15% from machine load).

## After-profile (sample(1), 512-token prefill, M9d build, 10 s window)

| frame | before | after |
|---|---|---|
| `__psynch_cvwait` (idle) | ~57% | ~53% |
| `apus_f32_linear_rows_scalar` | ~14.5% | **gone** (BLAS) |
| `apus_woa_rows` | ~4.1% | **gone** (BLAS) |
| `apus_fp4_gemm_neon_rows` | ~8.2% | ~20% |
| `apus_sparse_attn_head` | ~6.0% | ~13.5% |
| `pread` (I/O workers) | ~4.4% | ~6.7% |
| `apus_mhc_prepost_scalar` | ~1.1% | ~2.7% (now pooled rows) |
| `apus_indexer(_score_rows)` | ~0.6% serial | ~1.3% pooled |
| BLAS tiles (fp8 + dense + wo_a) | ~1.3% | ~4% |

The scalar f32 and wo_a NEON frames are eliminated; the remaining compute
is the routed-expert fp4 GEMMs and sparse_attn.

## Remaining bottlenecks (for the next milestone)

1. **Routed-expert fp4 GEMMs at M~6-24** (~20% of thread time): the
   medium-M gap between the M9a NEON path and the M>=256 BLAS dispatch.
   BLAS at 16<=M<256 is NOT gate-safe (the m4c/m5/m7b/m8 chunk-invariance
   gates compare batched-M forwards against M=1 decode and need
   per-row M-independence, which sgemm does not guarantee) and M9b
   measured BLAS losing at M<=8 anyway. The gate-safe option is a blocked
   NEON kernel that processes multiple weight rows per activation pass
   while keeping the EXACT per-row accumulation order (bitwise by
   construction, like M9a).
2. **sparse_attn** (~13.5%): pooled over s*h rows but the q*k dot is the
   pinned scalar-order dot (bitwise-pinned against decode by the smoke
   gate); only the P*V half is NEON. Needs a decode-side numerics
   renegotiation to move.
3. **cvwait ~53%**: the per-expert loop runs ~108 expert batches
   sequentially per layer (3 pool dispatches each at M~6-24 with small
   row counts), plus pool-join imbalance and the residual ~1 s of I/O
   waits. Grouping independent experts into fewer, larger dispatches is a
   scheduling-shaped win but must keep the per-(t,j) accumulation
   contract.

## Gate inventory (all green)

- test-m9d: 30 checks, 0 failures; digest `0dc73ce14af499ed` identical
  across APUS_THREADS=1/4/8 (UBSan: 30 checks, 0 failures, digest
  `b3e565cfcfa28af8` T=1/4-identical; it differs from the O2 digest via
  the pre-existing -O1/-O2 clang FP-contraction difference, same as the
  m6c/m9c model-level digests). f32/wo_a BLAS vs FP64 truth in the
  reorder class (worst 1.6e-07, bound 2e-5);
  dispatch boundary bitwise (M=255 scalar/NEON, M=256 BLAS); determinism;
  model-level s=300 forward (>cutoff: engages every new path) digest
  thread-count independent.
- m2 (5176+79), m3 (15503), m4a (143+111), m4c (1217), m5 (19): 0
  failures, tolerances untouched.
- m6a (85+5), m6b (47+7+recall checker): 0 failures.
- m6c: 514 checks, digest `8bc96af4504a5161` — byte-identical to the
  M9a/M9b/M9c value; T=1/4/8 diffed.
- m7a server: 34/34. m7b: 117+22 checks, 0 failures, Metal server 34/34
  (GPU==CPU contract preserved — zero Metal changes; all m7b shapes stay
  below the cutoff).
- m8: 41 checks, T=1/4/8 diffed. m9a: 164 checks, digest
  `5f7755151c61a979` unchanged. m9b: 25 checks, digest
  `b398ab7989152252` unchanged. m9c: 39 checks, digest
  `152005391be3b32b` unchanged.
- UBSan: m2, m3, m4a, m4c, m5, m6a, m6b, m6c, m8, m9a, m9b, m9c, m9d,
  m7b — all exit 0.
- Real-model 24-token smoke (production bin/apus): IDENTICAL tokens
  (`We need to respond to the user's query: "The capital of France is".
  This is a straightforward factual question.`) — expected by
  construction (the 9-token prompt keeps every GEMM at M<=54 and decode
  at M=1, all on the pinned paths) and verified.
- Real-model 615-token prompt: 16 greedy tokens IDENTICAL pre/post
  (reorder-class logits delta does not flip argmax on real text).

## Honest notes

- The 615-token real-prompt run (1.58x) is a fairer estimate of the
  end-to-end win than the 512-filler run (2.21x): real text routes to
  more unique experts per layer, so a larger share of its wall is the
  (untouched) expert I/O + medium-M expert GEMMs.
- The filler-prompt 24-token smoke can flip tokens under the reorder
  (observed: the degenerate 512x-id-100 prompt's second greedy token
  changed 100 -> 201 between builds — a near-tie among garbage logits on
  an out-of-distribution prompt). On real text all 16 verified tokens
  were identical. The accepted reorder tolerance class covers this; every
  pinned gate (which never sees M>=256) is bitwise-unchanged.
- Machine load during the A/B batch was high (~9-12); the interleaved
  protocol keeps ratios honest but absolute walls carry +-15% noise
  (documented since m6c/m9a).
- Decode is deliberately untouched: every M=1 path is bit-identical, the
  smoke tokens are identical, and the decode-24 median moved within load
  noise (56.33 -> 57.22 s, +-15% window).
- tests/m6b/tmp/recall_*.txt artifacts regenerate with different sampled
  tokens between O2 and UBSan builds (pre-existing clang FP-contraction
  difference between -O1 and -O2, verified by rebuilding the PRE-M9d
  source: identical tokens to the M9d build — not an M9d regression; the
  m6b checker validates recall accounting, not token identity).
