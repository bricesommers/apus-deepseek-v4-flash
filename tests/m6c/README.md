# tests/m6c — decode performance pass (threading, NEON routing, scratch, VM pressure)

Milestone M6c: make real-model decode fast on the M1 **without changing
numerics semantics**. Two hard gates: every existing suite stays green in
its documented tolerance class, and the real-model greedy smoke produces
BITWISE the same token stream as before (verified below). All threading is
fixed-partition + per-row accumulation, so results are thread-count
independent **bitwise** (APUS_THREADS=1/4/8 — asserted by the Makefile
diffing this test's stdout digest).

Run:

```
make test-m6c        # builds, then runs APUS_THREADS=1/4/8 and diffs digests
make ubsan-m6c       # same under -fsanitize=undefined
```

Checks (tests/m6c/test_m6c.c): pool coverage; scratch-arena LIFO/growth
incl. the dead-segment-grow regression (see Bug below); threaded fp8/fp4
GEMM BITWISE == the single-thread NEON kernels; f32/bf16 linears BITWISE
== the pre-M6c scalar order + M-independence; wo_a NEON dot bounded to
≤ 1 bf16 ulp vs the old scalar order; pool dispatch overhead (12–29 µs
avg at T=4/8, target < 50 µs).

## Result (real model, same command as the M6 smoke)

`./bin/apus run --model weights/apus --tiered --prompt "The capital of France is" --max-tokens 8 --seed 1 --temp 0`

| stage | decode (8 tok) | wall | sys | footprint | page reclaims |
|---|---|---|---|---|---|
| pre-M6c (documented M7b smoke) | 109.6 s (13.7 s/tok, 0.073 tok/s) | ~130 s | n/m | ~30 GB | n/m |
| +thread pool, NEON routing, scratch (T=8) | 93.6 s | 143.1 s | 320 s | 30.3 GB | 10.8 M |
| +wo_a NEON dot | 84.6 s | 123.4 s | 303 s | 30.3 GB | — |
| +slab buffer recycling, cache 4 GB | 35.4 s | 61.3 s | 150 s | 21.6 GB | 5.4 M |
| +wo_a BF16 storage, cache 4 GB/pin 1 GB | 21.0 s | 44.1 s | 93 s | 18.2 GB | 3.6 M |
| **final (defaults cache 4096/pin 512), median of 3** | **29.8 s (3.7 s/tok, 0.27 tok/s)** | 53.3 s | 110 s | 17.9 GB | 4.4 M |

(The three final runs: 30.9 / 29.8 / 22.3 s decode — run-to-run variance
on this shared desktop is ±25%; best-case 0.36 tok/s. peak RSS 14.0–14.1 GB
in every final run, ≤ ~14 GB requirement met; footprint 17.9 GB.)

Tokens every run: `We need to respond to the user's` — **identical** to the
pre-M6c smoke. Recall 1628/1920 (84.8%), 0 demand loads in all configs.
24-token run: decode 79.8 s (3.3 s/tok, 0.30 tok/s), wall 99.9 s vs the
pre-M6c 422 s — output `We need to respond to the user's query: "The
capital of France is". This is a straightforward factual question.` —
**identical** to the pre-M6c 24-token smoke.
APUS_THREADS=1/4/8: identical 8 tokens (decode 40.9 / 23.4 / 22.3–30.9 s).

Speedup end-to-end: **3.7× decode** (median) / **4.2× wall** on the
24-token run vs the documented baseline; 0.073 → 0.27–0.30 tok/s.

## What changed (and the numerics contract of each)

1. **Persistent pthread pool + TLS scratch arena (c/pool.h)**. Condvar
   pool, `APUS_THREADS` (default = perf-core count, 8). Row-range
   partition; each output row computed entirely by one lane with the same
   per-row code → bitwise thread-count independent. Scratch = grow-only
   segmented bump allocator, strictly LIFO, replaces per-call malloc/free
   in attn/moe/layer/model/pilot hot paths.
2. **Threaded fp8/fp4 GEMM (c/fp8.h, c/fp4.h `*_gemm_mt`)**. Output-row
   partition over the *same* NEON row body → **bitwise** identical to the
   pre-M6c single-thread kernels (asserted in-test). Act quant/scales/BF16
   boundaries untouched.
3. **f32/bf16 dense linears (c/attn.h)** threaded across rows but kept at
   the **exact scalar sequential-k order** (bitwise vs pre-M6c) — the
   router gate and compressor wkv/wgate feed QAT/top-k paths with no room
   for reorder noise.
4. **wo_a (c/attn.h `apus_woa_rows`)**: all G·ol·s rows in ONE pool
   dispatch; 4-accumulator NEON dot. **REORDER** vs the old scalar sum —
   the one accepted SIMD tolerance-class change in this milestone: FP32
   rel error ~1e-7 vs the per-output error scale, output is BF16-rounded
   (ulp 2⁻⁷), so the rounded result is almost always bitwise identical
   (0 flips in the test shapes; ≤1 ulp bound asserted). M- and
   thread-count-independent per row. m4c's out_h bounds (rel ~4e-3 class)
   absorb it; the real-model tokens are unchanged (verified).
5. **wo_a stored as BF16 bits (c/layer.h)** instead of f32: −2.9 GB
   anonymous memory on the real model (5.8 → 2.9 GB). Widening at use is
   exact, so values are **bitwise** identical to the f32 storage.
   `apus_bf16_bits`/`apus_bf16_f32` helpers added to c/fp4.h. Metal hook
   contract gains `APUS_HOOK_W_BF16` (c/backend_metal.h): the backend
   widens once and caches (c/backend_metal.mm) — the Metal binary's memory
   profile and numerics are unchanged.
6. **LM head GEMV (c/model.h)**: NEON bf16 widen + FP32 accumulate,
   threaded over the 129,280 vocab rows. Same near-bitwise reorder class
   as documented for the M7b Metal head (unrounded FP32 logits).
7. **Sparse attention (c/attn.h `apus_sparse_attn_head`)**: threaded over
   (token, head); NEON only across the independent i lanes, sequential j —
   **bitwise** vs the old scalar order.
8. **Expert-slab buffer recycling (c/cache.h)**: evicted/dropped slab
   payloads go to a free list (cap 64) instead of free(); the I/O workers
   pop them. Kills a 13.4 MB mmap/munmap + zero-fill soft-fault storm per
   expert load. RSS-guard drops still really free() (their purpose is
   memory relief). Pure allocation policy — no numerics.
9. **Default budget retune (c/cache.h)**: APUS_EXPERT_CACHE_MB
   12288→**4096**, APUS_PIN_MB 2048→**512**. See §VM below. No test depends
   on the defaults (m6a/m6b set explicit budgets; invariance is bitwise
   across cache sizes by design).

## Bug found by this milestone's regression (fixed in c/pool.h)

The scratch arena's reuse path never checked that a DEAD segment was big
enough for the new request: a request larger than the reused segment's
capacity bumped the offset past the segment end → **heap overflow**.
Symptoms: m5 "determinism prefill_len200 bitwise" failing at
APUS_THREADS=1, chunk-invariance broken, an O2 segfault at T=1.
Fix: grow dead segments in place (safe under the LIFO contract). The m6c
scratch test now covers the dead-segment-grow sequence.

## Profile (sample(1) on real-model decode, current build)

Top-of-stack over all threads (~49 k samples / 6 s):

| frame | share | note |
|---|---|---|
| `__psynch_cvwait` | ~49% | idle: 4 I/O threads (~84% idle at 0.9–1.4 GB/s of 5.5 GB/s) + pool lanes between the ~30 dispatches/layer |
| `apus_fp8_gemm_neon_rows` | ~17% | dense FP8 linears (wq_b 1024→32768, wo_b, wq_a, wkv, shared expert, indexer) |
| `apus_fp4_gemm_neon_rows` | ~13% | 6 routed experts/layer |
| `apus_woa_rows` | ~10% | was 23% before the NEON+bf16 work |
| `apus_f32_linear_rows_scalar` | ~4.5% | compressor wkv/wgate + router gate (bitwise-pinned) |
| `apus_head_gemv_rows` | ~2% | LM head, once per token |
| `pread` | ~1–2% | expert I/O is NOT the limit |

## Remaining bottlenecks (honest)

- **The fp8/fp4 NEON GEMM inner loops are FMA-latency-bound**: one vector
  accumulator with a 4-FMA chain per 16 codes ≈ 1 elem/cycle vs ~8
  possible with independent accumulators. They are pinned **bitwise** by
  the m7b Metal==CPU kernel gate (24,580/24,580) and the m3/m4a goldens;
  reordering CPU kernel + Metal shader identically is future work worth
  roughly 1.5–2× on ~30% of thread time.
- **~1/3 of pool capacity is idle**: the decode forward is a serial chain
  of ~30 small GEMV dispatches per layer (70–300 µs of work each) with
  serial main-thread glue between them (act quant, SwiGLU expf, top-k,
  norms, expert resolve loop). Batching the 6 experts' w1/w3 GEMVs into
  one dispatch was evaluated and deferred: the fp4 GEMVs are DRAM-bound,
  so batching saves only barrier latency (~26 ms/token), not bandwidth.
- **Page-reclaim churn persists** (~0.5 M minflt/token): the ~9 GB dense
  working set cycles active→inactive between tokens and refaults on every
  touch (~kernel time, no I/O). Further footprint cuts trade directly
  against more expert re-reads; current defaults sit at the knee.
- **Expert I/O**: ~4–5 GB/token of slab reads at ~1 GB/s effective ≈
  15–20% of decode wall — not the limit today, but it becomes visible if
  compute gets another ~2× faster (pilot recall 84.8%, 0 demand loads).
- Prefill is compute-bound the same way (GEMM M>1 paths share the same
  kernels); long-prompt prefill would benefit most from the fp8 ILP work.

## Thread scaling

| APUS_THREADS | 8-token decode | tokens |
|---|---|---|
| 1 | 40.9 s | identical |
| 4 | 23.4 s | identical |
| 8 (default) | 22.3–30.9 s (median 29.8) | identical |

T1→T8 scaling is only ~1.3–1.8×: the GEMVs are DRAM/latency-bound, so
adding lanes past ~4 mostly adds idle (the cvwait share above). The win
from threading is real but secondary to the VM-pressure and wo_a fixes.

In-process: this test's FNV digest of all kernel outputs is identical for
T=1/4/8 (Makefile diff), and m6a/m6b eager-vs-store invariance stays
bitwise.
