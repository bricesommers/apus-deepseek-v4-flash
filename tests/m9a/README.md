# tests/m9a — fp8/fp4 NEON GEMM inner-loop rework (ILP + exact F16/FMLAL conversion)

Milestone M9a: speed up the decode hot paths `apus_fp8_gemm_neon_rows` /
`apus_fp4_gemm_neon_rows` (~30% of thread time in the m6c profile) **without
dropping output quality**. FP32 accumulation-order change only — the
project's accepted scalar-vs-NEON tolerance class (tests/m3, tests/m4a) —
with the accumulation order now FIXED, documented, and mirrored bit-for-bit
by the Metal fp8 shaders.

```
make test-m9a    # 164 checks; digest diffed across APUS_THREADS=1/4/8
make ubsan-m9a   # same under -fsanitize=undefined
make bench-m9a   # microbenchmark at the real model shapes (single thread)
```

## What the profile actually said (premise correction)

The m6c README predicted the loops were "FMA-latency-bound (one vector
accumulator, ~1 elem/cycle vs ~8 possible with independent accumulators)".
Measured on this machine (M1 Pro): **wrong for fp8, moot for fp4** — an
A/B microbench (tests/m9a scratch experiment, L2-resident shapes) showed

| fragment | elem/ns |
|---|---|
| FMA-only dot (4 accumulators, f32×f32) | 13.8 |
| E4M3→FP32 expand alone (old two-path integer cvt) | 3.05 |
| full fp8 inner loop (old or 4-accumulator) | 3.03 |

The old E4M3 expansion (~16 ops per 4 lanes: two value constructions +
compare/select) was ~100% of the loop time; the FMAs were entirely hidden
behind it. Independent accumulators **alone** changed nothing (measured:
11.18 ms vs 11.26 ms on wq_b). So M9a did both:

1. **fp8: exact FP16-path E4M3 expansion** (`apus_e4m3_cvt8_neon`,
   c/fp8.h). Every E4M3 value is exactly representable in FP16; the bit
   placement `(c&0x80)<<8 | (c&0x7F)<<7` interpreted as f16 is exactly
   2⁻⁸ × the value for BOTH normals and subnormals (f16 subnormal
   m·2⁻¹⁷ × 256 = m·2⁻⁹), so `f32(h16) * 256.0f` reproduces all 256 codes
   EXACTLY — proven bitwise against the scalar dequant on all 256 codes in
   this test. ~31 ops per 16 elements vs ~67.
2. **fp8/fp4: four independent vector accumulators per row** (the planned
   ILP reorder) with a FIXED combine order — needed to keep the FMAs hidden
   once the expansion got cheap.
3. **fp4: FMLAL inner loop** (`apus_fp4_dot32_f16_neon`, c/fp4.h). FP4
   codes×2 (integers ≤ 12) and E4M3 acts are exact in FP16 (acts dequantize
   to f16 NORMALS — no subnormals anywhere in this path), the products are
   exact (≤ 9 significant bits), and Armv8.2 `FMLAL` adds each exact product
   into the FP32 lanes with the same single rounding as `vfmaq_f32` —
   **bitwise identical** to the f32 anchor `apus_fp4_dot32_neon` (asserted
   on 4096 random blocks here). FP4 codes expand via a 16-entry f16
   high-byte table (`vqtbl1q` + widening shift). Activations are dequantized
   straight to f16 in the same scratch bytes (2K of each row's 4K).

## The exact accumulation-order delta

Only the summation order inside dots changed. Scale application is
untouched: fp4 keeps `(dot·sa)·sb` per 32-block in two rounded steps (the
exact 0.5 folded into sa), fp8 keeps `dot·(sa·sb)` product-first then the
(contracted fmadd) fold; act-quant and BF16-rounding boundaries untouched;
the scalar reference kernels are untouched (semantic anchor).

- **fp8 block dot** (was: one acc, 4-FMA chain per 16-chunk, `vaddvq`
  reduce): `acc[j]` lane `l` accumulates elements `4j+l` of every
  16-element chunk in chunk order (j=0..3); combined lane-wise
  `((a0+a1)+(a2+a3))`, horizontally `((s0+s1)+(s2+s3))` (explicit, no longer
  the compiler's `vaddvq` lowering); a partial trailing chunk (< 16) is
  still added scalar after the reduction.
- **fp4 block dot** (was: one acc, 8-FMA chain): `acc[j]` lane `l`
  accumulates elements `4j+l` then `16+4j+l`; combined lane-wise
  `((a0+a1)+(a2+a3))` BEFORE the scale steps; final horizontal sum fixed as
  `((t0+t1)+(t2+t3))`.
- The GEMM row bodies are factored into single shared functions
  (`apus_fp8_gemm_rows_neon`, `apus_fp4_gemm_rows_neon`) used by both the
  plain and the threaded kernels — mt == single-thread **by construction**;
  the fp8 `mc==4` fast path and the per-row fallback compute the identical
  per-row sequence, so rows are M-independent bitwise (asserted M=1..5).
- **Metal in lockstep** (c/backend_metal.mm): `fp8_dot` mirrors the CPU
  canonical order exactly (four float4 accumulators, same combine, same
  tail); `fp8_fold` now uses `fma()` for the block fold, matching what the
  CPU's `total += dot * sc` compiles to (clang FP contraction, verified in
  the disassembly). Result: **GPU raw fp8 GEMM outputs are BITWISE equal to
  the CPU NEON kernel** (was: ~1.87e-07/esc, ~40–50% bitwise) — asserted in
  the m7b battery (test_kernels.c gained a `nbit_neon == 0` CHECK; 117
  checks). CPU-vs-Metal model logits dropped from rel 2.2e-2 to 3.2e-7;
  the m7b prefill argmax gate went from 8/200 excused near-ties to
  **0/200 IDENTICAL**.

Measured error vs FP64 truth (this test, esc metric): fp8 ≤ 4.4e-8,
fp4 ≤ 3.9e-9 — same 1e-7..1e-8 reorder class as before; m3 (15503 checks)
and m4a goldens pass un-loosened.

## Measured speedups (microbench, single thread, best-of-3)

Machine: MacBook Pro M1 Pro 32 GB, shared desktop (load avg ~4–6 during
both runs; before/after measured back-to-back in the same window from two
binaries: pre-M9a kernels vs M9a kernels).

| kernel | shape (O×K) | M | before | after | speedup |
|---|---|---|---|---|---|
| fp8 gemv | 32768×1024 (wq_b) | 1 | 11.24 ms (3.0 GB/s) | 6.11 ms (5.5 GB/s) | **1.84×** |
| fp8 gemm | 32768×1024 (wq_b) | 4 | 14.91 ms (18.0 GF/s) | 11.69 ms (23.0 GF/s) | 1.28× |
| fp8 gemv | 1024×4096 (wq_a) | 1 | 1.40 ms | 0.777 ms | 1.81× |
| fp8 gemv | 2048×4096 (shared w1/w3) | 1 | 2.79 ms | 1.54 ms | 1.81× |
| fp8 gemv | 4096×2048 (shared w2) | 1 | 2.88 ms | 1.56 ms | 1.85× |
| fp4 gemv | 2048×4096 (expert w1/w3) | 1 | 920 µs (4.9 GB/s) | 572 µs (7.8 GB/s) | **1.61×** |
| fp4 gemm | 2048×4096 | 4 | 1948 µs (34.5 GF/s) | 1585 µs (42.3 GF/s) | 1.23× |
| fp4 gemv | 4096×2048 (expert w2) | 1 | 906 µs | 576 µs | 1.57× |
| fp4 gemm | 4096×2048 | 4 | 1925 µs (34.9 GF/s) | 1603 µs (41.9 GF/s) | 1.20× |

M=1 (decode) is the shape that matters; M=4 (prefill) gains less — the
16-accumulator fp8 block spills registers, and the weight streams start
saturating DRAM.

## Real-model gate + timing

`./bin/apus run --model weights/apus --tiered --prompt "The capital of
France is" --max-tokens 24 --seed 1 --temp 0`, median of 3, same machine
conditions:

| | decode 24 tok | wall | tokens |
|---|---|---|---|
| before | 54.16 s (0.44 tok/s) | 71.9 s | `We need to respond to the user's query: "The capital of France is". This is a straightforward factual question.` |
| after | 50.83 s median (0.47 tok/s) | ~61 s | IDENTICAL 24/24 tokens in all 3 runs |

After-runs: 50.83 / 54.94 / 31.66 s (machine quieter than during the
before-run — the 31.66 s run hit 0.76 tok/s; conditions dominate the
wall number, so treat the microbench table above as the rigorous
before/after and the token identity as the quality gate).

## Digest / gate inventory (before → after)

- m3 test_fp4: 15503 checks, 0 failures (both) — max err/esc 2.53e-6 ≤ 2e-5.
- m4a: 143 + 111 checks, 0 failures (both).
- m4c: 1217 checks, 0 failures (both).
- m5: 19 checks, 0 failures (both); near-tie policies unchanged.
- m6a: 85 + 5 checks, 0 failures; m6b: 47 + 7 checks + recall, 0 failures
  (cross-config bitwise preserved — same kernel in every config).
- **m6c digest**: `1a9182e7be590531` → `8bc96af4504a5161` (514 checks,
  thread-count-independent in both; the fp8/fp4 kernel outputs changed by
  the reorder, everything else in the digest is unchanged).
- m7a server suite: 34/34 (both).
- **m7b**: 117 checks (was 98; +19 bitwise GPU==CPU assertions). Raw fp8
  GEMM gpu-vs-cpu 1.87e-07/esc → **0.0 (bitwise)**; full linear 24580/24580
  bitwise (both); model gate 0/24 greedy, prefill argmax 8/200 excused →
  **0/200 IDENTICAL**.
- m8: 41 checks, 0 failures; all spec/non-spec streams BITWISE (both).
- UBSan: m3, m4a, m4c, m5, m6a, m6b, m6c, m7b, m8, m9a all green; m9a UBSan
  digest == O2 digest (`5f7755151c61a979`).

## Makefile fix (pre-existing latent bug)

`M6C_DEPS`/`M8_DEPS` used `:=` expansion of `$(M5_DEPS)` before it was
defined, so the m6c/m8 binaries silently did NOT track c/fp4.h / c/fp8.h
(their "after" runs would have used stale kernels). Moved below the
`M5_DEPS` definition; verified the binaries now rebuild on kernel edits.

## Honest notes

- The win comes from the conversion rework, not the ILP reorder alone —
  the m6c "FMA-latency-bound" hypothesis was incorrect for fp8 (measured).
  ILP is still necessary: with the cheap expand, a single accumulator would
  re-become the limit.
- fp4's old expand was already LUT-based and cheap; its 1.59× comes from
  FMLAL (skipping 8 f32 converts + fusing the FMAs), not from accumulators.
- fp8 M=4 GEMM only gained 1.26×: 16 live accumulators spill; a 2-rows×4-acc
  or 4-rows×2-acc variant might do better but was left out to keep the
  canonical order single and simple.
- wq_b (32 MB stream/call) is approaching DRAM limits: 5.6 GB/s single
  thread of ~10+ GB/s achievable on this part under the current load.
- F16 conversion/FMLAL rely on Apple-Silicon FP16 SIMD (present since A11);
  the engine already requires NEON and targets M1+.
