# tests/m3 — MXFP4 kernel hard gate (c/fp4.h)

Milestone M3: MXFP4 (E2M1 weights + UE8M0 per-32 scales) dequant and
GEMV/GEMM kernels, scalar reference + NEON, C11 + libc + arm_neon.h only.

## Numerics path chosen: FP8-E4M3 activation quantization (normative)

The reference model **quantizes activations to FP8-E4M3 before every FP4
matmul** — this is not optional. Evidence:

- `reference/inference/model.py:109-115` (`linear()`): for
  `float4_e2m1fn_x2` weights, `x, s = act_quant(x, block_size, scale_fmt,
  scale_dtype)` then `fp4_gemm(x, s, weight, weight.scale, scale_dtype)`.
- `reference/inference/model.py:17-21,40-42,777-778` + `config.json`:
  `block_size=128`, `scale_fmt="ue8m0"` (generate.py defaults resolve to
  ue8m0 regardless of the `scale_fmt` CLI flag), scales are power-of-2.
- `reference/inference/kernel.py:441-515` (`fp4_gemm_kernel`): per 32-wide
  K block, FP4 codes are upcast FP4→FP32→FP8 (lossless), an FP8×FP8 GEMM
  accumulates the block dot in FP32, and the total is corrected per block:
  `accum += (dot * scale_a[kb//4]) * scale_b[kb]`.

`c/fp4.h` reproduces this exactly in FP32: E4M3 codes and E2M1 LUT values
are exactly representable in FP32, so the FP32 per-block dot is the same
computation (up to summation order, which the reference leaves
unspecified). No requantization, no approximations.

A second, **non-normative** path (`apus_fp4_gemv_f32_*`: raw FP32 activation
× fully dequantized weight, single accumulator) is kept as a diagnostic.
It does NOT match the reference model bit-for-bit (it skips the lossy FP8
activation quantization) and must not be used for inference output.

## What is tested

1. **Dequant exhaustive** — all 256 packed byte values at every byte
   position × scale bytes {0,1,2,63,126,127,128,129,190,254} plus the 255
   (→inf) case: scalar vs NEON **bitwise identical**, both exact vs the LUT
   formula `LUT[nibble] * 2^(scale-127)` evaluated in double. Block-boundary
   (elements 31/32) covered. Byte 0 = 2^-127 (FP32 subnormal) and byte 255
   = inf are exact per the M1-pinned numpy semantics.
2. **Activation quant golden** — `gen_golden.py` ports
   `act_quant(round_scale=True)` to numpy (brute-force RNE table lookup,
   independent of the C bit-twiddler); C codes and scales compare
   **bitwise**: 0 mismatches over 3×256 values incl. an all-zero row.
3. **Golden GEMM vs reference semantics** — `gen_golden.py` ports
   `fp4_quant_kernel` (amax floor 6·2^-126, scale = 2^ceil(log2(amax/6)),
   clamp ±6, RNE E2M1, low nibble = even K) and evaluates the fp4_gemm
   accumulation in float64 (M=3, O=64, K=256; fixtures include all-zero,
   all-negative, ±6-saturation and beyond-±6 blocks).
   Measured: **max_abs = 9.77e-4, max_rel = 6.84e-08** (GEMM),
   **max_rel = 4.04e-08** (GEMV), scalar-vs-NEON **1 ulp**. Assert < 1e-5.
4. **Shape sweep** — O×K ∈ {1×32, 3×32, 5×96, 17×160, 64×256, 128×512,
   2048×4096, 4096×2048} × M ∈ {1,2,4,7}, random packed weights/scales +
   zero blocks, vs in-test FP64 ground truth. Tolerances are fractions of
   the per-output error scale `esc = Σ_kb |dot·sa·sb|` (FP32 accumulation
   error scales with the sum of absolute contributions, not with |out|,
   which cancellation can drive to ~0):
   - normative path: max err/esc = **2.53e-06** (assert < 2e-5;
     linear bound at K=4096 is 128·2⁻²⁴ ≈ 7.6e-6)
   - scalar vs NEON: max diff/esc = **2.53e-06** (assert < 2e-5); **not
     bitwise** — FP32 accumulation-order differences only; ≤ 10 ulp on
     well-conditioned outputs at the real shapes, ≤ 32 ulp across the sweep
   - f32 diagnostic path (single accumulator over K): max err/esc =
     **4.71e-07** (assert < 1e-3; bound 4096·2⁻²⁴ ≈ 2.4e-4)
5. **Edge cases** — K=32 minimum block, all-zero weight blocks (with huge
   scale byte → still exactly 0), all-negative rows, ±6 saturation codes,
   zero activations (amax floor 1e-4 path).

`make test-m3` (15503 checks, 0 failures), `make ubsan-m3` clean
(Apple's ASan runtime is broken on this machine — UBSan only, same as M2).

## Benchmark (this MacBook Pro M1, single thread, `make bench-m3`)

| shape          | path   | time     | GB/s | GFLOP/s |
|----------------|--------|----------|------|---------|
| O=2048 K=4096  | scalar | 3764 µs  | 1.18 | 4.46    |
| O=2048 K=4096  | NEON   |  959 µs  | 4.65 | 17.50   |
| O=4096 K=2048  | scalar | 3726 µs  | 1.20 | 4.50    |
| O=4096 K=2048  | NEON   |  955 µs  | 4.67 | 17.57   |

GB/s counts packed weight + scale bytes streamed. NEON is compute-bound
(dequant + FMA), far from the ~60 GB/s M1 memory ceiling; optimization
(wider unrolling, SDOT on integerized codes, multithreading) is deferred —
exactness gate first.

## Surprises / notes for the M4 forward-pass implementer

- **Round activations to BF16 before `apus_fp4_act_quant_*`** to be
  bit-faithful: the reference `act_quant` input dtype is BF16
  (`kernel.py:41`), so amax and the divided values see BF16-rounded inputs.
  `apus_bf16_round()` is provided for this.
- **The FP8 activation step is lossy and mandatory.** Skipping it (f32
  diagnostic path) changes results beyond FP32 rounding — it is a different
  algorithm, not a precision tweak.
- **K % 128**: the reference asserts K % 128 == 0 for act quant. The C
  kernel also accepts K % 32 == 0 with a partial trailing 128-block (amax
  over the remainder) — an extension that cannot be validated against the
  reference; all real shapes (2048, 4096) are multiples of 128.
- **Zero weight blocks**: the amax floor (6·2⁻¹²⁶) yields scale byte 1
  (2⁻¹²⁶) in the numpy port — the f32 multiply `amax * (1/6)` lands exactly
  on 2⁻¹²⁶ (mantissa zero, no ceil bump). Codes are all 0, so the block
  contributes exactly 0 regardless; but fixture generators must not assume
  byte 0 for "zero scale". UE8M0 byte 0 means 2⁻¹²⁷ (FP32 subnormal), never
  emitted by the quant rule; byte 255 overflows FP32 to +inf (matches the
  M1 numpy semantics `np.exp2(128).astype(f32)`).
- **Accumulation order**: the reference applies scales as
  `(dot * scale_a) * scale_b` with an FP32 rounding at each step and clears
  the block accumulator per 32-block. The kernels mirror this (two plain
  multiplies, no FMA fusion at the scale step); the in-block dot uses FMA,
  which is at least as accurate as the tensor core's unspecified order.
- **act scale for block kb is `scales_a[kb // 4]`** (128/32); weight scale
  is `scales_b[o, kb]`. Getting this indexing wrong still "runs" — the
  golden test exists to catch it.
- Scales are stored FP32 in the API (`apus_fp4_act_quant_*` outputs); when
  `scale_dtype=fp8` in the reference they are the same power-of-2 values,
  so an FP32 store is lossless. Weight scales stay as UE8M0 bytes
  (`apus_ue8m0_f32` converts exactly).

## Files

- `c/fp4.h` — kernels (`APUS_FP4_IMPLEMENTATION` single-TU pattern)
- `tests/m3/gen_golden.py` — numpy port of the reference quant + gemm
- `tests/m3/test_fp4.c` — hard-gate tests
- `tests/m3/bench_fp4.c` — GEMV benchmark
- `tests/m3/golden/` — generated fixtures (via `make golden-m3` / `test-m3`)
