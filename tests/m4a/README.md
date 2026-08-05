# tests/m4a — FP8 dense kernel (c/fp8.h) + mHC (c/mhc.h) hard gate

Milestone M4a: FP8-E4M3 dense weights with blockwise 128×128 UE8M0 scales
(GEMV/GEMM), and the mHC residual-stream building blocks (per-token
pre/post/comb map generation, Sinkhorn-20, apply step, hc_head collapse).
Scalar reference + NEON, C11 + libc + arm_neon.h only.

## FP8: numerics path (normative)

`reference/inference/model.py:116-118` routes every FP8 linear through
`act_quant` + `fp8_gemm`; `kernel.py` is the normative semantics:

- **Activation quant** (`kernel.py:40-125`, `scale_fmt="ue8m0"`): per-128-along-K
  amax floored at 1e-4, scale = 2^ceil(log2(amax/448)) via the FP32 bit
  trick, clamp ±448, RNE to E4M3. This is *the same rule* as the FP4 path's
  activations, so `c/fp8.h` reuses the fp4.h helpers
  (`apus_fp4_act_quant_scalar`, `apus_ue8m0_f32`, `apus_e4m3_*`,
  `apus_bf16_round`) instead of duplicating them.
- **GEMM** (`kernel.py:203-254`): per 128-wide K block an FP8×FP8 dot in
  FP32, then `total += dot * (scale_a * scale_b)` — **the scale product is
  formed first** (kernel.py:242-249), one FP32 rounding, unlike the FP4
  path's `(dot*sa)*sb`. Weight scales are UE8M0 per 128×128 tile, indexed
  `[o/128, kb]`. The kernels mirror this step for step (plain multiplies,
  no FMA at the scale steps); the in-block dot uses FMA, which is at least
  as accurate as the tensor core's unspecified order.
- E4M3 codes are exactly representable in FP32, so the FP32 block dot is
  the same computation (up to summation order). No requantization anywhere.
- Output is FP32; the reference casts to BF16 (`out_dtype=BF16`). The
  caller must BF16-round at the same points to stay bit-faithful.

## mHC: numerics path (normative)

`model.py:673-686` (hc_pre/hc_post), `model.py:728-735` (hc_head),
`kernel.py:371-438` (hc_split_sinkhorn_kernel). FP32 throughout.

- `rsqrt = rsqrt(mean(x²) + norm_eps)` over the flattened n·d state;
  `mixes = (hc_fn @ x) * rsqrt` — **rsqrt is applied AFTER the matmul**
  (model.py:678). Mathematically identical, FP32-rounding different; the
  kernels follow the reference order.
- Gates: `pre = sigmoid(m·s0 + b) + hc_eps`, `post = 2·sigmoid(m·s1 + b)`,
  `comb_logits = m·s2 + b`; mix layout `[pre(n) | post(n) | comb(n²)]`.
- **Sinkhorn-20, exact order and eps placement** (kernel.py:398-423):
  1. row softmax (max-subtracted), then **per-element + eps**;
  2. column normalize with **eps in the denominator** (`/(colsum+eps)`);
  3. 19 × (row normalize `/(rowsum+eps)`; column normalize `/(colsum+eps)`).

  Total 1 row softmax + 20 column + 19 row normalizations; the last op is
  a column normalization, so column sums ≈ 1 (up to the eps effect ~1e-6)
  while row sums deviate more (golden range [0.959, 1.015]) — doubly
  stochastic only in the Sinkhorn sense, do not "fix" this.
- Apply: `y[j] = post[j]·f + Σ_k comb[k,j]·res[k]` (comb indexed
  [residual k][output j], model.py:685 — the pre-M5 version of c/mhc.h and
  of these goldens used the transposed convention consistently; both were
  corrected at M5 and the goldens regenerated); collapse:
  `y = Σ_j pre[j]·X[j]`; hc_head is pre-only (scalar scale, no Sinkhorn).
- The step functions (`apus_mhc_row_softmax_eps`, `apus_mhc_norm_rows_eps`,
  `apus_mhc_norm_cols_eps`) are exposed individually so the test verifies
  the reference iteration-for-iteration; `apus_mhc_sinkhorn` is the driver
  and is bitwise-identical to the stepwise sequence.

## What is tested / measured (this MacBook Pro M1, `make test-m4a`)

### FP8 (test_fp8: 143 checks, 0 failures)

1. **E4M3 exhaustive** — all 256 codes: `quant(dequant(c))` round trip
   (NaN codes 0x7F/0xFF decode as ±480 per fp4.h's LUT and requantize to
   ±448 — documented; they never occur in the checkpoint); NEON 16-code
   expand **bitwise** equal to scalar dequant at every position.
2. **Activation quant golden** — bitwise codes + scales vs the numpy port
   (incl. all-zero row → amax floor 1e-4 path): **0 mismatches**.
3. **Golden GEMM/GEMV** vs the reference fp8_gemm semantics evaluated in
   float64 (M=8, O=320, K=384; fixtures include zero tiles, ±448 saturation,
   min-subnormal codes, weight scale bytes 0/1/200/254): max err/esc =
   **1.56e-07** (GEMM), **6.09e-08** (GEMV), scalar-vs-NEON **1.47e-07**,
   ≤ **3 ulp**. Assert < 2e-5.
4. **Shape sweep** — real dense shapes {1024×4096, 32768×1024, 512×4096,
   8192×4096, 4096×8192} (GEMV + M=2 GEMM) plus odd/partial shapes
   {1×128, 3×128, 5×384, 17×448, 130×256, 64×384, 200×512} × M ∈ {1,2,5},
   vs in-test FP64 truth. Tolerances are fractions of the per-output error
   scale `esc = Σ_kb (Σ_i |a_i·w_i|)·sc` — **sum of absolute products, not
   |dot·sc|**, because intra-block cancellation can drive |dot| to ~0 while
   FP32 rounding scales with the absolute terms (an earlier |dot|-based esc
   showed spurious 1e-3 "errors" that were pure FP32 rounding):
   max err/esc = **1.53e-07** (assert < 2e-5), scalar-vs-NEON = **1.28e-07**,
   ≤ **5 ulp** on well-conditioned outputs (|out| ≥ esc/4).
5. **Edge cases** — K=128 single block, zero weight block with scale byte
   254 (stays exactly 0), scale byte 0 (2⁻¹²⁷ FP32 subnormal, exact),
   scale byte 255 (2¹²⁸ → inf, scalar==NEON), ±448 saturation, zero
   activations, weight-scale tile boundary rows 127/128 (O=129, exact).

### mHC (test_mhc: 111 checks, 0 failures)

1. **Golden maps** vs the float64 numpy port (T=6 tokens incl. zero /
   ×100 / ×1e-3 states, d=64, sigmoid-saturating bases ±30): max abs err
   mixes = **1.02e-06**, pre = **1.27e-07**, post = **4.97e-07**,
   comb = **1.59e-07**, ypre = **7.0e-05**, ypost = **3.07e-05**,
   yhead = **3.29e-05** (y magnitudes are O(10–100) for the ×100 state
   token; asserts 1e-4/1e-5/1e-3 respectively).
   Scalar-vs-NEON: comb/ypost max abs **4.58e-05**, comb ≤ **16 ulp**
   (NEON fma vs scalar mul+add in the vector ops; mixes dot lane order).
2. **Sinkhorn iteration-for-iteration** — all 40 normalization stages × 8
   logit sets (uniform, all-50, dominant-80, all-−60): max per-stage abs
   err vs float64 = **1.87e-07** (assert < 1e-4). Full driver bitwise-equal
   to the stepwise sequence. Stochasticity: max |colsum−1| = **1.09e-06**
   (the eps-in-denominator effect; assert < 1e-4), row/col sums vs golden
   **1.55e-07**.
3. **Random shapes** — d=4096 (real hidden) + d=64 n=4, d=48 n=3, d=8 n=2:
   vector ops (collapse/apply) vs in-test float64 from the C pre/post/comb:
   max **1.24e-06**.
4. **Edge cases** — zero state (rsqrt(0+1e-6f) = 999.99994, mixes exactly
   0, comb uniform 1/4), sigmoid saturation (pre ≡ 1+eps / ≈eps), huge
   state 1e30 (sum-of-squares overflows to inf → rsqrt 0, same as the
   reference would produce; no NaN), dominant comb logit (max subtraction),
   n=3 generic Sinkhorn doubly stochastic, d≢0 (mod 4) tail parity.

`make test-m4a` (143 + 111 checks, 0 failures), `make ubsan-m4a` clean
(Apple's ASan runtime is broken on this machine — UBSan only, same as
m2/m3).

## Surprises / notes for the M4c integration implementer

- **Scale application order differs between the FP4 and FP8 GEMMs.** FP8:
  `total += dot * (scale_a * scale_b)` (product first, kernel.py:242-249).
  FP4: `total += (dot * scale_a) * scale_b`. Both kernels mirror their
  reference exactly — don't unify them into a shared helper.
- **rsqrt placement in hc_pre/hc_head**: `mixes = (hc_fn @ x) * rsqrt`,
  applied AFTER the matmul (model.py:678,732), not `x·rsqrt` before it.
- **Sinkhorn eps placement is asymmetric on purpose**: per-element `+eps`
  only in the initial row softmax; every subsequent normalization has eps
  in the denominator. 20 iterations means 20 column normalizations and the
  last op is a column normalization — comb is column-stochastic to ~1e-6,
  row-stochastic only to ~5e-2. Verified stage-by-stage against the port.
- **BF16 boundaries**: round activations with `apus_bf16_round()` before
  `apus_fp8_act_quant_*` (reference act_quant input is BF16). FP8 GEMM
  output and mHC collapse/apply outputs must be BF16-rounded by the caller
  at the points where the reference does `.to(dtype)` / `.type_as(x)`:
  the fp8_gemm output (kernel out_dtype=BF16), the hc_pre collapse output
  (model.py:681), and the hc_post expand output (model.py:686). The mHC
  maps themselves (pre/post/comb) are FP32 end-to-end — do NOT round them.
- **Partial K blocks** (K % 128 ≠ 0): handled (amax over the remainder,
  partial dot) but cannot be validated against the reference, which asserts
  K % 128 == 0. All real dense shapes have K, O multiples of 128.
- **UE8M0 byte 0** = 2⁻¹²⁷ (FP32 subnormal, handled exactly); byte 255 =
  +inf (matches the M1 numpy semantics). Scale bytes are read from the
  checkpoint as-is; `apus_ue8m0_f32` converts exactly.
- **esc definition for tolerance checks**: use `Σ_kb (Σ|a·w|)·sc`, not
  `Σ|dot·sc|` — intra-block cancellation makes the latter understate FP32
  accumulation error by up to 1000× on random codes.
- mhc.h is self-contained (no fp4.h dependency). fp8.h's implementation
  CALLS fp4.h helpers, so some TU must define APUS_FP4_IMPLEMENTATION.
- The mHC mixes matmul (24 × 16384 per token) is the only O(hidden) mHC
  cost; NEON path already covers it. Sinkhorn itself is 4×4 scalar work.

## Files

- `c/fp8.h` — FP8-E4M3 blockwise 128×128 GEMV/GEMM (`APUS_FP8_IMPLEMENTATION`)
- `c/mhc.h` — mHC maps, Sinkhorn-20, apply, hc_head (`APUS_MHC_IMPLEMENTATION`)
- `tests/m4a/gen_golden.py` — numpy ports of fp8_gemm + hc_split_sinkhorn
- `tests/m4a/test_fp8.c`, `tests/m4a/test_mhc.c` — hard-gate tests
- `tests/m4a/golden/` — generated fixtures (via `make golden-m4a` / `test-m4a`)
