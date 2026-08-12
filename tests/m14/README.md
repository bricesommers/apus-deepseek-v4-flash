# tests/m14 — ARM NEON staged-product interleaved dots (M14)

Milestone M14: speed up the two remaining pinned scalar-order dot paths on
ARM **without changing one bit of output** — the NEON twin of the M12a-2
x86 pattern (`apus_dot4_f32_x86`, c/x86.h).

## The pattern (why it is bitwise, not a reorder class)

The pinned anchor is `apus_dot_f32_scalar` (c/attn.h): one IEEE multiply
per element, then a strictly sequential chain of scalar adds. Its cost is
not the multiplies (clang already vectorizes those) but the 512–4096-long
`fadd` dependency chain (~3–4 cycle latency each, nothing overlaps).

`apus_dot4_f32_neon` (c/attn.h) computes **four independent dots at once**:
products staged 4-wide with `vmulq_f32` (the exact same IEEE products as
the scalar `fmul`), then added as four interleaved scalar chains, each in
strict k order. Each dot's rounding sequence is identical to the scalar
anchor — only the latency is hidden. Accumulators/pointers are named
variables (indexed arrays spill in the inner loop — the Rosetta lesson,
tests/m12/README.md). This is the same construction M12a-2 proved bitwise
on x86; M14 ports it to NEON and applies it on ARM.

## What changed (c/attn.h only)

1. `apus_dot4_f32_neon` — the helper, `__ARM_NEON`-guarded.
2. `apus_sparse_attn_head` — the q·k loop runs four dots per dot4 group
   (kv_a/kv_b pointer selection identical to the existing x86 branch);
   `sc`/`mx` updates stay in jj order; scalar tail for n%4. This was the
   parked "sparse_attn reorder" item — resolved **bitwise**, so no
   numerics renegotiation was needed. P·V / softmax / sink untouched.
3. `apus_f32_linear_rows_neon` + dispatch from `apus_f32_linear_rows_scalar`
   / `_bf16` — four output rows per dot4 group (the ARM twin of
   `apus_f32_linear_rows_avx2`). Hits the pinned decode-time dense linears
   (compressor wkv/wgate, indexer weights_proj, router gate — M=1 never
   reaches the M≥256 BLAS cutoff). BLAS cutoff logic untouched.

x86 is untouched (it already had the pattern). Metal is untouched
(sparse_attn runs on CPU in both configurations; the m7b CPU==Metal gate
is preserved by construction). dspark.h shares `apus_sparse_attn` — the
spec-equivalence gates are preserved by construction.

## Gates (all bitwise — there is no tolerance to hide behind)

- `test-m14` (509 checks):
  - dot4 vs scalar anchor **bitwise**: 460 dot checks over a length sweep
    (0..17, 127..129, 511..513, 1024, 4095..4097 — every tail) × 5 value
    classes (uniform, mixed magnitude ±2^±40, denormals, cancellation,
    signed zeros). 0 mismatches.
  - `apus_sparse_attn` vs a reference copy of the pre-M14 per-head body
    (scalar q·k dot; platform P·V idiom unchanged): **bitwise** over
    s/h/d shapes (decode 1×64×512 full-640, small prefills incl. odd d=21)
    × 9 index patterns (n=0, n∈{1,3,5,6,7} tails, kv_a/kv_b boundary
    straddle + duplicates, full descending, random with -1 holes).
    0 mismatched elements.
  - `apus_f32_linear` / `apus_bf16_linear` (pool workers) vs a per-row
    scalar-dot reference: **bitwise** over an 11-shape M/K/O sweep
    (M ≤ 250 — below the BLAS cutoff — incl. M=1 decode shapes, K tails,
    O group tails, both round_out modes). 0 mismatched elements.
  - Digest diffed across APUS_THREADS=1/4/8: identical.
- `ubsan-m14`: clean, digest identical.
- On x86 the NEON-specific checks skip themselves; the sparse_attn level
  still runs (re-proves dot4_x86 + saxpy vs the scalar reference), so the
  suite is portable and runs in the CI battery on both platforms.

## Measured (M1 Pro, single thread, `make bench-m14`)

| shape | scalar | dot4 | speedup |
|---|---|---|---|
| q·k dots d=512, n=640 (one decode head) | 310.9 µs | 78.3 µs | **3.97×** |
| linear K=4096 O=2048 (decode compressor) | 7990 µs | 2198 µs | **3.64×** |
| linear K=4096 O=256 (decode gate) | 1003 µs | 257 µs | **3.90×** |
| linear K=4096 O=64 (decode weights_proj) | 251 µs | 65 µs | **3.89×** |

(The bench entry points are `noinline` — identical-input reps were being
CSE'd away, the same trap the m9a bench avoids with function pointers.)

Real-model walls and the 24-token token-identity proof are recorded in
docs/STATUS.md (M14 entry).

## Honest scope note

sparse_attn was ~12–13% of prefill in the m9d/m9e profiles, so a ~4× dot
win is a single-digit-% prefill-wall improvement; the decode-linear win is
real but decode remains RAM/disk-dominated on the dev machine. The
*reordered* NEON FMA dot (the originally parked idea) would be maybe
2–2.5× faster again on the dot portion but breaks bitwiseness — not done,
not needed for this result.
