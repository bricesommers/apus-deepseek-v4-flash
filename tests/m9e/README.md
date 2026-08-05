# tests/m9e — routed-expert GEMM/dispatch study: kernel falsification + cache-bound dispatch grouping

Milestone M9e attacked the two M9d-documented prefill bottlenecks in
order: (1) routed-expert fp4 GEMMs at M≈6–24 (~20% of the M9d prefill
profile) via a blocked NEON kernel, and (2) dispatch grouping of the
~108 sequential per-expert batches per layer (~53% cvwait). Outcome,
both measured honestly on this machine:

1. **The blocked-kernel premise is FALSIFIED**: the M9a kernel is
   FMLAL-throughput-bound, not weight-pass-bound; every blocked variant
   measured at or below parity. The M9a kernel ships UNCHANGED.
2. **Dispatch grouping is cache-bound**: any pool dispatch that streams
   more than one expert weight slab concurrently loses to the M9d
   one-slab-per-dispatch granularity. The shipped configuration keeps
   per-matrix per-expert GEMM dispatches (APUS_MOE_GROUP_MAX=1) and
   retains only the scheduling-only wins around them — act-quant
   sharing between w1/w3 and pooled BF16 output rounds — at M9d-parity
   prefill walls. The grouped-GEMM machinery (c/fp4.h
   `apus_fp4_gemm_mt_grouped`) is shipped, gate-covered, and used at
   group size 1.

```
make test-m9e    # 40 checks; digest diffed across APUS_THREADS=1/4/8
make ubsan-m9e   # same under -fsanitize=undefined
make bench-m9e   # pre-M9e 4-row kernel vs current apus_fp4_gemm_mt, interleaved
```

## Profile (premise check, this machine)

sample(1) of a 512-token prefill on the M9d build (12 s window):
`__psynch_cvwait` ~49%, `apus_fp4_gemm_neon_rows` ~20%,
`apus_sparse_attn_head` ~12%, `pread` ~5%, `apus_mhc_prepost_scalar`
~3%. Matches tests/m9d/README.md's after-profile. (Note: the 512×id-100
filler prompt does NOT route identically across tokens — attention
mixes positions before the first MoE layer — so the medium-M expert
path is genuinely exercised at s=512; per-layer unique experts ~108
with counts ~6–48.)

## 1. Blocked NEON kernel: FALSIFIED with numbers

M9d's hypothesis: at M=6–24 the kernel re-streams the 4 MB weight slab
ceil(M/4) times and a larger m-block should cut DRAM traffic 2–4×.
bench_m9e (same-process interleaved A/B against a bench-local verbatim
copy of the M9a 4-row-block body, best-of-7, APUS_THREADS=8) says
otherwise — the slab is L2-resident across the m-block passes, so
re-streaming is cheap, and the kernel sits at ~270–295 GF/s at M≥16
(w2 shape), i.e. at the FMLAL pipe roofline:

| variant | M=1 | M=4 | M=6 | M=8 | M=12–32 (w1/w2) |
|---|---|---|---|---|---|
| 8-row m-block (halves weight passes) | ~1.0× | ~1.0× | ~1.0× | 0.85× | 0.82–0.86× |
| paired weight rows (OW=2, act-load reuse) | 0.69–0.77× | 1.0–1.45× | 0.90–0.92× | 1.00× | 0.98–1.03× |

- 8-row blocks: register pressure/scheduling makes every M≥8 shape
  ~15% SLOWER; weight-pass reduction buys nothing (L2-resident slab).
- OW=2: parity at M≥8 (act loads were not the limiter either) and a
  real M=1/M=6 regression from the bloated shared function.
- Conclusion: the only headroom left in this kernel is removing
  non-FMLAL ops (~15–20% ceiling), not blocking. Reverted; c/fp4.h's
  kernel path is byte-for-byte the M9a/M9d one, and bench_m9e asserts
  bitwise parity against the M9a anchor at every M in {1,4,…,32}.

## 2. Dispatch grouping: cache-bound, resolved at group size 1

First implementation (groups of ≤16 experts / ≤128 rows, one dispatch
per matrix per GROUP) passed every bitwise gate but REGRESSED prefill
(pf512 62.7–64.1s vs 55.2–57.1s, pf128 39.1–40.1 vs 33.3–33.7, pf32
parity, decode +5%) and pushed cvwait to ~67%.

Root cause (measured, not guessed): the M9a kernel loops m-blocks of 4
activation rows over the FULL weight slab, re-reading it ~count/4 times
per GEMM. Those re-reads are fast only while the slab stays
L2-resident. The M9d per-expert dispatch streams ONE 4 MB slab at a
time (rows split across all 8 lanes → each lane's 0.5 MB slice is
L2-hot after the first pass). A grouped dispatch streams 2×gn slabs
concurrently; once the concurrent set exceeds L2 (~8–12 MB effective),
every m-block pass misses to DRAM and the GEMM wall inflates. Sweep at
512 tokens (single runs, M9d 55.0–56.0s): gn=1 54.3–55.1s, 2 experts
(matrix-split) 57.9s, 3 experts 53.9s*, 4 experts 55.2s*, 8 experts
64.8s*, w1+w3 merged into one dispatch (2 slabs/expert) 55.6–56.6s.
(*The first sweep also carried an entry-copy bug that clobbered the w3
entries, making the second dispatch re-read the SAME slab L2-hot — its
timings were flattered; m4c's bitwise gates caught the bug, the fix is
in the shipped code, and the honest trend is monotone degradation with
slabs-in-flight.)

Shipped resolution (`APUS_MOE_GROUP_MAX=1`, `APUS_MOE_MERGE_W13=0`):
GEMM dispatches stay one slab per matrix per expert — the L2-optimal
M9d granularity — through the new grouped API. Retained scheduling-only
wins around the GEMMs:

- act quant of the gathered xg rows computed ONCE per expert for both
  w1 and w3 (the per-expert path quantized the same rows twice,
  producing identical codes);
- the BF16 output rounds of g/u/eo POOLED over the row range (M9d ran
  them as serial per-element loops inside apus_fp4_linear while the
  pool idled);
- hi-lo ordering of group entries by row count (lane-balance insurance
  for any future group size > 1; trivial at gn=1).

Recovery A/B (interleaved after/before, 2 reps each, this machine under
desktop load): pf512 59.18/57.51 vs 56.77/57.89 (median 58.3 vs 57.3),
pf128 35.18/34.14 vs 34.50/34.22 (34.7 vs 34.4), pf32 parity,
decode-24 40.40/43.38 vs 52.13/52.40 with IDENTICAL token streams both
reps. Prefill is back at M9d parity within the documented ±15%
load-noise window; the decode wall delta is run-order/load noise — the
s==1 decode path is byte-identical code to M9d (grouping only exists in
the s>1 branch) and the tokens are identical.

## Real-model before/after summary (shipped gn=1 config)

Same-window interleaved A/B (BEFORE = the M9d build), this machine
under desktop load (the documented ±15% window applies):

| metric | M9d | M9e | note |
|---|---|---|---|
| prefill 512 (filler) | 56.77/57.89 | 59.18/57.51 | parity within noise |
| prefill 128 (filler) | 34.50/34.22 | 35.18/34.14 | parity |
| prefill 32 (filler) | 12.43–13.32 | 13.02–13.61 | parity (first finals batch) |
| prefill 615 (real text) | 119.16 | 118.21 | parity; 16 greedy tokens IDENTICAL |
| decode 24 wall | 52.13/52.40 | 40.40/43.38 | run-order/load noise; path untouched |
| 24-token smoke (9-tok prompt, seed 1 temp 0) | baseline string | IDENTICAL | quality gate |

After-profile (sample(1), 512-token prefill, shipped build, 12 s
window, 59.0 s wall): cvwait ~52% (M9d: ~49%), fp4 GEMM frames
(grouped_units + neon_rows) ~22% (M9d: ~20%), sparse_attn ~13%,
pread ~5%, mhc ~3% — composition essentially unchanged from M9d,
consistent with the parity walls.

(For the record, the REGRESSED multi-expert config measured: pf512
62.7–64.1s, pf128 39.1–40.1s, pf615 122.02s vs 116.41s, decode +5% —
see section 2.)

## Honest notes / what remains

- **M9e is a negative-results milestone for its two headline targets**:
  the blocked kernel and multi-expert dispatch grouping are both
  measured losses on this hardware; what ships is the falsification
  data, the cache-bound analysis, the grouped-GEMM API (gate-covered,
  used at gn=1), act-quant sharing, and pooled output rounds — at
  prefill parity with M9d.
- **sparse_attn (~12%) is the documented next decision point** — it is
  bitwise-pinned against decode (the 24-token smoke), so moving its
  q·k dot to NEON/BLAS requires a decode-side numerics renegotiation
  with the user (hard-gate territory). NOT touched here.
- The remaining cvwait (~50%) is dominated by expert-slab I/O waits and
  serial main-thread work between dispatches (resolve, gather/scatter,
  act dequant); the profile shows pread at only ~5% duty, so further
  gains likely need either deeper I/O pipelining or the sparse_attn
  renegotiation.
- The grouped-GEMM API imposes no numerics risk if a future milestone
  re-try's group size >1 (e.g. after an m-block restructure that keeps
  slab slices L2-resident): every gate here pins it bitwise at the
  shapes that matter.

## Gate inventory (all green)

- test-m9e: 40 checks, 0 failures; digest `d0ca6c28c005811d` identical
  across APUS_THREADS=1/4/8 (O2 and UBSan builds). Covers: grouped GEMM
  bitwise == per-entry apus_fp4_gemm_mt at the real w1/w3/w2 shapes +
  odd shapes (single entry, 70-entry chunk loop, M=1 entries); the M=1
  decode pin (mt AND grouped bitwise == the M9a 4-row-block kernel
  body); MoE group/solo mixes (engineered hash-routed counts, incl. a
  >128-row solo expert) bitwise vs per-token M=1 forwards.
- m2 (5176+79), m3 (15503), m4a (143+111), m4c (1217), m5 (19): 0
  failures, tolerances untouched.
- m6a (85+5), m6b (47+7+recall checker): 0 failures.
- m6c: 514 checks, digest `8bc96af4504a5161` — byte-identical to the
  M9a–M9d value; T=1/4/8 diffed.
- m7a server: 34/34. m7b: 117+22 checks, Metal server 34/34 (GPU==CPU
  contract preserved — zero Metal changes).
- m8: 41 checks, T=1/4/8 diffed. m9a digest `5f7755151c61a979`, m9b
  digest `b398ab7989152252`, m9c digest `152005391be3b32b`, m9d digest
  `0dc73ce14af499ed` — all byte-identical to their pre-M9e values.
- UBSan: m2, m3, m4a, m4c, m5, m6a, m6b, m6c, m8, m9a, m9b, m9c, m9d,
  m9e, m7b — all exit 0 (m9e UBSan digest `d0ca6c28c005811d`, same as
  O2).
- Real-model 24-token smoke (production bin/apus): IDENTICAL tokens
  (`We need to respond to the user's query: "The capital of France is".
  This is a straightforward factual question.`).
- Real-model 615-token prompt: 16 greedy tokens IDENTICAL to M9d.
