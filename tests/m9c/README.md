# tests/m9c — expert-I/O pipelining: demand-boost queue + prefill union lookahead

Milestone M9c: attack expert-I/O serialization in both phases **without
touching numerics** — same slabs, same kernels, same accumulation orders;
only WHEN/WHETHER slabs are in RAM changes. Every pre-M9c gate passes
unloosened (inventory below).

```
make test-m9c    # 39 checks: union equivalence, boost ordering, wait stats,
                 # pilot-ON bitwise invariance; digest diffed across APUS_THREADS=1/4/8
make ubsan-m9c   # same under -fsanitize=undefined
```

The instrumented real-model driver (`tests/m9c/run_model.c` →
`tests/m9c/bin/run_model`, same run path as `bin/apus run --tiered`)
prints the M9c scheduling stats: store `waits`/`wait_ns` (just-in-time
resolve stalls), `pread_ns` (I/O-pool read duty), pilot union/d2 counters.

## Profile evidence (what M9b thought vs what is)

M9b's "345 MB/s effective prefill read rate" was read as "reads serialized
with compute; ideal pipelined prefill could be ~15–20 s". M9c added the
instrumentation to test that (wait_ns inside the store's resolve path,
pread_ns inside the I/O workers) and it says otherwise:

| phase | config | wall | resolve waits | pread duty | bytes read |
|---|---|---|---|---|---|
| prefill 512 | M9b-equivalent (union OFF) | 123.6–140.4 s | **5.1 s (4%)** | 6% | 42.8 GB |
| prefill 512 | union ON (K=12) | 123.3–130.2 s | **0.7 s** | 10% | 60.9 GB |
| decode 24 | M9c, boost OFF (ablation) | 58.5–64.0 s | 13.8–15.5 s | 38% | 108.0 GB |
| decode 24 | M9c, boost ON | 50.4–52.3 s | 8.8–9.6 s | 44% | 108.0 GB |

- **Prefill was never I/O-stalled.** The M6b batch-union storm + 4 I/O
  threads already hid ~96% of expert wait time; the wall is
  compute/scheduling. sample(1) at 512 tokens: 57% of thread time is
  cvwait (idle pool lanes during serial sections), the active shares are
  `f32_linear_rows_scalar` ~12% (compressor/indexer/gate — bitwise-pinned
  scalar order), `fp4_gemm_neon_rows` ~10%, `woa_rows` ~6.6%,
  `sparse_attn_head` ~4.8%, `pread` ~4.8%. "Bytes/wall" conflated
  disk-idle-because-compute-bound with disk-slow-because-serialized.
- The I/O pool loafs at ~10% duty during prefill *because demand is
  satisfied*. Per-slab pread latency under load is ~11 ms
  (1.2 GB/s/thread) vs 1.6 ms (8.4 GB/s) in the quiet bench — kernel-copy
  CPU contention with the 8 compute lanes, not the SSD (bench-m6a
  re-measured **8.42 GB/s** during M9c). `APUS_BUF_FREE` 64 vs 192:
  identical pread_ns (page-fault theory rejected). `APUS_IO_THREADS` 4 vs
  8: identical wall, worse per-thread latency (contention-bound, not
  thread-starved).
- **Decode had real queue-inversion stalls**: storm/resolve demand jobs
  sat behind ~300 queued pilot hints per token. The demand-boost queue
  (below) removes ~5.5 s of waits per 24 tokens with IDENTICAL bytes.

## What changed

1. **Demand-boost I/O queue (c/cache.h)** — the decode win. Loads are two
   classes: demand (the MoE batch-union storm via the new
   `apus_store_hint_demand`, resolve re-submits, and any LOADING slot a
   resolve blocks on — boosted automatically) vs speculative
   (`apus_store_hint`, the pilot surface). Workers pop the first
   demand-class job before FIFO speculative ones (hot-first scan, FIFO
   preserved within a class, O(queue depth) under mu). Generation-tag
   straggler safety, the eviction guard (speculation never evicts a warm
   demand expert), and the RSS guard are unchanged. `APUS_STORE_BOOST=0`
   disables (ablation/debug only).
2. **Wait/read instrumentation (c/cache.h)**. `ApusStoreStats` gains
   `waits`/`wait_ns` (compute thread blocked in the just-in-time resolve
   path) and `pread_ns` (time inside `apus_st_lazy_pread`, summed over
   workers). Zero cost when not waiting.
3. **Prefill union lookahead (c/pilot.h, `prefill_k`, default 12)**.
   During prefill (s>1) the post-attention hook runs the SAME prediction
   math as the M6b per-token pilot, batched over all s tokens
   (`apus_pilot_predict_union`: pooled scalar-order mixes matmul with the
   per-token rsqrt applied after the dot, then the real gate's
   `apus_router_score` — the predicted sets are EXACTLY the union of s
   per-token `apus_pilot_predict` calls, asserted bitmap-equal in this
   test), then issues one speculative hint per unique expert. Layer L+1's
   slabs stream while layer L computes; the demand-boost queue guarantees
   the current layer's storm overtakes the speculation for the ~10% of
   experts the union misses. `APUS_PILOT_PREFILL_K=-1` restores the M6b
   last-token path. Net wall effect ≈ 0 (it removes the last 4.4 s of
   prefill waits and pays ~5 s of pooled prediction compute), kept ON as
   insurance for real prompts and slower disks — and it makes prefill
   waits ~0 by construction.
4. **Decode dL=2 lookahead (c/pilot.h, `d2`, default OFF)**. Predicts
   layer L+2 from layer L's post-attention state. Works as designed
   (waits 8.66 → 5.04 s/24 tok, rescue 160/870 d1-misses) but is a
   measured NET LOSS on the M1: the extra speculative slabs (+19.8 GB/24
   tok) cost more I/O-thread CPU contention than the waits save (61.6 s
   vs 45.6 s wall; K=12+d2: 90.1 s). The speculative-bytes/compute
   exchange rate is ~0.86 s of compute contention per extra GB read.
   Opt-in via `APUS_PILOT_D2=1`.
5. **APUS_BUF_FREE knob (c/cache.h)**: recycle-list capacity (default 64,
   unchanged; measured no effect on pread_ns).

## Coverage measurement (real model, 512-token prefill)

From a `--measure-locality` dump: union of predicted top-K per token vs
the union of actual routed experts, 40 non-hash layers:

| K | union coverage | mean union/layer | prefetch bytes/layer |
|---|---|---|---|
| 6 | 75.6% | 65 | 876 MB |
| 8 | 82.8% | 78 | 1042 MB |
| 12 (default) | 89.4% | 96 | 1281 MB |
| 16 | 92.2% | 109 | 1462 MB |
| 24 | 95.2% | 130 | 1737 MB |

Token-stride subsampling loses coverage fast (stride 2 @K=12 = 85.6%,
stride 4 = 73.3%), so the batch scores every token. Caveat: the standard
filler prompt (512 × id 100) underestimates the union diversity of real
text; K=12 keeps ~1.3× over-fetch vs the ~78 slabs/layer demand.

## Measured before/after (real model, this Mac)

Same-session baselines (quiet morning window, load ~3): prefill 32 =
14.08 s, 128 = 38.76 s, 512 = 116.84 s; decode 24 = 55.10 s; decode-100
(EOS at 47) = 93.82 s. Machine load varied 3→10 across the day (shared
desktop), so cross-window absolute numbers carry ±15–25%; decisions were
made only on interleaved same-window comparisons.

| metric | M9b baseline | M9c | verdict |
|---|---|---|---|
| prefill 512 wall | 116.84 s (quiet window) | 123.3–130.2 s median 3 (busy window); **waits 5.1 → 0.7 s** | wall ≈ unchanged (compute-bound); waits eliminated |
| prefill 512 eff. read rate | 366 MB/s (42.8 GB) | ~470 MB/s (60.9 GB) | rate was never the wall |
| prefill 32 wall | 14.08 s | 15.46 / 16.03 (ON/OFF, busy) | parity within noise |
| prefill 128 wall | 38.76 s | 42.62 / 41.81 (ON/OFF, busy) | parity within noise |
| decode 24 wall | 55.10 s (quiet window) | **50.40/52.31 s boost-ON vs 58.47/63.95 s boost-OFF** (same window, interleaved) | **−16% wall from the boost; waits 14.6 → 9.2 s** |
| decode 100 wall | 93.82 s (EOS 47) | 106.15 s (EOS 47, busy window) | noise-bound; see decode 24 ablation |
| 24-token smoke tokens | `We need to respond to the user's query: "The capital of France is". This is a straightforward factual question.` | IDENTICAL in every M9c run (9 verified) | gate green |

## Knobs

| knob | default | meaning |
|---|---|---|
| `APUS_PILOT_PREFILL_K` | 12 | prefill union lookahead top-N per token; -1 = off (M6b last-token path) |
| `APUS_PILOT_D2` | 0 | decode dL=2 lookahead (measured net loss on the M1; opt-in) |
| `APUS_STORE_BOOST` | 1 | demand-class loads overtake speculative ones (0 = ablation) |
| `APUS_BUF_FREE` | 64 | slab-buffer recycle list capacity |

## Honest notes

- The M9c mission premise — "prefill reads are serialized with compute" —
  does not hold on the measured build: M6b's storm + I/O pool already
  covered ~96% of prefill expert waits. M9c proves this with wait
  instrumentation (5.1 s of 123.6 s) rather than asserting it, removes the
  remaining 4% (union lookahead), and documents the actual prefill wall:
  compute at ~43% pool-lane utilization (serial sections + DRAM-bound
  GEMMs), which is out of scope for an I/O-scheduling milestone.
- Decode's real I/O defect was queue inversion, fixed by the boost
  (−16% wall, identical bytes). The remaining ~9 s/24-tok waits are
  dL=1-missed experts loading at storm time (~11 ms/slab under
  contention); dL=2/deeper K trade bytes for waits at a negative rate
  (measured) because speculative pread CPU competes with compute on 8
  cores. The decode wall itself is the FMA-latency-bound NEON GEMM inner
  loops (m6c/m9a notes) — a numerics-adjacent kernel topic, out of scope.
- m6b test configs pin `prefill_k=-1, d2=-1` to keep their M6b scenarios
  (exact pinned ring/recall counters) unchanged; the new default path is
  covered by test_m9c's invariance gates instead.
- Cross-window timing variance on this shared desktop is ±15–25%
  (documented in m6c/m9a); every claimed effect above is backed by an
  interleaved same-window A/B, not by cross-window absolutes.

## Gate inventory (all green)

- test-m9c: 39 checks; digest `152005391be3b32b` identical across
  APUS_THREADS=1/4/8. Union bitmap == per-token union (targets 3–5,
  k ∈ {4,8,12}); boost ordering (demand overtakes 24-deep speculative
  backlog; blocked resolve boosts its load); wait-stats accounting;
  prefill-union ON vs OFF vs eager: tokens + all 8192 logits bitwise.
- m2 (5176+79), m3 (15503), m4a (143+111), m4c (1217), m5 (19): 0
  failures.
- m6a (85+5), m6b (47+7+recall checker): 0 failures — store semantics and
  the pinned M6b scenarios unchanged (their cfgs pin prefill_k=-1, d2=-1).
- m6c: 514 checks, digest `8bc96af4504a5161` — byte-identical to the
  M9a/M9b value; T=1/4/8 diffed.
- m7a server: passed (in the m2..m9c chain, exit 0). m7b: 117+22 checks,
  0 failures (Metal build + server suite + UBSan variants, exit 0) — zero
  Metal changes, GPU==CPU contract preserved.
- m8: 41 checks, T=1/4/8 diffed. m9a: 164 checks, digest
  `5f7755151c61a979` unchanged. m9b: 25 checks, digest `b398ab7989152252`
  unchanged.
- UBSan: m2, m3, m4a, m4c, m5, m6a, m6b, m6c, m8, m9a, m9b, m9c, m7b — all
  exit 0 (m9c digest identical across T=1/4 under UBSan).
- Real-model 24-token smoke (production bin/apus): IDENTICAL tokens to
  the M9a/M9b baseline string (verified 10× across the day incl. the
  final build; baseline smoke saved in weights/work/m9c_smoke24_base.txt,
  final in weights/work/m9c_smoke24_final.txt).
