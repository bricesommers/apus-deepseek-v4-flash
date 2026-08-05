# tests/m6a — expert-store tiering (experts on NVMe, bounded RAM cache)

Milestone M6a: routed experts live on disk and are demand-loaded through a
bounded, pinned, LFRU-managed RAM cache — the colibri tiering design
(docs/ARCHITECTURE.md §4/§7) on the apus container. **Hard rule verified:
memory pressure costs speed only, never quality** — token streams and
logits are bitwise identical from "everything fits" down to "2 slots/layer
under a 1-byte RSS budget".

Run:

```
make test-m6a    # 85 store checks + 5 invariance checks, exit 0 iff pass
make ubsan-m6a   # same under -fsanitize=undefined
make bench-m6a   # NVMe slab-read bench + hit-rate/tok/s curve
make golden-m6a  # regenerate tests/m6a/fixtures deterministically
```

## Files

- `c/compat.h` — macOS shims: RSS via mach `task_info`, `F_NOCACHE` fd
  marking, `APUS_*` env parsing.
- `c/st.h` (addition) — `ApusStLazy`: header-only shard open (cached fd +
  `F_NOCACHE` twin), `apus_st_lazy_pread` (thread-safe positional reads,
  loops on short reads), atomic read counters. The slurp path is unchanged;
  all pre-M6a suites stay green. mmap deliberately NOT used.
- `c/cache.h` — the expert store (below).
- `c/moe.h` (hooks only) — `ApusMoeW` gains `hook_resolve/hook_hint/
  hook_layer_end/hook_ctx/layer_id`. When set, expert weights come from the
  hook; the router, SwiGLU, accumulation order are untouched. The hook
  first submits all unique routed experts of the block (batch-union miss
  overlap), then resolves just-in-time per expert.
- `c/layer.h` — skips eager expert-view resolution when the set is
  deferred (`apus_st_set_defer_experts`), sets `mw.layer_id`.
- `c/model.h` — `apus_model_load_ex(m, dir, tiered, ...)`; the old
  `apus_model_load` wraps it with `tiered=0`.
- `c/apus.c` — `--tiered` / `APUS_TIERED=1`: opens the store, attaches it
  to every layer, saves usage history + prints store stats at exit.
- `tests/m6a/gen_fixtures.py` — 6 layers × 64 experts (top-4, inter 128),
  same oracle machinery/seed as M5; expert slab = 3×(16384+1024) =
  **52,224 B**; asserts the 6-tensor coalescing per expert in Python.
  Fixtures ~25 MB, checked in.

## Expert store design (c/cache.h)

**Addressing.** (layer, eid) → slab record derived at open from the shard
headers (`weight_map` + each shard's JSON header): the six member tensors
`{w1,w1_scale,w2,w2_scale,w3,w3_scale}` must share one shard and tile a
contiguous byte range (canonical convert.py SLAB_MEMBERS order; the store
verifies, it does not assume). A miss is **one pread** of the whole slab
into a 4 KiB-aligned buffer (`posix_memalign`); `ApusFp4W` views are
zero-copy pointers into it. Open fails loudly on a split/non-contiguous
slab. The `apus.index.json` manifest is not needed — headers suffice.

**Slots.** Per layer: an LRU slot array, a pin array, and a per-forward
working set (heap slots). Misses load into the working set, never directly
into the LRU; at `layer_end` (end-of-block, hooked from `apus_moe_forward`)
each consumed working entry is promoted by swapping with the coldest LRU
slot (empty slots first). A batch-union overflow (more distinct experts
than slots) drops the excess after use — one-shot streaming experts do not
flush the cache. Hits bump an atomic clock.

**Miss accounting (for tuners):** a resolve counts as a *miss* when the
slab was not already cached (pin/LRU resident, or a working-set entry the
current block already consumed). First consume of a hint-loaded entry is a
miss — it came from disk for this block. In steady state
`misses == preads == unique (layer,eid) loads`.

**Pins.** `pins_per_layer` pinned slots, never evicted, payloads lazily
loaded on first touch. Seeded from a usage-history file (default
`<model_dir>/apus.usage`, plain text `layer eid count`, atomically
rewritten via tmp+fsync+rename, merge-with-max so counts are cumulative and
monotonic). Pin frequencies are seeded from the file so REPIN hysteresis is
meaningful immediately.

**LFRU REPIN.** `apus_store_repin` (call between turns): per layer, swap
the coldest pin with the hottest resident unpinned expert only if the
challenger's frequency ≥ pin × 1.25 + 4 (colibri hysteresis), recency as
tiebreak. Loops until no swap improves.

**RSS guard.** `task_info(TASK_BASIC_INFO_64)` RSS vs budget; over budget →
free coldest LRU payloads in place (slot keeps eid/freq identity, next
resolve reloads). Pins, in-flight loads, and working-set entries are never
touched; it runs at block boundaries only (never while a forward holds
views).

**Miss overlap.** pthread I/O pool (default 4). Workers pread into private
buffers and claim the slot under the store mutex **only if the generation
tag still matches** — a straggler whose slot was recycled drops its payload
and resets the slot, so the waiter re-submits; stale bytes can never alias
a newer generation. The compute thread never does I/O in pool mode
(`io_threads < 0` = synchronous mode for tests). `F_NOCACHE` twin fds keep
expert traffic out of the page cache (macOS has no
`posix_fadvise(DONTNEED)`; `compat.h` documents the no-op).

## Results (this Mac, M1)

**Quality invariance (THE hard test), `test_invariance`:** greedy decode,
8-token prompt, 24 steps, full forward — eager slurp path vs store at 64 /
16 / 2 slots-per-layer vs 2 slots under a 1-byte RSS budget (288 payload
drops!) vs 4 slots + 2 seeded pins: **token streams identical and all
12,288 logits bitwise identical in every configuration** (expected: same
slab bytes, same kernels, same accumulation order — any difference would be
a store bug). Store traffic: 282 preads (full) → 686 (+288 RSS reloads)
under maximal pressure; output unchanged.

**Unit tests, `test_store`:** 85 checks — slab derivation + one-pread-per-
expert (instrumented), view dims, hit/miss accounting, working-set
promotion with mid-block overflow, eviction order, LRU recency, pin
seeding, pin persistence across a simulated restart, LFRU hysteresis
(16 vs 29 thresholds), RSS guard (pins survive, identity kept, reload on
demand), generation-tag straggler (synthetic race via pre-claim hook;
stale payload discarded, re-read, bytes correct), concurrent-vs-serial
slab byte-identity (12 experts × 6 tensors memcmp), speculative-hint
eviction guard.

**NVMe (`bench_m6a`), 153 × 13,369,344 B random reads (real expert slab):**

| mode | GB/s | cold tok/s floor (3.45 GB/token, 100% miss) |
|---|---|---|
| F_NOCACHE pread | 5.5–9.2 (cold file → warm bench file) | 1.6–2.7 |
| cached pread | 4.0–7.9 | 1.2–2.3 |

(Variance is concurrent background load — the 160 GB download was writing
during these runs. F_NOCACHE ≥ cached consistently.) Implied decode floor
`tok/s = GB/s / (3.45 GB × miss-rate)`: at a conservative 5.5 GB/s,
m = 100% → 1.6, m = 50% → 3.2, m = 25% → 6.4 tok/s. The M1 SSD is much
faster than the 2 GB/s the architecture doc modelled — the tiering budget
buys real headroom.

**Fixture hit-rate vs tok/s (`bench_m6a`, mini-model, informational):**

| slots/layer | hit rate | preads | decode tok/s |
|---|---|---|---|
| 64 (all) | 62.1% | 282 | 348 |
| 32 | 55.2% | 333 | 328 |
| 16 (25%) | 37.6% | 464 | 281 |
| 8 | 23.0% | 573 | 289 |
| 4 | 15.9% | 626 | 283 |
| 2 | 10.9% | 663 | 282 |

(Fixture slabs are 52 KB and the whole 20 MB expert shard is OS-cache
warm, so decode tok/s is compute-bound and barely moves — the curve that
matters on the real model is hit-rate vs the NVMe floor above. The random
router reuses heavily: 282 unique experts of 384 over 32 tokens.)

## Knobs

| knob | default | meaning |
|---|---|---|
| `APUS_EXPERT_CACHE_MB` | 4096 | total LRU budget → slots/layer = budget / (layers × slab) (real model: ≈7/layer ≈ 300 experts; M6c retune from 12288 — see tests/m6c/README.md §VM) |
| `APUS_PIN_MB` | 512 | pin budget → pins/layer (real: ≈0–1/layer; M6c retune from 2048) |
| `APUS_RSS_GUARD_MB` | 26624 | RSS guard threshold (leaves macOS ~6 GB) |
| `APUS_IO_THREADS` | 4 | I/O pool size; <0 = synchronous |
| `APUS_NOCACHE` | 1 | F_NOCACHE streaming reads |
| `APUS_TIERED` | 0 | CLI: experts-on-demand (or `--tiered`) |
| usage file | `<model_dir>/apus.usage` | `""` in `ApusStoreCfg.usage_path` disables |

All of the above are also explicit `ApusStoreCfg` fields (tests use those).
Introspection: `apus_store_stats` (hits/misses/preads/evictions/drops/
pin_loads/repin_swaps), `apus_store_resident_bytes`,
`apus_store_debug_layer`.

## What M6b needs from this layer (already exposed)

- `apus_store_hint(store, layer, eid)` — thread-safe, dedup'ed, non-
  blocking; the pilot thread's real-load entry point. Unconsumed hints
  carry no use-clock: they take free slots but **never evict a warm
  demand-loaded expert** (the colibri eviction guard, tested).
- Hash-layer prefetch: layers 0–2 expert ids are known from `tid2eid` at
  tokenization time — just call `hint` for them before the forward.
- `apus_store_repin` between turns; `apus_store_save_usage` at exit.
- `apus_store_stats` for pilot-recall measurement (predicted vs resolved).
- Cross-layer loads: `hint` accepts any (layer, eid) at any time; the
  working set holds in-flight pilot slabs until a `layer_end` promotes or
  drops them.
- NOT provided (M6b work): the pilot router itself (run L+1's gate on the
  post-attention hidden), the 1P/1C hint ring, heat decay of usage counts,
  FADV_WILLNEED-style hint-only mode (hint currently always loads).
