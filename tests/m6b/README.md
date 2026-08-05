# tests/m6b — router-lookahead prefetch ("pilot") + locality measurement

Milestone M6b: the prefetch layer that feeds the M6a expert store, plus the
§8 router-locality measurement program. **Hard rule verified: the pilot
changes only WHEN/WHETHER an expert is in RAM, never numerics** — token
streams and logits are bitwise identical with pilot ON vs OFF across every
cache size (same slab bytes, same kernels, same accumulation order).

Run:

```
make test-m6b    # 47 pilot checks + 7 invariance checks + recall/audit, exit 0 iff pass
make ubsan-m6b   # same under -fsanitize=undefined
make golden-m6b  # regenerate tests/m6b/fixtures deterministically
```

## Files

- `c/pilot.h` — the pilot (below).
- `c/moe.h` — the router scoring is factored into a public
  `apus_router_score` (sqrtsoftplus + selection-only bias) used by BOTH the
  MoE gate and the pilot — one code path, no duplicated math. `ApusMoeW`
  gains `hook_hint_ctx`/`hook_layer_end_ctx` so the pilot can wrap those
  two hooks while resolve stays on the store's ctx.
- `c/layer.h` — optional post-attention hook on `ApusLayer`, invoked with
  the post-attention hidden state right after the attention half.
- `c/model.h` — `apus_model_forward_measure`: identical math, plus a
  per-layer callback exposing the actual `router_idx` sets and
  `post_attn_h` (the `--measure-locality` machinery).
- `c/cache.h` — `hint_loads`/`demand_loads` attribution in
  `ApusStoreStats`; configurable usage-count heat decay
  (`ApusStoreCfg.usage_decay`, env `APUS_USAGE_DECAY`, default 1.0 = M6a
  cumulative); `apus_store_debug_present` introspection hook; **bug fix:
  I/O job-queue growth now re-lays out wrapped entries** (latent M6a bug:
  `realloc` on a wrapped ring changes the modulo geometry; pops then read
  garbage slots — NULL `rec` → worker segfault. M6a never filled the
  32-entry queue while wrapped; pilot hint bursts do).
- `c/apus.c` — pilot wiring (tiered mode) + `--measure-locality FILE`
  (self-contained NDJSON: `ids`/`gen`/`A`/`P` lines).
- `tools/measure_router_locality.py` — the §8 measurement plan (below).
- `tests/m6b/gen_fixtures.py` — 6 layers × 64 experts (top-4, inter 128)
  with **3 hash layers** like the real model; same oracle seed/machinery
  as M5/M6a. Expert slab 52,224 B; coalescing verified.
- `tests/m6b/test_pilot.c`, `test_invariance.c`, `test_recall.c`,
  `check_recall.py`.

## Pilot design (c/pilot.h)

**Prediction (dL=1).** After layer L's attention, the pilot computes layer
L+1's router input — L+1's *own* ffn mHC `hc_pre` collapse (BF16) + ffn
RMSNorm, the exact helpers `c/layer.h` feeds the real router — on layer
L's post-attention hidden state, scores it with `apus_router_score`
(sqrtsoftplus + bias-for-selection, the real gate's code path) and takes
the stable top-N (`apus_topk_stable`, A6 ties). Hash-routed targets are
never predicted (their `tid2eid` sets are exact). No dL=2 router
predictions — cross-layer coupling tables are a separate offline artifact
(`coupling_d{1,2}.json` from the measurement tool).

**Delivery.** Predictions go onto a bounded single-producer/single-consumer
lock-free ring (`(pos:32|layer:16|eid:16)` slots, acquire/release head and
tail) consumed by a dedicated pilot thread that calls `apus_store_hint`.
The compute thread never blocks: a full ring **drops the newest** entry —
the ring is FIFO in issue order, which for dL=1 lookahead is also
time-to-need order, so the oldest queued hint is always the most urgent
and the newest is the cheapest to lose; hints that outlive their
usefulness are dropped at the consumer by a `(pos, layer)` watermark (a
hint strictly behind the compute thread's current MoE layer is wasted
bandwidth, not an error — the store dedups and never evicts a warm
demand-loaded expert for a speculation).

**Hash-layer prefetch.** At tokenization time (prompt) and before each
decode forward, `apus_pilot_prefetch_hash` enqueues the exact `tid2eid`
experts of all hash layers (real model: 0–2) tagged with the token's
position. Known with certainty → 100% coverage expected.

**Heat decay.** `apus_store_save_usage` scales the *old* usage-file counts
by `usage_decay` before the merge-with-max (0.5 = halve-on-save, so pins
track routing drift; seeded pin frequencies inherit the same factor or the
decay would never bite). Default 1.0 preserves M6a's cumulative semantics.

**Stats.** predictions / pred_experts / hints_enqueued / dropped_full /
issued / dropped_stale / hash_hints / actual_experts / actual_hits — the
pilot wraps the MoE `hook_hint`/`hook_layer_end` and compares each layer's
actual chosen experts against the pending prediction (decode path, s==1),
so `recall = actual_hits/actual_experts` is measurable live. Accounting
note (M6a): a first consume of a hint-loaded working-set entry counts as a
*miss* in store stats (the slab came from disk for this block) — prefetch
effectiveness shows up as `demand_loads == 0` / `present-at-ask`, not as
hits.

## Knobs (defaults and why)

| knob | default | meaning / rationale |
|---|---|---|
| `APUS_PILOT` | 1 in tiered mode | master switch (`--tiered` only; a store is required) |
| `APUS_PILOT_K` | 8 | predicted top-N cap. **Provisional**: colibri measured 71.6% top-8 recall on GLM; the recall-curve machinery below sets this from the real model (N∈{6,8,12,16,24}). 0 = no router lookahead (hash prefetch still runs) |
| `APUS_PILOT_RING` | 4096 | ring capacity (pow2). A decode token enqueues ≤ 43×8+18 ≈ 362 hints on the real model → ~11 tokens of backlog; never blocks the compute thread |
| `APUS_PILOT_HASH` | 1 | hash-layer (`tid2eid`) prefetch at tokenization time |
| `APUS_PILOT_DUMP` | off | NDJSON runtime P/A dump (decode path; used by test_recall) |
| `APUS_USAGE_DECAY` | 1.0 | heat decay at usage save; 0.5 = halve-on-save (candidate default once measured on the real model) |
| `prefill_last_only` | 1 | during prefill, predict only for the last token (the state that flows into decode); per-token prefill predictions would flood the ring while the I/O pool is busy with demand misses |

## Results (this Mac, M1; fixture = random weights — see caveat)

**Invariance (THE hard test), `test_invariance`:** greedy decode, 8-token
prompt, 24 steps: eager vs store-only vs pilot ON at 64 / 16 / 2
slots-per-layer vs 2 slots + RSS budget 1 byte (+288 payload drops) vs
4 slots + 2 pins — **token streams identical, all 12,288 logits bitwise
identical in every configuration** (7/7 checks), with the pilot thread
issuing real loads through a 4-thread I/O pool.

**Pilot correctness, `test_pilot` (47 checks):** pilot router input is
*bitwise identical* to the ffn_norm output the real router consumes, and
the predicted top-k prefix equals the MoE's actual `router_idx` (layers
3–5); `apus_router_score` bitwise equals the interm biased scores; ring
full/drain/wraparound with exact drop counts; thread start/stop/destroy
cycles incl. destroy-with-backlog; hash prefetch **100% present-at-ask
with 0 demand loads** (synchronous store, drained ring; the ask count is
unique (layer, expert) pairs over the prompt since M9b's batched prefill
resolves each expert once per layer, plus per-(t,j) decode asks);
heat decay 0.5 halves old counts at save (100→50, 50→25, 33→16), 1.0
preserves M6a behavior; hint/demand load attribution.

**Recall end-to-end, `test_recall` + `check_recall.py`:** the pilot's live
counters (`hits=241 actuals=276 predictions=72 enqueued=948 issued=948`)
recomputed from the dumped P/A sets in Python **match exactly**
(241/276 = 87.3%); runtime dump ≡ measure-mode dump (A sets identical, P
sets identical); hash audit 162/162 = 100%.

**§8 machinery, `tools/measure_router_locality.py` (both engines):**

| measurement | C engine | oracle engine |
|---|---|---|
| recall@8 overall | 0.878 | 0.894 |
| recall curve N=6..24 | 0.810/0.878/0.952/0.981/0.997 | 0.816/0.894/0.950/0.981/0.997 |
| hash audit | 240 sets, 100% | 144 sets, 100% |
| top-8 set overlap C vs oracle (pre-divergence) | 100% | — |

> **CAVEAT — these numbers are machinery validation, NOT tuning input.**
> The fixture model has random weights: its router usage is near-uniform
> and selection is dominated by the constant bias, so "recall" is
> structurally high and temporal reuse/coupling are near the uniform
> baseline (T=1 reuse 0.19, mean P(top-1 j|i) ≈ 0.14–0.19 vs 1/64 ≈ 0.016
> uniform — the lift comes from the constant bias ranking, not locality).
> What is validated: counts, recall computation, dump consistency across
> engines, audit correctness, artifact shapes. The real defaults come from
> the identical command on the real container.

**UBSan:** `make ubsan-m6b` clean (all three binaries + checker).

## Running the full §8 measurement on the real container

Once the 160 GB download + conversion finishes:

```
make apus
python tools/measure_router_locality.py --model weights/apus --engine c \
    --prompt "a real prompt (C engine tokenizes dsv4)" \
    --decode 512 --out docs/locality --keep-dump
```

Then read `docs/locality/locality_report.json` +
`coupling_d{1,2}.json`: recall curve → `APUS_PILOT_K`; Zipf pin coverage →
`APUS_PIN_MB`; temporal reuse → LRU sizing; coupling tables → the COUPLE
offline hint artifact. The oracle engine is fixture-only (cross-check of
the dump machinery); the C engine is the production path and takes the
identical flags on the real container.
