# tests/m8 — MTP speculative decoding

Milestone M8: the DeepSeek-V4-Flash MTP head (depth-1, V3-style;
docs/ARCHITECTURE.md §3.7, reference/inference/model.py `MTPBlock`) as a
draft proposer, plus an exact draft/verify decode loop — speculative
decoding whose emitted token stream is **bitwise identical** to
non-speculative decoding for the same seed, greedy and sampled.

Run:

```
make test-m8       # 41 checks x APUS_THREADS=1/4/8 (outputs diffed), exit 0 iff all pass
make ubsan-m8      # same under -fsanitize=undefined
make golden-m8     # regenerate tests/m8/fixtures deterministically
bin/apus run --model DIR --spec [--spec-k D] ...   # real-model speculative decode
```

`--spec` (also `APUS_SPEC=1`) defaults **OFF**; non-speculative behavior
is untouched. `--spec-k D` (also `APUS_SPEC_K`, default 2) is the draft
chain depth = verify-batch size: D=2 means one speculative token per
batch.

## The MTP block (loader + forward)

Tensor namespace `mtp.0.*` (real container: the `apus-mtp-*` shard group;
fixture: a third shard in tests/m8/fixtures/weights). The block is a
standard engine Block (SWA ratio 0, full bias-gate MoE, FP4 experts)
loaded by the unmodified `c/layer.h` machinery via the new
`apus_layer_load_named` prefix variant; `c/mtp.h` adds the glue
(reference model.py:757-766):

```
e = enorm(embed(ids)) -> e_proj (FP8)
x = hnorm(prev 4x-dim mHC hidden) -> h_proj (FP8, per hc row)
x = bf16(e_proj(e) + h_proj(x)) -> Block -> own hc_head collapse
    (sigmoid, NO Sinkhorn) -> own norm -> SHARED lm head
```

Config: the real schema declares the block via
`num_nextn_predict_layers` + the trailing `compress_ratios` entry (0 =
SWA); the fixture schema via `"mtp": [{"compress_ratio": 0, ...}]`.
`c/model.h` parses both into `n_mtp/mtp_ratio/mtp_hash` (0 when absent —
models without MTP load exactly as before).

Expert store (c/cache.h): `mtp.K.ffn.experts.*` names map to store layer
`n_main + K` (`ApusStoreCfg.n_mtp`, 0 = unchanged behavior). The store's
batch-union hinting inside `apus_moe_forward` covers the MTP experts like
any other layer; slab derivation/budgets are unchanged.

## The accept rule (the hard invariant)

**A draft token is accepted iff it equals the main model's own pick at
that position** — argmax for greedy; the main model's own `apus_sample`
draw for sampled. Every emitted token is sampled from the main model's
own logits row, consuming exactly one RNG uniform per emitted token in
position order; drafts consume no RNG. Since the verify batch's rows are
bitwise chunk-invariant to one-by-one decoding (below), the emitted
stream reproduces the non-speculative stream **bitwise, by construction**
— for greedy exactly, and for sampled exactly for a fixed seed.

Why not classical rejection sampling (Leviathan et al.)? It preserves the
target *distribution* using draft probabilities, but (a) the requirement
here is stronger: the deterministic per-seed stream must be identical,
not merely equal in distribution; (b) it needs no draft-probability
evaluation (the MTP head is only ever argmaxed); (c) greedy falls out as
the temp<=0 special case with zero extra machinery. DeepSeek's own
`generate.py` does not use MTP, so there is no reference accept rule to
match — this rule is the one that makes the invariant exact.

## The step shape (c/mtp.h `ApusSpec`)

Invariant at step top: main state fed <= q-1; `x_q` held (already sampled
from a valid main logits row, not yet emitted); drafts `d1..dD` chained
from `(h_{q-1}, x_{q-1})` (`d_i` = candidate for position q+i-1); MTP
state contains true `(h, id)` pairs through q-1.

1. Snapshot main + MTP state (below).
2. **Verify batch**: ONE batched main forward of `[x_q, d2, ..., d_D]` at
   positions q..q+D-1 -> rows `R[j]` (logits for q+j+1) + hiddens `H[j]`.
3. **Walk**: emit `x_q`; for j=1..D-1 sample `x_{q+j}` from `R[j-1]`;
   accept `d_{j+1}` iff `d_{j+1} == x_{q+j}` (accepted tokens emitted at
   once); stop at the first mismatch — the sampled replacement becomes
   the next held token. Full match: the bonus token from `R[D-1]` becomes
   held. (The last sampled token is always *held*, never double-emitted.)
4. **State fixup**: the fed-true prefix is `batch[0..matched]`. Full
   match: keep the batch state. Partial: restore the snapshot and re-feed
   the true prefix in one batched call (bitwise the one-by-one state).
   MTP: restore its snapshot, replay the true pairs `(H[j], batch[j])`
   j<=matched in one batched MTP forward, snapshot the clean state, then
   chain the next D drafts from the replay's last hidden (the depth-1
   head autoregresses on its own post-block hidden — draft quality only;
   correctness never depends on it).

RNG: one uniform per sampled token in position order == the non-spec
stream. Drafts are always argmax (maximizes acceptance; drafts never
affect outputs).

## Mid-stream batched forward (the c/attn.h change)

The engine previously had two forward shapes: causal prefill
(`start_pos == 0`, s>1) and decode (s == 1). The m4c chunk-invariance
property — prefill == prefill(split) + single decodes, bitwise — does
**not** cover a multi-token batch from a carried decode state: the
window-index set, compressed-entry set, indexer causal mask, and the
window-ring write were all start_pos==0- or s==1-only (the ring write
even wrote only the batch's first token). `c/attn.h` now handles
`start_pos > 0 && s > 1` as a **per-token interleave of the exact s=1
decode steps** on the s-wide precomputed q/kv rows (projections are
per-row bitwise; state updates and attention run per token in the decode
call order). State and outputs are bitwise "as if decoded one-by-one" by
construction — verified by debug diff (maxdiff 0) and by the state
digests below. The prefill and decode paths are byte-for-byte unchanged
(all pre-M8 suites stay green; real-model non-spec tokens unchanged).

Verify-batch cost note: the batched forward shares the s-wide dense
projections and the MoE expert batch-union (the dominant cost); only the
attention state machine runs per token.

## Rollback = truncation, not recompute-of-state

`ApusSnap` copies, per layer: attention `pos`, the SWA window ring, the
compressor carries (`kv`/`sc`) and compressed-entry count `nb` (cache
contents beyond `nb` are dead capacity and never read; rope tables are
stateless). Restore is memcpy + `nb` rewind. On a partial reject the main
snapshot is restored and the accepted prefix is re-fed in ONE batched
call — the m4c/m8 chunk invariance makes that bitwise the sequential
state; logits/hiddens from the verify batch remain valid and are reused
(no recompute of their values). The MTP block's own SWA state is
snapshot/restored around draft chains so its window always contains true
pairs only.

The test pins this directly: an FNV digest over the full model state
(`pos`, window rings, compressor carries, `nb`, live cache entries) after
a spec run is compared **bitwise** against the state after decoding
exactly the emitted tokens non-speculatively — rejected drafts leave no
trace.

## What the suite checks (41 checks)

1. `mtp_prefill` golden: per-prefix MTP replay logits vs oracle f32
   (rel bound 0.25; argmax flips excused only at golden top1-top2 gap
   <= 0.5 — the m5 near-tie policy). Measured: rel 0.184 (O2) / 0.2
   (O1/ubsan), 1/24 flips, 0 unexcused.
2. `mtp_chain` golden: 3-step greedy draft chain logits/drafts vs oracle
   (rel bound 0.25 — the first chain row IS the replay logits).
   Measured: rel 0.0396 (O2) / 0.184 (O1), drafts match.
3. **Equivalence (the gate)**: spec depth 1/2/3 vs non-spec, greedy and
   sampled (temp 0.8, top_p 0.95, fixed seed) — 32 emitted tokens,
   BITWISE identical streams.
4. **Rollback**: state digest after each spec run == non-spec state after
   the same emitted count, BITWISE.
5. **Forced draft patterns** (`draft_override`): truth-oracle drafts
   (100% accept incl. bonus-token path), garbage drafts (0% accept),
   mixed (partial) — streams and digests still bitwise == non-spec;
   accept stats match the pattern (16/16, 22/22, 0/32, 16/32).
6. Thread independence: APUS_THREADS=1/4/8 outputs diffed (Makefile);
   UBSan clean (ubsan-m8).

Fixture note: with random synthetic weights the real MTP head's acceptance
is ~0 (vocab 512; oracle sim measured 0/12) — the accept paths are
exercised deterministically via the forced patterns, the reject paths
both ways.

## Real-model results (weights/apus, tiered, greedy `--temp 0 --seed 1`)

Baseline (M6c, documented): decode ~3.3-3.7 s/tok (0.27-0.30 tok/s);
24-token smoke wall 99.9 s.

Measured (this milestone, same prompt/seed, tiered, Metal off):

| run | tokens | decode | tok/s | tok/batch | accept | stream |
|---|---|---|---|---|---|---|
| non-spec (M6c baseline) | 24 | ~77 s | 0.31 | 1.0 | - | reference |
| --spec (k=2, default) | 24 | 67.3 s | **0.36** | 1.85 | 11/13 (84.6%) d1 | BITWISE == baseline |
| --spec --spec-k 3 | 24 | 141.9 s | 0.17 | 2.08 | 13/20 (65%), 10 re-fed | BITWISE == baseline |

Conclusions: draft-1 (the default, k=2) is the sweet spot on this
hardware; deeper drafts lose - rejected tokens waste both verify
compute and expert reads (k3: 121 GB vs k2: 88 GB expert traffic for
the same 24 tokens; 0 demand-loads in all runs, pilot recall 88-96%).
47-token A/B (model emitted EOS at 47; same prompt/seed):
non-spec decode 0.34 tok/s vs --spec 0.23 tok/s; 1.62 tok/batch,
batch accept 18/29 (62%), d1 23/29 (79%), 11 re-feds; expert traffic
198 GB (spec) vs 163 GB. Streams BITWISE identical (both answered
"...</think>Paris<EOS>"). HONEST CONCLUSION: draft-1 spec wins on
high-acceptance stretches (24-token run: 0.36 vs 0.31 tok/s) but
LOSES when acceptance dips (re-feed cost + wasted verify compute and
expert reads). Default stays OFF. The known fixes if spec should
become a net win: per-position slot-restore instead of re-feed (gap
list above) and adaptive drafting (only draft when recent acceptance
is high).

## Remaining gaps / future work

- `apus serve` (M7a) does not offer speculative decoding yet (run-mode
  only). The engine pieces are request-agnostic; wiring is future work.
- `--spec` and `--measure-locality` are mutually exclusive (the measure
  forward has no hidden-state output variant).
- The re-feed on partial rejects costs one small batched forward; a
  per-position slot-restore (save the tail's ring slots + carries, skip
  the re-feed) is the known optimization if real acceptance stays low.
- MTP draft chaining uses the MTP block's own hidden (the depth-1
  approximation); a depth-2 head or a second MTP block would need config
  support the checkpoint doesn't have.
- m5-style free-running C-vs-oracle token comparison is not an
  acceptance criterion here either (random-weight near-tie cascade); the
  hard gate is C-spec vs C-non-spec, which is exact.
