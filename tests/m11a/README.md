# tests/m11a — DSpark oracle + fixtures (M11a)

Milestone M11a: a faithful numpy port of **DSpark**, the 0731-native
speculative decoding module (`reference-0731/inference/model.py` —
`DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead`,
`DSparkBlock`, `get_dspark_topk_idxs`, and the `Transformer.forward` /
`forward_spec` wiring), plus golden fixtures for the C port (M11b).
No C code in this milestone.

Run:

```
make golden-m11a    # regenerate tests/m11a/fixtures deterministically (~20 s)
make check-m11a     # 74 oracle self-consistency checks (~30 s), exit 0 iff all pass
```

Line refs below are `reference-0731/inference/model.py:N` unless noted.

## The DSpark data flow (as ported)

Three DSpark stages live in the `mtp.{0,1,2}` checkpoint namespace (layer_ids
43/44/45, `compress_ratios` entries 43–45 = 0/SWA; `__init__` 898–904). Each
stage is a full Block (SWA attention, bias-gate MoE with 256 FP4 experts,
mHC wiring). Extras: stage 0 has `main_proj` (FP8 [4096,12288]) + `main_norm`;
stage 2 has `norm`, own `hc_head_*`, `markov_head.markov_w{1,2}` (BF16
[129280,256]), `confidence_head.proj` (BF16 [1,4352]).

**Main forward (912–926).** While running the 43 main layers, the model
collects `h.mean(dim=2)` (mean over the 4 mHC streams) after each target
layer [40,41,42] (917–921), concatenates them → `main_hidden` [*, 3·4096].
The last-position logits row is sampled *inside* forward (924) — that sample
is the anchor. (Our engine samples outside `forward`; the anchor draw is
just the normal decode draw — same logits row, same uniform.)

**Draft (forward_spec, 928–936).**
1. `forward_embed` (851–858): `main_x = main_norm(main_proj(main_hidden))`;
   draft ids = `[anchor, noise, noise, noise, noise]` (noise = 128799);
   embed → 4× mHC expand → block input [5, 4, 4096].
2. Each stage (DSparkBlock.forward 845–849 → Block.forward 695–707 with
   DSparkAttention): stage's own `wkv/kv_norm` projection of `main_x` rope'd
   at `start_pos`, FP8-QAT'd, written to its KV ring at `start_pos % 128`
   (783). The 5 draft rows get q/kv rope'd at `start_pos+1..start_pos+5`
   (772), attend via `get_dspark_topk_idxs` (743–747): **one shared index
   row for all 5 queries** = window slots `0..min(128,start_pos+1)-1` ++
   `128+0..128+4` (the ring ++ the 5 draft KVs concatenated after it, 784).
   **Non-causal block attention** — every draft row attends to all 5 draft
   KVs. Then the usual inverse-rope, grouped o-proj, mHC post, MoE, mHC post.
3. `forward_head` (860–874) on stage 2: own hc_head collapse (sigmoid, no
   Sinkhorn) → `x` [5, 4096]; `logits = shared_head(norm(x))` [5, V]. For
   i = 0..4: `logits[i] += markov_w2(markov_w1(output_ids[i]))` (a bigram
   bias from the *preceding* token), then `output_ids[i+1] =
   sample(logits[i], temperature)`. Confidence = `proj(cat([x (pre-norm),
   markov_embeds[i]]))` [5] in fp32.

**Prefill (763–769).** forward_spec with `start_pos=0` only builds each
stage's KV ring from `main_x` over the whole prompt (same circular write as
SWA prefill); draft embeddings are discarded (848–849). All three stages
consume the *same* stage-0-projected `main_x` (930–933).

## THE ACCEPT RULE (what M11b must implement)

**Reference-0731 ships no speculative decode loop** (ambiguity D1), so this
rule is the M8 rule (`tests/m8/README.md`) restated for DSpark — the rule
that makes the emitted stream **bitwise equal to non-speculative decoding**:

> Every emitted token is drawn from the **main model's own logits row** for
> its position — argmax for greedy, the engine's pinned-uniform draw for
> sampled — consuming **exactly one uniform per emitted token in position
> order**. A draft token is **accepted iff it equals the main model's own
> draw at that position**. Drafts consume a **separate** uniform stream and
> never influence the emitted stream. The confidence scores take **no part**
> in acceptance (they are ported as faithful goldens; confidence-based
> early-exit is a stream-safe perf knob for later).

Round shape (invariant at round top: main state fed through position `f`,
all true; `held` token for `f+1` already drawn from a valid main row, not
yet emitted; main_hidden row of `f` available):

1. **Draft** at `start_pos=f` → drafts `d1..d5` for positions `f+2..f+6` +
   confidence. (Stage rings hold true positions 0..f — see 5.)
2. **Snapshot** main state. **Verify batch**: feed `[held, d1..d5]` at
   positions `f+1..f+6` in one chunk-invariant batched forward → rows
   `R[0..5]` (dists for `f+2..f+7`) + per-position main_hiddens `H[0..5]`.
3. **Walk**: emit `held`. For j=1..5: draw `m_j` from `R[j-1]` (one uniform);
   if `d_j == m_j` emit it (`a=j`) else stop — `m_j` becomes the next held
   (drawn, **not** emitted). Full match (`a=5`): held = draw from `R[5]`
   (bonus). Emitted per round = `1 + a`.
4. **State fixup**: full match keeps the batch state (all fed tokens true).
   Partial: restore the snapshot, re-feed the true prefix `[held, d1..d_a]`
   in one batched call (bitwise the sequential state by the M4c/M8
   chunk-invariance). Next round: `f' = f+1+a`, main_hidden = `H[a]`
   (reused from the verify batch), held from 3.
5. **Stage-ring maintenance** (D13): write each stage's KV slots for **all
   newly-true fed positions** `f+1..f'` from `H[0..a]`, so every stage ring
   always holds exactly the true positions 0..f'. The next round's
   in-attention write of slot `f' % 128` (783) rewrites the same value —
   idempotent.

## Ambiguities found (and resolutions)

- **D1 — there is no reference accept rule.** `reference-0731/inference/
  generate.py` is plain non-speculative decoding (its `generate()` at lines
  19–56 calls only `model.forward`); *nothing* consumes `forward_spec`'s
  confidence output — even model.py's own `__main__` demo (958–961) discards
  it. The task premise that generate.py defines the accept rule is wrong.
  Resolution: the M8 exact-match rule above (same situation as M8: "there is
  no reference accept rule to match — this rule is the one that makes the
  invariant exact"). Confidence is ported faithfully but is telemetry only.
- **D2 — reference sampling is not stream-safe under speculation.**
  `sample()` (939–946) is Gumbel-max on torch's global RNG (draft sampling
  would consume main-stream draws) and has no top-p. Resolution: the
  engine's existing contract (c/sample.h, m5-pinned): one pinned uniform per
  emitted token **indexed by absolute position**, CDF draw over f32 softmax
  with `top_p = 1.0`; drafts draw from a **separate** pinned PCG64 stream.
  Greedy (`temperature == 0`) = argmax everywhere, as in the reference.
- **D3 — `main_hidden = h.mean(dim=2)`** is a bf16-tensor mean (fp32
  accumulate, bf16 result). Oracle: mean in working precision, bf16-round in
  f32 mode only.
- **D4 — confidence hidden is the hc_head collapse *before* norm**
  (862 vs 873). Ported exactly (`conf_hidden` goldens).
- **D5 — stage MoE `input_ids`** receives only the anchor `[b,1]`
  (forward_spec passes `input_ids` through, 932/847) — unused because all
  three stages are non-hash (layer_ids 43–45 ≥ `n_hash_layers`). The fixture
  stages are non-hash too.
- **D6 — block attention is non-causal** among the 5 draft rows (single
  shared topk row, 746–747). Intentional in the port (checked).
- **D7 — all stages share the same `main_x`** (stage-0 projection) for their
  KV rings; each stage applies its *own* `wkv/kv_norm` (759/778).
- **D8 — rope positions**: draft rows at `start_pos+1..start_pos+5`
  (772, `seqlen=1` from `main_x`); the main-KV row at `start_pos` (758).
- **D9 — stage rope tables**: compress_ratio 0 → YaRN **off**, base
  `rope_theta` (483–485), same as main SWA layers.
- **D10 — the noise-token embeddings are real block inputs** (their KV is
  computed and attended over); the noise id must be a valid embed row.
  Fixture: 511 = vocab−1 (real: 128799, likewise near the vocab end).
- **D11 — window gather order at ring wrap**: `get_dspark_topk_idxs` uses
  plain `arange` slot order, *not* the chronological rotation
  `get_window_topk_idxs` uses (261–264). Attention is set-wise, so this is
  gather/softmax-order numerics only, but the C port must gather in the
  **same slot order** to stay bitwise.
- **D12 — draft KVs never enter the stage ring** (concatenated after it,
  784): stage rings only ever hold main-model positions → no per-round
  rollback of stage state, only forward maintenance (D13).
- **D13 — stage-ring maintenance under speculation is our design.** The
  reference (non-spec demo) writes one slot per decode step via 783; a spec
  driver accepts several positions per round, so the interior positions'
  slots must be written explicitly or later rounds would attend over stale
  slots. Rule: at each round's fixup, write slots for every newly-true fed
  position from the verify batch's per-position hiddens (cheap: per-stage
  `wkv` [512,4096] FP8 + norm + rope per accepted token). Draft quality only
  affects acceptance rate, never the stream — but the goldens pin this rule
  exactly, and check_oracle verifies every ring slot holds the newest true
  position's KV.
- **D14 — markov/confidence dtypes**: `markov_w1/w2` are BF16 in the
  checkpoint, used as fp32 params (embedding lookup exact; w2 is an fp32
  linear, no output rounding). `confidence_head.proj` BF16-stored, fp32
  compute, no output rounding (810). `main_proj` is FP8 blockwise-128
  (act_quant on `main_hidden` like any FP8 dense linear).

## Fixture inventory (tests/m11a/fixtures, ~17 MB; regenerate with golden-m11a)

Mini-model: m5 FULL_CFG extended to 5 layers (l0 swa+hash, l1 csa4, l2
hca128, l3/l4 csa4), DSpark targets [2,3,4] (the last three, mirroring
[40,41,42]), block_size 5 (real), markov_rank 64, noise 511, window 128.
Weights use the real naming (`mtp.K.*` incl. coalesced FP4 experts in the
`apus-mtp-00001` shard group, like the real container).

- `join_prefill_len24/`, `join_prefill_len140/` — prompt ids; per-position
  `main_hidden` [s, 3·256] and stage-0 `main_x` [s, 256] (f32+f64); the 3
  stage rings after prefill [128, 128]. len140 exercises the circular
  prefill write and full-window topk branch.
- `draft_rounds_len24/round{0,1,2}/` — 3 teacher-forced draft rounds:
  anchor, start_pos, drafts, greedy margins, confidence, `logits_base` [5,V],
  `markov_bias` [5,V], `logits_final` [5,V], `markov_embed` [5,64],
  `conf_hidden` [5,256], per-stage block outputs `stage{k}_h` [5,4,256]
  (f32; round0 also f64). `rounds.json`: per-round stage-ring digests.
- `draft_round_len140/` — one round at start_pos 139 (ring wrapped,
  full-window `arange(128)` topk branch), same arrays.
- `spec_episode_greedy/` — 8 rounds, temp 0: prompt, `tokens_spec/nonspec`
  (f32+f64), `round_f`, `anchors`, `drafts`, `accepted`, `bonus`,
  `emitted_counts`, `margins` (per main draw, greedy top1−top2 gaps),
  `confidence` (f32+f64), `digests.json` (spec/nonspec main + spec full
  state, f32+f64).
- `spec_episode_sampled/` — same at temp 0.8, top_p 1.0, plus
  `uniforms_main` (per absolute position), `uniforms_draft`,
  `draft_margins`.
- `forced_all_accept/` — drafts := true continuation → `accepted` [5]×6,
  bonus every round, 36 tokens (bonus path pinned exactly).
- `forced_all_reject/` — drafts := (true+1) % V → `accepted` [0]×6
  (full rollback every round).
- `forced_mixed/` — drafts reject at j=3 → `accepted` [2]×6; on the len140
  prompt, so stage/main rings wrap mid-episode (positions ≤ 157).
- `manifest.json` — seeds, RNG contract, per-sequence metadata.

With random synthetic weights the *natural* acceptance is 0/5 every round
(vocab 512; same as m8) — accept paths are pinned by the forced patterns,
draft numerics by the draft-round goldens.

## What check_oracle.py verifies (74 checks)

1. **Container sanity** — mtp.* naming/dtypes/shapes vs the real 0731
   scheme (FP8 main_proj + E8M0 scale, BF16 markov/confidence, F32 hc_head,
   per-stage FP4 experts + F32 gate bias; mtp.* ↔ apus-mtp shard group).
2. **Determinism** — recomputed joins, draft rounds (all arrays), and both
   natural episodes match goldens **bitwise** (f32).
3. **Equivalence (the gate)** — spec == non-spec emitted streams bitwise:
   greedy + sampled, f32 + f64, natural + 3 forced patterns; rollback
   digests equal.
4. **Forced patterns** — exact accept counts ([5]/[0]/[2] per round), bonus
   flags, streams == non-spec, digests == golden digests.
5. **Rollback** — post-episode main-model state == non-spec state
   array-by-array bitwise (not just digests), f32 + f64.
6. **Legality** — `get_dspark_topk_idxs` formula + non-causal rows;
   confidence finite [R,5]; `accepted ∈ [0,5]`; `emitted == 1+accepted` per
   round; bonus == full match; drafts/emitted in vocab; margins alignment;
   **stage-ring catch-up**: every ring slot holds the newest true position's
   KV (f32 storage; prompt positions in the batched-prefill chunk-noise
   class, decode positions bitwise).
7. **Divergence report** — f32-vs-f64 per draft-path stage (see below).

## Tolerancing guidance for M11b

Measured f32-vs-f64 (oracle self-check, draft round 0): `logits_final`
maxabs 5.0 at scale 5.2 (p99 3.5), argmax flip 4/5 rows (gaps 0.23–0.71,
*not* near-ties); confidence maxabs 1.43 at scale 1.7; `main_x` maxabs 1.2
at scale ~3.7. This is the known random-weight near-tie cascade, amplified
by 3 extra stages. Consequences:

- **Hard gates (bitwise, C-internal):** C-spec == C-non-spec streams (greedy
  + seeded sampled, natural + forced drafts); rollback state digests ==
  non-spec digests (FNV-1a over the canonical layout in
  `oracle_dspark.state_digest`); thread independence (T=1/4/8).
- **Golden comparisons vs the oracle f32 goldens:** per-row, given the same
  context, under the m5/m8 policy — logits rel bound 0.25 with argmax flips
  excused at golden margin ≤ 0.5; confidence rel 0.25 (it's one fp32
  linear on top of a deep cascade); sampled draws excused at small CDF
  margins (margins dumped per draw).
- **Not a gate:** free-running C-vs-oracle token streams and natural
  accept counts (near-tie cascade; m8 had the same policy). The forced
  episodes pin the accept/rollback *paths* exactly.
- Stage-ring contents: f32 storage; decode positions bitwise vs single-row
  recompute, prompt positions within one bf16 code step (m4b chunk noise).

## M11b implementation checklist (C port)

1. **Config**: parse `dspark_block_size`, `dspark_noise_token_id`,
   `dspark_target_layer_ids`, `dspark_markov_rank` (+ `n_mtp_layers` = 3 in
   the inference schema). The 3 trailing `compress_ratios` zeros are the
   stage ratios (all SWA) — the M10 parser already tolerates 46 entries.
2. **Loader**: `mtp.{0,1,2}` full blocks via the M8 `apus_layer_load_named`
   prefix path; stage-0 `main_proj` (FP8 [4096,12288] + scale) + `main_norm`;
   stage-2 `norm`, `hc_head_{fn,base,scale}` (F32), `markov_head.markov_w{1,
   2}` (BF16 [129280,256] — widen once to F32 like the lm head),
   `confidence_head.proj` (BF16 [1,4352] → F32).
3. **Expert store**: `mtp.K.ffn.experts.*` are already coalesced slabs in the
   0731 container (11,776 slabs = 46 layers × 256). Map stages to store
   layers `n_main + K` (M8 did `n_mtp = 1`; now 3) and let the existing
   batch-union hinting cover them: the draft forward's expert reads (3
   stages × 5 rows) should union-hint like the M8/M9b MoE batch path. Budgets
   unchanged.
4. **Main forward**: collect per-position hc-mean hiddens at the target
   layers during the normal forward (3 × mean-over-4-streams per token —
   negligible); return them with the logits row. The anchor draw is the
   existing `apus_sample` call — no new sampling machinery.
5. **Stage state** (per stage): SWA ring [128, 512] (f32, bf16-valued) +
   pos. Prefill: batched build from `main_x` (circular write). Per round:
   in-attention slot write + fixup writes (D13). Snapshot/restore: stage
   rings never need rollback (D12) — only the main model's M8 ApusSnap.
6. **Draft forward**: 3-stage block forward over the 5 draft rows reusing
   the existing block machinery (hc_pre/hc_post, MoE, sparse_attn with the
   shared non-causal topk row, gather in slot order — D11); forward_head:
   hc_head collapse, norm, shared lm head at M=5, markov bias loop
   (embed lookup + F32 GEMV per row), confidence GEMV on
   cat([pre-norm hidden, markov embeds]).
7. **Verify batch**: the M8 chunk-invariant batched decode over
   `[held, d1..d5]`, collecting per-position hiddens at target layers.
8. **Walk + rollback**: exactly the M8 ApusSpec walk with D=6 batch and the
   accept rule above; restore + batched re-feed on partial; reuse `H[a]`
   for the next round (no recompute).
9. **RNG**: main stream one uniform per emitted token in position order
   (existing apus_sample draws); separate draft stream (splitmix64 seeded
   apart). Draft draws never touch the main stream.
10. **Later perf knobs (stream-safe by construction)**: confidence-based
    early exit (skip verifying tail drafts below a threshold), adaptive
    drafting by recent acceptance (the m8 gap list), per-position
    slot-restore instead of re-feed.
