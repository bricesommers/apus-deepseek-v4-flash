# tests/m4b — golden-reference oracle for the M4 single-layer forward

Milestone M4b: `tools/oracle.py` is a numpy-only port of ONE DeepSeek-V4-Flash
transformer Block (attention + MoE + mHC), following
`reference/inference/model.py` (normative) with the tilelang kernel semantics
ported from `reference/inference/kernel.py`. It generates deterministic
fixtures (real-format safetensors weights + golden I/O + named intermediates)
that the C single-layer forward (M4c) is verified against, and self-checks
its own consistency (`check_oracle.py`).

Run:

```
.venv/bin/python tests/m4b/run_oracle.py    # regenerate fixtures/ deterministically
.venv/bin/python tests/m4b/check_oracle.py  # all checks must pass (exit 0)
```

numpy only. No torch/tilelang on this machine; the tilelang kernels were
ported by reading, not by running.

## Two numeric modes

- **f32 (dtype-faithful, the C target)** — bf16 activations where the
  reference is bf16, fp32 where fp32, every QAT quant simulation applied at
  the reference points. `*_f32.npy` goldens.
- **f64 (truth)** — the same algorithm, all arithmetic in float64, no bf16
  rounding. `*_f64.npy` goldens. Used to separate algorithmic (quant,
  selection) effects from precision effects.

Lossy quantization steps (FP8-E4M3 activation quant per-128/per-64, FP4-E2M1
act quant per-32, MXFP4/FP8 weight dequant, Hadamard-before-quant, bf16
index-score rounding) are **algorithmic** and applied in BOTH modes.

## Fixture dims (SMALL_CFG in tools/oracle.py)

Chosen to preserve every code path of the reference at ~1/16 scale:

| param | fixture | real | why |
|---|---|---|---|
| dim | 256 | 4096 | %128 for FP8 block scales + act quant |
| n_heads / head_dim | 4 / 128 | 64 / 512 | rope 64 + nope 64 (nope %64 for the group-64 kv QAT) |
| q_lora_rank | 128 | 1024 | %128 |
| o_groups / o_lora_rank | 2 / 64 | 8 / 1024 | grouped o-proj preserved; wo_b K=128 |
| moe_inter_dim | 256 | 2048 | %128 |
| n_routed_experts / top-k | 8 / 3 | 256 / 6 | router + dispatch paths |
| window_size | **128** | 128 | unchanged — exercises the real ring wrap |
| compress_ratios | **4 and 128** | 4 and 128 | unchanged — the state machine is ratio-sensitive |
| index heads/dim/topk | 4 / 64 / 8 | 64 / 128 / 512 | fp4 QAT needs %32; Hadamard needs pow2 |
| hc_mult / iters | 4 / 20 | 4 / 20 | unchanged |
| route_scale / swiglu_limit | 1.5 / 10.0 | 1.5 / 10.0 | unchanged |
| rope | θ=10⁴ (SWA), θ=1.6×10⁵ + YaRN(65536, f=16, βf=32, βs=1) | same | per model.py:476-482 |
| vocab_size | 512 | 129280 | only used for `tid2eid` rows / input ids |

Layers in the fixture container (`fixtures/weights/`):

- `layers.0` = **swa** — compress_ratio 0, hash routing (`gate.tid2eid`
  [512, 3] I64), mirrors real layers 0-2.
- `layers.1` = **csa** — ratio 4, overlapping compressor + lightning
  indexer + bias gate.
- `layers.2` = **hca** — ratio 128, non-overlapping, dense compressed
  attention, no indexer, bias gate.

Weight formats on disk use the REAL checkpoint naming and dtypes
(ARCHITECTURE §3.5): FP8-E4M3 + F8_E8M0 blockwise-128×128 scales for dense
linears (`wq_a/wq_b/wkv/wo_a/wo_b/indexer.wq_b/shared_experts.w{1,2,3}`),
I8 packed FP4 + F8_E8M0 per-32 scales for routed experts
(`ffn.experts.E.w{1,2,3}.{weight,scale}`, coalesced per expert in
`apus-00002.safetensors`), BF16 for norms/gate.weight/compressor
wkv/wgate/indexer.weights_proj, F32 for hc_*/attn_sink/gate.bias/ape, I64
for gate.tid2eid. Shards are written with the M1 manual writer
(`tests/m1/stutil.py`) + `model.safetensors.index.json`, so the C side
exercises its real loader path. The oracle itself only ever consumes
weights read back through `ShardSet` (dequantized), never the raw
generation arrays.

## Fixture layout (`tests/m4b/fixtures/`, ~68 MB, ~2300 files)

```
config.json                     # SMALL_CFG
manifest.json                   # seed, sequence inventory
weights/{model.safetensors.index.json, apus-00001.safetensors, apus-00002.safetensors}
golden/<layer>/<seq>/
  input_h.npy                   # [s,4,256] f32 (bf16-rounded), the mHC state input
  input_ids.npy                 # [s] i64 (needed by hash routing)
  out_h_f32.npy / out_h_f64.npy # [s,4,256] block output, both modes
  state_out/<name>.npy          # carried state after the sequence (f32 mode)
  interm/<stage>_f32.npy / <stage>_f64.npy
golden/<layer>/decode_from<N>/step00..step11/
  input_h.npy, input_ids.npy    # the single new token (stands in for embed output)
  state_in/<name>.npy           # == previous step's state_out (step00: prefill's)
  out_h_f32/f64.npy, interm/, state_out/
```

State arrays (all f32): `pos` (i64 scalar), `win_kv` [128,128] ring,
`comp_kv` [nb,128] compressed cache, `comp_kv_state`/`comp_score_state`
[2·ratio, 2·128] (CSA; [128,128] for HCA), and for CSA also `idx_kv`
[nb,64], `idx_kv_state`, `idx_score_state`. Values are bf16-valued where
the reference stores bf16 (caches), fp32 otherwise (states, per
model.py:303-304). NOTE: state slots beyond the live partial block hold
stale/never-written values (the reference never clears them — ambiguity
A10); the live region after `pos` tokens is: CSA slots `[0:4]` (overlap)
and `[4 : 4 + pos%4]`, HCA slots `[0 : pos%128]`.

Intermediates (each in both modes; shapes per sequence):
`attn_hc_pre/post/comb`, `attn_norm_out`, `q` [s,4,128], `win_kv`,
`comp_kv` (this call's compressed entries, CSA/HCA), `idx_comp_kv`,
`idx_scores` [s,nb], `idx_topk` [s,8] (CSA), `attn_out`, `o_out`,
`post_attn_h`, `ffn_hc_pre/post/comb`, `ffn_norm_out`, `router_scores`,
`router_scores_biased`, `router_idx`, `router_w`, `moe_routed`,
`moe_shared`, `moe_out`.

Sequences (per layer): prefill at several lengths crossing every boundary,
plus a 12-step decode chain with carried state:

- swa: prefill_len6, prefill_len140 (>128 window wrap), decode_from140×12
  (ring-wrap positions 140..151).
- csa: prefill_len6 (< one block + remainder), prefill_len199 (49 blocks +
  3-token tail; >128 window), decode_from199×12 — block 49 completes on the
  FIRST decode token (199+1)%4==0, then every 4th step.
- hca: prefill_len130 (1 block + 2 tail; window wrap), prefill_len250
  (1 block + 122 tail), decode_from250×12 — block 1 completes at step06
  (start_pos 255).

## Semantics ports (reference file:line → oracle)

model.py = `reference/inference/model.py`, kernel.py =
`reference/inference/kernel.py`.

- Linear dispatch + activation QAT: model.py:108-120 → `fp8_linear`,
  `fp4_linear`. Act quant per-128 along K from the BF16-rounded input
  (tests/m3/README.md pin), ue8m0 pow2 scales (kernel.py:40-125).
- FP8 dense GEMM: kernel.py:203-273 → per-128-K-block dot of raw codes,
  `accum += dot·(scale_a·scale_b)`; pow2 weight scales folded exactly.
- FP4 expert GEMM: kernel.py:441-515 → per-32-K-block dot, act scale
  `scales_a[kb//4]` (kernel.py:503-504), weight scale per 32, correction
  `(dot·sa)·sb` (kernel.py:508-509). Dequant LUT/low-nibble-even-K per
  tests/m1/test_3_dequant.py.
- FP4 act quant (indexer q/kv): kernel.py:128-200 → `fp4_qat` (amax floor
  6·2⁻¹²⁶, pow2 scale, clamp ±6, RNE).
- RMSNorm: model.py:183-196 → `rms_norm` (fp32 internal, weight mul, cast
  back). Per-head weight-free q norm: model.py:498.
- RoPE: model.py:199-229 (YaRN literal port), 232-244 (interleaved pairs,
  inverse = conjugate). SWA uses plain θ=10⁴ (model.py:477-479);
  CSA/HCA + compressors + indexer share θ=1.6×10⁵ YaRN freqs
  (model.py:476, 491-494).
- Hadamard: model.py:247-251 → `hadamard_rotate` (see A3).
- Window top-k idx incl. decode ring order: model.py:254-265 →
  `window_topk_idxs`. HCA dense compressed idx: model.py:268-276.
- Compressor: model.py:279-377 → `compressor_forward` +
  `CompressorState`. Overlap channel-split model.py:307-314; prefill
  state stores model.py:330-336; ape add model.py:335/338/345; softmax
  gating model.py:342/352/359; decode shift model.py:353-354; RMSNorm on
  bf16-rounded pooled kv model.py:362; RoPE at the block's FIRST-token
  position model.py:363-367; QAT model.py:368-372 (Hadamard+FP4 for the
  indexer, FP8 group-64 on nope dims otherwise); cache write
  model.py:373-376.
- Indexer: model.py:380-433 → `indexer_forward`. q from `qr`
  (q_norm output, NOT wq_b output) model.py:411, 496; RoPE 413;
  Hadamard+FP4 QAT 414-416; compressor 417; weights_proj ×
  (128⁻ᵅ·64⁻ᵅ equivalent) 418; score Σ_h w·ReLU(q·k) 420-421 (see A1);
  causal block mask 424-426; top-k 427; illegal→-1 + offset 428-432.
- Attention: model.py:436-543 → `attention_forward`. q path 496-499; kv
  path 502-506 (FP8-sim nope dims, group 64); idx concat 507-515 (offset
  = seqlen prefill / window_size decode, model.py:509); ring writes
  518-523 + decode write 530; compressor call order (indexer BEFORE attn
  compressor BEFORE sparse_attn) 511/525/532; sparse_attn 528/533;
  inverse RoPE on outputs 534; grouped o-proj 537-542.
- sparse_attn kernel: kernel.py:276-368 → `sparse_attn`. -1 indices
  skipped (322-327); scale = head_dim^-0.5 (285-286); probabilities
  rounded to bf16 before P·V (340); attn_sink in the DENOMINATOR only
  (345-348); output bf16. Blocked online softmax (64-wide) → serial
  equivalent (A7).
- Gate: model.py:546-584 → `gate_forward`. f32 `x@W` (565);
  sqrtsoftplus = √(softplus) with torch threshold-20 (571); bias added
  for SELECTION only (573-575); hash override via tid2eid (576-577);
  weights from UNBIASED scores (580), normalized to sum 1 (582), ×
  route_scale 1.5 (583).
- Expert: model.py:587-606 → `expert_forward`. FP32 SwiGLU, clamp up
  ±limit and gate max +limit (600-602), silu(gate)·up (603), bf16 round
  before w2 (606).
- MoE: model.py:609-644 → `moe_forward`. FP32 accumulation (633),
  per-expert dispatch (635-640), shared expert added (643).
- mHC: model.py:647-700 + kernel.py:371-438 → `hc_pre`, `hc_post`,
  `hc_split_sinkhorn`. mixes = (x_flat @ fnᵀ)·rsqrt(mean(x²)+1e-6) on the
  UNNORMALIZED flattened state (model.py:676-678); pre =
  sigmoid(m·s₀+b)+1e-6 (kernel.py:391-392); post = 2·sigmoid (393-394);
  comb logits (395-396); row-softmax + eps, then col-norm, then 19 ×
  (row-norm, col-norm) = 20 iterations (401-423); y = Σ pre·x
  (model.py:680); hc_post y_j = post_j·x + Σ_i comb[i,j]·res_i
  (model.py:683-686); block wiring model.py:688-700.

## Ambiguities / interpretations (risk items)

Places where the reference behavior is not fully determined by the code we
can read (or cannot be run here). The oracle FIXES one interpretation; the
C side must match the oracle on these fixtures, and these are the items to
re-verify against a live reference (R4/R5) before M5.

- **A1 — index-score precision.** model.py:420-421 runs the einsum on bf16
  tensors (bf16 out, fp32 acc on CUDA) and the ReLU·weight mul + head sum
  in bf16 ("We performed QAT here", model.py:419). Port: f32 compute with
  a bf16 round after the einsum, after the elementwise mul, and after the
  head sum. Top-k sees the bf16-rounded scores ("FP32→BF16 index-score
  rounding" of ARCHITECTURE §3.3).
- **A2 — per-head q RMSNorm dtype** (model.py:498). Torch computes on the
  bf16 tensor (bf16 elementwise, fp32-acc mean). Port: f32 chain, one
  bf16 round at the end.
- **A3 — Hadamard convention.** `fast_hadamard_transform`
  (model.py:247-251) is not runnable here; its sign/permutation
  convention is unspecified in the repo. The matrix is orthogonal, so
  q·k is preserved pre-quant and only the FP4 quant error differs across
  conventions. The oracle uses the Sylvester construction × d^-0.5. The
  C side must match THIS convention on these fixtures; residual risk vs
  the real model carries to M5 (R4).
- **A4 — wo_a is FP8 on disk but BF16 in the reference** (model.py:462
  explicit dtype, 539-540 comment). Port: dequantize FP8→f32, round to
  bf16 at use (e4m3→bf16 is lossless; the pow2 scale product rounds
  once).
- **A5 — GEMM output dtype.** fp8_gemm/fp4_gemm allocate output in
  `torch.get_default_dtype()` (kernel.py:270, 533) — assumed bf16 (the
  generate-time default). All dense/expert gemm outputs are bf16-rounded.
- **A6 — top-k tie-breaking.** torch.topk's order for exactly-equal
  scores is unspecified. Port: stable descending argsort (lower index
  first). Applies to indexer top-k and router top-k.
- **A7 — sparse_attn online softmax.** The kernel processes 64-wide
  blocks with running-max rescaling; the oracle uses the serial
  equivalent. Mathematically identical; differs only in fp32 summation
  order (same class as any C accumulation order). The bf16 probability
  cast (kernel.py:340) and sink-in-denominator (345-348) ARE reproduced.
- **A8 — all-(-1) index rows.** Kernel: max stays -inf → exp(sink+inf) =
  inf denominator → output 0. Oracle emits 0 explicitly. (Does not occur
  in the fixtures; every query has ≥1 window entry.)
- **A9 — FP8 weight-quant rule for fixtures.** The checkpoint ships
  pre-quantized; no reference code defines the offline weight quant.
  Fixtures use blockwise-128×128 amax (floor 1e-4) → pow2 ue8m0 scale →
  RNE E4M3 (same rule family as act_quant). Self-consistent by
  construction; irrelevant to C (loader reads codes+scales verbatim).
- **A10 — dead state slots.** The reference never clears compressor state
  slots beyond the live partial block (the decode shift model.py:353-354
  leaves the previous block duplicated in slots [ratio:]). Semantically
  dead (overwritten before next use); `check_oracle.py::_live_slots`
  masks them in comparisons. C may zero them freely.
- **A11 — freqs dtype.** Reference computes freqs in float32
  (model.py:220-228). f32 mode replicates; f64 mode uses f64.
- **A12 — ring-mode entry point.** Decode window idx uses ring order as
  soon as `start_pos >= window_size - 1` (model.py:256), i.e. AT position
  127 already. Ported literally.

## Measured f32-vs-f64 divergence (check_oracle.py §3, long prefills)

First stage (mHC mixes, pre-quant): p99 ≈ 3e-7 — pure bf16 input rounding.
After attn_norm (one bf16 chain): p99 ≈ 8e-3 (≈1 bf16 ulp at scale 4).
Downstream stages show LARGER max/p99 gaps — this is expected and
algorithmic, not a bug: (a) FP8/FP4 act-quant scale = 2^ceil(log2(amax)),
so a bf16 rounding of the input can flip a block's scale by 2× and change
every code in it; (b) discrete selections differ between modes on these
random-weight fixtures: indexer top-k ≈ 37% of queries (csa), router ≈
3.6-3.9% of tokens (csa/hca), 0% (swa/hash). Representative rows (csa,
len199): q p99 8.8e-2, comp_kv p99 6.2e-2, attn_out p99 5.7e-2, moe_out
p99 3.4e-1, out_h p99 3.4e-1 (max 1.9). The f32 goldens remain
deterministic (bitwise reproducible — check §2) and are the C target; the
f64 goldens quantify how much of the gap is precision vs algorithm.

## Chunk-invariance (check_oracle.py §7)

One-shot prefill vs prefill(split)+single-token decodes, same inputs
(splits: swa 140@60, csa 199@99, hca 250@130):

- **f64: EXACT.** out diff ≤ 4.9e-15, window ring / compressed caches /
  indexer cache BITWISE identical, compressor kv/score states ≤ 4.5e-15
  (live slots), all router and indexer selections identical, all layers.
  The decode-time compressor state machine (partial-block carry, overlap
  pooling against the previous block, RoPE positions, ring wrap) is
  chunk-invariant — this is the property the C port most easily breaks.
- **f32: invariant up to single-code flips.** Caches ≥ 99.8% bitwise
  (csa comp 0.08% elements differ, each by ONE fp8 code step, max
  6.25e-2; hca win 0.15%; swa 0). Router selection flips: 0% swa/csa,
  1.7% hca; indexer block-set flips 0%. Output p99 on selection-matching
  tokens ≤ 8×2 bf16 ulps (csa 6.2, hca 8.0, swa 0.0); the tail comes from
  ONE flipped fp8 code in a compressed entry cascading into every query
  that selects that block. Cause: raw f32 matmul reorder noise (~1e-5)
  flipping `amax` at a binade boundary — the same class of difference any
  C accumulation order will produce. M4c tolerances must be set in these
  terms (code steps / flip fractions), not absolute epsilons.

## What M4c must reproduce, stage by stage

In pipeline order; the fixture intermediate for each stage is in
parentheses. All f32 goldens; tolerance per the chunk-invariance paragraph
(bitwise for discrete indices except near-ties, code-step granularity for
quantized values).

1. **Loader** — parse `model.safetensors.index.json` + 2 shards; MXFP4
   dequant (E2M1 LUT, low nibble = even K, UE8M0 per-32 along K), FP8
   E4M3 + blockwise-128×128 UE8M0, BF16/F32/I64. (container sanity check)
2. **mHC pre (attn)** — flatten [s,4,256]→[s,1024], rsqrt over the
   unnormalized vector, mixes in f32, sinkhorn-20 in the EXACT kernel
   order (row-softmax+eps → col → 19×(row, col)), pre/post/comb, y = Σ
   pre·x → bf16. (`attn_hc_pre/post/comb`; comb doubly stochastic to
   1.6e-4 rows / 1.2e-6 cols)
3. **attn_norm** RMSNorm. (`attn_norm_out`)
4. **Q path** — wq_a FP8 GEMM (act-quant per-128 from the bf16-rounded
   input), q_norm, wq_b FP8 GEMM, per-head weight-free RMSNorm (A2),
   partial RoPE on last 64 dims. (`q`)
5. **Window KV** — wkv FP8 GEMM, kv_norm, RoPE, FP8 QAT nope dims
   (group 64, ue8m0). (`win_kv`)
6. **Window indices** — prefill triangular windows; decode ring order
   `[sp+1..128) ++ [0..sp]` (A12).
7. **Compressor state machine** — f32 wkv/wgate, ape bias per
   in-block position, softmax-gated pooling; CSA overlap pools previous
   block's FIRST channel half + current block's SECOND half (channel
   split!); RMSNorm; RoPE at the block's first-token position; FP8 QAT
   (group 64, nope dims) for attn compressor / Hadamard+FP4 QAT for the
   indexer compressor; partial blocks carried in kv/score state; decode
   shift; first block's overlap half is 0/-inf. (`comp_kv`,
   `idx_comp_kv`, state files)
8. **Indexer** — q from `qr` via FP8 GEMM, RoPE, Hadamard, FP4 QAT;
   scores = Σ_h w_h·ReLU(q_h·k) with bf16 rounding (A1); causal block
   mask (only FULLY COMPLETED earlier blocks); top-k (A6); -1 for
   illegal; offset = seqlen (prefill) / 128 (decode). (`idx_scores`,
   `idx_topk` — causal legality checked)
9. **sparse_attn** — gather by index (-1 skip), scale 128^-0.5, softmax
   with the per-head FP32 sink in the DENOMINATOR only, probabilities
   bf16-rounded before P·V, f32 accumulation, bf16 out, INVERSE RoPE on
   the last 64 dims at the query's position. (`attn_out`)
10. **Grouped o-proj** — per-group bf16 matmul with wo_a (A4), then wo_b
    FP8 GEMM. (`o_out`)
11. **hc_post** — y_j = post_j·x + Σ_i comb[i,j]·res_i, bf16.
    (`post_attn_h`)
12. **mHC pre (ffn) + ffn_norm** — same as 2/3 with ffn params.
    (`ffn_hc_*`, `ffn_norm_out`)
13. **Router** — f32 x@W, sqrtsoftplus (softplus threshold 20), +bias for
    SELECTION only, top-3 (or tid2eid lookup on swa), weights from
    UNBIASED scores, normalize to sum 1, ×1.5. (`router_*`; weights sum
    to 1.5 ± 2.4e-7)
14. **Experts** — FP4 GEMMs (act-quant per-128), FP32 SwiGLU with clamps
    (up ±10, gate max 10), per-token weight mul, FP32 accumulation across
    routed experts; shared expert via FP8 GEMMs, same SwiGLU; bf16 out.
    (`moe_routed`, `moe_shared`, `moe_out`)
15. **hc_post (ffn)** → block output. (`out_h_f32.npy`)
16. **Decode stepping** — consume `state_in/*.npy`, produce
    `state_out/*.npy` for all 12 steps × 3 layers; replay must be
    bitwise-vs-oracle up to accumulation-order noise (check §2 replays
    step00 bitwise in numpy; C gets reorder tolerance). Compressor state
    slots outside the live region are don't-care (A10).

Call order within the attention decode step (model.py:496-533): q → win kv
→ window idx → INDEXER (incl. its compressor) → ring write → ATTN
compressor → sparse_attn. CSA decode: a token that completes a block sees
its own block in the indexer/top-k (it is completed by definition) — HCA
decode likewise appends the new compressed entry before attention.

## Files

- `tools/oracle.py` — oracle implementation + fixture generator + loader.
- `tests/m4b/run_oracle.py` — deterministic regeneration (seed 20260729).
- `tests/m4b/check_oracle.py` — self-consistency checks (exit 0 = green).
- `tests/m4b/fixtures/` — generated (weights + golden I/O + intermediates).
