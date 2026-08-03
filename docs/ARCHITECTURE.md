# apus — Architecture & Implementation Plan

Local inference engine for **DeepSeek-V4-Flash** (284B-total / 13B-active MoE,
FP4 experts + FP8 dense, ~160 GB of weights) on consumer hardware, by streaming
routed experts from disk through a GPU → RAM → NVMe hierarchy.

Design lineage: [colibri](https://github.com/JustVugg/colibri) (Apache-2.0),
which does the same for GLM-5.2 in pure C. Where colibri code is reused,
attribution and license notices are carried per-file.

**Status: Phase 0 plan approved by user (2026-07-29). Engine code may proceed
per the milestone gates.**

---

## 1. Goals, non-goals, hard invariants

Goals:
- Run DeepSeek-V4-Flash locally with an OpenAI-compatible server.
- C11 core, no dependencies beyond libc/pthreads (OpenMP optional, GPU backends
  optional and strictly additive).
- Insufficient fast memory costs **speed only, never quality**.

Hard invariants (any violation requires explicit user approval — hard gate 3):
- Never silently change numeric precision of any tensor or computation.
- Never change router semantics (scoring fn, bias correction, top-k, scaling,
  hash-routing tables).
- Expert weights are stored and computed from the original MXFP4 + UE8M0
  representation. No transcoding to int4/int8 as a "convenience".

Non-goals (for now): training/fine-tuning, multi-user high-throughput serving.
The dev target is Apple Silicon; Linux/Windows support and AVX2 kernels come
after the M1/NEON path works.

## 2. Target hardware profile (confirmed with user)

- Machine: **MacBook Pro, Apple M1 family, 32 GB unified memory**.
- CPU: ARM NEON with dot-product instructions (SDOT/UDOT); no AVX2 on the dev
  target → **NEON kernels are the primary path**, AVX2 is the later x86 port.
- GPU: Apple GPU via **Metal, zero-copy unified memory** — the "GPU tier" is
  wired Metal buffers carved from the same 32 GB, not a separate device
  (Metal's recommended working set ≈ 2/3 of RAM ≈ 21 GB). No PCIe copies;
  colibri's `backend_metal.mm` already proves this exact pattern.
- io_uring is Linux-only; the macOS I/O path is `F_NOCACHE` pread + a pthread
  I/O pool (colibri's `compat.h` pattern).
- Storage: internal NVMe (M1 SSDs bench ~3–5 GB/s sequential; random-read
  bandwidth at 13 MB blocks to be measured with an iobench-style tool in M6).
- Weights: not yet downloaded; download + convert is part of the roadmap.
- Priority when memory falls short: balanced, with a tunable cache budget.

## 3. DeepSeek-V4-Flash — confirmed architecture reference

Sources: HF model card + `config.json`
(huggingface.co/deepseek-ai/DeepSeek-V4-Flash), technical report
arXiv:2606.19348, mHC paper arXiv:2512.24880, and the repo's
`inference/model.py` / `inference/kernel.py` (TileLang, CUDA) which are the
**normative numerics reference**, plus `encoding/encoding_dsv4.py` for the
message format. Released 2026-04-24, MIT license (weights and code).
Local copies of the small spec files live in `reference/`.

### 3.1 Top level

| Field | Value |
|---|---|
| Layers | 43 (+1 MTP block) |
| Hidden size | 4096 |
| Vocab | 129,280 (128,000 BPE + 1,280 added) |
| MoE intermediate | 2048 per expert |
| Routed experts | **256 per layer, in all 43 layers** (+ MTP) = 11,264 total |
| Shared experts | 1 per layer (same dims as routed) |
| Top-k | 6 |
| Attention | 64 q heads, MQA (1 shared KV vector of 512 dims/position) |
| head_dim | 512 (448 nope + 64 rope) |
| q_lora_rank | 1024 |
| Output proj | grouped: 8 groups × 8 heads, wo_a [8192,4096] → wo_b [4096,8192] |
| mHC | residual stream ×4, Sinkhorn-20 (see §3.4) |
| Context | 1,048,576 (YaRN factor 16 over 65536, θ=160000 for compressed branches) |
| rms_norm_eps | 1e-6, swiglu_limit 10.0, untied embed/head |

Layer pattern (`compress_ratios`): layers 0–1 pure sliding-window (SWA);
layers 2–42 alternate CSA(ratio 4, even layers) / HCA(ratio 128, odd layers)
→ 21 CSA + 20 HCA; MTP block is SWA.

### 3.2 Router (must be bit-faithful)

- Scoring: **sqrtsoftplus** = √(softplus(x)) (replaces V3's sigmoid).
- `noaux_tc` bias correction: FP32 `gate.bias[256]` added to scores for
  **top-k selection only**; weights gathered from *unbiased* scores, then
  normalized to sum 1, × `routed_scaling_factor` 1.5.
- No group-limited routing.
- **Hash routing in the first 3 layers** (`num_hash_layers: 3`): expert choice
  is a fixed int64 lookup `gate.tid2eid[129280, 6]` keyed by token id.
  ⇒ Expert needs of layers 0–2 are known at tokenization time — free,
  perfect prefetch (see §8).
- Expert MLP: SwiGLU, **FP32 compute**, clamp up to ±10 and gate to max 10
  (`swiglu_limit`).

### 3.3 Attention: CSA + HCA + SWA (must be bit-faithful)

Common path per layer:
- Low-rank Q: `wq_a` 4096→1024, RMSNorm, `wq_b` 1024→64×512. Per-head
  weight-free RMSNorm of q.
- Shared-KV MQA: one 512-dim KV per position is both K and V for all 64 heads;
  learned `kv_norm`; partial RoPE on last 64 dims of q and KV; **inverse RoPE
  (de-rotation) applied to attention outputs**; learnable per-head FP32
  attention sink logit added to the softmax denominator; softmax scale 512^-0.5.
- **Sliding-window branch in every layer**: each query also attends to the most
  recent 128 uncompressed KV entries (`wkv` 4096→512 + kv_norm + RoPE). Window
  indices are concatenated with compressed top-k indices into one gather list.

Compressor (both CSA & HCA): learned gated pooling `wkv`/`wgate`
4096→coff×512 (BF16), learnable positional bias `ape[ratio, coff×512]` (FP32),
output RMSNorm, RoPE (position = block's first token).
- CSA (m=4): overlapping — each compressed entry pools 2m=8 token-KVs
  (current block of 4 + previous block of 4, split by channel halves), net
  rate 1/4.
- HCA (m′=128): non-overlapping, rate 1/128, dense attention over compressed
  entries, no indexer.

CSA sparse selection — Lightning Indexer (V3.2 DSA lineage):
- Own compressor (same m=4 overlap, head_dim 128, **Hadamard rotation before
  quantization**); indexer queries from the shared q latent via
  `indexer.wq_b` 1024→64×128; per-head weights from `weights_proj` 4096→64,
  scaled 128^-0.5 × 64^-0.5; score I(t,s) = Σ_h w_h·ReLU(q_h·k_s); **top-512**
  compressed entries per query; causal masking by block; index scores
  quantized FP32→BF16 (QAT — must match).

KV cache storage (report §2.3.4): BF16 for the 64 RoPE dims, FP8 for the 448
non-RoPE dims per KV entry; indexer cache FP4. Estimated ≈3.8 KB/token total
across layers → ~4 GB at full 1M context (≈2% of a BF16 GQA-8 baseline).
Block management: KV blocks span lcm(4,128)=128 tokens (32 CSA + 1 HCA entry
per block) plus fixed-size state for SWA windows and un-compressed tails.

### 3.4 mHC — Manifold-Constrained Hyper-Connections

Residual stream is 4× (`hc_mult: 4`): hidden state [s, 4, 4096]; embedding
output replicated 4×. Per sublayer: X_{l+1} = B·X_l + C·F(A·X_l) where A
(pre), C (post), B (comb) are generated per token: flatten X → RMSNorm →
`hc_{attn,ffn}_fn [24, 16384]` (FP32) → 3 mixes, scaled (`hc_*_scale[3]`) and
biased (`hc_*_base[24]`); then:
- pre  = sigmoid(mix·s0 + b) + 1e-6
- post = 2·sigmoid(mix·s1 + b)
- comb = exp(mix·s2 + b) → **20 Sinkhorn-Knopp iterations** (first row-softmax
  +eps, then alternate col/row normalization +eps) → doubly stochastic 4×4.

The head collapses 4→1 with its own sigmoid-gated `hc_head_*` (no Sinkhorn);
the MTP block has its own set. All mHC params FP32 in the checkpoint.

### 3.5 Weight formats and on-disk layout

Checkpoint: 46 safetensors shards, `model.safetensors.index.json`, 69,187
tensors, 159.6 GB. Dtype totals: I8 141.7 GB (packed FP4), F8_E8M0 8.86 GB
(scales), F8_E4M3 6.0 GB (dense), BF16 1.42 GB, F32 36 MB, I64 2.3 M.

- **Routed experts — MXFP4**: w1/w2/w3 stored I8, 2× FP4-E2M1 per byte along
  K: w1/w3 [2048, 2048] B (= 2048×4096 logical), w2 [4096, 1024] B
  (= 4096×2048). Scales: one **UE8M0 (power-of-2) per 32 elements along K**:
  w1/w3 [2048,128], w2 [4096,64]. **Per expert: 12,582,912 B weights +
  786,432 B scales = 13,369,344 B ≈ 12.75 MiB.**
  Quant rule (reference `fp4_quant_kernel`): scale = 2^ceil(log2(amax/6)),
  amax floor 6·2^-126, values clamp ±6. FP4→FP8 upcast is lossless; GEMM
  quantizes activations to FP8-E4M3 per-128-along-K and accumulates FP32 with
  per-block scale correction.
- **Dense — FP8 E4M3, blockwise 128×128 UE8M0 scales**: wq_a, wq_b, wkv,
  wo_a, wo_b, indexer.wq_b, shared-expert w1/w2/w3, MTP e_proj/h_proj.
- **BF16**: embed [129280,4096], head (lm_head), gate.weight, RMSNorms,
  compressor wkv/wgate, indexer.weights_proj, indexer.compressor.*.
- **FP32**: all hc_* params, attn_sink[64], gate.bias[256], compressor ape.
- **I64**: gate.tid2eid (3 hash layers), loaded as int32.

### 3.6 Tokenizer & message format

- HF `PreTrainedTokenizerFast` BPE: 128,000 base vocab, 127,741 merges, no
  byte fallback. BOS id 0, EOS id 1, pad 2; `add_bos_token: false` (the
  encoder script adds BOS).
- No Jinja template. `encoding/encoding_dsv4.py` is the spec, with 4
  input/output conformance pairs. Turns:
  `BOS {system}<｜User｜>{content}<｜Assistant｜>{think}{content}{tool_calls}<EOS>`;
  generation prompt ends `<｜Assistant｜>` + `<think>` (thinking) or `</think>`
  (chat). Tool calls are DSML markup; tool results merged into user messages
  wrapped in `<tool_result>`. `reasoning_effort` (0731: low/high/max;
  "high" = the preview's old "max" prefix, "max" = a new prefix, low/None =
  none) prepends a fixed system prefix in thinking mode. Key special ids: `<｜User｜>` 128803, `<｜Assistant｜>` 128804,
  `<think>` 128821, `</think>` 128822, `｜DSML｜` 128825.
- Sampling defaults: temperature 1.0, top_p 1.0.

### 3.7 MTP head

Depth 1, V3-style: shared embed → enorm → e_proj (FP8); previous 4×4096 mHC
hidden → hnorm → h_proj; sum → full Block (SWA + full 256-expert MoE with
bias gate, FP4 experts) → own hc_head collapse → norm → shared head.
Used for speculative decoding (milestone M8).

## 4. colibri: what we reuse vs. reimplement

colibri (Apache-2.0, v1.1.0) targets GLM-5.2, itself a DeepSeek-family model
(MLA 512+64 latent KV, sigmoid+bias noaux_tc router, DSA lightning indexer,
V3-style MTP). Its engine is a single `c/colibri.c` (~7.1k lines) plus
single-purpose headers — **no custom weight container**: ordinary safetensors
shards read via a hardened pread-based index (`st.h`).

Reuse as-is or with light adaptation (model-agnostic infrastructure):
- `st.h` — safetensors shard index, pread + uncached-fd twins, dual-SSD
  mirror, N-drive split, FNV hash lookup. Proven on ~120k-tensor indexes.
- Tiering: `ESlot` slab machinery, per-layer LRU with end-of-block promotion,
  RSS guard, `tier.h` LFRU score (frequency-primary, recency tiebreak) with
  25%+4 hysteresis, hot-pin store seeded from a persistent usage-history file,
  between-turns REPIN pass.
- Miss overlap: generation-tagged lock-free cursor + ≤16 I/O pthreads
  (compute thread never does I/O), `posix_fadvise(DONTNEED)` hygiene.
  (io_uring is Linux-only — not on the M1 target.)
- Prefetch ("pilot"): run layer L+1's real router on the post-attention
  hidden state, feed a 1P/1C ring to a dedicated pilot thread; hint-only
  (FADV_WILLNEED) or real cross-layer loads with an eviction guard. Measured
  71.6% top-8 one-layer-ahead recall on GLM-5.2 (vs 41.3% previous-token
  reuse). The math ("next layer's linear router on current hidden") transfers
  directly; the scoring function differs (sqrtsoftplus vs sigmoid) and must
  be V4's.
- `compat.h` (macOS F_NOCACHE etc.), `kv_persist.h` pattern, `telemetry.h`,
  `iobench.c`, `json.h`, `sample.h`, converter's shard-by-shard disk-safe
  driver, `openai_server.py` gateway pattern, doctor/resource-planner CLI.
- `backend_metal.mm` precedent: zero-copy unified-memory slabs, full-layer
  command buffers fusing norm+attention+router+shared-expert on GPU while the
  CPU resolves experts, expert MoE GEMV in Metal.
- The coalesced-expert-read trick: write each expert's tensors adjacently in
  the output shard so a miss = one ~13.4 MB pread into one slab, with
  zero-copy views.

Reimplement / parametrize for V4-Flash:
- **FP4 storage + compute — the main new kernel work.** colibri's formats are
  int8/int4/int3/int2/E8; NVFP4 exists only as converter *input*. We add an
  MXFP4 format (§6) + NEON kernels (§9). Note the precedent colibri sets
  (transcode FP4→int4-gs64) is **rejected** here: quality invariant.
- Attention: CSA/HCA compressors, overlapping-pool state machine, Hadamard
  rotation, attention sink, output de-rotation, grouped o-proj. GLM's MLA
  latent-KV path does not carry over (V4 replaced the latent with
  sequence-compressed shared-KV MQA; only low-rank Q survives).
- Router: sqrtsoftplus, unbiased-weight gather, hash-routing tables.
- mHC residual stream (4× state, Sinkhorn-20) — entirely new; colibri/GLM
  has nothing like it.
- KV cache layout: CSA/HCA compressed entries + SWA windows + tails, FP8/BF16
  mixed storage, 128-token blocks.
- Tokenizer (129,280 vocab tables) and the `encoding_dsv4.py` message format
  (colibri renders Jinja in Python; we implement the DSML/think format
  exactly, with the 4 conformance pairs as tests).
- Config parsing for the V4 schema; tensor naming map.

## 5. File layout

```
apus/
├── Makefile                     # clang -O3, -lm -lpthread only; OpenMP opt-in
├── docs/{ARCHITECTURE.md, STATUS.md}
├── reference/                   # small spec files from the HF repo (config,
│                                #   tokenizer.json, encoding/, inference/*.py)
├── c/
│   ├── apus.c                   # config, model load, forward driver, CLI, serve loop
│   ├── st.h                     # safetensors index + pread/F_NOCACHE (from colibri)
│   ├── fp4.h                    # MXFP4 E2M1 storage, dequant, NEON GEMV/GEMM kernels
│   ├── fp8.h                    # FP8 E4M3 blockwise-128×128 dense GEMM kernels
│   ├── attn.h                   # CSA/HCA/SWA attention, indexer, KV cache
│   ├── mhc.h                    # mHC residual stream + Sinkhorn-20
│   ├── moe.h                    # router (sqrtsoftplus/noaux_tc/hash) + expert dispatch
│   ├── cache.h                  # expert store: slots, LRU, pins, PIPE miss overlap
│   ├── tier.h                   # LFRU scoring, REPIN, heat decay (from colibri)
│   ├── pilot.h                  # router-lookahead prefetch
│   ├── compat.h                 # macOS/Linux shims (from colibri)
│   ├── kv.h / kv_persist.h      # compressed KV cache + crash-safe persistence
│   ├── tok.h                    # BPE tokenizer (129,280 vocab)
│   ├── encoding.h               # dsv4 message format / DSML rendering
│   ├── sample.h / json.h / telemetry.h
│   └── backend_metal.mm         # optional Metal tier/offload (M7)
├── tools/
│   ├── convert.py               # HF shards → apus container (disk-safe, resumable)
│   ├── download.py              # shard-by-shard download+convert driver
│   ├── oracle.py                # reference-inference golden I/O generator
│   ├── measure_router_locality.py  # recall/coupling/entropy measurements
│   └── route_pairs.py           # cross-layer coupling tables
└── tests/                       # dependency-free C + Python; one dir per milestone
```

## 6. Weight container format (FP4 experts)

Decision: **keep colibri's proven design — ordinary safetensors shards +
`st.h` index — and keep expert bytes exactly as shipped by DeepSeek.**
No custom binary container, no transcode.

- One output shard set, written shard-by-shard (disk-safe: peak disk = one
  ~5 GB input shard + growing output), resumable.
- FP4 expert tensors stored byte-identical to the HF checkpoint:
  expert w1/w2/w3 (I8, 2×E2M1/byte) and their `weight_scale` (F8_E8M0).
  Converter copies, never requantizes; verification is a byte-compare plus a
  dequant spot-check. (Exact tensor names are taken from the checkpoint's
  `model.safetensors.index.json`, not assumed.)
- **Coalesced per-expert layout**: the converter orders tensors so each
  expert's {w1,w1_scale,w2,w2_scale,w3,w3_scale} are contiguous in the shard
  → a cache miss is one 13.37 MB pread into a single slab, zero-copy views.
  Scales ride along inside the slab (unlike colibri's separate fslab) to keep
  it a single read; the extra 0.77 MB/expert is negligible.
- Dense tensors converted to the same safetensors output unchanged (FP8 E4M3
  + blockwise UE8M0 scales, BF16, FP32, I64 tables) — resident set §7.
- Format self-description: extend colibri's `qt_resolve_fmt` idea with an
  MXFP4 kind identified by (packed bytes == O·K/2) ∧ (scale bytes ==
  O·K/32) ∧ scale dtype E8M0. Group size 32 is implied by the byte counts,
  validated at load.
- MTP and indexer tensors in separate shard groups (`apus-mtp-*`,
  `apus-idx-*`) like colibri, so the indexer set can be lazy-loaded.
- Manifest: `apus.index.json` (config hash, tensor map, format version) next
  to `model.safetensors.index.json`.

Total container size ≈ 160 GB (same as source). We may *additionally* ship an
optional lossless re-pack (align slabs to 4 K for uncached reads) — alignment
only, never values.

## 7. Tiering design (GPU → RAM → NVMe, unified memory)

Per-token expert demand (decode, no cache hits): 43 layers × 6 = 258 experts
× 13.37 MB ≈ **3.45 GB/token** worst-case cold reads (+MTP when active).
This is the number every tier decision fights.

On the M1 there is one 32 GB physical pool; the "tiers" are budget classes
within it, plus the disk. Budget (tunable, `APUS_*` env):
- Resident dense set: ~6.0 GB FP8 + 1.4 GB BF16 + embed/head ~2.1 GB +
  FP32/I64 smalls ≈ **~9.6 GB**, wired.
- KV cache: budget 2 GB default (~500K tokens), growable.
- macOS + working set / I/O buffers: ~5 GB (unified memory must leave the OS
  real headroom or everything swaps — the failure mode to avoid).
- **Expert cache + pins (CPU-side): ~12–14 GB ≈ 900–1,050 of 11,264 experts.**
- **Metal tier (M7)**: zero-copy means no upload cost — "GPU-resident"
  experts are wired Metal buffers over the same pool, traded against the CPU
  expert cache within one shared budget. Likely first win is dense-on-GPU
  (fused norm+attention+router+shared-expert, colibri's full-layer command
  buffers), freeing CPU cycles for expert resolve/disk.

Cache policy (from colibri, retuned):
- Per-layer LRU with end-of-block promotion; hits bump an atomic clock.
- Hot-pin store seeded from a persistent usage-history file, scaled by
  history confidence; between-turns REPIN with LFRU hysteresis.
- RSS guard: free LRU slabs in place if measured RSS exceeds budget.
- Miss overlap: generation-tagged lock-free cursor + pthread I/O pool with
  `F_NOCACHE` pread; compute thread only waits just-in-time per expert.
- Prefetch (pilot): run layer L+1's router on the post-attention hidden
  state; top-`APUS_PILOT_K` predictions (default from recall curve, §8) into
  a 1P/1C ring consumed by the pilot thread; real cross-layer loads with the
  eviction guard (never evict a warm demand-loaded expert for a speculation).
- **Hash-layer shortcut**: layers 0–2 experts are known exactly at
  tokenization/prefill time (`tid2eid`) — enqueue them for prefetch before
  the forward even starts; 18 of 258 expert loads per token become free.
- Batch-union: in prefill/batched decode, read each unique expert once.

Performance model (to be validated in M6): with NVMe delivering ~B GB/s at
13 MB random reads, miss-rate m gives tok/s ≈ B / (3.45 GB × m). At
B = 2 GB/s, m = 50% → ~1.15 tok/s; m = 25% → ~2.3 tok/s. M1 SSDs are fast;
the cache-hit and pilot-recall measurements in §8 set realistic expectations.
We will not quote a tok/s target until measured.

## 8. Router-locality measurement plan

Before tuning cache sizes, measure (tools/measure_router_locality.py, against
the real router on real prompts, using the reference model or our M5 forward):
1. **Pilot recall curve**: fraction of layer L+1's actual top-6 contained in
   the pilot's predicted top-N, for N ∈ {6, 8, 12, 16, 24}, per layer-type
   (post-CSA vs post-HCA). Sets `APUS_PILOT_K`. colibri got 71.6% at top-8 on
   GLM; V4's sqrtsoftplus + 256-expert/43-layer profile may differ.
2. **Temporal reuse**: P(expert reused within next T tokens), T ∈ {1,2,4,8} —
   sizes the LRU benefit vs pure streaming.
3. **Cross-layer coupling**: P(e_{L+d}=j | e_L=i), d ∈ {1,2} — the cheap
   no-router-matmul hint table (colibri's COUPLE mode).
4. **Expert frequency distribution** per layer over a corpus → pin candidates
   (Zipf-ness decides how much the pin store helps at ~10% capacity).
5. **Hash-layer audit**: confirm tid2eid prefetch hits 100% on layers 0–2.
Deliverable: a report in docs/ with concrete cache/pin/pilot defaults.

## 9. Kernel plan

- `fp4.h` (gate 2 applies): **NEON first** (M1 dev target; SDOT/UDOT
  dot-product insns and FP16 lanes where exactness allows), AVX2 as the
  portable x86 path second. A scalar reference implementation is written
  first and both SIMD paths are tested against it.
  - Dequant: nibble → E2M1 value via 16-entry LUT, scale applied as
    power-of-2 exponent add on FP32 lanes (UE8M0 = exponent byte); per-32
    block scale broadcast.
  - GEMV (decode): quantize activation row to FP8-E4M3 per-128 (lossless
    FP4→FP8 upcast per the report) or dequant-to-FP32 — whichever matches the
    reference within tolerance; FP32 accumulation with per-block scale
    correction. Kernel choice is a numerics decision, verified against the
    reference kernel, not a performance shortcut.
  - GEMM (prefill): blocked, same accumulation rules.
  - Tests: dequant exact vs LUT over all 256 byte patterns × all scale
    exponents; GEMV/GEMM vs a FP32 reference on random tensors (abs/rel
    tolerance recorded in tests/README); then vs the reference TileLang
    kernel outputs on identical inputs.
- `fp8.h`: blockwise-128×128 UE8M0 dense GEMM, same discipline.
- `mhc.h`: Sinkhorn-20 in FP32 exactly as kernel.py (row-softmax first, +eps
  placement, 20 iterations) — exactness matters, it's in the residual path.
- `attn.h`: compressor overlap state machine, Hadamard rotation (indexer),
  FP32→BF16 index-score rounding, attention sink, output inverse-RoPE,
  index-gather + online softmax.
- Threading: Grand Central Dispatch or pthreads for matmul parallelism on
  macOS (OpenMP optional via Homebrew libomp), pthreads for I/O; atomics for
  cross-thread state. Metal backend deferred to M7; until then the GPU is
  unused and the engine is CPU+disk.

## 10. Milestone roadmap

- **M0 — Phase 0 docs** (this file + STATUS.md). ✅ Approved at gate 1.
- **M1 — Converter + downloader** (`tools/convert.py`, `download.py`).
  Verify: byte-identical expert tensors vs source shards; dequant spot-check
  vs torch reference; resume-from-interruption test; manifest validation.
  (Full 160 GB download is deferred — needs disk-space confirmation; M1 is
  validated on synthetic fixtures + the small spec files.)
- **M2 — Tokenizer + encoding** (`c/tok.h`, `c/encoding.h`).
  Verify: round-trip vs `tokenizer.json` on a corpus incl. all special ids;
  pass all 4 `encoding/` conformance pairs; thinking/chat/DSML tool-call
  rendering matches byte-for-byte.
- **M3 — FP4 NEON kernel** standalone (`c/fp4.h` + tests).
  Verify: LUT exhaustive test; GEMV/GEMM tolerance vs FP32 scalar reference
  and vs reference kernel outputs.
  **HARD GATE 2: kernel numerics approved before integration.**
- **M4 — Single-layer forward on CPU**: one CSA layer and one HCA layer,
  incl. mHC, router, experts-from-disk (uncached).
  Verify: golden I/O vs reference `model.py` on synthetic random weights
  (small fixtures generated by `tools/oracle.py`), elementwise tolerances.
- **M5 — Full 43-layer forward + sampling.**
  Verify: golden-token comparison against DeepSeek's reference inference on
  fixed prompts/seeds (see risk R1 — needs a machine that can run the
  reference; fallback is layerwise golden I/O + hash-layer determinism
  checks + MTP-consistency checks).
- **M6 — Tiering**: LRU/pins/pilot, RSS guard, usage-history learning.
  Verify: identical tokens with minimal vs maximal cache budget (quality
  invariant, automated); measured tok/s + RSS within budget; router-locality
  report (§8) landed as tuned defaults.
- **M7 — Metal tier + server**: zero-copy Metal expert tier / dense offload
  (per colibri's `backend_metal.mm` precedent), OpenAI-compatible endpoints
  (SSE streaming), DSML tool calls, KV persistence.
  Verify: endpoint conformance, streaming correctness, tokens unchanged vs
  CPU-only.
- **M8 — MTP speculative decoding.**
  Verify: output tokens identical to non-speculative (same seed); measured
  acceptance rate and tok/forward ≥ ~2 target (colibri measured 2.2–2.8 on
  GLM; V4 TBD).

## 11. Risks / open questions

- R1: **Golden-token reference needs the full model running somewhere**
  (~160 GB; reference kernels are CUDA/TileLang). The 32 GB M1 can't host
  it. Options: rent a big-GPU instance once to generate golden outputs, use
  HF transformers (if `deepseek_v4` support exists) on a large host, or
  restrict golden tests to layerwise fixtures + real-weight single-layer
  I/O. Decision needed by M5 — will raise it again there.
- R2: 32 GB unified memory is tight: ~9% expert cache capacity means decode
  speed is pilot-recall-bound. Mitigations: hash-layer prefetch, coupling
  tables, pinning from usage history, Metal tier in M7. No quality impact by
  design. Also: leaving macOS < ~5 GB free risks swap thrash — the RSS guard
  is a hard requirement, not a nicety.
- R3: ~~GPU backend choice~~ Resolved: Apple M1 → Metal only, zero-copy
  unified memory; macOS I/O via F_NOCACHE pread + pthread pool.
- R4: Indexer QAT details (FP32→BF16 score rounding, Hadamard before FP4)
  exist only in kernel.py — treat as normative; add focused tests in M4.
- R5: Report defers "tiny details" to code (weight-free per-head q RMSNorm,
  decode-time compressor state machine). model.py/kernel.py win over prose
  wherever they disagree.
