#!/usr/bin/env python3
"""tools/oracle_dspark.py — M11a golden-reference oracle for DSpark, the
0731-native speculative decoding module.

A numpy-only port of the DSpark draft/verify flow from
reference-0731/inference/model.py (normative; line refs below refer to that
file), built ON TOP of tools/oracle.py (the M4b/M5 block port — imported as
`base`). Two modes exactly like base: "f64" truth and "f32" dtype-faithful
(bf16 activation rounding + QAT quant simulation at the reference points).

Ported pieces (model.py):
  * Transformer.forward target-layer hidden collection (912-926):
    main_hidden = concat over target layers of h.mean(dim=2)  [*, 3*dim]
  * DSparkBlock.forward_embed (851-858): stage-0 main_proj (FP8) + main_norm;
    draft ids [anchor, noise x (B-1)]; embed -> hc expand.
  * DSparkAttention prefill branch (763-769): per-stage KV window built from
    main_x (the stage-0 projection of the MAIN model's hidden states), same
    circular write as Attention prefill.
  * DSparkAttention decode branch (771-792): q/kv over the B=block_size draft
    rows rope'd at start_pos+1..start_pos+B, main_kv written at
    start_pos % win, topk = get_dspark_topk_idxs (743-747: window slots
    0..min(win,start_pos+1)-1 ++ win+0..win+B-1, SAME row for all B queries —
    non-causal block attention), sparse_attn, inverse rope, grouped o-proj.
  * DSparkBlock.forward (845-849): prefill = KV build only; decode = full
    Block (hc_pre/attn/hc_post/hc_pre/MoE/hc_post).
  * DSparkBlock.forward_head (860-874): own hc_head collapse (sigmoid, no
    Sinkhorn), own norm, SHARED lm head with full logits; per draft position
    the Markov bias markov_w2(markov_w1(token)) is added to the logits row
    and the next draft token is sampled; confidence head over
    cat([pre-norm hidden, markov embeds]).
  * DSparkMarkovHead (795-804) / DSparkConfidenceHead (807-815).

THE ACCEPT RULE: reference-0731 ships NO speculative decode loop — its
generate.py is plain non-speculative decoding and nothing consumes the
confidence scores (see tests/m11a/README.md ambiguity D1). The accept rule
ported here is the M8 rule (tests/m8/README.md), which is the one that makes
the emitted stream bitwise equal to non-speculative decoding:

  Every emitted token is sampled from the MAIN model's own logits row for
  that position (argmax for greedy; the engine's pinned-uniform draw for
  sampled), consuming exactly one uniform per emitted token in position
  order. A draft token is accepted iff it equals the main model's own draw
  at that position. Drafts consume a SEPARATE pinned uniform stream and never
  affect the emitted stream. Confidence scores are faithful goldens but take
  no part in acceptance (stream-safe early-exit is an M11b perf knob).

The step shape (mirrors c/mtp.h's ApusSpec, adapted to DSpark):
  Round state: main model fed through position f (a TRUE position); held
  token x (position f+1) already sampled from a valid main row, not yet
  emitted; main_hidden row of position f available.
  1. Draft: forward_spec decode at start_pos=f — each stage writes slot
     f%win with its own projection of main_x (model.py:783), then the 3-stage
     block forward over the 5 draft rows -> drafts d1..d5 for positions
     f+2..f+6 + confidence. (Stage windows hold all true positions 0..f at
     this point — see step 5.)
  2. Snapshot main state. Verify batch: feed [x, d1..d5] at positions
     f+1..f+6 (the oracle models the C engine's bitwise chunk-invariant
     batched forward as per-token interleaved decode steps, exactly like
     c/attn.h's M8 rule) -> rows R[0..5] (dists for f+2..f+7) + per-position
     main hiddens H[0..5].
  3. Walk: emit x; for j=1..5 draw m_j from R[j-1] (one uniform); accept
     d_j iff d_j == m_j; stop at the first mismatch — the drawn replacement
     becomes the next held token (sampled, NOT yet emitted). Full match:
     held = draw from R[5] (bonus).
  4. State fixup: full match keeps the batch state (all fed tokens true).
     Partial: restore the snapshot and re-feed the true prefix
     [x, d1..d_a] in one batched call (== sequential by chunk invariance).
     Next round: f' = f+1+a (last true fed), main_hidden = H[a] reused from
     the verify batch, held = the replacement/bonus token.
  5. Stage-window maintenance (ambiguity D13 — the reference has no spec
     loop, so it never needs this; a spec driver does): write each stage's
     KV slots for ALL newly-true fed positions f+1..f' from the verify
     batch's hiddens H[0..a], so every stage window always holds exactly
     the true positions 0..f'. The next round's in-attention write of slot
     f'%win (model.py:783) rewrites the same value — idempotent.

Fixture mini-model: the m5 FULL_CFG extended to 5 layers (target layers
[2,3,4] = the last three, mirroring real [40,41,42]) + 3 synthetic DSpark
stages under the real mtp.* naming (full blocks incl. FP4 experts, stage-0
main_proj/main_norm, stage-2 norm/hc_head/markov/confidence).
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m3"))

import oracle as base                # the M4b/M5 block/model port
import stutil                        # manual safetensors writer (M1)

MASTER_SEED = base.MASTER_SEED

# ---------------------------------------------------------------------------
# Fixture config: m5 FULL_CFG + one more CSA layer + the DSpark section.
# window 128 is kept so prompt 140 exercises the ring-wrap/full-window
# branch; prompts 24 exercise the partial-window branch. noise id 511 =
# vocab_size-1 (real 128799 is likewise near the vocab end).
# ---------------------------------------------------------------------------

M11A_CFG = dict(base.FULL_CFG)
M11A_CFG["layers"] = [
    {"name": "l0_swa", "compress_ratio": 0, "hash": True},
    {"name": "l1_csa", "compress_ratio": 4, "hash": False},
    {"name": "l2_hca", "compress_ratio": 128, "hash": False},
    {"name": "l3_csa", "compress_ratio": 4, "hash": False},
    {"name": "l4_csa", "compress_ratio": 4, "hash": False},
]
M11A_CFG["dspark"] = {
    "block_size": 5,                    # real: 5
    "noise_token_id": 511,              # real: 128799
    "target_layer_ids": [2, 3, 4],      # real: [40, 41, 42]
    "markov_rank": 64,                  # real: 256
    "n_mtp_layers": 3,                  # real: 3
}

SAMPLE_SEED = 20260811                # main-stream uniforms (per position)
DRAFT_SEED = 20260812                 # draft-stream uniforms (per round*5+i)
SAMPLE_TEMP = 0.8                     # sampled episode temperature (top_p 1.0)
N_UNIFORMS = 640                      # pinned main uniforms, indexed by position

GREEDY_PROMPT_SEED_OFF = 40000        # join/draft-rounds prompt (len 24)
LONG_PROMPT_SEED_OFF = 41000          # ring-wrap prompt (len 140)
GREEDY_EP_PROMPT_SEED_OFF = 43000     # greedy spec episode prompt (len 24)
SAMPLED_EP_PROMPT_SEED_OFF = 44000    # sampled spec episode prompt (len 24)

EPISODE_ROUNDS = 8
FORCED_ROUNDS = 6


# ---------------------------------------------------------------------------
# Fixture weights: real mtp.* naming (verified against the 0731 index).
# ---------------------------------------------------------------------------


def gen_dspark_tensors(cfg, rng):
    """Synthetic DSpark stage tensors under the real mtp.{0,1,2} namespace:
    3 full blocks (SWA ratio 0, bias gate, FP4 experts) + stage-0 main_proj
    (FP8 [dim, 3*dim]) / main_norm + stage-2 norm / hc_head_* (F32) /
    markov_head.markov_w{1,2} (BF16 [V, rank]) / confidence_head.proj
    (BF16 [1, dim+rank])."""
    recs = []
    B_rank = cfg["dspark"]["markov_rank"]
    dim, V, hc = cfg["dim"], cfg["vocab_size"], cfg["hc_mult"]
    n_tgt = len(cfg["dspark"]["target_layer_ids"])

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))).astype(np.float32)

    for k in range(cfg["dspark"]["n_mtp_layers"]):
        srng = np.random.default_rng(MASTER_SEED + 6000 + 100 * k)
        layer = {"compress_ratio": 0, "hash": False}
        recs.extend(base.gen_layer_tensors(cfg, layer, k, srng,
                                           prefix=f"mtp.{k}"))
    # stage 0 glue
    cb, sb, csh, ssh = base.fp8_store(rnd(dim, n_tgt * dim))
    recs.append(("mtp.0.main_proj.weight", "F8_E4M3", csh, cb))
    recs.append(("mtp.0.main_proj.scale", "F8_E8M0", ssh, sb))
    recs.append(("mtp.0.main_norm.weight", "BF16", (dim,),
                 base.f32_to_bf16_bytes(
                     (1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))))
    # stage 2 glue
    last = cfg["dspark"]["n_mtp_layers"] - 1
    recs.append((f"mtp.{last}.norm.weight", "BF16", (dim,),
                 base.f32_to_bf16_bytes(
                     (1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))))
    recs.append((f"mtp.{last}.hc_head_fn", "F32", (hc, hc * dim),
                 np.ascontiguousarray(rnd(hc, hc * dim), np.float32).tobytes()))
    recs.append((f"mtp.{last}.hc_head_base", "F32", (hc,),
                 np.ascontiguousarray(0.25 * rng.standard_normal(hc),
                                      np.float32).tobytes()))
    recs.append((f"mtp.{last}.hc_head_scale", "F32", (1,),
                 np.ascontiguousarray(1.0 + 0.1 * rng.standard_normal(1),
                                      np.float32).tobytes()))
    recs.append((f"mtp.{last}.markov_head.markov_w1.weight", "BF16", (V, B_rank),
                 base.f32_to_bf16_bytes(rnd(V, B_rank, std=1.0))))
    recs.append((f"mtp.{last}.markov_head.markov_w2.weight", "BF16", (V, B_rank),
                 base.f32_to_bf16_bytes(rnd(V, B_rank))))
    recs.append((f"mtp.{last}.confidence_head.proj.weight", "BF16",
                 (1, dim + B_rank),
                 base.f32_to_bf16_bytes(rnd(1, dim + B_rank))))
    return recs


def write_m11a_weights(cfg, out_dir):
    """write_full_weights + a third apus-mtp shard with all mtp.* tensors
    (real container layout: DSpark stages live in their own shard group)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED + 90000)
    dense_recs = base.gen_toplevel_tensors(cfg, rng)
    expert_recs = []
    for li, layer in enumerate(cfg["layers"]):
        lr = np.random.default_rng(MASTER_SEED + 1000 * li)
        for rec in base.gen_layer_tensors(cfg, layer, li, lr):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    mtp_recs = gen_dspark_tensors(cfg, np.random.default_rng(MASTER_SEED + 7000))
    shards = [("apus-00001.safetensors", dense_recs),
              ("apus-00002.safetensors", expert_recs),
              ("apus-mtp-00001.safetensors", mtp_recs)]
    weight_map, total = {}, 0
    for fname, recs in shards:
        stutil.write_shard(os.path.join(out_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  f, indent=1)


def load_dspark_params(shards, cfg, f64):
    """Oracle params for the 3 DSpark stages: full block params (prefix
    mtp.K) + the stage-0/stage-2 glue."""
    last = cfg["dspark"]["n_mtp_layers"] - 1
    Pd = []
    for k in range(cfg["dspark"]["n_mtp_layers"]):
        layer = {"compress_ratio": 0, "hash": False}
        P = base.load_layer_params(shards, cfg, layer, k, f64,
                                   prefix=f"mtp.{k}")
        if k == 0:
            P["main_proj"] = shards.fp8("mtp.0.main_proj")
            P["main_norm"] = shards.f32("mtp.0.main_norm.weight")
        if k == last:
            P["norm"] = shards.f32(f"mtp.{last}.norm.weight")
            P["hc_head_fn"] = shards.f32(f"mtp.{last}.hc_head_fn")
            P["hc_head_scale"] = shards.f32(f"mtp.{last}.hc_head_scale")
            P["hc_head_base"] = shards.f32(f"mtp.{last}.hc_head_base")
            P["markov_w1"] = shards.f32(f"mtp.{last}.markov_head.markov_w1.weight")
            P["markov_w2"] = shards.f32(f"mtp.{last}.markov_head.markov_w2.weight")
            P["conf_proj"] = shards.f32(f"mtp.{last}.confidence_head.proj.weight")
        Pd.append(P)
    return Pd


def new_dspark_states(cfg, f64):
    """Per-stage state: a ratio-0 LayerState (SWA window ring + pos)."""
    layer = {"compress_ratio": 0, "hash": False}
    return [base.LayerState(cfg, layer, f64)
            for _ in range(cfg["dspark"]["n_mtp_layers"])]


# ---------------------------------------------------------------------------
# Main-model forward with DSpark target-layer collection
# (Transformer.forward, model.py:912-926).
# ---------------------------------------------------------------------------


def main_forward_dspark(Ps, top, cfg, ids, states, start_pos, f64):
    """base.model_forward + main_hidden collection. ids [s]; states mutated.
    Returns (logits [s, V], main_hidden [s, n_targets*dim]) where
    main_hidden = concat over target layers of hc-mean(h) (model.py:920-921:
    h.mean(dim=2) — fp32 accumulate, bf16 result in f32 mode)."""
    dt = base._dt(f64)
    ids = np.asarray(ids, dtype=np.int64)
    h = top["embed"][ids].astype(dt)
    h = np.repeat(h[:, None, :], cfg["hc_mult"], axis=1)
    targets = cfg["dspark"]["target_layer_ids"]
    mains = []
    for i, P in enumerate(Ps):
        h, _ = base.block_forward(P, cfg, h, ids, start_pos, states[i], f64)
        if i in targets:
            mains.append(base._B(h.astype(dt).mean(axis=1), f64))
    y = base.hc_head_collapse(h, top, cfg, f64)
    yn = base.rms_norm(y, top["norm"], cfg["norm_eps"], f64)
    logits = base._mm(yn.astype(dt), top["head"].astype(dt).T)
    mh = np.stack(mains, axis=1).reshape(len(ids), -1)
    return logits, mh


def stage0_project(P0, cfg, mh, f64):
    """forward_embed's main_x = main_norm(main_proj(main_hidden))
    (model.py:853). mh [..., n_targets*dim] -> [..., dim] bf16."""
    x = base.fp8_linear(mh, *P0["main_proj"], f64)
    return base.rms_norm(x, P0["main_norm"], cfg["norm_eps"], f64)


# ---------------------------------------------------------------------------
# DSpark attention pieces (model.py:743-792).
# ---------------------------------------------------------------------------


def stage_main_kv(P, cfg, main_x_row, pos, f64):
    """One stage's KV row for MAIN-model position pos from the stage-0
    projection (model.py:759-761): kv_norm(wkv(main_x)) -> rope @ pos ->
    FP8 QAT on non-rope dims -> bf16. Returns [head_dim]."""
    rd = cfg["rope_head_dim"]
    kv = base.rms_norm(base.fp8_linear(main_x_row[None], *P["wkv"], f64),
                       P["kv_norm"], P["eps"], f64)
    kv = kv.copy()
    kv[..., -rd:] = base.apply_rope(kv[..., -rd:], P["cos"][pos:pos + 1],
                                    P["sin"][pos:pos + 1], False, f64)
    kv[..., :-rd] = base.fp8_qat(kv[..., :-rd], 64, f64)
    return base._B(kv, f64)[0]


def dspark_prefill(Pd, cfg, main_x_all, dstates, f64):
    """DSparkAttention prefill branch (model.py:763-769): build each stage's
    KV window from main_x over the whole prompt; circular write identical to
    Attention prefill. Draft embeddings are computed but discarded there
    (model.py:848-849) — nothing else happens."""
    s = main_x_all.shape[0]
    win = cfg["window_size"]
    rd = cfg["rope_head_dim"]
    for P, st in zip(Pd, dstates):
        kv = base.rms_norm(base.fp8_linear(main_x_all, *P["wkv"], f64),
                           P["kv_norm"], P["eps"], f64)
        kv = kv.copy()
        kv[..., -rd:] = base.apply_rope(kv[..., -rd:], P["cos"][:s],
                                        P["sin"][:s], False, f64)
        kv[..., :-rd] = base.fp8_qat(kv[..., :-rd], 64, f64)
        kv = base._B(kv, f64)
        if s <= win:
            st.win[:s] = kv
        else:
            cut = s % win
            st.win[cut:win] = kv[s - win:s - win + (win - cut)]
            st.win[:cut] = kv[s - cut:]
        st.pos = s


def dspark_catchup(Pd, cfg, writes, dstates, f64):
    """Write stage-window slots for newly-true fed positions.
    writes = [(pos, main_x_row), ...] ascending. See ambiguity D13."""
    for p, row in writes:
        for P, st in zip(Pd, dstates):
            st.win[p % P["window"]] = stage_main_kv(P, cfg, row, p, f64)


def dspark_topk_idxs(win, block_size, start_pos):
    """get_dspark_topk_idxs (model.py:743-747): one row, shared by all B
    queries — window slots 0..min(win,start_pos+1)-1 ++ win+0..win+B-1."""
    row = np.concatenate([np.arange(min(win, start_pos + 1)),
                          win + np.arange(block_size)])
    return np.tile(row, (block_size, 1))


def dspark_attn_decode(P, x, start_pos, st, f64):
    """DSparkAttention decode branch (model.py:771-792). x [B, dim] is the
    stage's attention input for the B draft rows; st.win already holds all
    true positions through start_pos (slot start_pos%win written by the
    caller, matching the in-attention write at model.py:783)."""
    dt = base._dt(f64)
    B = x.shape[0]
    h, d, rd = P["n_heads"], P["head_dim"], P["rope_head_dim"]
    win = P["window"]
    # draft rope positions start_pos+1 .. start_pos+B (model.py:772)
    fc = P["cos"][start_pos + 1:start_pos + 1 + B]
    fs = P["sin"][start_pos + 1:start_pos + 1 + B]

    qr = base.rms_norm(base.fp8_linear(x, *P["wq_a"], f64),
                       P["q_norm"], P["eps"], f64)                    # 774
    q = base.fp8_linear(qr, *P["wq_b"], f64).reshape(B, h, d)         # 775
    qf = q.astype(dt)
    q = base._B(qf * (1.0 / np.sqrt((qf * qf).mean(-1, keepdims=True)
                                    + P["eps"])), f64)                # 776
    q = q.copy()
    q[..., -rd:] = base.apply_rope(q[..., -rd:], fc, fs, False, f64)  # 777

    kv = base.rms_norm(base.fp8_linear(x, *P["wkv"], f64),
                       P["kv_norm"], P["eps"], f64)                   # 778
    kv = kv.copy()
    kv[..., -rd:] = base.apply_rope(kv[..., -rd:], fc, fs, False, f64)  # 779
    kv[..., :-rd] = base.fp8_qat(kv[..., :-rd], 64, f64)              # 780
    kv = base._B(kv, f64)

    idxs = dspark_topk_idxs(win, B, start_pos)                        # 782
    kv_all = np.concatenate([st.win, kv], axis=0)                     # 784
    o = base.sparse_attn(q, kv_all, P["attn_sink"], idxs, d ** -0.5, f64)  # 785
    o = o.copy()
    o[..., -rd:] = base.apply_rope(o[..., -rd:], fc, fs, True, f64)   # 786

    G, o_lora = P["o_groups"], P["o_lora"]
    og = o.reshape(B, G, h * d // G)
    wo_a = P["wo_a"].reshape(G, o_lora, h * d // G)
    y = np.stack([base._B(base._mm(og[:, g, :].astype(dt), wo_a[g].astype(dt).T), f64)
                  for g in range(G)], axis=1)                         # 788-790
    return base.fp8_linear(base._B(y.reshape(B, G * o_lora), f64),
                           *P["wo_b"], f64)                           # 791


def dspark_stage_forward(P, cfg, h, draft_ids, start_pos, st, f64):
    """DSparkBlock decode (model.py:846-847 -> Block.forward 695-707 with
    DSparkAttention). h [B, hc, dim]; returns same."""
    x, post, comb = base.hc_pre(h, P["hc_attn_fn"], P["hc_attn_scale"],
                                P["hc_attn_base"], cfg, f64, None, "")
    x = base.rms_norm(x, P["attn_norm"], cfg["norm_eps"], f64)
    x = dspark_attn_decode(P, x, start_pos, st, f64)
    h = base.hc_post(x, h, post, comb, f64)

    x, post, comb = base.hc_pre(h, P["hc_ffn_fn"], P["hc_ffn_scale"],
                                P["hc_ffn_base"], cfg, f64, None, "")
    x = base.rms_norm(x, P["ffn_norm"], cfg["norm_eps"], f64)
    x = base.moe_forward(P, x, draft_ids, f64, None)   # ids unused (non-hash, D5)
    h = base.hc_post(x, h, post, comb, f64)
    return h


# ---------------------------------------------------------------------------
# Sampling draws: the engine contract (c/sample.h, m5-pinned): greedy argmax
# (temp==0) else f32 softmax + CDF draw with ONE pinned uniform (top_p=1.0 —
# the reference's sample() has no top-p, model.py:939-946; see D2).
# ---------------------------------------------------------------------------


def draw_token(logits, temp, u):
    """Returns (token, margin): greedy margin = top1-top2 gap; sampled
    margin = min |u - cdf boundary| (m5 flip indicator)."""
    if temp == 0:
        lg = np.asarray(logits, np.float64)
        part = np.partition(lg, -2)
        return int(np.argmax(logits)), float(part[-1] - part[-2])
    tok, margin = base.top_p_draw(base.probs_from_logits(logits, temp),
                                  1.0, float(u))
    return tok, margin


# ---------------------------------------------------------------------------
# Draft round: forward_embed + 3 stages + forward_head (model.py:928-936).
# ---------------------------------------------------------------------------


def dspark_draft_round(Pd, top, cfg, anchor, main_x_row, start_pos, dstates,
                       temp, draft_us, f64):
    """One DSpark draft round (forward_spec decode, model.py:929-936).
    Writes each stage's slot start_pos%win (model.py:783) — the caller must
    have catch-up written all interior true positions already. Returns a
    dict with drafts/conf/logits goldens. draft_us: B pinned uniforms
    (sampled mode) or None (greedy)."""
    B = cfg["dspark"]["block_size"]
    noise = cfg["dspark"]["noise_token_id"]
    dt = base._dt(f64)
    # forward_embed (model.py:851-858): [anchor, noise x (B-1)] -> embed
    draft_ids = np.array([anchor] + [noise] * (B - 1), dtype=np.int64)
    h = top["embed"][draft_ids].astype(dt)
    h = np.repeat(h[:, None, :], cfg["hc_mult"], axis=1)              # 857
    stage_h = []
    for P, st in zip(Pd, dstates):
        st.win[start_pos % P["window"]] = stage_main_kv(
            P, cfg, main_x_row, start_pos, f64)                       # 783
        st.pos = start_pos + 1
        h = dspark_stage_forward(P, cfg, h, draft_ids, start_pos, st, f64)
        stage_h.append(h.copy())
    # forward_head (model.py:860-874)
    P2 = Pd[-1]
    x = base.hc_head_collapse(h, P2, cfg, f64)                        # 862
    yn = base.rms_norm(x, P2["norm"], cfg["norm_eps"], f64)           # 863
    logits = base._mm(yn.astype(dt), top["head"].astype(dt).T)                 # 863
    logits_base = logits.copy()
    biases, embeds, margins = [], [], []
    out_ids = [int(anchor)]                                           # 864-865
    for i in range(B):
        e = P2["markov_w1"][out_ids[i]]                               # 802
        bias = base._mm(e.astype(dt), P2["markov_w2"].astype(dt).T)            # 803
        logits[i] = logits[i] + bias                                  # 869
        biases.append(bias)
        embeds.append(e)
        tok, marg = draw_token(logits[i], temp,
                               None if draft_us is None else draft_us[i])
        out_ids.append(tok)                                           # 871
        margins.append(marg)
    # confidence over cat([pre-norm hidden, markov embeds]) (model.py:872-873)
    conf_in = np.concatenate([x, np.stack(embeds).astype(dt)], axis=-1)
    conf = base._mm(conf_in, P2["conf_proj"].astype(dt).T)[:, 0]             # 815
    return {"drafts": out_ids[1:], "conf": conf, "margins": margins,
            "logits_base": logits_base, "markov_bias": np.stack(biases),
            "logits_final": logits, "markov_embed": np.stack(embeds),
            "conf_hidden": x, "stage_h": stage_h, "draft_ids": draft_ids}


# ---------------------------------------------------------------------------
# State snapshot/restore + digests (rollback goldens).
# ---------------------------------------------------------------------------


def snapshot_states(states):
    snap = []
    for st in states:
        d = {"pos": st.pos, "win": st.win.copy()}
        if st.ratio:
            d["comp_cache"] = st.comp.cache.copy()
            d["comp_kv"] = st.comp.kv.copy()
            d["comp_sc"] = st.comp.sc.copy()
            if st.ratio == 4:
                d["idx_cache"] = st.idx_comp.cache.copy()
                d["idx_kv"] = st.idx_comp.kv.copy()
                d["idx_sc"] = st.idx_comp.sc.copy()
        snap.append(d)
    return snap


def restore_states(states, snap):
    for st, d in zip(states, snap):
        st.pos = d["pos"]
        st.win = d["win"].copy()
        if st.ratio:
            st.comp.cache = d["comp_cache"].copy()
            st.comp.kv = d["comp_kv"].copy()
            st.comp.sc = d["comp_sc"].copy()
            if st.ratio == 4:
                st.idx_comp.cache = d["idx_cache"].copy()
                st.idx_comp.kv = d["idx_kv"].copy()
                st.idx_comp.sc = d["idx_sc"].copy()


def _fnv1a(chunks):
    """FNV-1a 64 over a list of byte chunks (the M11b C digest must match)."""
    h = 14695981039346656037
    for c in chunks:
        for b in c:
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def state_digest(states, dstates=None):
    """Digest over native-byte state arrays in canonical order: per main
    layer (pos, win, [comp cache/kv/sc, [idx cache/kv/sc]]), then per DSpark
    stage (pos, win). Same-mode comparison only (dtypes differ f32/f64)."""
    chunks = []
    for st in states:
        chunks.append(np.asarray(st.pos, np.int64).tobytes())
        chunks.append(np.ascontiguousarray(st.win).tobytes())
        if st.ratio:
            chunks.append(np.ascontiguousarray(st.comp.cache).tobytes())
            chunks.append(np.ascontiguousarray(st.comp.kv).tobytes())
            chunks.append(np.ascontiguousarray(st.comp.sc).tobytes())
            if st.ratio == 4:
                chunks.append(np.ascontiguousarray(st.idx_comp.cache).tobytes())
                chunks.append(np.ascontiguousarray(st.idx_comp.kv).tobytes())
                chunks.append(np.ascontiguousarray(st.idx_comp.sc).tobytes())
    if dstates:
        for st in dstates:
            chunks.append(np.asarray(st.pos, np.int64).tobytes())
            chunks.append(np.ascontiguousarray(st.win).tobytes())
    return _fnv1a(chunks)


# ---------------------------------------------------------------------------
# Episode simulators (incremental state, snapshot/restore, the accept rule).
# ---------------------------------------------------------------------------


def decode_batch(Ps, top, cfg, tokens, start_pos, states, f64):
    """Feed tokens at positions start_pos.. with per-token interleaved decode
    steps — the oracle model of the C engine's bitwise chunk-invariant
    batched forward (M8 c/attn.h rule). Returns (rows [n,V], hiddens
    [n, 3*dim]). Does NOT touch st.pos (caller's bookkeeping)."""
    rows, Hs = [], []
    for i, t in enumerate(tokens):
        lg, mh = main_forward_dspark(Ps, top, cfg, [int(t)], states,
                                     start_pos + i, f64)
        rows.append(lg[0])
        Hs.append(mh[0])
    return rows, Hs


def nonspec_episode(Ps, top, cfg, prompt_ids, n_tokens, temp, uniforms, f64):
    """Plain decode, one uniform per emitted token in position order."""
    states = base.new_model_states(cfg, f64)
    logits, _ = main_forward_dspark(Ps, top, cfg, prompt_ids, states, 0, f64)
    s = len(prompt_ids)
    for st in states:
        st.pos = s
    held, _ = draw_token(logits[-1], temp,
                         None if temp == 0 else uniforms[s])
    out = []
    while len(out) < n_tokens:
        out.append(held)
        pos = s + len(out) - 1                    # position of held
        lg, _ = main_forward_dspark(Ps, top, cfg, [held], states, pos, f64)
        for st in states:
            st.pos = pos + 1
        held, _ = draw_token(lg[0], temp,
                             None if temp == 0 else uniforms[pos + 1])
    return out, states


def spec_episode(Ps, top, Pd, cfg, prompt_ids, n_rounds, temp, uniforms,
                 draft_uniforms, f64, draft_override=None):
    """DSpark draft/verify episode with real incremental state. Returns
    {"emitted", "rounds", "states", "dstates"}. draft_override(r, f) ->
    list of B drafts bypasses the DSpark forward (forced patterns; stage
    window maintenance still runs, so state digests stay meaningful)."""
    B = cfg["dspark"]["block_size"]
    states = base.new_model_states(cfg, f64)
    dstates = new_dspark_states(cfg, f64)
    logits, mh_all = main_forward_dspark(Ps, top, cfg, prompt_ids, states,
                                         0, f64)
    s = len(prompt_ids)
    for st in states:
        st.pos = s
    main_x_all = stage0_project(Pd[0], cfg, mh_all, f64)
    dspark_prefill(Pd, cfg, main_x_all, dstates, f64)
    f = s - 1                                     # last true fed position
    held, _ = draw_token(logits[-1], temp,
                         None if temp == 0 else uniforms[s])
    mh_last = mh_all[-1]
    emitted, rounds = [], []
    du = 0
    for r in range(n_rounds):
        main_x_row = stage0_project(Pd[0], cfg, mh_last, f64)
        if draft_override is None:
            us = None
            if temp != 0:
                us = draft_uniforms[du:du + B]
                du += B
            dr = dspark_draft_round(Pd, top, cfg, held, main_x_row, f,
                                    dstates, temp, us, f64)
            drafts, conf, dmargins = dr["drafts"], dr["conf"], dr["margins"]
        else:
            for P, st in zip(Pd, dstates):       # slot f write still happens
                st.win[f % P["window"]] = stage_main_kv(P, cfg, main_x_row,
                                                        f, f64)
                st.pos = f + 1
            drafts = [int(t) for t in draft_override(r, f)]
            conf, dmargins = None, None
        snap = snapshot_states(states)
        batch = [held] + drafts
        rows, Hs = decode_batch(Ps, top, cfg, batch, f + 1, states, f64)
        for st in states:
            st.pos = f + 1 + len(batch)
        # --- the walk (THE ACCEPT RULE) ---
        out = [held]
        margins = []          # one margin per main draw this round (a+1 total)
        a = 0
        newheld = None
        for j in range(1, B + 1):
            m, mg = draw_token(rows[j - 1], temp,
                               None if temp == 0 else uniforms[f + 1 + j])
            margins.append(mg)
            if drafts[j - 1] == m:
                out.append(m)
                a = j
            else:
                newheld = m                       # held, NOT emitted
                break
        bonus = a == B
        if bonus:
            newheld, bmg = draw_token(rows[B], temp,
                                      None if temp == 0 else uniforms[f + B + 2])
            margins.append(bmg)
        # --- state fixup ---
        if bonus:
            f_new, mh_new = f + B + 1, Hs[B]
        else:
            restore_states(states, snap)
            decode_batch(Ps, top, cfg, batch[:a + 1], f + 1, states, f64)
            f_new, mh_new = f + 1 + a, Hs[a]
        # stage-window maintenance (D13): write slots for ALL newly-true fed
        # positions f+1..f_new, so each stage window always holds exactly the
        # true positions 0..f_new. The next round's in-attention slot write
        # (model.py:783) rewrites slot f_new with the identical value.
        dspark_catchup(Pd, cfg,
                       [(f + 1 + j, stage0_project(Pd[0], cfg, Hs[j], f64))
                        for j in range(a + 1)], dstates, f64)
        for st in states:
            st.pos = f_new + 1
        rounds.append({"round": r, "f": f, "anchor": held, "drafts": drafts,
                       "conf": conf, "accepted": a, "bonus": bonus,
                       "emitted": out, "margins": margins,
                       "draft_margins": dmargins})
        emitted.extend(out)
        held, mh_last, f = newheld, mh_new, f_new
    return {"emitted": emitted, "rounds": rounds, "states": states,
            "dstates": dstates}


# ---------------------------------------------------------------------------
# Fixture driver.
# ---------------------------------------------------------------------------


def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(arr))


def _gen_ids(cfg, off, s):
    return base._gen_ids(cfg, MASTER_SEED + off, s)


def _uniforms():
    return np.random.default_rng(SAMPLE_SEED).random(N_UNIFORMS)


def _draft_uniforms(n):
    return np.random.default_rng(DRAFT_SEED).random(n)


def _dump_episode(d, ep, ep64, nonspec, nonspec64, digests, sampled,
                  uniforms, s):
    """Shared golden writer for spec episodes."""
    B_rounds = ep["rounds"]
    _save(os.path.join(d, "tokens_spec_f32.npy"),
          np.asarray(ep["emitted"], np.int64))
    _save(os.path.join(d, "tokens_nonspec_f32.npy"),
          np.asarray(nonspec[0], np.int64))
    _save(os.path.join(d, "round_f.npy"),
          np.asarray([r["f"] for r in B_rounds], np.int64))
    _save(os.path.join(d, "anchors.npy"),
          np.asarray([r["anchor"] for r in B_rounds], np.int64))
    _save(os.path.join(d, "drafts_f32.npy"),
          np.asarray([r["drafts"] for r in B_rounds], np.int64))
    _save(os.path.join(d, "accepted.npy"),
          np.asarray([r["accepted"] for r in B_rounds], np.int64))
    _save(os.path.join(d, "bonus.npy"),
          np.asarray([r["bonus"] for r in B_rounds], bool))
    _save(os.path.join(d, "emitted_counts.npy"),
          np.asarray([len(r["emitted"]) for r in B_rounds], np.int64))
    _save(os.path.join(d, "margins.npy"),
          np.asarray([m for r in B_rounds for m in r["margins"]], np.float64))
    if ep["rounds"][0]["conf"] is not None:
        _save(os.path.join(d, "confidence_f32.npy"),
              np.stack([r["conf"] for r in B_rounds]).astype(np.float32))
    if ep64 is not None:
        _save(os.path.join(d, "tokens_spec_f64.npy"),
              np.asarray(ep64["emitted"], np.int64))
        _save(os.path.join(d, "tokens_nonspec_f64.npy"),
              np.asarray(nonspec64[0], np.int64))
        _save(os.path.join(d, "drafts_f64.npy"),
              np.asarray([r["drafts"] for r in ep64["rounds"]], np.int64))
        _save(os.path.join(d, "accepted_f64.npy"),
              np.asarray([r["accepted"] for r in ep64["rounds"]], np.int64))
        if ep64["rounds"][0]["conf"] is not None:
            _save(os.path.join(d, "confidence_f64.npy"),
                  np.stack([r["conf"] for r in ep64["rounds"]]))
    if sampled:
        E = len(ep["emitted"])
        _save(os.path.join(d, "uniforms_main.npy"),
              uniforms[s:s + E + 8])
        _save(os.path.join(d, "uniforms_draft.npy"),
              _draft_uniforms(len(B_rounds) * ep["rounds"][0]["conf"].shape[0]))
        _save(os.path.join(d, "draft_margins.npy"),
              np.asarray([m for r in B_rounds for m in r["draft_margins"]],
                         np.float64))
    with open(os.path.join(d, "digests.json"), "w") as fp:
        json.dump(digests, fp, indent=1)


def generate(fixtures_dir, clean=True, seed=MASTER_SEED):
    """Generate the full M11a fixture set (weights + DSpark goldens)."""
    cfg = dict(M11A_CFG)
    if clean and os.path.isdir(fixtures_dir):
        import shutil
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    weights_dir = os.path.join(fixtures_dir, "weights")
    write_m11a_weights(cfg, weights_dir)
    shards = base.ShardSet(weights_dir)
    Ps32, top32 = base.load_model_params(shards, cfg, False)
    Ps64, top64 = base.load_model_params(shards, cfg, True)
    Pd32 = load_dspark_params(shards, cfg, False)
    Pd64 = load_dspark_params(shards, cfg, True)
    U = _uniforms()
    gdir = os.path.join(fixtures_dir, "golden")
    manifest = {"seed": seed, "config": "config.json", "weights": "weights/",
                "sampling": {"main_rng": "numpy PCG64, one f64 uniform per "
                                          "emitted token, indexed by absolute "
                                          "position",
                             "draft_rng": "numpy PCG64, one f64 uniform per "
                                          "draft token, round*block+i order",
                             "seed": SAMPLE_SEED, "draft_seed": DRAFT_SEED,
                             "temp": SAMPLE_TEMP, "top_p": 1.0},
                "sequences": {}}
    div_rows = []

    def div(tag, a32, a64):
        a32 = np.asarray(a32, np.float64)
        a64 = np.asarray(a64, np.float64)
        d = np.abs(a32 - a64)
        row = (tag, float(d.max()),
               float(d.max() / max(np.abs(a64).max(), 1e-30)),
               float(np.mean(np.argmax(a32, -1) != np.argmax(a64, -1))))
        div_rows.append(row)
        return row

    def run_main(Ps, top, ids, f64):
        states = base.new_model_states(cfg, f64)
        lg, mh = main_forward_dspark(Ps, top, cfg, ids, states, 0, f64)
        for st in states:
            st.pos = len(ids)
        return lg, mh, states

    # --- join_prefill_len24 / join_prefill_len140: target-layer joining +
    #     stage-0 projection + per-stage prefill window build ---
    prompts = {}
    for tag, off, plen in (("join_prefill_len24", GREEDY_PROMPT_SEED_OFF, 24),
                           ("join_prefill_len140", LONG_PROMPT_SEED_OFF, 140)):
        ids = _gen_ids(cfg, off, plen)
        prompts[tag] = ids
        lg32, mh32, _ = run_main(Ps32, top32, ids, False)
        lg64, mh64, _ = run_main(Ps64, top64, ids, True)
        mx32 = stage0_project(Pd32[0], cfg, mh32, False)
        mx64 = stage0_project(Pd64[0], cfg, mh64, True)
        ds32, ds64 = new_dspark_states(cfg, False), new_dspark_states(cfg, True)
        dspark_prefill(Pd32, cfg, mx32, ds32, False)
        dspark_prefill(Pd64, cfg, mx64, ds64, True)
        d = os.path.join(gdir, tag)
        _save(os.path.join(d, "input_ids.npy"), ids)
        _save(os.path.join(d, "main_hidden_f32.npy"), mh32.astype(np.float32))
        _save(os.path.join(d, "main_hidden_f64.npy"), mh64)
        _save(os.path.join(d, "main_x_f32.npy"), mx32.astype(np.float32))
        _save(os.path.join(d, "main_x_f64.npy"), mx64)
        for k in range(cfg["dspark"]["n_mtp_layers"]):
            _save(os.path.join(d, f"stage{k}_win_f32.npy"),
                  ds32[k].win.astype(np.float32))
            _save(os.path.join(d, f"stage{k}_win_f64.npy"),
                  ds64[k].win)
        div(tag + "/main_x", mx32, mx64)
        manifest["sequences"][tag] = {"kind": "join_prefill", "len": plen}
        print(f"  {tag} done")

    # --- draft_rounds_len24: 3 teacher-forced draft rounds (partial-window
    #     branch), full per-round draft goldens ---
    ids = prompts["join_prefill_len24"]
    rounds_meta = []
    for mode, Ps, top, Pd, f64 in (("f32", Ps32, top32, Pd32, False),
                                   ("f64", Ps64, top64, Pd64, True)):
        logits, mh, states = run_main(Ps, top, ids, f64)
        dstates = new_dspark_states(cfg, f64)
        dspark_prefill(Pd, cfg, stage0_project(Pd[0], cfg, mh, f64), dstates,
                       f64)
        f = len(ids) - 1
        anchor, _ = draw_token(logits[-1], 0, None)
        mh_last = mh[-1]
        for r in range(3):
            main_x_row = stage0_project(Pd[0], cfg, mh_last, f64)
            dr = dspark_draft_round(Pd, top, cfg, anchor, main_x_row, f,
                                    dstates, 0, None, f64)
            if mode == "f32" or r == 0:
                d = os.path.join(gdir, "draft_rounds_len24", f"round{r}")
                suf = "_" + mode
                _save(os.path.join(d, "prompt_ids.npy"), ids)
                _save(os.path.join(d, "anchor" + suf + ".npy"),
                      np.asarray(anchor, np.int64))
                _save(os.path.join(d, "start_pos" + suf + ".npy"),
                      np.asarray(f, np.int64))
                _save(os.path.join(d, "drafts" + suf + ".npy"),
                      np.asarray(dr["drafts"], np.int64))
                _save(os.path.join(d, "margins" + suf + ".npy"),
                      np.asarray(dr["margins"], np.float64))
                _save(os.path.join(d, "confidence" + suf + ".npy"), dr["conf"])
                _save(os.path.join(d, "logits_base" + suf + ".npy"),
                      dr["logits_base"])
                _save(os.path.join(d, "markov_bias" + suf + ".npy"),
                      dr["markov_bias"])
                _save(os.path.join(d, "logits_final" + suf + ".npy"),
                      dr["logits_final"])
                _save(os.path.join(d, "markov_embed" + suf + ".npy"),
                      dr["markov_embed"])
                _save(os.path.join(d, "conf_hidden" + suf + ".npy"),
                      dr["conf_hidden"])
                for k, sh in enumerate(dr["stage_h"]):
                    _save(os.path.join(d, f"stage{k}_h" + suf + ".npy"), sh)
            if mode == "f32":
                rounds_meta.append({
                    "round": r, "start_pos": f, "anchor": anchor,
                    "stage_win_digest": state_digest([], dstates)})
            # advance truth: feed the anchor through the main model
            rows, Hs = decode_batch(Ps, top, cfg, [anchor], f + 1, states, f64)
            for st in states:
                st.pos = f + 2
            anchor, _ = draw_token(rows[0], 0, None)
            mh_last = Hs[0]
            f = f + 1
    div("draft_rounds/logits_final",
        np.load(os.path.join(gdir, "draft_rounds_len24", "round0",
                             "logits_final_f32.npy")),
        np.load(os.path.join(gdir, "draft_rounds_len24", "round0",
                             "logits_final_f64.npy")))
    div("draft_rounds/confidence",
        np.load(os.path.join(gdir, "draft_rounds_len24", "round0",
                             "confidence_f32.npy")),
        np.load(os.path.join(gdir, "draft_rounds_len24", "round0",
                             "confidence_f64.npy")))
    with open(os.path.join(gdir, "draft_rounds_len24", "rounds.json"), "w") as fp:
        json.dump(rounds_meta, fp, indent=1)
    manifest["sequences"]["draft_rounds_len24"] = {
        "kind": "draft_rounds", "prompt_len": 24, "rounds": 3}
    print("  draft_rounds_len24 done")

    # --- draft_round_len140: one round, ring-wrap/full-window branch ---
    ids140 = prompts["join_prefill_len140"]
    for mode, Ps, top, Pd, f64 in (("f32", Ps32, top32, Pd32, False),
                                   ("f64", Ps64, top64, Pd64, True)):
        logits, mh, states = run_main(Ps, top, ids140, f64)
        dstates = new_dspark_states(cfg, f64)
        dspark_prefill(Pd, cfg, stage0_project(Pd[0], cfg, mh, f64), dstates,
                       f64)
        f = len(ids140) - 1
        anchor, _ = draw_token(logits[-1], 0, None)
        dr = dspark_draft_round(Pd, top, cfg, anchor,
                                stage0_project(Pd[0], cfg, mh[-1], f64), f,
                                dstates, 0, None, f64)
        d = os.path.join(gdir, "draft_round_len140")
        suf = "_" + mode
        _save(os.path.join(d, "prompt_ids.npy"), ids140)
        _save(os.path.join(d, "anchor" + suf + ".npy"),
              np.asarray(anchor, np.int64))
        _save(os.path.join(d, "start_pos" + suf + ".npy"),
              np.asarray(f, np.int64))
        _save(os.path.join(d, "drafts" + suf + ".npy"),
              np.asarray(dr["drafts"], np.int64))
        _save(os.path.join(d, "confidence" + suf + ".npy"), dr["conf"])
        _save(os.path.join(d, "logits_final" + suf + ".npy"),
              dr["logits_final"])
        _save(os.path.join(d, "conf_hidden" + suf + ".npy"), dr["conf_hidden"])
        for k, sh in enumerate(dr["stage_h"]):
            _save(os.path.join(d, f"stage{k}_h" + suf + ".npy"), sh)
    manifest["sequences"]["draft_round_len140"] = {
        "kind": "draft_round", "prompt_len": 140}
    print("  draft_round_len140 done")

    # --- spec_episode_greedy ---
    ids_g = _gen_ids(cfg, GREEDY_EP_PROMPT_SEED_OFF, 24)
    ep32 = spec_episode(Ps32, top32, Pd32, cfg, ids_g, EPISODE_ROUNDS, 0,
                        None, None, False)
    ep64 = spec_episode(Ps64, top64, Pd64, cfg, ids_g, EPISODE_ROUNDS, 0,
                        None, None, True)
    ns32 = nonspec_episode(Ps32, top32, cfg, ids_g, len(ep32["emitted"]), 0,
                           None, False)
    ns64 = nonspec_episode(Ps64, top64, cfg, ids_g, len(ep64["emitted"]), 0,
                           None, True)
    dig = {"spec_main_f32": state_digest(ep32["states"]),
           "nonspec_main_f32": state_digest(ns32[1]),
           "spec_full_f32": state_digest(ep32["states"], ep32["dstates"]),
           "spec_main_f64": state_digest(ep64["states"]),
           "nonspec_main_f64": state_digest(ns64[1])}
    assert dig["spec_main_f32"] == dig["nonspec_main_f32"], "greedy rollback"
    assert dig["spec_main_f64"] == dig["nonspec_main_f64"], "greedy rollback64"
    assert ep32["emitted"] == ns32[0], "greedy spec != nonspec (f32)"
    assert ep64["emitted"] == ns64[0], "greedy spec != nonspec (f64)"
    d = os.path.join(gdir, "spec_episode_greedy")
    _save(os.path.join(d, "prompt_ids.npy"), ids_g)
    _dump_episode(d, ep32, ep64, ns32, ns64, dig, False, None, len(ids_g))
    manifest["sequences"]["spec_episode_greedy"] = {
        "kind": "spec", "prompt_len": 24, "rounds": EPISODE_ROUNDS,
        "temp": 0, "emitted": len(ep32["emitted"]),
        "accepted": [r["accepted"] for r in ep32["rounds"]]}
    print(f"  spec_episode_greedy done (accepted "
          f"{[r['accepted'] for r in ep32['rounds']]})")

    # --- spec_episode_sampled ---
    ids_s = _gen_ids(cfg, SAMPLED_EP_PROMPT_SEED_OFF, 24)
    du = _draft_uniforms(EPISODE_ROUNDS * cfg["dspark"]["block_size"])
    ep32 = spec_episode(Ps32, top32, Pd32, cfg, ids_s, EPISODE_ROUNDS,
                        SAMPLE_TEMP, U, du, False)
    ep64 = spec_episode(Ps64, top64, Pd64, cfg, ids_s, EPISODE_ROUNDS,
                        SAMPLE_TEMP, U, du, True)
    ns32 = nonspec_episode(Ps32, top32, cfg, ids_s, len(ep32["emitted"]),
                           SAMPLE_TEMP, U, False)
    ns64 = nonspec_episode(Ps64, top64, cfg, ids_s, len(ep64["emitted"]),
                           SAMPLE_TEMP, U, True)
    dig = {"spec_main_f32": state_digest(ep32["states"]),
           "nonspec_main_f32": state_digest(ns32[1]),
           "spec_full_f32": state_digest(ep32["states"], ep32["dstates"]),
           "spec_main_f64": state_digest(ep64["states"]),
           "nonspec_main_f64": state_digest(ns64[1])}
    assert dig["spec_main_f32"] == dig["nonspec_main_f32"], "sampled rollback"
    assert ep32["emitted"] == ns32[0], "sampled spec != nonspec (f32)"
    assert ep64["emitted"] == ns64[0], "sampled spec != nonspec (f64)"
    d = os.path.join(gdir, "spec_episode_sampled")
    _save(os.path.join(d, "prompt_ids.npy"), ids_s)
    _dump_episode(d, ep32, ep64, ns32, ns64, dig, True, U, len(ids_s))
    manifest["sequences"]["spec_episode_sampled"] = {
        "kind": "spec", "prompt_len": 24, "rounds": EPISODE_ROUNDS,
        "temp": SAMPLE_TEMP, "top_p": 1.0, "emitted": len(ep32["emitted"]),
        "accepted": [r["accepted"] for r in ep32["rounds"]]}
    print(f"  spec_episode_sampled done (accepted "
          f"{[r['accepted'] for r in ep32['rounds']]})")

    # --- forced draft patterns (greedy; drafts injected from the true
    #     continuation, acceptance exact by construction) ---
    def true_tok_fn(nonspec_tokens, plen):
        def fn(pos):
            return int(nonspec_tokens[pos - plen])
        return fn

    V = cfg["vocab_size"]
    B = cfg["dspark"]["block_size"]

    def make_override(kind, tt):
        if kind == "forced_all_accept":
            return lambda r, f: [tt(f + 2 + j) for j in range(B)]
        if kind == "forced_all_reject":
            return lambda r, f: [(tt(f + 2 + j) + 1) % V for j in range(B)]
        # mixed: accept drafts 1,2,4,5; reject draft 3 -> accepted == 2
        return lambda r, f: [tt(f + 2 + j) if j != 2 else (tt(f + 2 + j) + 1) % V
                             for j in range(B)]

    for kind, pids, expect in (
            ("forced_all_accept", ids_g, B),
            ("forced_all_reject", ids_g, 0),
            ("forced_mixed", ids140, 2)):
        plen = len(pids)
        ns_true = nonspec_episode(Ps32, top32, cfg, pids,
                                  FORCED_ROUNDS * (B + 2) + 8, 0, None, False)
        tt = true_tok_fn(ns_true[0], plen)
        ep32 = spec_episode(Ps32, top32, Pd32, cfg, pids, FORCED_ROUNDS, 0,
                            None, None, False,
                            draft_override=make_override(kind, tt))
        ns32 = nonspec_episode(Ps32, top32, cfg, pids, len(ep32["emitted"]),
                               0, None, False)
        dig = {"spec_main_f32": state_digest(ep32["states"]),
               "nonspec_main_f32": state_digest(ns32[1]),
               "spec_full_f32": state_digest(ep32["states"], ep32["dstates"])}
        assert dig["spec_main_f32"] == dig["nonspec_main_f32"], kind
        assert ep32["emitted"] == ns32[0], kind + " stream"
        got = [r["accepted"] for r in ep32["rounds"]]
        assert all(g == expect for g in got), (kind, got)
        d = os.path.join(gdir, kind)
        _save(os.path.join(d, "prompt_ids.npy"), pids)
        _dump_episode(d, ep32, None, ns32, None, dig, False, None, plen)
        manifest["sequences"][kind] = {
            "kind": "spec_forced", "prompt_len": plen,
            "rounds": FORCED_ROUNDS, "temp": 0,
            "emitted": len(ep32["emitted"]), "accepted_per_round": expect}
        print(f"  {kind} done (accepted {got})")

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("f32-vs-f64 divergence (self-check):")
    print(f"  {'sequence':<26} {'maxabs':>10} {'rel2scale':>10} {'argmax-flip':>12}")
    for tag, ma, rel, flip in div_rows:
        print(f"  {tag:<26} {ma:10.3g} {rel:10.3g} {flip:11.2%}")
    print(f"fixtures written to {fixtures_dir}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="apus M11a DSpark oracle fixture generator")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "m11a",
                                                  "fixtures"))
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args(argv)
    generate(args.out, clean=not args.no_clean)


if __name__ == "__main__":
    main()
