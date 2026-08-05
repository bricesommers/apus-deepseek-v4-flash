#!/usr/bin/env python3
"""M4b oracle self-consistency checks.

Usage: .venv/bin/python tests/m4b/check_oracle.py

Verifies, against tests/m4b/fixtures (regenerate with run_oracle.py):

  1. Container sanity — index/shard consistency, real naming scheme, dtypes.
  2. Determinism — re-running the oracle reproduces the saved goldens
     bitwise (f32 mode), including decode-step replay from the serialized
     carried state (proves the state files are sufficient to continue).
  3. f32-vs-f64 divergence per pipeline stage (report + early-stage assert).
  4. Sinkhorn comb is doubly stochastic.
  5. Indexer top-k causal legality (selected blocks are fully completed
     before the query position).
  6. Router weights sum to route_scale (1.5); selected experts match the
     biased-score top-k recomputed from saved scores.
  7. Chunk invariance of the compressor/window state machine: one-shot
     prefill vs prefill+single-token decodes yields (near-)identical
     caches, states and outputs. This is the property the C port most
     easily breaks.

Exit code 0 iff all checks pass.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

import oracle
import stutil

FIX = os.path.join(ROOT, "tests", "m4b", "fixtures")
GOLD = os.path.join(FIX, "golden")

FAILURES = []


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def load_cfg():
    with open(os.path.join(FIX, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. container sanity
# ---------------------------------------------------------------------------

def check_container(cfg):
    print("== container sanity ==")
    wdir = os.path.join(FIX, "weights")
    with open(os.path.join(wdir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    total = 0
    n = 0
    for shard in sorted(set(wmap.values())):
        header, _ = stutil.read_shard(os.path.join(wdir, shard))
        for name, meta in header.items():
            total += meta["data_offsets"][1] - meta["data_offsets"][0]
            n += 1
            check(f"index contains {name}", wmap.get(name) == shard, "", ) if wmap.get(name) != shard else None
    check("all shard tensors listed in index", True, f"{n} tensors")
    # spot-check the real naming/format scheme on layer 1 (csa)
    shards = oracle.ShardSet(wdir)
    pk, sc = shards.fp4("layers.1.ffn.experts.0.w1")
    check("expert w1 packed shape", pk.shape == (cfg["moe_inter_dim"], cfg["dim"] // 2), str(pk.shape))
    check("expert w1 scale shape", sc.shape == (cfg["moe_inter_dim"], cfg["dim"] // 32), str(sc.shape))
    codes, wsc = shards.fp8("layers.1.attn.wq_a")
    check("wq_a fp8 shapes", codes.shape == (cfg["q_lora_rank"], cfg["dim"])
          and wsc.shape == (cfg["q_lora_rank"] // 128, cfg["dim"] // 128),
          f"{codes.shape} {wsc.shape}")
    t = shards.i64("layers.0.ffn.gate.tid2eid")
    check("tid2eid i64 + rows unique", t.shape == (cfg["vocab_size"], cfg["n_activated_experts"])
          and all(len(set(r)) == len(r) for r in t), str(t.shape))
    check("total payload", total > 0, f"{total / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# 2. determinism + state replay
# ---------------------------------------------------------------------------

def check_determinism(cfg):
    print("== determinism / state replay (f32, bitwise) ==")
    shards = oracle.ShardSet(os.path.join(FIX, "weights"))
    for li, layer in enumerate(cfg["layers"]):
        P = oracle.load_layer_params(shards, cfg, layer, li, f64=False)
        # long prefill rerun
        long_seq = {"swa": "prefill_len140", "csa": "prefill_len199",
                    "hca": "prefill_len250"}[layer["name"]]
        d = os.path.join(GOLD, layer["name"], long_seq)
        h = np.load(os.path.join(d, "input_h.npy"))
        ids = np.load(os.path.join(d, "input_ids.npy"))
        out, _, st = oracle.run_prefill(P, cfg, layer, h, ids, False)
        saved = np.load(os.path.join(d, "out_h_f32.npy"))
        check(f"{layer['name']} prefill bitwise", np.array_equal(out, saved),
              f"maxabs={np.abs(out - saved).max():.2e}")
        # decode step00 replay from its serialized state_in
        dec = {"swa": "decode_from140", "csa": "decode_from199",
               "hca": "decode_from250"}[layer["name"]]
        sd = os.path.join(GOLD, layer["name"], dec, "step00")
        st_in = {k[:-4]: np.load(os.path.join(sd, "state_in", k))
                 for k in os.listdir(os.path.join(sd, "state_in"))}
        st = oracle.load_state_arrays(cfg, layer, st_in, False)
        h1 = np.load(os.path.join(sd, "input_h.npy"))
        id1 = np.load(os.path.join(sd, "input_ids.npy"))
        out1, _ = oracle.run_decode_step(P, cfg, layer, h1, id1, st, False)
        saved1 = np.load(os.path.join(sd, "out_h_f32.npy"))
        check(f"{layer['name']} decode step00 replay bitwise",
              np.array_equal(out1, saved1),
              f"maxabs={np.abs(out1 - saved1).max():.2e}")
        for k in os.listdir(os.path.join(sd, "state_out")):
            a = oracle.state_arrays(st)[k[:-4]]
            b = np.load(os.path.join(sd, "state_out", k))
            if not np.array_equal(a, b):
                check(f"{layer['name']} state_out[{k}] bitwise", False,
                      f"maxabs={np.abs(a - b).max():.2e}")
        check(f"{layer['name']} state_out bitwise", True)


# ---------------------------------------------------------------------------
# 3. f32 vs f64 divergence per stage
# ---------------------------------------------------------------------------

def check_divergence(cfg):
    print("== f32 vs f64 divergence per stage (prefill, long sequences) ==")
    print(f"  {'stage':24s} {'layer':5s} {'scale':>9s} {'maxabs':>9s} "
          f"{'p99abs':>9s} {'medabs':>9s} {'sel_mismatch':>12s}")
    worst_early = 0.0
    for lname, seq in (("swa", "prefill_len140"), ("csa", "prefill_len199"),
                       ("hca", "prefill_len250")):
        d = os.path.join(GOLD, lname, seq, "interm")
        keys = sorted(set(f[:-8] for f in os.listdir(d) if f.endswith("_f32.npy")))
        for k in keys:
            a = np.load(os.path.join(d, k + "_f32.npy"))
            b = np.load(os.path.join(d, k + "_f64.npy"))
            if a.dtype.kind in "iu":
                mm = (a != b).mean()
                print(f"  {k:24s} {lname:5s} {'':>9s} {'':>9s} {'':>9s} "
                      f"{'':>9s} {mm:12.4f}")
                continue
            dd = np.abs(a - b)
            p99 = np.percentile(dd, 99)
            med = np.median(dd)
            print(f"  {k:24s} {lname:5s} {np.abs(b).max():9.3f} {dd.max():9.2e} "
                  f"{p99:9.2e} {med:9.2e} {'':>12s}")
            if k in ("attn_hc_pre", "attn_hc_post", "attn_hc_comb"):
                worst_early = max(worst_early, p99)
        a = np.load(os.path.join(GOLD, lname, seq, "out_h_f32.npy"))
        b = np.load(os.path.join(GOLD, lname, seq, "out_h_f64.npy"))
        dd = np.abs(a - b)
        print(f"  {'out_h':24s} {lname:5s} {np.abs(b).max():9.3f} {dd.max():9.2e} "
              f"{np.percentile(dd, 99):9.2e} {np.median(dd):9.2e}")
    check("first-stage (mHC mixes) p99 divergence < 1e-4", worst_early < 1e-4,
          f"p99={worst_early:.2e}")
    print("  note: downstream stages include QAT scale flips and discrete "
          "top-k/router selection changes between modes; see README §divergence.")


# ---------------------------------------------------------------------------
# 4. sinkhorn doubly stochastic
# ---------------------------------------------------------------------------

def check_sinkhorn(cfg):
    print("== sinkhorn doubly stochastic ==")
    worst_r = worst_c = 0.0
    for lname, seq in (("csa", "prefill_len199"), ("hca", "prefill_len250"),
                       ("swa", "prefill_len140")):
        for mode in ("f32", "f64"):
            c = np.load(os.path.join(GOLD, lname, seq, "interm",
                                     f"attn_hc_comb_{mode}.npy"))
            worst_r = max(worst_r, np.abs(c.sum(-1) - 1).max())
            worst_c = max(worst_c, np.abs(c.sum(-2) - 1).max())
    check("comb row sums ~ 1", worst_r < 1e-3, f"max err {worst_r:.2e}")
    check("comb col sums ~ 1", worst_c < 1e-4, f"max err {worst_c:.2e}")


# ---------------------------------------------------------------------------
# 5. indexer causal legality
# ---------------------------------------------------------------------------

def check_indexer_legality(cfg):
    print("== indexer top-k causal legality (csa) ==")
    ratio, win = 4, cfg["window_size"]
    # prefill len199: offset = seqlen
    s = 199
    idx = np.load(os.path.join(GOLD, "csa", "prefill_len199",
                               "interm", "idx_topk_f32.npy"))
    bad = 0
    for t in range(s):
        for v in idx[t]:
            if v < 0:
                continue
            blk = v - s
            if not (0 <= blk < (t + 1) // ratio):
                bad += 1
    check("prefill selections only fully-completed earlier blocks", bad == 0,
          f"{bad} illegal")
    # decode steps: offset = win; token position = prefill_len + step
    bad = 0
    for k in range(12):
        pos = 199 + k
        idx = np.load(os.path.join(GOLD, "csa", "decode_from199",
                                   f"step{k:02d}", "interm", "idx_topk_f32.npy"))
        for v in idx[0]:
            if v < 0:
                continue
            blk = v - win
            if not (0 <= blk and (blk + 1) * ratio - 1 <= pos):
                bad += 1
    check("decode selections only completed blocks", bad == 0, f"{bad} illegal")
    # also: number of available blocks grows at the right steps
    nb0 = np.load(os.path.join(GOLD, "csa", "decode_from199", "step00",
                               "state_in", "idx_kv.npy")).shape[0]
    nb11 = np.load(os.path.join(GOLD, "csa", "decode_from199", "step11",
                                "state_out", "idx_kv.npy")).shape[0]
    check("indexer cache grows on block completion",
          nb0 == 49 and nb11 == 52, f"{nb0} -> {nb11}")


# ---------------------------------------------------------------------------
# 6. router
# ---------------------------------------------------------------------------

def check_router(cfg):
    print("== router semantics ==")
    worst = 0.0
    for lname, seq in (("swa", "prefill_len140"), ("csa", "prefill_len199"),
                       ("hca", "prefill_len250")):
        for mode, tol in (("f32", 1e-4), ("f64", 1e-9)):
            w = np.load(os.path.join(GOLD, lname, seq, "interm",
                                     f"router_w_{mode}.npy"))
            err = np.abs(w.sum(-1) - cfg["route_scale"]).max()
            worst = max(worst, err) if mode == "f32" else worst
            check(f"{lname} {mode} weights sum to 1.5", err < tol,
                  f"max err {err:.2e}")
    # selection consistency: idx == topk of biased scores (non-hash layers)
    for lname, seq in (("csa", "prefill_len199"), ("hca", "prefill_len250")):
        sb = np.load(os.path.join(GOLD, lname, seq, "interm",
                                  "router_scores_biased_f32.npy"))
        idx = np.load(os.path.join(GOLD, lname, seq, "interm",
                                   "router_idx_f32.npy"))
        recomputed = oracle.topk_stable(sb, cfg["n_activated_experts"])
        check(f"{lname} idx == topk(biased)", np.array_equal(idx, recomputed))
    # hash layer: idx == tid2eid[ids]
    ids = np.load(os.path.join(GOLD, "swa", "prefill_len140", "input_ids.npy"))
    idx = np.load(os.path.join(GOLD, "swa", "prefill_len140", "interm",
                               "router_idx_f32.npy"))
    shards = oracle.ShardSet(os.path.join(FIX, "weights"))
    t = shards.i64("layers.0.ffn.gate.tid2eid")
    check("swa hash idx == tid2eid[ids]", np.array_equal(idx, t[ids]))


# ---------------------------------------------------------------------------
# 7. chunk invariance
# ---------------------------------------------------------------------------

def _code_step_ratio(a, b, mult=2.0, floor=0.5):
    """max |a-b| / (mult bf16 ulps of b), ulp = 2^-8 * max(|b|, floor).
    Ratio <= 1 means every element matches within `mult` bf16 code steps
    (robust around zero, unlike raw code-index differences)."""
    tol = mult * 0.0078125 * np.maximum(np.abs(b), floor)
    return float((np.abs(a - b) / tol).max())


def _live_slots(ratio, overlap, pos):
    """Which compressor state slots are semantically live after `pos` tokens.
    Slots beyond the current partial block hold stale/never-written values
    (reference never clears them) and are excluded from comparison."""
    n = (2 if overlap else 1) * ratio
    m = np.zeros(n, dtype=bool)
    if overlap:
        m[:ratio] = True                      # overlap window (previous block)
        m[ratio:ratio + pos % ratio] = True   # current partial block
    else:
        m[:pos % ratio] = True
    return m


def _comp_blocks(idx_row, offset):
    """Compressed-block numbers selected by a query (offset = window part)."""
    v = idx_row[idx_row >= offset]
    return np.sort(v - offset)


def _run_chunked_compare(P, cfg, layer, h, ids, split, f64):
    """One-shot vs prefill(split)+decode comparison. Returns dict of metrics."""
    ratio = layer["compress_ratio"]
    out1, i1, st1 = oracle.run_prefill(P, cfg, layer, h, ids, f64)
    st2 = oracle.LayerState(cfg, layer, f64)
    outs = [oracle.block_forward(P, cfg, h[:split], ids[:split], 0, st2, f64)[0]]
    st2.pos = split
    s = h.shape[0]
    router_flip, block_flip = [], []
    for t in range(split, s):
        o, i2 = oracle.block_forward(P, cfg, h[t:t + 1], ids[t:t + 1],
                                     st2.pos, st2, f64)
        st2.pos += 1
        outs.append(o)
        router_flip.append(not np.array_equal(i1["router_idx"][t],
                                              i2["router_idx"][0]))
        if ratio == 4:
            # index VALUES differ by design (prefill offset = s, decode
            # offset = win); compare selected compressed-block SETS.
            b1 = _comp_blocks(i1["idx_topk"][t], s)
            b2 = _comp_blocks(i2["idx_topk"][0], cfg["window_size"])
            block_flip.append(not np.array_equal(b1, b2))
    out2 = np.concatenate(outs)
    router_flip = np.array(router_flip)
    block_flip = (np.array(block_flip) if ratio == 4
                  else np.zeros(s - split, bool))
    sel_ok = ~(router_flip | block_flip)

    m = {"router_flip_rate": float(router_flip.mean()),
         "block_flip_rate": float(block_flip.mean()) if ratio == 4 else 0.0,
         "win_ratio_max": _code_step_ratio(st2.win, st1.win, mult=1.0),
         "win_flip_frac": float((np.abs(st1.win - st2.win) > 1e-6).mean()),
         "win_abs": float(np.abs(st1.win - st2.win).max()),
         "out_abs": float(np.abs(out1 - out2).max())}
    if sel_ok.any():
        a = out1[split:][sel_ok]
        b = out2[split:][sel_ok]
        tol = 2.0 * 0.0078125 * np.maximum(np.abs(a), 0.5)
        r = np.abs(a - b) / tol
        m["out_p99_ratio"] = float(np.percentile(r, 99))
        m["out_max_ratio"] = float(r.max())
    else:
        m["out_p99_ratio"] = m["out_max_ratio"] = np.inf
    if ratio:
        overlap = ratio == 4
        lm = _live_slots(ratio, overlap, st1.pos)
        dc = np.abs(st1.comp.cache - st2.comp.cache)
        m["comp_abs"] = float(dc.max())
        m["comp_flip_frac"] = float((dc > 1e-6).mean())
        m["kv_state_abs"] = float(np.abs(st1.comp.kv[lm] - st2.comp.kv[lm]).max())
        fin = np.isfinite(st1.comp.sc[lm]) & np.isfinite(st2.comp.sc[lm])
        m["sc_state_finite_ok"] = bool(
            (np.isfinite(st1.comp.sc[lm]) == np.isfinite(st2.comp.sc[lm])).all())
        m["sc_state_abs"] = float(
            np.abs(st1.comp.sc[lm] - st2.comp.sc[lm])[fin].max()
            if fin.any() else 0.0)
        if ratio == 4:
            di = np.abs(st1.idx_comp.cache - st2.idx_comp.cache)
            m["idx_abs"] = float(di.max())
            m["idx_flip_frac"] = float((di > 1e-6).mean())
    return m


def check_chunk_invariance(cfg):
    print("== chunk invariance of the state machine ==")
    shards = oracle.ShardSet(os.path.join(FIX, "weights"))
    cases = [("swa", "prefill_len140", 60, 0), ("csa", "prefill_len199", 99, 1),
             ("hca", "prefill_len250", 130, 2)]
    for lname, seq, split, li in cases:
        layer = cfg["layers"][li]
        ratio = layer["compress_ratio"]
        d = os.path.join(GOLD, lname, seq)
        h = np.load(os.path.join(d, "input_h.npy"))
        ids = np.load(os.path.join(d, "input_ids.npy"))

        # --- f64: the proof. No bf16/QAT boundary sensitivity, reorder noise
        # ~1e-14. The state machine must be EXACTLY chunk-invariant.
        P64 = oracle.load_layer_params(shards, cfg, layer, li, f64=True)
        m = _run_chunked_compare(P64, cfg, layer, h, ids, split, True)
        ok = (m["out_abs"] < 1e-9 and m["win_abs"] == 0
              and m["router_flip_rate"] == 0 and m["block_flip_rate"] == 0)
        det = [f"out {m['out_abs']:.2e}", f"win {m['win_abs']:.2e}"]
        if ratio:
            det += [f"comp {m['comp_abs']:.2e}"]
            ok = ok and m["comp_abs"] == 0 and m["kv_state_abs"] < 1e-9 \
                and m["sc_state_finite_ok"] and m["sc_state_abs"] < 1e-9
            if ratio == 4:
                det += [f"idx {m['idx_abs']:.2e}"]
                ok = ok and m["idx_abs"] == 0
        check(f"{lname} f64 exact state-machine invariance", ok, ", ".join(det))

        # --- f32: same property at bf16/fp8 storage granularity. Raw f32
        # matmul reorder noise (~1e-5) can flip ONE bf16/E4M3 code when an
        # amax sits at a binade boundary; a flipped compressed-cache code
        # then cascades into every query that selects that block. Assert:
        # caches >= 99.8% bitwise, cache diffs bounded by one quant step,
        # selection flips <= 2%, and the output p99 within 16x2 bf16 ulps
        # on selection-matching tokens (max reported, not asserted).
        P32 = oracle.load_layer_params(shards, cfg, layer, li, f64=False)
        m = _run_chunked_compare(P32, cfg, layer, h, ids, split, False)
        ok = (m["win_flip_frac"] <= 0.002
              and m["out_p99_ratio"] <= 16
              and m["router_flip_rate"] <= 0.02 and m["block_flip_rate"] <= 0.02)
        det = [f"out_p99 {m['out_p99_ratio']:.1f} (max {m['out_max_ratio']:.1f})",
               f"win_flips {m['win_flip_frac']:.5f}",
               f"router_flips {m['router_flip_rate']:.3f}"]
        if ratio:
            det += [f"comp {m['comp_abs']:.2e}/{m['comp_flip_frac']:.4f}",
                    f"kv_state {m['kv_state_abs']:.2e}"]
            ok = ok and m["comp_abs"] <= 0.25 and m["comp_flip_frac"] <= 0.01 \
                and m["kv_state_abs"] < 1e-3 and m["sc_state_finite_ok"] \
                and m["sc_state_abs"] < 1e-3
            if ratio == 4:
                det += [f"idx {m['idx_abs']:.2e}/{m['idx_flip_frac']:.4f}",
                        f"block_flips {m['block_flip_rate']:.3f}"]
                ok = ok and m["idx_abs"] <= 0.25 and m["idx_flip_frac"] <= 0.01
        check(f"{lname} f32 invariance up to single-code flips", ok,
              ", ".join(det))


def main():
    cfg = load_cfg()
    check_container(cfg)
    check_determinism(cfg)
    check_divergence(cfg)
    check_sinkhorn(cfg)
    check_indexer_legality(cfg)
    check_router(cfg)
    check_chunk_invariance(cfg)
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("all m4b oracle checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
