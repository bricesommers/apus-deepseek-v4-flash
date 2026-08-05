#!/usr/bin/env python3
"""M11a oracle self-consistency checks.

Usage: .venv/bin/python tests/m11a/check_oracle.py

Verifies, against tests/m11a/fixtures (regenerate with gen_fixtures.py):

  1. Container sanity — mtp.* naming/dtypes/shapes vs the real 0731 scheme.
  2. Determinism — recomputing the join goldens, draft rounds and episodes
     reproduces the checked-in goldens bitwise (f32).
  3. Equivalence (the gate) — spec == non-spec emitted streams BITWISE,
     greedy + sampled, f32 + f64, natural + forced drafts.
  4. Forced draft patterns — exact accept counts (5/0/2), bonus flags,
     streams == non-spec, rollback digests equal.
  5. Rollback — full main-model state after a spec episode == state after
     non-spec decoding of the same emitted tokens, array-by-array bitwise.
  6. Legality — get_dspark_topk_idxs formula + non-causal block attention;
     confidence finite/shape; accept-prefix/emission bookkeeping; stage
     window catch-up writes hold the newest true position per slot.
  7. f32-vs-f64 divergence report (informational + loose sanity bound).

Exit code 0 iff all checks pass.
"""

import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

import oracle as base
import oracle_dspark as od
import stutil

FIX = os.path.join(ROOT, "tests", "m11a", "fixtures")
GOLD = os.path.join(FIX, "golden")

FAILURES = []

# M12a-1: the checked-in f32 goldens are bitwise-reproducible only on the
# macOS/ARM host that generated them — numpy's f32 BLAS kernels and the
# libm transcendentals are not bitwise-stable across platforms, and the
# random-weight near-tie cascade (tests/m11a/README) amplifies ulp
# differences to O(1) within a few layers. Off that host the f32-vs-golden
# comparisons below are INFORMATIONAL (reported, not gated); every
# within-platform bitwise gate (spec == non-spec, rollback digests, state
# arrays, accept counts) and the f64-vs-golden comparisons (bitwise-stable
# across platforms) stay hard.
GOLDEN_BITWISE = (sys.platform == "darwin" and platform.machine() == "arm64")


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def check_f32_golden(name, ok, detail=""):
    """f32-vs-golden bitwise on the golden-generating host; informational
    elsewhere (see GOLDEN_BITWISE note). Same check name and count."""
    if GOLDEN_BITWISE:
        check(name, ok, detail)
    else:
        note = "matches" if ok else "diverges as expected (f32 goldens are macOS/ARM-pinned)"
        check(name, True, f"[informational: {note}] {detail}")


def load_cfg():
    with open(os.path.join(FIX, "config.json")) as f:
        return json.load(f)


def load_params(cfg, f64):
    shards = base.ShardSet(os.path.join(FIX, "weights"))
    Ps, top = base.load_model_params(shards, cfg, f64)
    Pd = od.load_dspark_params(shards, cfg, f64)
    return Ps, top, Pd


# ---------------------------------------------------------------------------
# 1. container sanity
# ---------------------------------------------------------------------------

def check_container(cfg):
    print("== container sanity ==")
    wdir = os.path.join(FIX, "weights")
    with open(os.path.join(wdir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    n = 0
    for shard in sorted(set(wmap.values())):
        header, _ = stutil.read_shard(os.path.join(wdir, shard))
        for name in header:
            n += 1
            if wmap.get(name) != shard:
                check(f"index contains {name}", False)
    check("all shard tensors listed in index", True, f"{n} tensors")
    dim, V = cfg["dim"], cfg["vocab_size"]
    rank = cfg["dspark"]["markov_rank"]
    shards = base.ShardSet(wdir)
    dt, sh = shards.meta("mtp.0.main_proj.weight")
    check("main_proj FP8 [dim, 3*dim]", dt == "F8_E4M3"
          and tuple(sh) == (dim, 3 * dim), f"{dt} {sh}")
    dt, sh = shards.meta("mtp.0.main_proj.scale")
    check("main_proj scale E8M0", dt == "F8_E8M0"
          and tuple(sh) == (dim // 128, 3 * dim // 128), f"{dt} {sh}")
    dt, sh = shards.meta("mtp.0.main_norm.weight")
    check("main_norm BF16", dt == "BF16" and tuple(sh) == (dim,), f"{dt} {sh}")
    last = cfg["dspark"]["n_mtp_layers"] - 1
    for name, shape in ((f"mtp.{last}.markov_head.markov_w1.weight", (V, rank)),
                        (f"mtp.{last}.markov_head.markov_w2.weight", (V, rank))):
        dt, sh = shards.meta(name)
        check(f"{name} BF16 {shape}", dt == "BF16" and tuple(sh) == shape,
              f"{dt} {sh}")
    dt, sh = shards.meta(f"mtp.{last}.confidence_head.proj.weight")
    check("confidence proj BF16 [1, dim+rank]", dt == "BF16"
          and tuple(sh) == (1, dim + rank), f"{dt} {sh}")
    dt, sh = shards.meta(f"mtp.{last}.hc_head_fn")
    check("hc_head_fn F32", dt == "F32"
          and tuple(sh) == (cfg["hc_mult"], cfg["hc_mult"] * dim), f"{dt} {sh}")
    for k in range(cfg["dspark"]["n_mtp_layers"]):
        pk, sc = shards.fp4(f"mtp.{k}.ffn.experts.0.w1")
        check(f"mtp.{k} expert w1 fp4 shapes",
              pk.shape == (cfg["moe_inter_dim"], dim // 2)
              and sc.shape == (cfg["moe_inter_dim"], dim // 32),
              f"{pk.shape} {sc.shape}")
        dt, _ = shards.meta(f"mtp.{k}.ffn.gate.bias")
        check(f"mtp.{k} gate bias F32 (non-hash stage)", dt == "F32", dt)
    # every mtp tensor lives in the apus-mtp shard group (real layout)
    bad = [k for k, v in wmap.items()
           if k.startswith("mtp.") != v.startswith("apus-mtp-")]
    check("mtp.* namespace <-> apus-mtp shard group", not bad,
          f"{len(bad)} misplaced")


# ---------------------------------------------------------------------------
# 2. determinism (f32, bitwise vs goldens)
# ---------------------------------------------------------------------------

def check_determinism(cfg):
    print("== determinism (f32, bitwise) ==")
    Ps, top, Pd = load_params(cfg, False)
    for tag in ("join_prefill_len24", "join_prefill_len140"):
        d = os.path.join(GOLD, tag)
        ids = np.load(os.path.join(d, "input_ids.npy"))
        states = base.new_model_states(cfg, False)
        _, mh = od.main_forward_dspark(Ps, top, cfg, ids, states, 0, False)
        mx = od.stage0_project(Pd[0], cfg, mh, False)
        saved = np.load(os.path.join(d, "main_x_f32.npy"))
        check_f32_golden(f"{tag} main_x bitwise", np.array_equal(mx, saved),
                         f"maxabs={np.abs(mx - saved).max():.2e}")
        saved_mh = np.load(os.path.join(d, "main_hidden_f32.npy"))
        check_f32_golden(f"{tag} main_hidden bitwise",
                         np.array_equal(mh, saved_mh),
                         f"maxabs={np.abs(mh - saved_mh).max():.2e}")
        ds = od.new_dspark_states(cfg, False)
        od.dspark_prefill(Pd, cfg, mx, ds, False)
        ok = True
        for k in range(cfg["dspark"]["n_mtp_layers"]):
            sw = np.load(os.path.join(d, f"stage{k}_win_f32.npy"))
            ok = ok and np.array_equal(ds[k].win, sw)
        check_f32_golden(f"{tag} stage windows bitwise", ok)
    # draft rounds
    d0 = os.path.join(GOLD, "draft_rounds_len24")
    ids = np.load(os.path.join(d0, "round0", "prompt_ids.npy"))
    states = base.new_model_states(cfg, False)
    logits, mh = od.main_forward_dspark(Ps, top, cfg, ids, states, 0, False)
    ds = od.new_dspark_states(cfg, False)
    od.dspark_prefill(Pd, cfg, od.stage0_project(Pd[0], cfg, mh, False), ds,
                      False)
    f = len(ids) - 1
    anchor, _ = od.draw_token(logits[-1], 0, None)
    mh_last = mh[-1]
    ok_all = True
    for r in range(3):
        dr = od.dspark_draft_round(Pd, top, cfg, anchor,
                                   od.stage0_project(Pd[0], cfg, mh_last, False),
                                   f, ds, 0, None, False)
        d = os.path.join(d0, f"round{r}")
        ok = (np.array_equal(dr["drafts"], np.load(os.path.join(d, "drafts_f32.npy")))
              and np.array_equal(dr["conf"], np.load(os.path.join(d, "confidence_f32.npy")))
              and np.array_equal(dr["logits_final"], np.load(os.path.join(d, "logits_final_f32.npy")))
              and np.array_equal(dr["logits_base"], np.load(os.path.join(d, "logits_base_f32.npy")))
              and np.array_equal(dr["markov_bias"], np.load(os.path.join(d, "markov_bias_f32.npy")))
              and np.array_equal(dr["conf_hidden"], np.load(os.path.join(d, "conf_hidden_f32.npy"))))
        for k in range(cfg["dspark"]["n_mtp_layers"]):
            ok = ok and np.array_equal(
                dr["stage_h"][k], np.load(os.path.join(d, f"stage{k}_h_f32.npy")))
        ok_all = ok_all and ok
        rows, Hs = od.decode_batch(Ps, top, cfg, [anchor], f + 1, states, False)
        anchor, _ = od.draw_token(rows[0], 0, None)
        mh_last = Hs[0]
        f += 1
    check_f32_golden("draft_rounds_len24 all rounds bitwise", ok_all)
    # episodes
    for tag, temp in (("spec_episode_greedy", 0),
                      ("spec_episode_sampled", od.SAMPLE_TEMP)):
        d = os.path.join(GOLD, tag)
        ids = np.load(os.path.join(d, "prompt_ids.npy"))
        U = od._uniforms()
        du = od._draft_uniforms(od.EPISODE_ROUNDS * cfg["dspark"]["block_size"])
        ep = od.spec_episode(Ps, top, Pd, cfg, ids, od.EPISODE_ROUNDS, temp,
                             U if temp else None, du if temp else None, False)
        ok = (np.array_equal(ep["emitted"], np.load(os.path.join(d, "tokens_spec_f32.npy")))
              and np.array_equal([r["drafts"] for r in ep["rounds"]],
                                 np.load(os.path.join(d, "drafts_f32.npy")))
              and np.array_equal([r["accepted"] for r in ep["rounds"]],
                                 np.load(os.path.join(d, "accepted.npy")))
              and np.allclose(np.stack([r["conf"] for r in ep["rounds"]]),
                              np.load(os.path.join(d, "confidence_f32.npy")),
                              rtol=0, atol=0))
        check_f32_golden(f"{tag} rerun bitwise", ok)


# ---------------------------------------------------------------------------
# 3+4. equivalence + forced patterns (rerun fresh, compare streams/digests)
# ---------------------------------------------------------------------------

def check_equivalence(cfg):
    print("== equivalence: spec == non-spec streams (bitwise) ==")
    U = od._uniforms()
    for mode, f64 in (("f32", False), ("f64", True)):
        Ps, top, Pd = load_params(cfg, f64)
        for tag, temp in (("spec_episode_greedy", 0),
                          ("spec_episode_sampled", od.SAMPLE_TEMP)):
            d = os.path.join(GOLD, tag)
            ids = np.load(os.path.join(d, "prompt_ids.npy"))
            du = od._draft_uniforms(od.EPISODE_ROUNDS
                                    * cfg["dspark"]["block_size"])
            ep = od.spec_episode(Ps, top, Pd, cfg, ids, od.EPISODE_ROUNDS,
                                 temp, U if temp else None,
                                 du if temp else None, f64)
            ns = od.nonspec_episode(Ps, top, cfg, ids, len(ep["emitted"]),
                                    temp, U if temp else None, f64)
            check(f"{tag} {mode} stream bitwise", ep["emitted"] == ns[0],
                  f"{len(ep['emitted'])} tokens")
            check(f"{tag} {mode} rollback digest",
                  od.state_digest(ep["states"]) == od.state_digest(ns[1]))
            gold = np.load(os.path.join(d, f"tokens_spec_{mode}.npy"))
            if f64:
                # f64 goldens are bitwise-stable across platforms: hard gate
                check(f"{tag} {mode} stream == golden",
                      ep["emitted"] == gold.tolist())
            else:
                check_f32_golden(f"{tag} {mode} stream == golden",
                                 ep["emitted"] == gold.tolist())
    print("== forced draft patterns ==")
    Ps, top, Pd = load_params(cfg, False)
    B = cfg["dspark"]["block_size"]
    V = cfg["vocab_size"]
    for kind, expect, bonus in (("forced_all_accept", B, True),
                                ("forced_all_reject", 0, False),
                                ("forced_mixed", 2, False)):
        d = os.path.join(GOLD, kind)
        ids = np.load(os.path.join(d, "prompt_ids.npy"))
        plen = len(ids)
        ns_true = od.nonspec_episode(Ps, top, cfg, ids,
                                     od.FORCED_ROUNDS * (B + 2) + 8, 0, None,
                                     False)
        tt = lambda pos: int(ns_true[0][pos - plen])
        if kind == "forced_all_accept":
            ov = lambda r, f: [tt(f + 2 + j) for j in range(B)]
        elif kind == "forced_all_reject":
            ov = lambda r, f: [(tt(f + 2 + j) + 1) % V for j in range(B)]
        else:
            ov = lambda r, f: [tt(f + 2 + j) if j != 2
                               else (tt(f + 2 + j) + 1) % V for j in range(B)]
        ep = od.spec_episode(Ps, top, Pd, cfg, ids, od.FORCED_ROUNDS, 0,
                             None, None, False, draft_override=ov)
        got = [r["accepted"] for r in ep["rounds"]]
        check(f"{kind} accept counts exact", got == [expect] * od.FORCED_ROUNDS,
              str(got))
        check(f"{kind} bonus flags",
              [r["bonus"] for r in ep["rounds"]] == [bonus] * od.FORCED_ROUNDS)
        ns = od.nonspec_episode(Ps, top, cfg, ids, len(ep["emitted"]), 0,
                                None, False)
        check(f"{kind} stream == non-spec bitwise", ep["emitted"] == ns[0],
              f"{len(ep['emitted'])} tokens")
        check_f32_golden(f"{kind} stream == golden",
                         ep["emitted"] == np.load(os.path.join(d, "tokens_spec_f32.npy")).tolist())
        check(f"{kind} rollback digest",
              od.state_digest(ep["states"]) == od.state_digest(ns[1]))
        with open(os.path.join(d, "digests.json")) as fp:
            dig = json.load(fp)
        check_f32_golden(f"{kind} golden digests consistent",
                         dig["spec_main_f32"] == dig["nonspec_main_f32"]
                         and od.state_digest(ep["states"]) == dig["spec_main_f32"])


# ---------------------------------------------------------------------------
# 5. rollback: full array-by-array state comparison (not just digests)
# ---------------------------------------------------------------------------

def _state_arrays_equal(a, b):
    if a.pos != b.pos or not np.array_equal(a.win, b.win):
        return False
    if a.ratio:
        if not (np.array_equal(a.comp.cache, b.comp.cache)
                and np.array_equal(a.comp.kv, b.comp.kv)
                and np.array_equal(a.comp.sc, b.comp.sc)):
            return False
        if a.ratio == 4:
            return (np.array_equal(a.idx_comp.cache, b.idx_comp.cache)
                    and np.array_equal(a.idx_comp.kv, b.idx_comp.kv)
                    and np.array_equal(a.idx_comp.sc, b.idx_comp.sc))
    return True


def check_rollback_arrays(cfg):
    print("== rollback: array-by-array state equality ==")
    for mode, f64 in (("f32", False), ("f64", True)):
        Ps, top, Pd = load_params(cfg, f64)
        ids = np.load(os.path.join(GOLD, "spec_episode_greedy", "prompt_ids.npy"))
        ep = od.spec_episode(Ps, top, Pd, cfg, ids, od.EPISODE_ROUNDS, 0,
                             None, None, f64)
        ns = od.nonspec_episode(Ps, top, cfg, ids, len(ep["emitted"]), 0,
                                None, f64)
        ok = all(_state_arrays_equal(a, b)
                 for a, b in zip(ep["states"], ns[1]))
        check(f"greedy {mode} all layer states bitwise", ok)


# ---------------------------------------------------------------------------
# 6. legality
# ---------------------------------------------------------------------------

def check_legality(cfg):
    print("== legality ==")
    win, B = cfg["window_size"], cfg["dspark"]["block_size"]
    # get_dspark_topk_idxs formula + non-causal block rows (model.py:743-747)
    ok = True
    for start_pos in (1, 23, win - 1, win, 139, 300):
        idx = od.dspark_topk_idxs(win, B, start_pos)
        n = min(win, start_pos + 1)
        row = np.concatenate([np.arange(n), win + np.arange(B)])
        ok = ok and idx.shape == (B, n + B)
        ok = ok and all(np.array_equal(idx[i], row) for i in range(B))
    check("dspark topk idxs formula + all rows identical (non-causal block)",
          ok)
    # episode bookkeeping legality
    for tag in ("spec_episode_greedy", "spec_episode_sampled",
                "forced_all_accept", "forced_all_reject", "forced_mixed"):
        d = os.path.join(GOLD, tag)
        acc = np.load(os.path.join(d, "accepted.npy"))
        cnt = np.load(os.path.join(d, "emitted_counts.npy"))
        bon = np.load(os.path.join(d, "bonus.npy"))
        toks = np.load(os.path.join(d, "tokens_spec_f32.npy"))
        check(f"{tag} accepted in [0,B] + counts == 1+accepted + bonus==full",
              bool((acc >= 0).all() and (acc <= B).all()
                   and np.array_equal(cnt, 1 + acc)
                   and np.array_equal(bon, acc == B)
                   and cnt.sum() == len(toks)),
              f"acc={acc.tolist()}")
        V = cfg["vocab_size"]
        dr = np.load(os.path.join(d, "drafts_f32.npy"))
        check(f"{tag} drafts/emitted in vocab",
              bool((dr >= 0).all() and (dr < V).all()
                   and (toks >= 0).all() and (toks < V).all()))
    # confidence legality (natural episodes)
    for tag in ("spec_episode_greedy", "spec_episode_sampled"):
        c = np.load(os.path.join(GOLD, tag, "confidence_f32.npy"))
        check(f"{tag} confidence finite [R,B]",
              c.shape == (od.EPISODE_ROUNDS, B) and bool(np.isfinite(c).all()),
              f"range [{c.min():.3f}, {c.max():.3f}]")
    # margins alignment: one margin per main draw = sum(accepted+1)
    for tag in ("spec_episode_greedy", "spec_episode_sampled"):
        d = os.path.join(GOLD, tag)
        acc = np.load(os.path.join(d, "accepted.npy"))
        m = np.load(os.path.join(d, "margins.npy"))
        check(f"{tag} margins count == sum(accepted)+R",
              len(m) == int(acc.sum()) + len(acc), f"{len(m)}")


def check_stage_window_catchup(cfg):
    """Stage windows must hold, per slot, the KV of the newest TRUE position
    (prefill writes + per-round maintenance writes). Decode positions (fed
    by single-row decode steps): bitwise in both modes. Prompt positions
    (written by the batched stage prefill): f64 within 1e-6 (gemm reorder),
    f32 within one bf16 code step (the m4b-documented chunk-noise class)."""
    print("== stage window catch-up correctness ==")
    B = cfg["dspark"]["block_size"]
    win = cfg["window_size"]
    for mode, f64 in (("f64", True), ("f32", False)):
        Ps, top, Pd = load_params(cfg, f64)
        ids = np.load(os.path.join(GOLD, "forced_mixed", "prompt_ids.npy"))
        plen = len(ids)
        ns_true = od.nonspec_episode(Ps, top, cfg, ids,
                                     od.FORCED_ROUNDS * (B + 2) + 8, 0, None,
                                     f64)
        V = cfg["vocab_size"]
        tt = lambda pos: int(ns_true[0][pos - plen])
        ov = lambda r, f: [tt(f + 2 + j) if j != 2 else (tt(f + 2 + j) + 1) % V
                           for j in range(B)]
        ep = od.spec_episode(Ps, top, Pd, cfg, ids, od.FORCED_ROUNDS, 0,
                             None, None, f64, draft_override=ov)
        f_final = ep["rounds"][-1]["f"] + 1 + ep["rounds"][-1]["accepted"]
        # truth: main_hidden per position from a straight decode of the truth
        states = base.new_model_states(cfg, f64)
        _, mh_all = od.main_forward_dspark(Ps, top, cfg, ids, states, 0, f64)
        rows = [mh_all[i] for i in range(plen)]
        for i, t in enumerate(ns_true[0][:f_final + 1 - plen]):
            _, mh = od.main_forward_dspark(Ps, top, cfg, [t], states,
                                           plen + i, f64)
            rows.append(mh[0])
        worst = 0.0
        n_bit = n_tol = 0
        bad = 0
        for p in range(0, f_final + 1):
            if p + win <= f_final:
                continue                      # slot overwritten by p+win
            mx = od.stage0_project(Pd[0], cfg, rows[p], f64)
            for k in range(cfg["dspark"]["n_mtp_layers"]):
                # stage windows are f32-typed (base.LayerState.win), so the
                # f64-mode KV is truncated to f32 on storage — compare at
                # storage precision.
                exp = od.stage_main_kv(Pd[k], cfg, mx, p, f64).astype(np.float32)
                got = ep["dstates"][k].win[p % win]
                if np.array_equal(exp, got):
                    n_bit += 1
                    continue
                if p < plen:
                    if f64:
                        ok_tol = float(np.abs(exp - got).max()) < 1e-6
                    else:
                        tol = 2 * 0.0078125 * np.maximum(np.abs(exp), 0.5)
                        r_ = float((np.abs(exp - got) / tol).max())
                        worst = max(worst, r_)
                        ok_tol = r_ <= 1.0
                    n_tol += 1 if ok_tol else 0
                    bad += 0 if ok_tol else 1
                else:
                    bad += 1
        check(f"forced_mixed {mode} stage slots hold newest true position",
              bad == 0,
              f"bitwise={n_bit} tolerated={n_tol} bad={bad} worst={worst:.2f}")


# ---------------------------------------------------------------------------
# 7. divergence report
# ---------------------------------------------------------------------------

def check_divergence(cfg):
    print("== f32 vs f64 divergence (draft path) ==")
    d = os.path.join(GOLD, "draft_rounds_len24", "round0")
    rows = []
    for k in ("logits_base", "markov_bias", "logits_final", "confidence",
              "conf_hidden", "stage0_h", "stage2_h"):
        a = np.load(os.path.join(d, k + "_f32.npy")).astype(np.float64)
        b = np.load(os.path.join(d, k + "_f64.npy")).astype(np.float64)
        dd = np.abs(a - b)
        rows.append((k, float(np.abs(b).max()), float(dd.max()),
                     float(np.percentile(dd, 99))))
        print(f"  {k:14s} scale={np.abs(b).max():8.3f} maxabs={dd.max():10.3g} "
              f"p99={np.percentile(dd, 99):10.3g}")
    worst_conf = [r for r in rows if r[0] == "confidence"][0][2]
    check("confidence f32-vs-f64 maxabs < 2.0 (cascade sanity)",
          worst_conf < 2.0, f"{worst_conf:.3f}")
    print("  note: random-weight cascade through 5 main layers + 3 stages; "
          "M11b must use the m5 near-tie policy (margins dumped per draw).")


def main():
    cfg = load_cfg()
    check_container(cfg)
    check_determinism(cfg)
    check_equivalence(cfg)
    check_rollback_arrays(cfg)
    check_legality(cfg)
    check_stage_window_catchup(cfg)
    check_divergence(cfg)
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("all m11a oracle checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
