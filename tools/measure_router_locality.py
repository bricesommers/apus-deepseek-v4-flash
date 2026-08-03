#!/usr/bin/env python3
"""tools/measure_router_locality.py — the ARCHITECTURE.md §8 measurement
plan: router-locality measurements that set the tiering defaults
(APUS_PILOT_K, pin budget, cache expectations).

Engines (both produce the same NDJSON set dump, then identical analysis):
  --engine c       run the C engine: bin/apus run --tiered
                   --measure-locality DUMP (works on the fixture AND on the
                   real 160 GB container — same command, see README).
  --engine oracle  pure-Python reference forward (tools/oracle.py) with the
                   same instrumentation; fixture-schema models only
                   (cross-check of the C dump machinery).

IMPORTANT CAVEAT (read the report header): the synthetic fixture model has
RANDOM weights. Its router has near-uniform expert usage and its "recall"
is dominated by the constant selection bias — every number here is
machinery validation only, NOT a tuning input. On the real container the
same command produces the numbers that set the defaults.

§8 measurements:
  1. pilot recall curve: |pred top-N ∩ actual top-k| / topk for
     N in {6,8,12,16,24}, per source-layer type (post-SWA/CSA/HCA).
  2. temporal reuse: |A(pos) ∩ A(pos+T)| / topk, T in {1,2,4,8}.
  3. cross-layer coupling: P(e_{L+d}=j | e_L=i), d in {1,2} (sparse
     artifact: coupling_d{d}.json).
  4. per-layer expert frequency + Zipf fit; pin-candidate coverage at
     5/10/20% of E pinned.
  5. hash-layer audit: A(pos, hash layer) == tid2eid[token] — must be 100%.

Usage:
  python tools/measure_router_locality.py --model tests/m6b/fixtures \
      --engine c --decode 64 --out tests/m6b/results
  python tools/measure_router_locality.py --model tests/m6b/fixtures \
      --engine oracle --decode 64 --out tests/m6b/results_oracle
  # real container, once downloaded (identical analysis):
  python tools/measure_router_locality.py --model weights/apus \
      --engine c --prompt "real prompt text" --decode 512 \
      --out docs/locality
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

RECALL_N = [6, 8, 12, 16, 24]
REUSE_T = [1, 2, 4, 8]
COUPLE_D = [1, 2]
PIN_FRAC = [0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# model config (fixture schema and real-checkpoint schema)
# ---------------------------------------------------------------------------

def load_model_cfg(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    layers = cfg.get("layers")
    if layers is None:                      # real checkpoint schema
        ratios = cfg["compress_ratios"]
        n_hash = cfg.get("num_hash_layers", 0)
        layers = [{"compress_ratio": r, "hash": i < n_hash}
                  for i, r in enumerate(ratios)]
    out = {
        "layers": layers,
        "n_layers": len(layers),
        "E": cfg.get("n_routed_experts"),
        "topk": cfg.get("n_activated_experts", cfg.get("num_experts_per_tok")),
        "vocab": cfg.get("vocab_size"),
        "raw": cfg,
    }
    return out


def ratio_name(r):
    return {0: "swa", 4: "csa", 128: "hca"}.get(r, f"r{r}")


# ---------------------------------------------------------------------------
# engine: C (bin/apus --measure-locality)
# ---------------------------------------------------------------------------

def run_c_engine(model_dir, prompt_ids, prompt_text, decode, seed, dump_path):
    apus = os.path.join(ROOT, "bin", "apus")
    if not os.path.isfile(apus):
        sys.exit("bin/apus not found — run `make apus` first")
    cmd = [apus, "run", "--model", model_dir, "--tiered", "--greedy",
           "--seed", str(seed), "--max-tokens", str(decode),
           "--measure-locality", dump_path, "--quiet"]
    if prompt_text is not None:
        cmd += ["--prompt", prompt_text]
    else:
        cmd += ["--ids", ",".join(str(i) for i in prompt_ids)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"apus failed ({r.returncode}):\n{r.stderr[-2000:]}")
    return r.stderr


# ---------------------------------------------------------------------------
# engine: pure oracle (fixture-schema models only)
# ---------------------------------------------------------------------------

def run_oracle_engine(model_dir, prompt_ids, decode, dump_path):
    import oracle
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    if "layers" not in cfg:
        sys.exit("oracle engine: fixture-schema config only (no 'layers'); "
                 "use --engine c for the real container")
    shards = oracle.ShardSet(os.path.join(model_dir, "weights"))
    Ps, top = oracle.load_model_params(shards, cfg, False)
    states = oracle.new_model_states(cfg, False)
    E, topk = cfg["n_routed_experts"], cfg["n_activated_experts"]
    n_pred = min(24, E)

    def predict(target, h4):
        """The pilot rule, replicated on oracle ops: layer `target`'s ffn
        hc_pre collapse + rms_norm + gate scores + stable top-N."""
        P = Ps[target]
        if P["hash"]:
            return None
        x, _, _ = oracle.hc_pre(h4[None], P["hc_ffn_fn"], P["hc_ffn_scale"],
                                P["hc_ffn_base"], cfg, False, None, "ffn_hc")
        x = oracle.rms_norm(x, P["ffn_norm"], cfg["norm_eps"], False)
        scores = oracle.f32_linear(x, P["gate_w"], False)
        sp = np.sqrt(oracle.softplus(scores))
        biased = sp + P["gate_bias"]
        return oracle.topk_stable(biased, n_pred)[0].astype(int).tolist()

    lines = []

    def forward(ids, start_pos):
        ids = np.asarray(ids, dtype=np.int64)
        h = top["embed"][ids].astype(np.float32)
        h = np.repeat(h[:, None, :], cfg["hc_mult"], axis=1)
        for i, P in enumerate(Ps):
            h, it = oracle.block_forward(P, cfg, h, ids, start_pos,
                                         states[i], False)
            for t in range(len(ids)):
                lines.append({"type": "A", "pos": start_pos + t, "layer": i,
                              "eids": [int(x) for x in
                                       it["router_idx"][t]]})
                if i + 1 < len(Ps):
                    pred = predict(i + 1, it["post_attn_h"][t])
                    if pred is not None:
                        lines.append({"type": "P", "pos": start_pos + t,
                                      "layer": i + 1, "eids": pred})
        y = oracle.hc_head_collapse(h, top, cfg, False)
        yn = oracle.rms_norm(y, top["norm"], cfg["norm_eps"], False)
        return yn.astype(np.float32) @ top["head"].astype(np.float32).T

    lines.append({"type": "ids", "pos0": 0,
                  "ids": [int(x) for x in prompt_ids]})
    logits = forward(prompt_ids, 0)
    pos = len(prompt_ids)
    for _ in range(decode):
        nxt = int(np.argmax(logits[-1]))
        lines.append({"type": "gen", "pos": pos, "id": nxt})
        logits = forward([nxt], pos)
        pos += 1
    for st in states:      # keep states consistent (pos advanced manually)
        pass
    with open(dump_path, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")


# ---------------------------------------------------------------------------
# analysis (shared)
# ---------------------------------------------------------------------------

def load_dump(path):
    ids_at = {}
    A = defaultdict(dict)      # layer -> pos -> eids
    P = defaultdict(dict)
    for line in open(path):
        e = json.loads(line)
        if e["type"] == "ids":
            for i, t in enumerate(e["ids"]):
                ids_at[e["pos0"] + i] = t
        elif e["type"] == "gen":
            ids_at[e["pos"]] = e["id"]
        elif e["type"] == "A":
            A[e["layer"]][e["pos"]] = e["eids"]
        elif e["type"] == "P":
            P[e["layer"]][e["pos"]] = e["eids"]
    return ids_at, A, P


def m1_recall(mcfg, A, P):
    """Recall curve per source-layer type + overall."""
    topk = mcfg["topk"]
    by_src = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # type -> N -> [hits, tot]
    overall = defaultdict(lambda: [0, 0])
    for layer, per_pos in A.items():
        if layer not in P or mcfg["layers"][layer]["hash"]:
            continue
        src_type = ratio_name(mcfg["layers"][layer - 1]["compress_ratio"])
        for pos, a in per_pos.items():
            pred = P[layer].get(pos)
            if not pred:
                continue
            for N in RECALL_N:
                if N > len(pred):
                    continue
                h = sum(1 for e in a if e in pred[:N])
                by_src[src_type][N][0] += h
                by_src[src_type][N][1] += len(a)
                overall[N][0] += h
                overall[N][1] += len(a)
    out = {"per_source_type": {}, "overall": {}}
    for t, d in by_src.items():
        out["per_source_type"][t] = {
            N: (v[0] / v[1] if v[1] else None) for N, v in d.items()}
    out["overall"] = {N: (v[0] / v[1] if v[1] else None)
                      for N, v in overall.items()}
    out["topk"] = topk
    return out


def m2_temporal_reuse(mcfg, A):
    topk = mcfg["topk"]
    per_T = {T: [] for T in REUSE_T}
    for layer, per_pos in A.items():
        positions = sorted(per_pos)
        pos_set = set(positions)
        for p in positions:
            for T in REUSE_T:
                if p + T in pos_set:
                    inter = len(set(per_pos[p]) & set(per_pos[p + T]))
                    per_T[T].append(inter / topk)
    return {T: (float(np.mean(v)) if v else None)
            for T, v in per_T.items()}


def m3_coupling(mcfg, A, out_dir):
    E = mcfg["E"]
    artifacts = []
    for d in COUPLE_D:
        per_layer = []
        for L in range(mcfg["n_layers"] - d):
            if L not in A or L + d not in A:
                continue
            counts = defaultdict(int)
            for pos, a in A[L].items():
                b = A[L + d].get(pos)
                if not b:
                    continue
                for i in a:
                    for j in b:
                        counts[(i, j)] += 1
            top = defaultdict(list)
            for (i, j), c in counts.items():
                top[i].append((j, c))
            top = {i: sorted(v, key=lambda x: (-x[1], x[0]))[:8]
                   for i, v in top.items()}
            tot_i = defaultdict(int)
            for (i, j), c in counts.items():
                tot_i[i] += c
            p1 = [top[i][0][1] / tot_i[i] for i in top if top[i]]
            per_layer.append({
                "layer": L, "d": d,
                "mean_p_top1": float(np.mean(p1)) if p1 else None,
                "top_j_given_i": {str(i): v for i, v in top.items()},
            })
        path = os.path.join(out_dir, f"coupling_d{d}.json")
        with open(path, "w") as f:
            json.dump({"d": d, "E": E, "layers": per_layer}, f)
        artifacts.append(path)
    return {d: [x["mean_p_top1"] for x in
                json.load(open(os.path.join(
                    out_dir, f"coupling_d{d}.json")))["layers"]]
            for d in COUPLE_D}, artifacts


def m4_frequency_zipf(mcfg, A):
    E = mcfg["E"]
    per_layer = {}
    for layer, per_pos in A.items():
        cnt = np.zeros(E)
        for a in per_pos.values():
            for e in a:
                cnt[e] += 1
        total = cnt.sum()
        nz = np.sort(cnt[cnt > 0])[::-1]
        ranks = np.arange(1, len(nz) + 1)
        # Zipf fit: log(freq) = a - s*log(rank)
        if len(nz) > 2:
            s, _ = np.polyfit(np.log(ranks), np.log(nz), 1)
            slope = float(-s)
        else:
            slope = None
        pins = {}
        for frac in PIN_FRAC:
            npin = max(1, round(E * frac))
            pins[f"{int(frac*100)}%"] = float(np.sort(cnt)[::-1][:npin].sum()
                                              / total) if total else None
        per_layer[str(layer)] = {
            "zipf_slope": slope,
            "pin_coverage": pins,
            "top_counts": [int(x) for x in np.sort(cnt)[::-1][:16]],
        }
    return per_layer


def m5_hash_audit(mcfg, model_dir, ids_at, A):
    import stutil
    hash_layers = [i for i, l in enumerate(mcfg["layers"]) if l["hash"]]
    if not hash_layers:
        return {"hash_layers": [], "checked": 0, "mismatches": 0,
                "hit_rate": None}
    wdir = model_dir
    if not os.path.isfile(os.path.join(wdir, "model.safetensors.index.json")):
        wdir = os.path.join(model_dir, "weights")
    idx = json.load(open(os.path.join(wdir, "model.safetensors.index.json")))
    shards = {}
    topk = mcfg["topk"]
    checked = mismatches = 0
    for L in hash_layers:
        name = f"layers.{L}.ffn.gate.tid2eid"
        shard = idx["weight_map"].get(name)
        if shard is None:
            return {"error": f"{name} not in weight_map"}
        if shard not in shards:
            shards[shard] = stutil.read_tensor_bytes(
                os.path.join(wdir, shard))
        raw = shards[shard][name]
        tbl = np.frombuffer(raw, dtype=np.int64).reshape(-1, topk)
        for pos, a in A.get(L, {}).items():
            tid = ids_at.get(pos)
            checked += 1
            if tid is None or sorted(a) != sorted(int(x) for x in tbl[tid]):
                mismatches += 1
    return {"hash_layers": hash_layers, "checked": checked,
            "mismatches": mismatches,
            "hit_rate": 1.0 - mismatches / checked if checked else None}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["c", "oracle"], default="c")
    ap.add_argument("--prompt", default=None,
                    help="real text (C engine tokenizes it); default: seeded "
                         "random --prompt-len ids")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-dump", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mcfg = load_model_cfg(args.model)
    fixture = os.path.isfile(os.path.join(args.model, "manifest.json"))

    rng = np.random.default_rng(args.seed + 777)
    prompt_ids = [int(x) for x in rng.integers(0, mcfg["vocab"],
                                               size=args.prompt_len)]

    dump_path = os.path.join(args.out, "locality_dump.ndjson")
    engine_note = ""
    if args.engine == "c":
        stderr = run_c_engine(args.model, prompt_ids, args.prompt,
                              args.decode, args.seed, dump_path)
        engine_note = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    else:
        run_oracle_engine(args.model, prompt_ids, args.decode, dump_path)

    ids_at, A, P = load_dump(dump_path)
    n_a = sum(len(v) for v in A.values())
    n_p = sum(len(v) for v in P.values())

    recall = m1_recall(mcfg, A, P)
    reuse = m2_temporal_reuse(mcfg, A)
    coupling_summary, coupling_files = m3_coupling(mcfg, A, args.out)
    zipf = m4_frequency_zipf(mcfg, A)
    audit = m5_hash_audit(mcfg, args.model, ids_at, A)

    report = {
        "model": args.model,
        "engine": args.engine,
        "fixture_random_weights": fixture,
        "decode_tokens": args.decode,
        "dump": {"A_sets": n_a, "P_sets": n_p,
                 "path": dump_path if args.keep_dump else None},
        "m1_pilot_recall": recall,
        "m2_temporal_reuse": reuse,
        "m3_coupling_mean_p_top1": coupling_summary,
        "m4_frequency_zipf": zipf,
        "m5_hash_audit": audit,
    }
    with open(os.path.join(args.out, "locality_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    # --- console report
    print("=" * 72)
    print(f"router-locality report: {args.model} (engine: {args.engine})")
    if fixture:
        print("!! FIXTURE MODEL WITH RANDOM WEIGHTS — every number below is")
        print("!! MACHINERY VALIDATION ONLY (near-uniform router, recall is")
        print("!! dominated by the constant selection bias). NOT tuning input.")
    print("=" * 72)
    print(f"dump: {n_a} actual sets, {n_p} predicted sets "
          f"({args.decode} decode tokens)")
    if engine_note:
        print(f"engine: {engine_note}")
    print("\n[1] pilot recall curve (top-k = %d):" % recall["topk"])
    hdr = "  source   " + "".join(f"  N={N:<3}" for N in RECALL_N)
    print(hdr)
    for t, d in sorted(recall["per_source_type"].items()):
        print(f"  {t:<8} " + "".join(
            f"  {d[N]:.3f}" if d.get(N) is not None else "     -"
            for N in RECALL_N))
    print(f"  {'overall':<8} " + "".join(
        f"  {recall['overall'][N]:.3f}" if recall['overall'].get(N) is not None
        else "     -" for N in RECALL_N))
    print("\n[2] temporal reuse |A(pos) ∩ A(pos+T)|/topk:")
    print("  " + "  ".join(f"T={T}: {v:.3f}" if v is not None else f"T={T}: -"
                            for T, v in reuse.items()))
    print("\n[3] cross-layer coupling (mean P(top-1 j | i)):")
    for d, vals in coupling_summary.items():
        print(f"  d={d}: " + (" ".join(f"{v:.3f}" for v in vals)
                              if vals else "-"))
    print(f"  artifacts: {', '.join(coupling_files)}")
    print("\n[4] frequency/Zipf (slope; pin coverage at 5/10/20% of E):")
    for layer, z in sorted(zipf.items(), key=lambda x: int(x[0])):
        cov = z["pin_coverage"]
        print(f"  layer {layer}: slope={z['zipf_slope']:.3f} "
              f"cov={cov['5%']:.3f}/{cov['10%']:.3f}/{cov['20%']:.3f}"
              if z["zipf_slope"] is not None else f"  layer {layer}: n/a")
    print("\n[5] hash-layer audit:")
    if audit.get("hit_rate") is not None:
        print(f"  layers {audit['hash_layers']}: {audit['checked']} sets, "
              f"{audit['mismatches']} mismatches "
              f"(hit rate {audit['hit_rate']*100:.1f}%)")
    else:
        print(f"  {audit}")
    print(f"\nreport: {os.path.join(args.out, 'locality_report.json')}")
    if not args.keep_dump:
        os.remove(dump_path)


if __name__ == "__main__":
    main()
