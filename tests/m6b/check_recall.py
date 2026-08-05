#!/usr/bin/env python3
"""tests/m6b/check_recall.py — validate the pilot's recall machinery
end-to-end:

  1. parse the RUNTIME dump (written by the live pilot during a greedy
     decode) and recompute recall = sum|P ∩ A| / sum|A| in Python;
  2. compare against the pilot's own stats counters (RECALL line printed
     by test_recall) — must match EXACTLY;
  3. cross-check the MEASURE dump (apus_model_forward_measure path): for
     every decode position, the A sets must be identical and the runtime
     P set must equal the measure P set's prefix (same prediction code);
  4. hash-layer audit: every A set on layers 0-2 must equal the tid2eid
     row of the token at that position (100%, by construction);
  5. dump shape: A lines have topk eids, runtime P lines pilot_k eids.

Usage: check_recall.py RUNTIME_DUMP MEASURE_DUMP RECALL_LINE_FILE [FIXTURE_DIR]
Exit 0 iff all checks pass.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def load_ndjson(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    rt_path, mm_path, rec_path = sys.argv[1], sys.argv[2], sys.argv[3]
    fix = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        ROOT, "tests", "m6b", "fixtures")
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            print(f"FAIL: {msg}")
            fails += 1

    rt = load_ndjson(rt_path)
    mm = load_ndjson(mm_path)
    with open(rec_path) as f:
        rec = next(l for l in f.read().splitlines() if l.startswith("RECALL"))
    kv = dict(tok.split("=") for tok in rec.split()
              if "=" in tok and not tok.startswith("tokens="))
    hits_c = int(kv["hits"])
    actuals_c = int(kv["actuals"])
    tokens = [int(t) for t in
              rec.split("tokens=")[1].strip().split(",")]

    cfg = json.load(open(os.path.join(fix, "config.json")))
    topk = cfg["n_activated_experts"]
    hash_layers = {i for i, l in enumerate(cfg["layers"]) if l["hash"]}
    n_layers = len(cfg["layers"])

    # token id at each position: prompt then sampled tokens
    prompt = [3, 41, 7, 200, 511, 0, 128, 65]
    ids_at = prompt + tokens          # pos p held ids_at[p]

    # tid2eid for the hash audit
    import stutil
    import numpy as np
    shard = os.path.join(fix, "weights", "apus-00001.safetensors")
    tb = stutil.read_tensor_bytes(shard)
    tid2eid = {}
    for L in hash_layers:
        raw = tb[f"layers.{L}.ffn.gate.tid2eid"]
        tid2eid[L] = np.frombuffer(raw, dtype=np.int64).reshape(
            cfg["vocab_size"], topk)

    # --- 1/2: recall recompute vs counters (runtime dump, decode positions)
    a_rt = {}
    p_rt = {}
    for e in rt:
        key = (e["pos"], e["layer"])
        if e["type"] == "A":
            a_rt[key] = e["eids"]
        else:
            p_rt[key] = e["eids"]

    hits_py = 0
    actuals_py = 0
    for (pos, layer), a in a_rt.items():
        check(len(a) == topk, f"A set size {len(a)} != topk at {pos},{layer}")
        if layer in hash_layers:
            continue
        pred = p_rt.get((pos, layer))
        check(pred is not None, f"missing P for A at pos={pos} layer={layer}")
        if pred is None:
            continue
        check(len(pred) == 8, f"runtime P size {len(pred)} != pilot_k 8")
        actuals_py += len(a)
        hits_py += sum(1 for e in a if e in pred)
    check(actuals_py == actuals_c,
          f"actuals: python {actuals_py} != C counter {actuals_c}")
    check(hits_py == hits_c,
          f"hits: python {hits_py} != C counter {hits_c}")
    recall = hits_py / actuals_py if actuals_py else 0.0

    # --- 3: measure-dump cross-check (decode positions only)
    a_mm = {(e["pos"], e["layer"]): e["eids"] for e in mm if e["type"] == "A"}
    p_mm = {(e["pos"], e["layer"]): e["eids"] for e in mm if e["type"] == "P"}
    n_mm = 0
    for key, a in a_rt.items():
        am = a_mm.get(key)
        check(am == a, f"measure A mismatch at {key}")
        pm = p_mm.get(key)
        pr = p_rt.get(key)
        if pr is not None:
            check(pm is not None and pm[:len(pr)] == pr,
                  f"measure P prefix mismatch at {key}")
            n_mm += 1
    check(n_mm > 0, "no P cross-checks performed")

    # --- 4: hash audit (both dumps, all positions)
    n_hash = 0
    for (pos, layer), a in list(a_rt.items()) + list(a_mm.items()):
        if layer not in hash_layers:
            continue
        check(0 <= pos < len(ids_at), f"hash audit: bad pos {pos}")
        want = [int(x) for x in tid2eid[layer][ids_at[pos]]]
        check(a == want,
              f"hash audit: layer {layer} pos {pos}: {a} != tid2eid {want}")
        n_hash += 1
    want_hash = len(hash_layers) * (len(a_rt) // n_layers
                                    + len(a_mm) // n_layers)
    check(n_hash == want_hash,
          f"hash audit: {n_hash} sets checked, expected {want_hash}")

    # --- 5: measure dump covers prefill too (the runtime dump does not)
    prefill_a = [k for (k, v) in a_mm.items() if 0 <= k[0] < len(prompt) - 1]
    check(len(prefill_a) == (len(prompt) - 1) * n_layers,
          f"measure dump prefill A coverage {len(prefill_a)}")

    print(f"check_recall: recall(recomputed)={hits_py}/{actuals_py}="
          f"{recall:.4f} == C counters; runtime==measure P/A sets; "
          f"hash audit {n_hash}/{n_hash} = 100%")
    print("check_recall: " + ("FAILED" if fails else "ok"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
