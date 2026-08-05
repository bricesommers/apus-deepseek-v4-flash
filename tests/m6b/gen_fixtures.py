#!/usr/bin/env python3
"""M6b fixture generator — like M6a (6 layers x 64 experts, coalesced
52,224 B slabs) but with **3 hash-routed layers** (num_hash_layers = 3,
like the real model) so hash-layer prefetch applies to layers 0-2 and
router-lookahead pilot prediction to targets 3, 4, 5.

Same oracle machinery/seed as M5/M6a (tools/oracle.py, MASTER_SEED).
NOTE: weights are random — the router has near-uniform expert usage, so
recall/locality NUMBERS on this fixture are meaningless by construction;
the fixture exists to validate the machinery (counts, invariance, audit
correctness). See tests/m6b/README.md.
"""

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle

M6B_CFG = dict(oracle.SMALL_CFG)
M6B_CFG["n_routed_experts"] = 64
M6B_CFG["n_activated_experts"] = 4
M6B_CFG["moe_inter_dim"] = 128
M6B_CFG["layers"] = [
    {"name": "l0_swa", "compress_ratio": 0, "hash": True},
    {"name": "l1_csa", "compress_ratio": 4, "hash": True},
    {"name": "l2_hca", "compress_ratio": 128, "hash": True},
    {"name": "l3_csa", "compress_ratio": 4, "hash": False},
    {"name": "l4_hca", "compress_ratio": 128, "hash": False},
    {"name": "l5_csa", "compress_ratio": 4, "hash": False},
]


def generate(fixtures_dir, clean=True):
    cfg = dict(M6B_CFG)
    if clean and os.path.isdir(fixtures_dir):
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    oracle.write_full_weights(cfg, os.path.join(fixtures_dir, "weights"))
    manifest = {
        "seed": oracle.MASTER_SEED,
        "config": "config.json",
        "weights": "weights/",
        "note": "M6b pilot fixtures: 6 layers x 64 experts, 3 hash layers "
                "(0-2), coalesced 52,224 B slabs; quality reference is the "
                "C eager path itself (bitwise invariance, pilot ON vs OFF).",
    }
    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    # verify the per-expert 6-tensor coalescing the store relies on
    sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
    import stutil
    wdir = os.path.join(fixtures_dir, "weights")
    hdr, _ = stutil.read_shard(os.path.join(wdir, "apus-00002.safetensors"))
    n_hash = 0
    for L, layer in enumerate(cfg["layers"]):
        if layer["hash"]:
            n_hash += 1
        for e in range(cfg["n_routed_experts"]):
            offs = []
            for w in ("w1", "w2", "w3"):
                for k in ("weight", "scale"):
                    offs.append(hdr[f"layers.{L}.ffn.experts.{e}.{w}.{k}"]
                                ["data_offsets"])
            offs.sort()
            for i in range(1, 6):
                assert offs[i][0] == offs[i - 1][1], (
                    f"slab not contiguous: layer {L} expert {e}")
    assert n_hash == 3, "fixture must have exactly 3 hash layers"
    print(f"m6b fixtures: {len(cfg['layers'])} layers x "
          f"{cfg['n_routed_experts']} experts, {n_hash} hash layers, "
          "coalescing verified")


if __name__ == "__main__":
    generate(os.path.join(ROOT, "tests", "m6b", "fixtures"))
