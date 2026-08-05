#!/usr/bin/env python3
"""M6a fixture generator — wider/experts-heavy synthetic model for the
expert-store tiering tests.

Same machinery as M5 (tools/oracle.py, MASTER_SEED, apus container layout
with per-expert 6-tensor coalesced slabs) but with **64 routed experts per
layer** and **6 layers** so cache-size fractions are meaningful: the M6a
invariance test runs the C forward against itself with (a) cache >= all
experts, (b) ~25%, (c) a few slots/layer — no python goldens are needed
(the tiered path must reproduce the eager path bit-for-bit).

Expert slab: w1/w3 [128,256] -> I8 [128,128] + E8M0 [128,8],
             w2 [256,128]    -> I8 [256,64]  + E8M0 [256,4]
= 3 * (16384 + 1024) = 52,224 B/expert; 64 * 6 * 52,224 = 20.05 MB experts.
"""

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle

M6A_CFG = dict(oracle.SMALL_CFG)
M6A_CFG["n_routed_experts"] = 64
M6A_CFG["n_activated_experts"] = 4
M6A_CFG["moe_inter_dim"] = 128
M6A_CFG["layers"] = [
    {"name": "l0_swa", "compress_ratio": 0, "hash": True},
    {"name": "l1_csa", "compress_ratio": 4, "hash": False},
    {"name": "l2_hca", "compress_ratio": 128, "hash": False},
    {"name": "l3_csa", "compress_ratio": 4, "hash": False},
    {"name": "l4_hca", "compress_ratio": 128, "hash": False},
    {"name": "l5_csa", "compress_ratio": 4, "hash": False},
]


def generate(fixtures_dir, clean=True):
    cfg = dict(M6A_CFG)
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
        "note": "M6a tiering fixtures: 6 layers x 64 experts, coalesced "
                "52,224 B slabs; quality reference is the C eager path "
                "itself (bitwise invariance across cache sizes).",
    }
    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    # verify the coalescing assumption the store relies on (also asserted
    # in C at store open): per-expert 6 tensors contiguous in the shard
    sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
    import stutil
    wdir = os.path.join(fixtures_dir, "weights")
    hdr, _ = stutil.read_shard(os.path.join(wdir, "apus-00002.safetensors"))
    for L in range(len(cfg["layers"])):
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
    print(f"m6a fixtures: {len(cfg['layers'])} layers x "
          f"{cfg['n_routed_experts']} experts, coalescing verified")


if __name__ == "__main__":
    generate(os.path.join(ROOT, "tests", "m6a", "fixtures"))
