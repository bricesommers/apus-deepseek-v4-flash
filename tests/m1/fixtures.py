"""Synthetic DeepSeek-V4-Flash checkpoint fixtures for M1 tests.

Mimics the real checkpoint's naming scheme and dtypes at tiny scale
(verified against reference/model.safetensors.index.json):

  layers.{L}.ffn.experts.{E}.w{1,2,3}.weight   I8, packed 2x FP4-E2M1 / byte
  layers.{L}.ffn.experts.{E}.w{1,2,3}.scale    F8_E8M0, 1 scale / 32 elems / K
  layers.{L}.ffn.shared_experts.w{1,2,3}.{weight,scale}
  layers.{L}.ffn.gate.weight                   BF16
  layers.{L}.ffn.gate.bias                     F32   (layers >= 3)
  layers.{L}.ffn.gate.tid2eid                  I64   (layers 0..2, hash routing)
  layers.{L}.attn.wq_a.weight / .scale         F8_E4M3 / F8_E8M0
  layers.{L}.attn.indexer.wq_b.weight / .scale (even layers >= 2)
  layers.{L}.hc_attn_fn, ...                   F32
  mtp.0.*                                      same scheme, incl. 256->few experts
  embed.weight / head.weight / norm.weight     BF16

Dims are chosen so the MXFP4 self-description invariants hold exactly like
the real model (packed bytes == O*K/2, scale elements == O*K/32):

  hidden H = 64, expert intermediate M = 32
  w1/w3: [M, H/2] packed bytes ([M, H] logical), scales [M, H/32]
  w2:    [H, M/2] packed bytes ([H, M] logical), scales [H, M/32]
  -> per expert: 2*(32*32 + 32*2) + (64*16 + 64*1) = 3264 bytes

Payloads are random bytes (byte-identity tests don't need meaningful
values). The fixture writes 3 input shards with a weight_map index, exactly
like the real layout, with every expert wholly inside one input shard.
"""

import json
import os

import numpy as np

import stutil

H = 64        # hidden size
M = 32        # moe intermediate size
N_EXPERTS = 6
N_LAYERS = 5  # 0..4: hash layers 0-2 (tid2eid), bias layers 3-4
VOCAB = 128   # tiny stand-in for 129,280
TOPK = 6
N_HEADS = 8

PER_EXPERT_BYTES = 2 * (M * (H // 2) + M * (H // 32)) + H * (M // 2) + H * (M // 32)
assert PER_EXPERT_BYTES == 3264


def _rand(rng, dtype, shape):
    return rng.randint(0, 256, size=stutil.tensor_nbytes(dtype, shape),
                       dtype=np.uint8).tobytes()


def _layer_tensors(layer):
    """All tensors of one main-model layer, as (name, dtype, shape) specs."""
    t = []
    pre = f"layers.{layer}"
    for e in range(N_EXPERTS):
        epre = f"{pre}.ffn.experts.{e}"
        t += [
            (f"{epre}.w1.weight", "I8", [M, H // 2]),
            (f"{epre}.w1.scale", "F8_E8M0", [M, H // 32]),
            (f"{epre}.w2.weight", "I8", [H, M // 2]),
            (f"{epre}.w2.scale", "F8_E8M0", [H, M // 32]),
            (f"{epre}.w3.weight", "I8", [M, H // 2]),
            (f"{epre}.w3.scale", "F8_E8M0", [M, H // 32]),
        ]
    for w in ("w1", "w2", "w3"):
        rows, cols = (M, H) if w in ("w1", "w3") else (H, M)
        t.append((f"{pre}.ffn.shared_experts.{w}.weight", "F8_E4M3",
                  [rows, cols]))
        t.append((f"{pre}.ffn.shared_experts.{w}.scale", "F8_E8M0",
                  [max(rows // 128, 1), max(cols // 128, 1)]))
    t.append((f"{pre}.ffn.gate.weight", "BF16", [N_EXPERTS, H]))
    if layer < 3:
        t.append((f"{pre}.ffn.gate.tid2eid", "I64", [VOCAB, TOPK]))
    else:
        t.append((f"{pre}.ffn.gate.bias", "F32", [N_EXPERTS]))
    # attention (wq_a FP8 blockwise, rest BF16/F32)
    t.append((f"{pre}.attn.wq_a.weight", "F8_E4M3", [H, H]))
    t.append((f"{pre}.attn.wq_a.scale", "F8_E8M0", [1, 1]))
    t.append((f"{pre}.attn.kv_norm.weight", "BF16", [H]))
    t.append((f"{pre}.attn.attn_sink", "F32", [N_HEADS]))
    if layer >= 2:
        t.append((f"{pre}.attn.compressor.wkv.weight", "BF16", [H, H]))
        if layer % 2 == 0:
            idx = f"{pre}.attn.indexer"
            t.append((f"{idx}.wq_b.weight", "F8_E4M3", [N_HEADS * 16, H]))
            t.append((f"{idx}.wq_b.scale", "F8_E8M0", [1, 1]))
            t.append((f"{idx}.weights_proj.weight", "BF16", [N_HEADS, H]))
    # mHC params (FP32)
    for kind in ("attn", "ffn"):
        t.append((f"{pre}.hc_{kind}_fn", "F32", [24, 4 * H]))
        t.append((f"{pre}.hc_{kind}_scale", "F32", [3]))
        t.append((f"{pre}.hc_{kind}_base", "F32", [24]))
    t.append((f"{pre}.attn_norm.weight", "BF16", [H]))
    t.append((f"{pre}.ffn_norm.weight", "BF16", [H]))
    return t


def _mtp_tensors():
    t = []
    pre = "mtp.0"
    for e in range(N_EXPERTS):
        epre = f"{pre}.ffn.experts.{e}"
        t += [
            (f"{epre}.w1.weight", "I8", [M, H // 2]),
            (f"{epre}.w1.scale", "F8_E8M0", [M, H // 32]),
            (f"{epre}.w2.weight", "I8", [H, M // 2]),
            (f"{epre}.w2.scale", "F8_E8M0", [H, M // 32]),
            (f"{epre}.w3.weight", "I8", [M, H // 2]),
            (f"{epre}.w3.scale", "F8_E8M0", [M, H // 32]),
        ]
    t.append((f"{pre}.e_proj.weight", "F8_E4M3", [H, H]))
    t.append((f"{pre}.e_proj.scale", "F8_E8M0", [1, 1]))
    t.append((f"{pre}.h_proj.weight", "F8_E4M3", [4 * H, H]))
    t.append((f"{pre}.h_proj.scale", "F8_E8M0", [1, 1]))
    t.append((f"{pre}.enorm.weight", "BF16", [H]))
    t.append((f"{pre}.hnorm.weight", "BF16", [H]))
    t.append((f"{pre}.ffn.gate.weight", "BF16", [N_EXPERTS, H]))
    t.append((f"{pre}.ffn.gate.bias", "F32", [N_EXPERTS]))
    return t


def _global_tensors():
    return [
        ("embed.weight", "BF16", [VOCAB, H]),
        ("head.weight", "BF16", [VOCAB, H]),
        ("norm.weight", "BF16", [H]),
        ("hc_head_fn", "F32", [24, 4 * H]),
        ("hc_head_scale", "F32", [3]),
        ("hc_head_base", "F32", [24]),
    ]


def make_fixture_tree(root, n_shards=3, seed=1234):
    """Write a fake checkpoint under root/. Returns list of shard file names.

    Layout mimics the real one: embed alone in shard 1 (real shard 1 holds
    exactly 1 tensor), layers spread over the remaining shards, MTP last.
    Every expert's 6 tensors stay within one input shard.
    """
    rng = np.random.RandomState(seed)
    assert n_shards >= 3
    groups = [[] for _ in range(n_shards)]
    groups[0] += _global_tensors()
    for layer in range(N_LAYERS):
        groups[1 + (layer % (n_shards - 2))] += _layer_tensors(layer)
    groups[-1] += _mtp_tensors()

    weight_map = {}
    names = []
    for i, specs in enumerate(groups):
        fname = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        tensors = [(name, dtype, shape, _rand(rng, dtype, shape))
                   for name, dtype, shape in specs]
        stutil.write_shard(os.path.join(root, fname), tensors)
        for name, _, _, _ in tensors:
            weight_map[name] = fname
        names.append(fname)

    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        total = sum(
            stutil.tensor_nbytes(d, s)
            for specs in groups for _, d, s in specs
        )
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({
            "model_type": "deepseek_v4",
            "hidden_size": H,
            "moe_intermediate_size": M,
            "n_routed_experts": N_EXPERTS,
            "num_hidden_layers": N_LAYERS,
            "num_hash_layers": 3,
        }, f)
    return names
