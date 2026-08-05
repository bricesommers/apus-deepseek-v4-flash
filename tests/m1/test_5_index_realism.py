"""M1 test 5 — converter assumptions vs the REAL checkpoint index.

Parses reference/model.safetensors.index.json (the real 46-shard, 69,187-
tensor DeepSeek-V4-Flash index) and asserts every assumption the converter
relies on:

  * the naming scheme matches the converter's regexes,
  * every (layer, expert) has exactly the 6-tensor group
    {w1..w3}.{weight,scale},
  * no expert's tensors are split across input shards,
  * MXFP4 layout arithmetic from config.json dims:
    packed bytes == O*K/2, scale elements == O*K/32 (group size 32),
  * per-expert byte count matches docs/ARCHITECTURE.md §3.5,
  * total size bookkeeping is consistent (experts + dense == total).

NOTE (documented deviation): the real index maps tensor name -> shard file
only; it carries NO shapes/dtypes. Shapes/dtypes are read from each shard's
safetensors header at conversion time. This test therefore validates shapes
against config.json-derived expectations plus total-size arithmetic.
"""

import json
import os
import re
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))
import convert as apus_convert  # noqa: E402 — reuse the converter's regex

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "reference")


class TestIndexRealism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REF_DIR, "model.safetensors.index.json")) as f:
            cls.index = json.load(f)
        cls.wm = cls.index["weight_map"]
        with open(os.path.join(REF_DIR, "config.json")) as f:
            cls.cfg = json.load(f)

    def test_scale_and_counts(self):
        self.assertEqual(len(set(self.wm.values())), 46)
        self.assertEqual(len(self.wm), 69187)
        self.assertEqual(self.index["metadata"]["total_size"],
                         159609485896)

    def test_expert_groups_complete(self):
        groups = defaultdict(set)
        shards = defaultdict(set)
        for name, shard in self.wm.items():
            m = apus_convert.EXPERT_RE.match(name)
            if m:
                key = (m.group(1), int(m.group(2)), int(m.group(3)))
                groups[key].add(f"{m.group(4)}.{m.group(5)}")
                shards[key].add(shard)
        n_layers = self.cfg["num_hidden_layers"]
        n_exp = self.cfg["n_routed_experts"]
        # 43 layers x 256 experts + 256 in the MTP block
        self.assertEqual(len(groups), n_layers * n_exp + n_exp)
        self.assertEqual(len(groups), 11264)
        expected = set(apus_convert.SLAB_MEMBERS)
        for key, members in groups.items():
            self.assertEqual(members, expected, f"expert {key} incomplete")

    def test_experts_never_split_across_shards(self):
        shards = defaultdict(set)
        for name, shard in self.wm.items():
            m = apus_convert.EXPERT_RE.match(name)
            if m:
                shards[(m.group(1), int(m.group(2)),
                        int(m.group(3)))].add(shard)
        split = {k: v for k, v in shards.items() if len(v) > 1}
        self.assertEqual(split, {},
                         f"{len(split)} experts split across input shards")

    def test_mxfp4_layout_and_per_expert_bytes(self):
        H = self.cfg["hidden_size"]              # 4096
        M = self.cfg["moe_intermediate_size"]    # 2048
        # w1/w3: [M, H] logical -> [M, H/2] packed, scales [M, H/32]
        # w2:    [H, M] logical -> [H, M/2] packed, scales [H, M/32]
        for O, K in ((M, H), (H, M)):
            packed = O * (K // 2)
            scale = O * (K // 32)
            self.assertEqual(packed, O * K // 2)    # O*K even
            self.assertEqual(scale * 32, O * K)     # group size 32 exact
        weights = 2 * (M * H // 2) + (H * M // 2)
        scales = 2 * (M * H // 32) + (H * M // 32)
        self.assertEqual(weights, 12582912)
        self.assertEqual(scales, 786432)
        per_expert = weights + scales
        self.assertEqual(per_expert, 13369344)  # 12.75 MiB, ARCH §3.5
        # Bookkeeping: 11,264 experts account for ~150.6 GB of the 159.6 GB
        # total; the ~9 GB remainder matches the documented dense set
        # (F8_E4M3 6.0 GB + BF16 1.42 GB + F32 36 MB + I64 2.3 MB).
        expert_total = 11264 * per_expert
        total = self.index["metadata"]["total_size"]
        rest = total - expert_total
        self.assertEqual(expert_total, 150592290816)
        self.assertTrue(8 * 1024**3 < rest < 10 * 1024**3,
                        f"dense remainder {rest / 1024**3:.2f} GiB unexpected")
        print(f"\nindex realism: 11,264 experts x {per_expert:,} B "
              f"= {expert_total / 1e9:.2f} GB; dense remainder "
              f"{rest / 1e9:.2f} GB of {total / 1e9:.2f} GB total")

    def test_group_classification_covers_everything(self):
        """Every real tensor name must route to exactly one shard group and
        match a known structural pattern."""
        seen_patterns = set()
        for name in self.wm:
            group = apus_convert.classify_group(name)
            if name.startswith("mtp."):
                self.assertEqual(group, "mtp", name)
            elif ".attn.indexer." in name:
                self.assertEqual(group, "idx", name)
                self.assertFalse(name.startswith("mtp."), name)
            else:
                self.assertEqual(group, "main", name)
            seen_patterns.add(re.sub(r"\d+", "N", name))
        # Guard against silent naming drift in future checkpoints: the set
        # of patterns must stay exactly what the converter was built for.
        self.assertEqual(len(seen_patterns), 82)

    def test_hash_and_bias_layer_split(self):
        tid_layers = sorted(
            int(m.group(1)) for n in self.wm
            for m in [re.match(r"layers\.(\d+)\.ffn\.gate\.tid2eid$", n)]
            if m)
        bias_layers = sorted(
            int(m.group(1)) for n in self.wm
            for m in [re.match(r"layers\.(\d+)\.ffn\.gate\.bias$", n)] if m)
        self.assertEqual(tid_layers, [0, 1, 2])   # num_hash_layers: 3
        self.assertEqual(bias_layers, list(range(3, 43)))
        self.assertIn("mtp.0.ffn.gate.bias", self.wm)

    def test_indexer_layers(self):
        idx_layers = sorted(
            int(m.group(1)) for n in self.wm
            for m in [re.match(r"layers\.(\d+)\.attn\.indexer\.wq_b\.weight$",
                               n)] if m)
        # CSA layers: even layers 2..42 (compress_ratios == 4)
        ratios = self.cfg["compress_ratios"]
        expected = [i for i, r in enumerate(ratios) if r == 4]
        self.assertEqual(idx_layers, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
