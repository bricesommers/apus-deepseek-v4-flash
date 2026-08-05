"""M1 test 2 — coalesced per-expert layout.

Parses the output shard headers manually and asserts, for every expert:

  * its 6 tensors {w1,w1_scale,w2,w2_scale,w3,w3_scale} appear consecutively
    in the shard header, in that exact order,
  * their data regions are contiguous and adjacent (end == next start),
  * the whole slab lives in one shard,
  * the manifest's slab record (shard, offset, nbytes) matches the actual
    header offsets exactly,
  * slab size equals the expected per-expert byte count.
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024
EXPERT_RE = re.compile(
    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
SLAB_ORDER = ["w1.weight", "w1.scale", "w2.weight", "w2.scale",
              "w3.weight", "w3.scale"]


class TestCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src")
        cls.dst = os.path.join(cls.tmp.name, "out")
        os.makedirs(cls.src)
        fixtures.make_fixture_tree(cls.src)
        conv = apus_convert.Converter(cls.src, cls.dst,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()
        with open(os.path.join(cls.dst, "apus.index.json")) as f:
            cls.manifest = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _experts_from_headers(self):
        """Rebuild per-expert layout facts from the raw output headers."""
        experts = {}  # (block, eid) -> {"shard":..., "members":[(suffix, off, nbytes)], "order":[names]}
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(os.path.join(self.dst,
                                                                shard))
            names = [n for n in header if n != "__metadata__"]
            for pos, name in enumerate(names):
                m = EXPERT_RE.match(name)
                if not m:
                    continue
                block = f"{m.group(1)}.{m.group(2)}"
                key = (block, int(m.group(3)))
                begin, end = header[name]["data_offsets"]
                rec = experts.setdefault(
                    key, {"shard": shard, "members": [], "positions": [],
                          "names_in_header": names})
                self.assertEqual(rec["shard"], shard,
                                 f"expert {key} straddles shards")
                rec["members"].append(
                    (f"{m.group(4)}.{m.group(5)}", data_start + begin,
                     end - begin))
                rec["positions"].append(pos)
        return experts

    def test_slabs_contiguous_and_adjacent(self):
        experts = self._experts_from_headers()
        self.assertEqual(len(experts),
                         (fixtures.N_LAYERS + 1) * fixtures.N_EXPERTS)
        for key, rec in experts.items():
            # consecutive positions in the header, in the fixed slab order
            positions = rec["positions"]
            self.assertEqual(positions, list(range(positions[0],
                                                   positions[0] + 6)),
                             f"{key}: slab members not adjacent in header")
            # sort members by header position
            by_pos = [m for _, m in sorted(
                zip(positions, rec["members"]), key=lambda pm: pm[0])]
            self.assertEqual([s for s, _, _ in by_pos], SLAB_ORDER,
                             f"{key}: wrong intra-slab order")
            # data regions contiguous and adjacent
            for (s0, off0, nb0), (s1, off1, nb1) in zip(by_pos, by_pos[1:]):
                self.assertEqual(off0 + nb0, off1,
                                 f"{key}: gap between {s0} and {s1}")
            total = sum(nb for _, _, nb in by_pos)
            self.assertEqual(total, fixtures.PER_EXPERT_BYTES,
                             f"{key}: unexpected per-expert byte count")

    def test_manifest_slab_records_match_headers(self):
        experts = self._experts_from_headers()
        slabs = {(s["block"], s["expert"]): s
                 for s in self.manifest["expert_slabs"]}
        self.assertEqual(set(slabs), set(experts))
        for key, rec in experts.items():
            by_pos = [m for _, m in sorted(
                zip(rec["positions"], rec["members"]), key=lambda pm: pm[0])]
            slab_start = by_pos[0][1]
            slab_bytes = sum(nb for _, _, nb in by_pos)
            rec_m = slabs[key]
            self.assertEqual(rec_m["shard"], rec["shard"], key)
            self.assertEqual(rec_m["offset"], slab_start, key)
            self.assertEqual(rec_m["nbytes"], slab_bytes, key)

    def test_manifest_offsets_match_headers_for_all_tensors(self):
        """Every tensor_map entry must point at the same bytes the output
        shard header describes."""
        tmap = self.manifest["tensor_map"]
        n = 0
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(os.path.join(self.dst,
                                                                shard))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                rec = tmap[name]
                self.assertEqual(rec["shard"], shard, name)
                self.assertEqual(rec["offset"],
                                 data_start + meta["data_offsets"][0], name)
                self.assertEqual(rec["nbytes"],
                                 meta["data_offsets"][1]
                                 - meta["data_offsets"][0], name)
                n += 1
        self.assertEqual(n, len(tmap))


if __name__ == "__main__":
    unittest.main()
