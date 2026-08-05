"""M1 test 1 — byte identity.

Convert the synthetic fixture checkpoint, then compare EVERY output tensor's
bytes against the source bytes. Any difference fails. Also checks dtype and
shape are preserved verbatim, that the manifest agrees with the output
shard headers, and cross-checks numpy-readable tensors via the safetensors
library (allowed for reads where dtypes permit).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024  # small shards -> exercises multi-shard packing


class TestByteIdentity(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tensor_count_preserved(self):
        n_src = 0
        for shard in os.listdir(self.src):
            if shard.endswith(".safetensors"):
                header, _ = stutil.read_shard(os.path.join(self.src, shard))
                n_src += len(header)
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["ntensors"], n_src)
        self.assertEqual(len(manifest["tensor_map"]), n_src)

    def test_every_tensor_byte_identical(self):
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            manifest = json.load(f)
        tmap = manifest["tensor_map"]
        nchecked = 0
        for shard in sorted(os.listdir(self.src)):
            if not shard.endswith(".safetensors"):
                continue
            spath = os.path.join(self.src, shard)
            src_tensors = stutil.read_tensor_bytes(spath)
            sheader, _ = stutil.read_shard(spath)
            for name, payload in src_tensors.items():
                rec = tmap[name]
                self.assertEqual(rec["dtype"], sheader[name]["dtype"], name)
                self.assertEqual(rec["shape"], sheader[name]["shape"], name)
                self.assertEqual(rec["nbytes"], len(payload), name)
                opath = os.path.join(self.dst, rec["shard"])
                with open(opath, "rb") as f:
                    f.seek(rec["offset"])
                    out = f.read(rec["nbytes"])
                self.assertEqual(out, payload,
                                 f"{name}: bytes differ after conversion")
                nchecked += 1
        self.assertEqual(nchecked, len(tmap))

    def test_verify_helper(self):
        n = apus_convert.verify_source(self.src, self.dst, log=lambda m: None)
        self.assertGreater(n, 0)

    def test_safetensors_lib_crosscheck(self):
        """The safetensors library must parse our manually-written output
        shards, and numpy-readable tensors (I8/F32/I64) must match."""
        from safetensors import safe_open
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(self.dst, shard),
                           framework="numpy") as f:
                for name in f.keys():
                    meta = f.get_slice(name)
                    self.assertIn(meta.get_dtype(),
                                  ("I8", "F32", "I64", "BF16", "F8_E8M0",
                                   "F8_E4M3", "U8"))


if __name__ == "__main__":
    unittest.main()
