"""M1 test 4 — resume from interruption.

Simulates a crash mid-conversion by raising from the converter's progress
callback after N committed tensors, then re-runs the converter and asserts
the final output is byte-for-byte identical to an uninterrupted run:

  * same set of output files,
  * every output shard byte-identical,
  * identical manifests.

Also covers the torn-write path: garbage bytes appended to the open shard
after the crash (simulating a partial tensor write killed mid-copy) must be
detected, truncated, and rewritten — final bytes still identical.
"""

import filecmp
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import convert as apus_convert

TARGET_BYTES = 64 * 1024


class Crash(Exception):
    pass


def count_source_tensors(src):
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        return len(json.load(f)["weight_map"])


def tree_bytes_equal(dir_a, dir_b, skip=()):
    files_a = sorted(f for f in os.listdir(dir_a)
                     if not f.startswith(".") and f not in skip)
    files_b = sorted(f for f in os.listdir(dir_b)
                     if not f.startswith(".") and f not in skip)
    if files_a != files_b:
        return False, f"file lists differ: {files_a} vs {files_b}"
    for f in files_a:
        if not filecmp.cmp(os.path.join(dir_a, f), os.path.join(dir_b, f),
                           shallow=False):
            return False, f"{f} differs"
    return True, "ok"


class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src")
        os.makedirs(cls.src)
        fixtures.make_fixture_tree(cls.src)
        cls.n_tensors = count_source_tensors(cls.src)
        # Reference: uninterrupted conversion.
        cls.ref = os.path.join(cls.tmp.name, "ref")
        conv = apus_convert.Converter(cls.src, cls.ref,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _crashed_run(self, name, crash_after, corrupt_open=False):
        dst = os.path.join(self.tmp.name, name)
        calls = {"n": 0}

        def cb(event, **kw):
            if event == "tensor":
                calls["n"] += 1
                if calls["n"] >= crash_after:
                    raise Crash()

        conv = apus_convert.Converter(self.src, dst, target_bytes=TARGET_BYTES)
        with self.assertRaises(Crash):
            conv.convert(progress_cb=cb)

        if corrupt_open:
            state = json.load(open(os.path.join(
                dst, apus_convert.STATE_FILE)))
            for g in state["groups"].values():
                if g["open"] is not None:
                    with open(os.path.join(dst, g["open"]["file"]),
                              "ab") as f:
                        f.write(os.urandom(12345))  # torn tail

        # Resume: must not redo or corrupt completed work.
        conv2 = apus_convert.Converter(self.src, dst,
                                       target_bytes=TARGET_BYTES)
        conv2.convert()
        conv2.finalize()
        return dst

    def test_resume_mid_conversion(self):
        dst = self._crashed_run("out_mid", crash_after=self.n_tensors // 3)
        ok, msg = tree_bytes_equal(self.ref, dst)
        self.assertTrue(ok, msg)

    def test_resume_with_torn_write(self):
        dst = self._crashed_run("out_torn", crash_after=self.n_tensors // 2,
                                corrupt_open=True)
        ok, msg = tree_bytes_equal(self.ref, dst)
        self.assertTrue(ok, msg)

    def test_resume_at_seal_boundary(self):
        """Crash exactly on a seal event (after state flush)."""
        dst = os.path.join(self.tmp.name, "out_seal")

        def cb(event, **kw):
            if event == "seal":
                raise Crash()

        conv = apus_convert.Converter(self.src, dst, target_bytes=TARGET_BYTES)
        with self.assertRaises(Crash):
            conv.convert(progress_cb=cb)
        conv2 = apus_convert.Converter(self.src, dst,
                                       target_bytes=TARGET_BYTES)
        conv2.convert()
        conv2.finalize()
        ok, msg = tree_bytes_equal(self.ref, dst)
        self.assertTrue(ok, msg)

    def test_rerun_complete_is_noop(self):
        """Re-running on a finished output must change nothing."""
        before = {f: open(os.path.join(self.ref, f), "rb").read()
                  for f in os.listdir(self.ref) if not f.startswith(".")}
        conv = apus_convert.Converter(self.src, self.ref,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()
        after = {f: open(os.path.join(self.ref, f), "rb").read()
                 for f in os.listdir(self.ref) if not f.startswith(".")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
