"""M1 test 6 — download driver (offline mode).

Exercises tools/download.py's state machine without network: a local
"remote" directory stands in for HuggingFace (--source-dir mode uses the
same resumable .part-copy logic and the same download -> convert -> verify
-> delete pipeline).

Asserts:
  * killed mid-run (during a download copy and during conversion), the
    driver restarts cleanly and finishes,
  * the final container is byte-identical to a direct one-shot conversion
    (incremental shard-by-shard conversion is deterministic),
  * source shards are deleted from the landing zone only after conversion,
    bounding peak disk usage.
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
import download as apus_download

TARGET_BYTES = 64 * 1024


class Crash(Exception):
    pass


class TestDownloadDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.remote = os.path.join(cls.tmp.name, "remote")
        os.makedirs(cls.remote)
        fixtures.make_fixture_tree(cls.remote)
        # Reference: direct one-shot conversion of the whole tree.
        cls.ref = os.path.join(cls.tmp.name, "ref")
        conv = apus_convert.Converter(cls.remote, cls.ref,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _compare_to_ref(self, out):
        # The driver (not the one-shot Converter) also copies small support
        # files (config.json, tokenizer*, generation_config) into out/ —
        # intended since M12c: the container dir must be self-contained.
        skip = {apus_convert.STATE_FILE, *apus_download.Driver.SUPPORT_FILES}
        ref_files = sorted(f for f in os.listdir(self.ref)
                           if not f.startswith(".") and f not in skip)
        out_files = sorted(f for f in os.listdir(out)
                           if not f.startswith(".") and f not in skip)
        self.assertEqual(ref_files, out_files)
        for f in ref_files:
            self.assertTrue(
                filecmp.cmp(os.path.join(self.ref, f), os.path.join(out, f),
                            shallow=False), f"{f} differs")
        # support files present in the fixture remote must land in out/
        for f in apus_download.Driver.SUPPORT_FILES:
            src = os.path.join(self.remote, f)
            if os.path.exists(src):
                self.assertTrue(
                    filecmp.cmp(src, os.path.join(out, f), shallow=False),
                    f"support file {f} missing/different in out/")

    def test_clean_run(self):
        work = os.path.join(self.tmp.name, "work1")
        out = os.path.join(self.tmp.name, "out1")
        apus_download.Driver(work, out, source_dir=self.remote,
                             target_bytes=TARGET_BYTES).run()
        self._compare_to_ref(out)
        # landing zone holds no leftover shards (only small config/index)
        leftover = [f for f in os.listdir(os.path.join(work, "src"))
                    if f.endswith(".safetensors")]
        self.assertEqual(leftover, [])
        # remote untouched: offline mode copies, never deletes the source
        self.assertTrue(any(f.endswith(".safetensors")
                            for f in os.listdir(self.remote)))

    def test_kill_and_restart(self):
        work = os.path.join(self.tmp.name, "work2")
        out = os.path.join(self.tmp.name, "out2")
        calls = {"n": 0}

        def cb(event, **kw):
            # Kill during the second shard's conversion.
            if event == "tensor":
                calls["n"] += 1
                if calls["n"] == 50:
                    raise Crash()

        driver = apus_download.Driver(work, out, source_dir=self.remote,
                                      target_bytes=TARGET_BYTES,
                                      progress_cb=cb)
        with self.assertRaises(Crash):
            driver.run()

        # Restart with a second crash point, this time mid-download.
        # (Fixture shards are smaller than one 8 MB copy chunk, so the first
        # download_chunk event is the right kill point: the .part file then
        # holds the complete-but-unrenamed payload.)
        calls2 = {"n": 0}

        def cb2(event, **kw):
            if event == "download_chunk":
                calls2["n"] += 1
                if calls2["n"] == 1:
                    raise Crash()

        driver2 = apus_download.Driver(work, out, source_dir=self.remote,
                                       target_bytes=TARGET_BYTES,
                                       progress_cb=cb2)
        with self.assertRaises(Crash):
            driver2.run()
        # A partial .part file must exist now (download resume state).
        parts = [f for f in os.listdir(os.path.join(work, "src"))
                 if f.endswith(".part")]
        self.assertEqual(len(parts), 1)

        # Final run to completion.
        apus_download.Driver(work, out, source_dir=self.remote,
                             target_bytes=TARGET_BYTES).run()
        self._compare_to_ref(out)
        leftover = [f for f in os.listdir(os.path.join(work, "src"))
                    if f.endswith((".safetensors", ".part"))]
        self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()
