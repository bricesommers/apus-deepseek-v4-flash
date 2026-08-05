#!/usr/bin/env python3
"""R1 kit tests: compare.py classification logic with fabricated goldens.

Fabricates tiny r1_goldens.json / apus-output files and checks both the
classify_case() unit behavior and the --offline end-to-end exit codes.
No torch, no model, no GPU — pure stdlib. Run:

    .venv/bin/python tests/r1b/test_compare.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "r1"))
import compare  # noqa: E402

EOS = 1
COMPARE_PY = ROOT / "tools" / "r1" / "compare.py"


def gsteps(ids, margins, top1s=None):
    top1s = top1s or ids
    return [{"pos": i, "emitted": t, "top1": p, "margin": m}
            for i, (t, p, m) in enumerate(zip(ids, top1s, margins))]


def gcase(ids, margins, temp=0.0, mode="raw"):
    return {"status": "ok", "mode": mode, "temp": temp,
            "prompt_ids": [10, 11, 12], "steps": gsteps(ids, margins)}


class TestClassify(unittest.TestCase):
    def test_identical(self):
        v = compare.classify_case("c", gcase([5, 6, 7], [1.0, 1.0, 1.0]),
                                  [5, 6, 7], "length", EOS)
        self.assertEqual(v["verdict"], "identical")
        self.assertEqual(v["compared"], 3)
        self.assertNotIn("capped", v)

    def test_identical_eos_normalized(self):
        # Reference records the eos step; apus stops silently (finish "stop").
        v = compare.classify_case("c", gcase([5, EOS], [1.0, 0.9]),
                                  [5], "stop", EOS)
        self.assertEqual(v["verdict"], "identical")

    def test_eos_position_mismatch(self):
        # apus continued where the reference emitted eos -> flip at that pos.
        v = compare.classify_case("c", gcase([5, EOS], [1.0, 3.0]),
                                  [5, 9, 9], "length", EOS)
        self.assertEqual(v["verdict"], "hard_mismatch")
        self.assertEqual(v["flip_pos"], 1)
        self.assertEqual(v["margin"], 3.0)

    def test_near_tie_flip(self):
        v = compare.classify_case("c", gcase([5, 6, 7], [1.0, 0.3, 1.0]),
                                  [5, 8, 7], "length", EOS, near_tie=0.5)
        self.assertEqual(v["verdict"], "near_tie_flip")
        self.assertEqual(v["flip_pos"], 1)
        self.assertEqual(v["apus_token"], 8)
        self.assertEqual(v["ref_token"], 6)
        self.assertEqual(v["ref_top1"], 6)

    def test_near_tie_boundary_is_hard(self):
        v = compare.classify_case("c", gcase([5, 6], [1.0, 0.5]),
                                  [5, 8], "length", EOS, near_tie=0.5)
        self.assertEqual(v["verdict"], "hard_mismatch")

    def test_hard_mismatch(self):
        v = compare.classify_case("c", gcase([5, 6], [1.0, 2.1]),
                                  [5, 8], "length", EOS)
        self.assertEqual(v["verdict"], "hard_mismatch")

    def test_sampled_flip_is_rng_divergence(self):
        v = compare.classify_case("c", gcase([5, 6], [1.0, 9.9], temp=1.0),
                                  [5, 8], "length", EOS)
        self.assertEqual(v["verdict"], "rng_divergence")

    def test_sampled_identical(self):
        v = compare.classify_case("c", gcase([5, 6], [1.0, 1.0], temp=1.0),
                                  [5, 6], "length", EOS)
        self.assertEqual(v["verdict"], "identical")

    def test_capped_prefix_is_identical(self):
        v = compare.classify_case("c", gcase([5, 6, 7, 8], [1.0] * 4),
                                  [5, 6], "length", EOS)
        self.assertEqual(v["verdict"], "identical")
        self.assertTrue(v["capped"])
        self.assertEqual(v["compared"], 2)


class TestOfflineCLI(unittest.TestCase):
    """End-to-end --offline runs: exit 0 iff no hard mismatch/error."""

    def run_offline(self, cases_g, cases_a, encode=None, e2e=None):
        with tempfile.TemporaryDirectory() as td:
            g = {"meta": {"eos_token_id": EOS}, "cases": cases_g}
            a = {"cases": cases_a, "encode_checks": encode or {},
                 "e2e": e2e or {}}
            gp = os.path.join(td, "g.json")
            ap = os.path.join(td, "a.json")
            with open(gp, "w") as f:
                json.dump(g, f)
            with open(ap, "w") as f:
                json.dump(a, f)
            r = subprocess.run(
                [sys.executable, str(COMPARE_PY), "--goldens", gp,
                 "--offline", ap],
                capture_output=True, text=True)
            return r

    def test_pass_identical_plus_near_tie_plus_rng(self):
        g = {"ok1": gcase([5, 6], [1.0, 1.0]),
             "ok2": gcase([5, 6], [1.0, 0.1]),
             "ok3": gcase([5, 6], [1.0, 9.0], temp=1.0)}
        a = {"ok1": {"ids": [5, 6], "finish_reason": "length"},
             "ok2": {"ids": [5, 7], "finish_reason": "length"},
             "ok3": {"ids": [5, 7], "finish_reason": "length"}}
        r = self.run_offline(g, a)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("near_tie_flip: 1", r.stdout)
        self.assertIn("rng_divergence: 1", r.stdout)

    def test_fail_hard_mismatch(self):
        g = {"bad": gcase([5, 6], [1.0, 4.0])}
        a = {"bad": {"ids": [5, 7], "finish_reason": "length"}}
        r = self.run_offline(g, a)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("hard_mismatch", r.stdout)

    def test_fail_apus_error(self):
        g = {"c": gcase([5], [1.0])}
        a = {"c": {"error": "apus serve: EOF on stdout"}}
        r = self.run_offline(g, a)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_fail_encode_check(self):
        g = {"c": gcase([5], [1.0], mode="chat")}
        a = {"c": {"ids": [5], "finish_reason": "length"}}
        enc = {"c": {"mode": "chat", "ok": False, "apus_ids": [1, 2]}}
        r = self.run_offline(g, a, encode=enc)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_pass_with_e2e(self):
        g = {"c": gcase([5, 6], [1.0, 1.0])}
        a = {"c": {"ids": [5, 6], "finish_reason": "length"}}
        e2e = {"c": {"ids": [5, 6], "finish_reason": "length"}}
        r = self.run_offline(g, a, e2e=e2e)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("identical: 2", r.stdout)

    def test_failed_golden_case_skipped(self):
        g = {"c": gcase([5], [1.0]),
             "hostfail": {"status": "failed", "error": "boom"}}
        a = {"c": {"ids": [5], "finish_reason": "length"}}
        r = self.run_offline(g, a)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("hostfail", r.stdout)  # noted as skipped


class TestBattery(unittest.TestCase):
    def test_prompts_json_parses_and_is_wellformed(self):
        with open(ROOT / "tests" / "r1b" / "prompts.json") as f:
            b = json.load(f)
        self.assertEqual(len(b["cases"]), 8)
        modes = {"raw", "chat", "thinking"}
        n_sampled = 0
        for c in b["cases"]:
            self.assertIn(c["mode"], modes)
            self.assertIsInstance(c["prompt"], str)
            self.assertGreater(c["max_tokens"], 0)
            if float(c.get("temp", 0)) > 0:
                n_sampled += 1
                self.assertIn("seed", c)
        self.assertEqual(n_sampled, 1)  # exactly one fixed-seed sampled case


if __name__ == "__main__":
    unittest.main()
