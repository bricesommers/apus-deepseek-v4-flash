#!/usr/bin/env python3
"""M11a fixture generator — deterministic regeneration of tests/m11a/fixtures.

Wraps tools/oracle_dspark.py's generate: the m5-style mini-model extended to
5 layers (DSpark target layers [2,3,4]) + 3 synthetic DSpark stages under the
real mtp.* naming, with goldens for the target-layer join, stage KV prefill,
draft rounds, spec episodes (greedy/sampled), forced draft patterns and
rollback digests. Same master seed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle_dspark

if __name__ == "__main__":
    oracle_dspark.generate(os.path.join(ROOT, "tests", "m11a", "fixtures"))
