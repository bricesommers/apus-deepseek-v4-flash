#!/usr/bin/env python3
"""M5 fixture generator — deterministic regeneration of tests/m5/fixtures.

Wraps tools/oracle.py's generate_full (full mini-model: embed -> 4 stacked
layers (swa/hash + csa + hca + csa) -> hc_head -> norm -> head -> logits),
same master seed as M4b. Prints the f32-vs-f64 logit divergence table and
asserts f32 determinism.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle

if __name__ == "__main__":
    oracle.generate_full(os.path.join(ROOT, "tests", "m5", "fixtures"))
