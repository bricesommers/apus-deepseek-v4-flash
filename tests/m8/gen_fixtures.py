#!/usr/bin/env python3
"""M8 fixture generator — deterministic regeneration of tests/m8/fixtures.

Wraps tools/oracle.py's generate_m8: the M5 synthetic mini-model plus a
depth-1 MTP block (SWA ratio 0, bias-gate MoE, e/h glue, own hc_head,
shared head), with goldens for the MTP forward (true-pair replay + greedy
draft chain) and a greedy draft/verify spec episode. Same master seed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle

if __name__ == "__main__":
    oracle.generate_m8(os.path.join(ROOT, "tests", "m8", "fixtures"))
