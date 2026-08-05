#!/usr/bin/env python3
"""Regenerate all M4b oracle fixtures deterministically.

Usage: .venv/bin/python tests/m4b/run_oracle.py
Deletes and recreates tests/m4b/fixtures/ from tools/oracle.py with the
pinned master seed (20260729). Output is byte-for-byte reproducible.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle

if __name__ == "__main__":
    oracle.generate(os.path.join(ROOT, "tests", "m4b", "fixtures"), clean=True)
