# tests/m1 — converter + downloader (M1)

Dependency: the project venv at `../../.venv` (numpy; safetensors used for
read cross-checks only). No pytest — plain `unittest`.

Run all:

```sh
../../.venv/bin/python -m unittest discover -s . -v
```

or individually, e.g.:

```sh
../../.venv/bin/python -m unittest test_3_dequant -v
```

What each file does:

- `stutil.py` — minimal *manual* safetensors reader/writer (8-byte LE header
  length + JSON header + raw data). The safetensors library is deliberately
  not used for writing: numpy has no F8_E8M0/F8_E4M3/BF16 dtypes and the
  fixtures must be raw bytes we fully control.
- `fixtures.py` — synthetic DeepSeek-V4-Flash checkpoint at tiny scale
  (H=64, M=32, 5 layers x 6 experts + MTP): same naming scheme and dtypes
  as the real index, MXFP4 layout invariants intact (packed == O*K/2,
  scales == O*K/32), per-expert slab 3264 B, 3 input shards with a
  weight_map index. Payloads are random bytes — sufficient for
  byte-identity testing.
- `test_1_byte_identity.py` — every output tensor's bytes equal the source
  bytes; dtype/shape preserved; manifest consistent; safetensors-lib parse
  cross-check.
- `test_2_coalescing.py` — each expert's 6 tensors are consecutive in the
  shard header, contiguous+adjacent in the data region, single-shard, and
  the manifest slab records match the actual header offsets.
- `test_3_dequant.py` — numpy MXFP4 dequant reference for the M3 C kernel:
  E2M1 16-entry LUT (s ee m, bias 1), low nibble = even K index, UE8M0
  scale = 2^(byte-127), one scale per 32 elements along K. Hand-computed
  and boundary cases.
- `test_4_resume.py` — crash mid-conversion (progress-callback fault
  injection), torn-write tail on the open shard, crash at seal boundary,
  no-op rerun: final output always byte-identical to the uninterrupted run.
- `test_5_index_realism.py` — validates the converter's assumptions against
  the REAL `reference/model.safetensors.index.json`: naming scheme,
  complete 6-tensor expert groups, no expert split across input shards,
  MXFP4 shape arithmetic from config.json, per-expert 13,369,344 B, size
  bookkeeping. NOTE: the real index carries no shapes/dtypes (name -> shard
  only); shapes come from shard headers at conversion time.
- `test_6_download_driver.py` — offline (local "remote" dir) end-to-end
  download.py run: kill mid-download and mid-conversion, restart, partial
  `.part` resume, source shards deleted only after byte-verification, final
  container byte-identical to direct conversion.
