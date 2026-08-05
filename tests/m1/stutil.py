"""Minimal manual safetensors reader/writer for M1 test fixtures.

Deliberately does NOT use the safetensors library: writing is done as raw
bytes (8-byte little-endian header length + JSON header + raw data) exactly
like tools/convert.py, so the fixtures exercise the real format rather than
a library round-trip, and dtypes numpy lacks (F8_E8M0, F8_E4M3, BF16) can
be written without lossy detours.
"""

import json
import os
import struct

DTYPE_SIZES = {
    "I8": 1, "U8": 1, "F8_E8M0": 1, "F8_E4M3": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8,
}


def tensor_nbytes(dtype, shape):
    n = DTYPE_SIZES[dtype]
    for d in shape:
        n *= d
    return n


def write_shard(path, tensors):
    """Write tensors = [(name, dtype, shape, payload_bytes), ...] as one
    safetensors shard, manually."""
    header = {}
    off = 0
    data = []
    for name, dtype, shape, payload in tensors:
        assert len(payload) == tensor_nbytes(dtype, shape), (
            f"{name}: payload {len(payload)} bytes != "
            f"{tensor_nbytes(dtype, shape)} expected for {dtype} {shape}")
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(payload)],
        }
        off += len(payload)
        data.append(payload)
    doc = json.dumps(header, separators=(",", ":")).encode()
    # safetensors writers pad the header with spaces; harmless and realistic.
    pad = (-len(doc)) % 8
    doc += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(doc)))
        f.write(doc)
        for payload in data:
            f.write(payload)


def read_shard(path):
    """Parse a shard manually. Returns (header_dict, data_start) where
    header_dict maps name -> {dtype, shape, data_offsets} and data_start is
    the absolute file offset of the data region."""
    with open(path, "rb") as f:
        raw = f.read(8)
        assert len(raw) == 8, f"{path}: too small for safetensors"
        (hlen,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def read_tensor_bytes(path, header=None, data_start=None):
    """Returns dict name -> raw payload bytes."""
    if header is None:
        header, data_start = read_shard(path)
    out = {}
    with open(path, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            out[name] = f.read(end - begin)
    return out
