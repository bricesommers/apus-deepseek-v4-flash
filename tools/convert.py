#!/usr/bin/env python3
"""apus M1 — HF safetensors -> apus container converter.

Converts a directory of HuggingFace safetensors shards (DeepSeek-V4-Flash)
into the apus weight container described in docs/ARCHITECTURE.md §6:

  * Byte-identical copy of every tensor. Payloads are treated as raw bytes;
    this tool never requantizes, transcodes, or even interprets tensor data.
  * Coalesced per-expert layout: within each output shard, the 6 tensors of
    every routed expert {w1,w1_scale,w2,w2_scale,w3,w3_scale} are contiguous
    and adjacent, so the engine fetches a whole expert with one pread.
  * Output shards are ordinary safetensors files, written MANUALLY (8-byte
    little-endian header length + JSON header + raw data). safetensors.numpy
    is not used for writing: numpy has no F8_E8M0/F8_E4M3 dtype and we need
    exact control over tensor order and byte identity.
  * Shard groups: main weights in `apus-NNNNN.safetensors`, MTP block in
    `apus-mtp-NNNNN.safetensors`, lightning-indexer tensors in
    `apus-idx-NNNNN.safetensors` (so the engine can lazy-load those sets).
  * Manifest `apus.index.json`: format version, config hash, full tensor map
    (name -> shard, absolute file offset, nbytes, dtype, shape) and
    per-expert slab records (layer, expert -> shard, offset, total bytes).

Resumability / crash safety
---------------------------
Conversion is driven input-shard by input-shard. A state file
(`apus.convert.state.json` in the output dir) is rewritten atomically after
every single tensor append, after every shard seal, and after every finished
input shard. Re-running after an interruption:

  1. Sealed output shards are verify-before-trust checked (size + header
     hash recorded in the state).
  2. The open output shard is validated against the state: bytes committed
     in the state must be present on disk; any tail beyond the last commit
     (a torn tensor write) is truncated and rewritten.
  3. Completed input shards are skipped entirely.

The append sequence is a deterministic function of the input set, so an
interrupted+resumed run produces byte-identical output to an uninterrupted
run (covered by tests/m1/test_4_resume.py).

Output shard format detail: shards are created with a fixed-capacity header
region (HEADER_RESERVE bytes) holding a placeholder JSON document. This lets
us append tensor data streaming-style and "seal" the shard later by simply
rewriting the header region in place — no data copies, and the file is a
structurally valid safetensors file at every point in time. Sealing pads the
JSON with trailing spaces, which the safetensors format permits.

Usage:
    python tools/convert.py convert  SRC_DIR DST_DIR [--shard NAME ...]
                                     [--target-bytes N]
    python tools/convert.py verify   SRC_DIR DST_DIR [--shard NAME ...]
    python tools/convert.py finalize SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile

FORMAT_VERSION = 1

# Fixed header capacity of every output shard. ~100 bytes of JSON per tensor
# entry => room for well over 100k tensors per shard; validated on seal.
HEADER_RESERVE = 16 * 1024 * 1024

# Target output shard size (~5 GB per docs/ARCHITECTURE.md §6). A shard is
# sealed before a tensor/expert-slab append that would exceed this, so the
# actual size stays within target + one expert slab (~13.4 MB).
DEFAULT_TARGET_BYTES = 5 * 1024**3

COPY_CHUNK = 8 * 1024 * 1024

STATE_FILE = "apus.convert.state.json"
MANIFEST_FILE = "apus.index.json"

MAIN_PREFIX = "apus"
MTP_PREFIX = "apus-mtp"
IDX_PREFIX = "apus-idx"

# layers.{L}.ffn.experts.{E}.w{1,2,3}.{weight,scale}  (same under mtp.{N}.)
EXPERT_RE = re.compile(
    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
# Fixed intra-slab order: w1, w1_scale, w2, w2_scale, w3, w3_scale.
SLAB_MEMBERS = tuple(
    f"{w}.{kind}" for w in ("w1", "w2", "w3") for kind in ("weight", "scale")
)


# --------------------------------------------------------------------------
# safetensors header reading (source shards)
# --------------------------------------------------------------------------

class SrcTensor:
    """One tensor in a source shard: location + self-description, no data."""
    __slots__ = ("name", "dtype", "shape", "src_path", "file_offset", "nbytes")

    def __init__(self, name, dtype, shape, src_path, file_offset, nbytes):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.src_path = src_path
        self.file_offset = file_offset  # absolute offset in src_path
        self.nbytes = nbytes


def read_st_header(path):
    """Parse a safetensors header. Returns dict name -> SrcTensor.

    Only the header is read; tensor payloads are never touched here.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (too small)")
        (hlen,) = struct.unpack("<Q", raw)
        hjson = f.read(hlen)
        if len(hjson) != hlen:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(hjson)
    data_start = 8 + hlen
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        tensors[name] = SrcTensor(
            name, meta["dtype"], list(meta["shape"]), path,
            data_start + begin, end - begin,
        )
    return tensors


def list_input_shards(src_dir):
    """Input shard file names in deterministic processing order.

    Uses model.safetensors.index.json when present (the real checkpoint
    layout), else every *.safetensors file in the directory.
    """
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))
    return sorted(
        n for n in os.listdir(src_dir)
        if n.endswith(".safetensors") and not n.startswith("apus")
    )


def config_hash(src_dir):
    """Stable hash identifying the model configuration being converted."""
    for name in ("config.json", "model.safetensors.index.json"):
        path = os.path.join(src_dir, name)
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(COPY_CHUNK), b""):
                    h.update(chunk)
            return f"sha256:{h.hexdigest()}"
    return "sha256:none"


# --------------------------------------------------------------------------
# Tensor classification and ordering
# --------------------------------------------------------------------------

def classify_group(name):
    """Route a tensor name to its output shard group."""
    if name.startswith("mtp."):
        return "mtp"
    if ".attn.indexer." in name:
        return "idx"
    return "main"


def order_contribution(tensors):
    """Deterministic write order for one input shard's contribution to a
    single group.

    Expert tensors come first, grouped into per-expert slabs ordered by
    (block, expert id) with the fixed SLAB_MEMBERS order inside; every slab
    is therefore contiguous in the output. Remaining tensors follow, sorted
    by name. Raises if an expert's 6 tensors are not all present — the real
    checkpoint never splits an expert across input shards (verified by
    tests/m1/test_5_index_realism.py), so a partial slab means a corrupt or
    unexpected input and we fail loudly rather than emit a broken layout.
    """
    slabs = {}   # (kind, block_id, expert_id) -> {member_suffix: name}
    others = []
    for name in tensors:
        m = EXPERT_RE.match(name)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            slabs.setdefault(key, {})[f"{m.group(4)}.{m.group(5)}"] = name
        else:
            others.append(name)

    ordered = []
    for key in sorted(slabs):
        members = slabs[key]
        missing = [s for s in SLAB_MEMBERS if s not in members]
        if missing:
            raise ValueError(
                f"expert {key} incomplete in this input shard "
                f"(missing {missing}); experts must not be split across "
                f"input shards"
            )
        ordered.extend(members[s] for s in SLAB_MEMBERS)
    ordered.extend(sorted(others))
    return ordered


def slab_prefix(name):
    """'layers.3.ffn.experts.17' style prefix for an expert tensor name."""
    m = EXPERT_RE.match(name)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}.ffn.experts.{m.group(3)}"


# --------------------------------------------------------------------------
# Output shard writer
# --------------------------------------------------------------------------

def _placeholder_header():
    doc = json.dumps({"__metadata__": {"apus_state": "open"}}).encode()
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _sealed_header_bytes(entries):
    """Final JSON header for a shard, padded with spaces to HEADER_RESERVE."""
    header = {}
    off = 0
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype"],
            "shape": e["shape"],
            "data_offsets": [off, off + e["nbytes"]],
        }
        off += e["nbytes"]
    doc = json.dumps(header, separators=(",", ":")).encode()
    if len(doc) > HEADER_RESERVE:
        raise ValueError(
            f"shard header needs {len(doc)} bytes > HEADER_RESERVE "
            f"({HEADER_RESERVE}); raise HEADER_RESERVE in tools/convert.py"
        )
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _header_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(8 + HEADER_RESERVE))
    return h.hexdigest()


class GroupStream:
    """Manages the output shard sequence of one group (main/mtp/idx).

    The "open" shard is created lazily on the first append, appended to
    tensor by tensor, and sealed (header rewritten in place) once the next
    append would exceed the target size. All mutations are reflected in the
    converter state and flushed to disk before the next mutation, so a kill
    at any point leaves a resumable on-disk state.
    """

    PREFIXES = {"main": MAIN_PREFIX, "mtp": MTP_PREFIX, "idx": IDX_PREFIX}

    def __init__(self, group, out_dir, state, save_state):
        self.group = group
        self.prefix = self.PREFIXES[group]
        self.out_dir = out_dir
        self.state = state          # state["groups"][group], mutated in place
        self.save_state = save_state

    # -- state helpers ----------------------------------------------------

    @property
    def gstate(self):
        return self.state["groups"][self.group]

    def _shard_name(self, idx):
        return f"{self.prefix}-{idx:05d}.safetensors"

    def _open_path(self):
        open_ = self.gstate["open"]
        if open_ is None:
            return None
        return os.path.join(self.out_dir, open_["file"])

    # -- write path --------------------------------------------------------

    def _create_open(self):
        idx = self.gstate["next_shard_idx"]
        name = self._shard_name(idx)
        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(_placeholder_header())
        self.gstate["open"] = {"file": name, "data_bytes": 0, "entries": []}
        self.save_state()

    def append_tensor(self, tensor, progress_cb=None):
        """Append one tensor's raw bytes to the open shard."""
        open_ = self.gstate["open"]
        if open_ is None:
            self._create_open()
            open_ = self.gstate["open"]
        path = os.path.join(self.out_dir, open_["file"])
        data_off = open_["data_bytes"]
        with open(path, "r+b") as out, open(tensor.src_path, "rb") as src:
            out.seek(8 + HEADER_RESERVE + data_off)
            src.seek(tensor.file_offset)
            remaining = tensor.nbytes
            while remaining:
                chunk = src.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(f"{tensor.src_path}: short read on {tensor.name}")
                out.write(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        open_["entries"].append({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "offset": data_off,   # relative to data region start
            "nbytes": tensor.nbytes,
        })
        open_["data_bytes"] += tensor.nbytes
        self.save_state()
        if progress_cb:
            progress_cb("tensor", group=self.group, name=tensor.name,
                        shard=open_["file"])

    def ensure_capacity(self, nbytes, target_bytes, progress_cb=None):
        """Seal the open shard if appending nbytes would exceed the target.

        Called before a whole expert slab (or a single dense tensor), so a
        slab is never split across shards.
        """
        open_ = self.gstate["open"]
        if open_ and open_["data_bytes"] > 0 and \
                open_["data_bytes"] + nbytes > target_bytes:
            self.seal(progress_cb)

    def seal(self, progress_cb=None):
        """Finalize the open shard: rewrite its header region in place."""
        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        header = _sealed_header_bytes(open_["entries"])
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(header)
            f.flush()
            os.fsync(f.fileno())
        size = os.path.getsize(path)
        self.gstate["sealed"][open_["file"]] = {
            "size": size,
            "header_sha256": _header_sha256(path),
            "ntensors": len(open_["entries"]),
        }
        sealed_name = open_["file"]
        self.gstate["open"] = None
        self.gstate["next_shard_idx"] += 1
        self.save_state()
        if progress_cb:
            progress_cb("seal", group=self.group, shard=sealed_name,
                        ntensors=self.gstate["sealed"][sealed_name]["ntensors"])

    # -- resume validation --------------------------------------------------

    def validate(self):
        """Verify-before-trust check of this group's on-disk output.

        Sealed shards must match recorded size + header hash. The open shard
        must contain at least the committed bytes; a torn tail (crash mid
        tensor write, after the previous state flush) is truncated. A shard
        whose seal was written but never recorded in the state is adopted as
        sealed. Anything else is corruption: fail loudly.
        """
        for name, rec in self.gstate["sealed"].items():
            path = os.path.join(self.out_dir, name)
            if not os.path.exists(path):
                raise ValueError(f"missing sealed output shard {name}")
            if os.path.getsize(path) != rec["size"]:
                raise ValueError(f"sealed shard {name}: size mismatch "
                                 f"(state {rec['size']}, disk "
                                 f"{os.path.getsize(path)})")
            if _header_sha256(path) != rec["header_sha256"]:
                raise ValueError(f"sealed shard {name}: header hash mismatch")

        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        if not os.path.exists(path):
            raise ValueError(f"missing open output shard {open_['file']}")
        data_start = 8 + HEADER_RESERVE
        expected = data_start + open_["data_bytes"]
        size = os.path.getsize(path)
        if size < expected:
            raise ValueError(
                f"open shard {open_['file']}: {size} bytes on disk but state "
                f"records {expected}; output is corrupt, delete the apus-* "
                f"shards and state file and reconvert"
            )
        if size > expected:
            # Torn write of the tensor that was being appended when we were
            # killed: the state flush for it never happened. Drop the tail.
            with open(path, "r+b") as f:
                f.truncate(expected)

        # Distinguish "placeholder header" from "sealed but unrecorded".
        with open(path, "rb") as f:
            f.read(8)
            raw = f.read(HEADER_RESERVE)
        header = json.loads(raw)
        if "__metadata__" not in header:
            expected_header = _sealed_header_bytes(open_["entries"])
            if raw != expected_header:
                raise ValueError(
                    f"open shard {open_['file']}: header neither placeholder "
                    f"nor the expected sealed header; refusing to trust it"
                )
            # Seal completed but the state flush did not: adopt it.
            self.gstate["sealed"][open_["file"]] = {
                "size": expected,
                "header_sha256": hashlib.sha256(
                    struct.pack("<Q", HEADER_RESERVE) + raw).hexdigest(),
                "ntensors": len(open_["entries"]),
            }
            self.gstate["open"] = None
            self.gstate["next_shard_idx"] += 1
            self.save_state()


# --------------------------------------------------------------------------
# Converter state
# --------------------------------------------------------------------------

def _empty_state(cfg_hash, target_bytes):
    return {
        "format_version": FORMAT_VERSION,
        "config_hash": cfg_hash,
        "target_shard_bytes": target_bytes,
        "header_reserve": HEADER_RESERVE,
        "inputs_done": [],
        "groups": {
            g: {"next_shard_idx": 1, "open": None, "sealed": {}}
            for g in ("main", "mtp", "idx")
        },
        "complete": False,
    }


class Converter:
    def __init__(self, src_dir, dst_dir, target_bytes=DEFAULT_TARGET_BYTES):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.target_bytes = target_bytes
        self.state_path = os.path.join(dst_dir, STATE_FILE)
        self.cfg_hash = config_hash(src_dir)
        os.makedirs(dst_dir, exist_ok=True)
        self.state = self._load_or_init()
        self.streams = {
            g: GroupStream(g, dst_dir, self.state, self.save_state)
            for g in ("main", "mtp", "idx")
        }
        for stream in self.streams.values():
            stream.validate()
        # Names already written to the output (sealed shards + committed
        # open-shard entries). Conversion of an interrupted input shard
        # resumes exactly after these, never redoing them.
        self.written = set()
        for stream in self.streams.values():
            for shard in stream.gstate["sealed"]:
                self.written.update(read_st_header(
                    os.path.join(dst_dir, shard)))
            open_ = stream.gstate["open"]
            if open_ is not None:
                self.written.update(e["name"] for e in open_["entries"])

    def _load_or_init(self):
        if not os.path.exists(self.state_path):
            existing = [
                n for n in os.listdir(self.dst_dir)
                if n.endswith(".safetensors") and n.startswith("apus")
            ]
            if existing:
                raise ValueError(
                    f"{self.dst_dir} contains apus shards but no state "
                    f"file; refusing to guess — remove them or restore "
                    f"{STATE_FILE}"
                )
            return _empty_state(self.cfg_hash, self.target_bytes)
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state["format_version"] != FORMAT_VERSION:
            raise ValueError("state format version mismatch")
        if state["config_hash"] != self.cfg_hash:
            raise ValueError(
                "config hash mismatch: the source directory changed since "
                "conversion started; refusing to mix outputs"
            )
        if state["target_shard_bytes"] != self.target_bytes:
            raise ValueError(
                f"target shard size changed ({state['target_shard_bytes']} "
                f"-> {self.target_bytes}); keep it constant across a run"
            )
        return state

    def save_state(self):
        """Atomic state flush: every crash window collapses to the last
        fully recorded step."""
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".state-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    # -- main flow ---------------------------------------------------------

    def convert(self, shard_names=None, progress_cb=None):
        """Convert input shards (all pending, or the named ones)."""
        pending = [
            n for n in list_input_shards(self.src_dir)
            if n not in self.state["inputs_done"]
        ]
        if shard_names is not None:
            wanted = set(shard_names)
            missing = wanted - set(list_input_shards(self.src_dir))
            if missing:
                raise ValueError(f"unknown input shard(s): {sorted(missing)}")
            pending = [n for n in pending if n in wanted]
        for name in pending:
            self._convert_input_shard(name, progress_cb)
            self.state["inputs_done"].append(name)
            self.save_state()
            if progress_cb:
                progress_cb("input_done", shard=name)

    def _convert_input_shard(self, shard_name, progress_cb):
        tensors = read_st_header(os.path.join(self.src_dir, shard_name))
        by_group = {"main": [], "mtp": [], "idx": []}
        for name in tensors:
            by_group[classify_group(name)].append(name)
        for group, names in by_group.items():
            if not names:
                continue
            stream = self.streams[group]
            ordered = order_contribution({n: tensors[n] for n in names})
            todo = [n for n in ordered if n not in self.written]
            i = 0
            while i < len(todo):
                prefix = slab_prefix(todo[i])
                if prefix is not None:
                    # Whole slab: capacity check covers all 6 members so the
                    # slab can never straddle a shard boundary. (After an
                    # interruption only the unwritten members remain here;
                    # they still land contiguously at the open shard's tail.)
                    slab = []
                    while i + len(slab) < len(todo) and \
                            slab_prefix(todo[i + len(slab)]) == prefix:
                        slab.append(todo[i + len(slab)])
                    total = sum(tensors[s].nbytes for s in slab)
                    stream.ensure_capacity(total, self.target_bytes,
                                           progress_cb)
                    for s in slab:
                        stream.append_tensor(tensors[s], progress_cb)
                    i += len(slab)
                else:
                    t = tensors[todo[i]]
                    stream.ensure_capacity(t.nbytes, self.target_bytes,
                                           progress_cb)
                    stream.append_tensor(t, progress_cb)
                    i += 1
            self.written.update(todo)

    def finalize(self, progress_cb=None):
        """Seal any open shards and (re)write the manifest. Idempotent."""
        for stream in self.streams.values():
            stream.seal(progress_cb)
        manifest = self.build_manifest()
        path = os.path.join(self.dst_dir, MANIFEST_FILE)
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".manifest-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Standard safetensors index for the C loader (st.h/cache.h read
        # model.safetensors.index.json, not the apus manifest).
        std_index = {
            "metadata": {"total_size": sum(t["nbytes"] for t in
                                           manifest["tensor_map"].values())},
            "weight_map": {name: t["shard"]
                           for name, t in manifest["tensor_map"].items()},
        }
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".stindex-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(std_index, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.dst_dir,
                                     "model.safetensors.index.json"))
        self.state["complete"] = True
        self.save_state()
        return manifest

    # -- introspection ------------------------------------------------------

    def output_tensor_map(self):
        """name -> (shard_file, absolute_offset, nbytes, dtype, shape) for
        everything written so far, from sealed headers + open entries."""
        out = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            for shard in gstate["sealed"]:
                path = os.path.join(self.dst_dir, shard)
                for name, t in read_st_header(path).items():
                    out[name] = (shard, t.file_offset, t.nbytes, t.dtype,
                                 t.shape)
            open_ = gstate["open"]
            if open_ is not None:
                base = 8 + HEADER_RESERVE
                for e in open_["entries"]:
                    out[e["name"]] = (open_["file"], base + e["offset"],
                                      e["nbytes"], e["dtype"], e["shape"])
        return out

    def build_manifest(self):
        tensor_map = {}
        slabs = []
        groups = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            files = sorted(gstate["sealed"])
            groups[group] = files
            expert_parts = {}
            for shard in files:
                for name, t in read_st_header(
                        os.path.join(self.dst_dir, shard)).items():
                    tensor_map[name] = {
                        "shard": shard,
                        "offset": t.file_offset,   # absolute file offset
                        "nbytes": t.nbytes,
                        "dtype": t.dtype,
                        "shape": t.shape,
                    }
                    m = EXPERT_RE.match(name)
                    if m:
                        key = (f"{m.group(1)}.{m.group(2)}",
                               int(m.group(3)))
                        expert_parts.setdefault(key, []).append(
                            (t.file_offset, t.nbytes, shard))
            for (block, expert), parts in sorted(expert_parts.items()):
                if len(parts) != len(SLAB_MEMBERS):
                    raise ValueError(
                        f"expert {block}/{expert}: {len(parts)} tensors in "
                        f"output, expected {len(SLAB_MEMBERS)}"
                    )
                parts.sort()
                shards = {p[2] for p in parts}
                if len(shards) != 1:
                    raise ValueError(
                        f"expert {block}/{expert} straddles shards {shards}")
                start = parts[0][0]
                contiguous = all(
                    parts[k][0] + parts[k][1] == parts[k + 1][0]
                    for k in range(len(parts) - 1)
                )
                if not contiguous:
                    raise ValueError(
                        f"expert {block}/{expert} tensors not contiguous")
                slabs.append({
                    "block": block,
                    "expert": expert,
                    "shard": parts[0][2],
                    "offset": start,
                    "nbytes": sum(p[1] for p in parts),
                })
        return {
            "format_version": FORMAT_VERSION,
            "config_hash": self.cfg_hash,
            "offset_base": "file",
            "header_reserve": HEADER_RESERVE,
            "shard_groups": groups,
            "ntensors": len(tensor_map),
            "tensor_map": tensor_map,
            "expert_slabs": slabs,
        }


# --------------------------------------------------------------------------
# Verification: byte-compare source tensors against the output
# --------------------------------------------------------------------------

def verify_source(src_dir, dst_dir, shard_names=None, log=print):
    """Byte-compare every tensor of the given (or all converted) source
    shards against its copy in the output. Returns the number of tensors
    verified; raises on the first mismatch."""
    conv = Converter.__new__(Converter)   # lightweight: no validation writes
    conv.src_dir, conv.dst_dir = src_dir, dst_dir
    conv.state_path = os.path.join(dst_dir, STATE_FILE)
    with open(conv.state_path, "r", encoding="utf-8") as f:
        conv.state = json.load(f)
    conv.target_bytes = conv.state["target_shard_bytes"]
    conv.streams = {
        g: GroupStream(g, dst_dir, conv.state, lambda: None)
        for g in ("main", "mtp", "idx")
    }
    out_map = conv.output_tensor_map()

    shards = list_input_shards(src_dir)
    if shard_names is not None:
        shards = [s for s in shards if s in set(shard_names)]
    done = set(conv.state["inputs_done"])
    nverified = 0
    for shard in shards:
        if shard not in done:
            continue
        src_tensors = read_st_header(os.path.join(src_dir, shard))
        for name, t in src_tensors.items():
            if name not in out_map:
                raise ValueError(f"{name}: missing from output")
            oshard, ooff, onbytes, odtype, oshape = out_map[name]
            if onbytes != t.nbytes or odtype != t.dtype or oshape != t.shape:
                raise ValueError(f"{name}: metadata mismatch vs output")
            with open(t.src_path, "rb") as fs, \
                    open(os.path.join(dst_dir, oshard), "rb") as fo:
                fs.seek(t.file_offset)
                fo.seek(ooff)
                remaining = t.nbytes
                while remaining:
                    a = fs.read(min(COPY_CHUNK, remaining))
                    b = fo.read(min(COPY_CHUNK, remaining))
                    if a != b:
                        raise ValueError(
                            f"{name}: BYTE MISMATCH between source and "
                            f"output")
                    remaining -= len(a)
            nverified += 1
        log(f"verify: {shard}: {len(src_tensors)} tensors byte-identical")
    return nverified


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("convert", "verify", "finalize"):
        sp = sub.add_parser(cmd)
        sp.add_argument("src_dir")
        sp.add_argument("dst_dir")
        sp.add_argument("--shard", action="append", default=None,
                        help="limit to these input shard(s); repeatable")
        if cmd == "convert":
            sp.add_argument("--target-bytes", type=int,
                            default=DEFAULT_TARGET_BYTES)
    args = p.parse_args(argv)

    if args.cmd == "convert":
        conv = Converter(args.src_dir, args.dst_dir, args.target_bytes)
        conv.convert(shard_names=args.shard)
        conv.finalize()
    elif args.cmd == "finalize":
        conv = Converter(args.src_dir, args.dst_dir)
        conv.finalize()
    elif args.cmd == "verify":
        n = verify_source(args.src_dir, args.dst_dir, args.shard)
        print(f"verify: OK, {n} tensors byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
