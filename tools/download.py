#!/usr/bin/env python3
"""apus M1 — disk-safe download + convert driver for DeepSeek-V4-Flash.

Downloads the 46 HF checkpoint shards ONE AT A TIME and converts each into
the apus container before fetching the next, so peak disk usage stays at
roughly (growing apus output) + (one ~3.5 GB source shard) instead of
160 GB + 160 GB.

Per shard the pipeline is:

    download (resumable)  ->  convert (resumable, tools/convert.py)
                          ->  byte-verify against output
                          ->  record "done" in the state file
                          ->  delete the source shard

The driver can be killed at any point and restarted cleanly:

  * A download interrupted mid-stream resumes from the partial bytes
    (HTTP Range via huggingface_hub; byte-range resume in --source-dir mode).
  * A conversion interrupted mid-shard is resumed by convert.py's own
    state machine (verify-before-trust + truncation of torn writes).
  * A source shard is deleted only AFTER its content is byte-verified in
    the output and the "done" mark is durably recorded. If the process dies
    between recording and deleting, startup garbage collection removes the
    leftover source file.

Network mode uses huggingface_hub (already in the project venv). For tests
and offline rehearsal, --source-dir DIR treats a local directory as the
"remote", copying shards with the same resumable state machine.

Usage:
    python tools/download.py --repo deepseek-ai/DeepSeek-V4-Flash \
        --work DIR --out DIR [--token TOKEN] [--target-bytes N]
    python tools/download.py --source-dir DIR --work DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as apus_convert  # noqa: E402

STATE_FILE = "apus.download.state.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
COPY_CHUNK = 8 * 1024 * 1024
# Socket timeout for connect AND reads: a silently stalled connection
# (e.g. network drop mid-stream) must raise so the retry loop kicks in,
# instead of hanging forever at 0% CPU.
HTTP_TIMEOUT = 60


class Driver:
    def __init__(self, work_dir, out_dir, repo=None, token=None,
                 source_dir=None, target_bytes=apus_convert.DEFAULT_TARGET_BYTES,
                 progress_cb=None):
        self.work_dir = work_dir
        self.src_dir = os.path.join(work_dir, "src")  # landing zone, 1 shard
        self.out_dir = out_dir
        self.repo = repo
        self.token = token
        self.source_dir = source_dir
        self.target_bytes = target_bytes
        self.progress_cb = progress_cb
        self._size_cache = {}
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        self.state_path = os.path.join(work_dir, STATE_FILE)
        self.state = self._load_state()

    # -- state --------------------------------------------------------------

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"format_version": 1, "files": {}}

    def save_state(self):
        fd, tmp = tempfile.mkstemp(dir=self.work_dir, prefix=".dl-state-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def fstate(self, name):
        return self.state["files"].setdefault(name, {"status": "pending"})

    # -- shard list ----------------------------------------------------------

    # Small support files the engine needs next to the converted container:
    # tokenizer + generation config (weights are useless without them).
    SUPPORT_FILES = ("tokenizer.json", "tokenizer_config.json",
                     "generation_config.json")

    def shard_names(self):
        index_path = os.path.join(self.work_dir, INDEX_NAME)
        if not os.path.exists(index_path):
            self._fetch_small(INDEX_NAME)
        if not os.path.exists(os.path.join(self.work_dir, CONFIG_NAME)):
            try:
                self._fetch_small(CONFIG_NAME)
            except Exception:
                pass  # config is optional for conversion
        for name in self.SUPPORT_FILES:
            dst = os.path.join(self.out_dir, name)
            if not os.path.exists(dst):
                try:
                    self._fetch_small(name)
                    shutil.copyfile(os.path.join(self.work_dir, name), dst)
                except Exception:
                    pass  # tokenizer files are only needed at run time
        # convert.py expects config/index next to the shards it reads.
        for small in (INDEX_NAME, CONFIG_NAME):
            src = os.path.join(self.work_dir, small)
            dst = os.path.join(self.src_dir, small)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))

    def _fetch_small(self, name):
        if self.source_dir:
            shutil.copyfile(os.path.join(self.source_dir, name),
                            os.path.join(self.work_dir, name))
        else:
            self._hf_download(name, os.path.join(self.work_dir, name))

    # -- download backends ----------------------------------------------------

    def _hf_download(self, name, dest):
        """Download via huggingface_hub (resumes partial downloads itself),
        then move into place."""
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=self.repo, filename=name, token=self.token,
        )
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copyfile(path, dest)
        return dest

    def _expected_size(self, name):
        """Authoritative byte size of a remote file (HF xet: x-linked-size).

        Retries transient network failures forever with capped backoff —
        an unattended 160 GB download must survive long outages."""
        import time
        import urllib.request
        if name in self._size_cache:
            return self._size_cache[name]
        url = f"https://huggingface.co/{self.repo}/resolve/main/{name}"
        backoff = 1.0
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(url, method="HEAD")
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    h = resp.headers
                    size = h.get("x-linked-size") or h.get("Content-Length")
                if size is None:
                    raise IOError(f"cannot determine remote size for {name}")
                size = int(size)
                self._size_cache[name] = size
                return size
            except Exception as e:
                print(f"[download] HEAD {name}: attempt {attempt} failed: "
                      f"{e}; retrying in {backoff:.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _download_shard(self, name):
        """Resumable download of one shard into the landing zone."""
        dest = os.path.join(self.src_dir, name)
        if self.source_dir:
            self._local_resume_copy(os.path.join(self.source_dir, name), dest)
        else:
            expected = self._expected_size(name)
            part = dest + ".part"
            # Heal a truncated "final" file from a pre-fix run: demote it to
            # the .part file and resume instead of trusting it.
            if os.path.exists(dest):
                have = os.path.getsize(dest)
                if have == expected:
                    return dest
                if os.path.exists(part):
                    os.remove(part)
                os.replace(dest, part)
            self._http_resume_copy(name, part, expected)
            if os.path.getsize(part) != expected:
                raise IOError(f"{name}: size mismatch after download")
            os.replace(part, dest)
        return dest

    def _local_resume_copy(self, src, dest):
        """Offline stand-in for HTTP resume: append-copy honoring an existing
        .part file, then rename. Exercises the same state machine."""
        part = dest + ".part"
        total = os.path.getsize(src)
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > total:
            os.remove(part)
            have = 0
        with open(src, "rb") as fin, open(part, "ab") as fout:
            fin.seek(have)
            while True:
                chunk = fin.read(COPY_CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
                fout.flush()
                os.fsync(fout.fileno())
                if self.progress_cb:
                    self.progress_cb("download_chunk", file=name_of(src),
                                     have=fout.tell(), total=total)
        os.replace(part, dest)

    def _http_resume_copy(self, name, part, expected):
        """Size-verified HTTP download with Range resume and unlimited retries.

        Never treats an early EOF (dropped connection) as completion: loops
        until the .part file holds exactly `expected` bytes. Safe against
        long outages — retries forever with capped exponential backoff."""
        import time
        import urllib.request
        url = f"https://huggingface.co/{self.repo}/resolve/main/{name}"
        backoff = 1.0
        attempt = 0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have == expected:
                return
            if have > expected:
                os.remove(part)  # cannot be a prefix of the real file
                have = 0
            attempt += 1
            req = urllib.request.Request(url)
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    if have and resp.status != 206:
                        # Server ignored the range: start over.
                        have = 0
                    mode = "ab" if have else "wb"
                    with open(part, mode) as fout:
                        while True:
                            chunk = resp.read(COPY_CHUNK)
                            if not chunk:
                                break
                            fout.write(chunk)
                            have += len(chunk)
                        fout.flush()
                        os.fsync(fout.fileno())
                backoff = 1.0  # a clean EOF resets the backoff
            except Exception as e:
                print(f"[download] {name}: attempt {attempt} failed at "
                      f"{have}/{expected} bytes: {e}; retrying in "
                      f"{backoff:.0f}s", file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    # -- pipeline ---------------------------------------------------------------

    def gc(self):
        """Remove source shards whose conversion was recorded done but whose
        deletion was lost to a crash."""
        for name, st in self.state["files"].items():
            path = os.path.join(self.src_dir, name)
            if st.get("status") == "done" and os.path.exists(path):
                os.remove(path)

    def run(self):
        self.gc()
        names = self.shard_names()
        for name in names:
            st = self.fstate(name)
            if st["status"] == "done":
                continue
            dest = os.path.join(self.src_dir, name)
            if st["status"] == "downloaded" and not self.source_dir:
                # Revalidate against the authoritative remote size: a shard
                # truncated by a dropped connection must be re-fetched, not
                # handed to the converter.
                try:
                    good = (os.path.exists(dest) and
                            os.path.getsize(dest) == self._expected_size(name))
                except Exception:
                    good = os.path.exists(dest)  # offline: defer to converter
                if not good:
                    st["status"] = "pending"
                    self.save_state()
            if st["status"] == "pending":
                self._download_shard(name)
                st["status"] = "downloaded"
                st["size"] = os.path.getsize(dest)
                self.save_state()
                if self.progress_cb:
                    self.progress_cb("downloaded", file=name)
            # convert (idempotent/resumable) + byte-verify
            conv = apus_convert.Converter(self.src_dir, self.out_dir,
                                          self.target_bytes)
            conv.convert(shard_names=[name], progress_cb=self.progress_cb)
            apus_convert.verify_source(self.src_dir, self.out_dir, [name],
                                       log=lambda _msg: None)
            st["status"] = "done"
            self.save_state()
            os.remove(dest)
            if self.progress_cb:
                self.progress_cb("shard_done", file=name)
        # All shards converted: seal the last open shards, write manifest.
        conv = apus_convert.Converter(self.src_dir, self.out_dir,
                                      self.target_bytes)
        conv.finalize()
        if self.progress_cb:
            self.progress_cb("complete")


def name_of(path):
    return os.path.basename(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HF repo id, e.g. deepseek-ai/DeepSeek-V4-Flash")
    src.add_argument("--source-dir", help="offline mode: treat DIR as the remote")
    p.add_argument("--work", required=True,
                   help="work dir: state file + one-shard landing zone")
    p.add_argument("--out", required=True, help="apus container output dir")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--target-bytes", type=int,
                   default=apus_convert.DEFAULT_TARGET_BYTES)
    args = p.parse_args(argv)

    driver = Driver(args.work, args.out, repo=args.repo, token=args.token,
                    source_dir=args.source_dir,
                    target_bytes=args.target_bytes,
                    progress_cb=lambda ev, **kw: print(f"[{ev}] {kw}"))
    driver.run()
    print("download+convert complete:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
