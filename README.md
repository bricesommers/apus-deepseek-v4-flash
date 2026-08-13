# apus

[![CI](https://github.com/bricesommers/apus-deepseek-v4-flash/actions/workflows/ci.yml/badge.svg)](https://github.com/bricesommers/apus-deepseek-v4-flash/actions/workflows/ci.yml)

Local inference engine for **DeepSeek-V4-Flash** (284B-total / 13B-active
MoE, FP4 experts + FP8 dense) on consumer hardware. apus runs the full
model — ~160 GB of weights, 1M-token context — on a Mac with 32 GB of
unified memory by streaming routed experts from NVMe through a bounded RAM
cache, instead of holding the model in memory.

Measured on a MacBook Pro M1 Pro 32 GB:

- **Decode:** ~0.5–1.07 tok/s (cache-warm vs cold expert working set)
- **Prefill:** ~9 tok/s at 512-token prompts
- **Weights:** ~160 GB in the apus container (MXFP4 experts, FP8 dense)
- **Context:** up to 1,048,576 tokens (YaRN-extended, per the model config)

This is a single-user, single-machine engine: correctness and exactness
first, throughput second.

## Features

- **Exact quantized kernels** — MXFP4 (E2M1 + UE8M0 scales) and FP8 (E4M3)
  GEMM/GEMV kernels in hand-written ARM NEON, plus an optional Metal GPU
  backend for the dense compute (zero-copy unified memory). Numerics follow
  DeepSeek's reference kernels; insufficient RAM costs speed, never quality.
- **Tiered expert store** — routed experts live on NVMe in coalesced slabs
  and are demand-paged through a bounded LFRU RAM cache with an RSS guard.
- **Router-lookahead prefetch ("pilot")** — predicts the next layer's
  experts from the current layer's hidden state and warms the cache ahead
  of the demand load; measured 84.5% recall. Prefetching changes only
  *when* an expert is in RAM, never the numerics (verified bitwise).
- **Speculative decoding** — classic MTP and DSpark multi-stage drafting;
  emitted tokens are bitwise identical to non-speculative decoding.
- **OpenAI-compatible server** (`tools/server.py`) and a terminal chat
  client (`tools/chat.py`).
- **Byte-exact tokenizer and chat encoding** — validated against DeepSeek's
  reference `tokenizer.json` and `encoding_dsv4.py` (both revisions).
- **Externally validated** — R1b compared apus against DeepSeek-V4-Flash-0731
  hosted on DeepSeek's API side-by-side: zero contradictions, zero factual
  errors across the test battery (see `docs/R1B-RESULTS.md`).
- **Portable** — builds and passes the full test battery on macOS/ARM
  (clang, NEON, optional Metal), Linux/x86_64 (gcc, AVX2 kernels with
  scalar fallbacks; libc + pthreads only, no BLAS dependency), and
  Windows/x86_64 (MinGW-w64 gcc via MSYS2 UCRT64; the POSIX surface is
  shimmed in `c/compat.h`).

## Requirements

- **macOS/Apple Silicon** (M1 or later) with Xcode command line tools, or
  **Linux/x86_64** with gcc and make (AVX2 auto-detected; scalar fallback
  otherwise), or **Windows 10/11 x86_64** with
  [MSYS2](https://www.msys2.org/) — from the **UCRT64** shell:
  `pacman -S mingw-w64-ucrt-x86_64-gcc make`, then the same build/run
  commands as Linux. (WSL2 also works: follow the Linux path inside it.)
- **≥ 16 GB unified memory** (32 GB recommended; the tiered store trades
  speed for memory headroom)
- **~180 GB free disk** (weights container + one source shard during
  download/convert)
- Python 3.11+ for the tools (download/convert, chat, server)

## Quickstart (step by step, no experience needed)

You need: a Mac with Apple Silicon (M1 or later), a Linux PC, **or** a
Windows PC — and about **180 GB of free disk space** for the model.
Windows users: install [MSYS2](https://www.msys2.org/) first, then use
the **Windows** command blocks below (WSL2 works too — that follows the
Linux path).

Every grey box below is a command to paste into your terminal (on Mac:
open **Terminal** from Applications → Utilities; on Windows: open
**MSYS2 UCRT64** from the Start menu). Paste one box at a time, press
Enter, wait for it to finish.

### Step 1 — get the code onto your computer

Either clone it (if you have git):

```sh
git clone https://github.com/bricesommers/apus-deepseek-v4-flash.git
cd apus-deepseek-v4-flash
```

Or without git: click the green **Code** button on the GitHub page →
**Download ZIP** → double-click the downloaded zip → open a terminal
inside the unzipped folder (on Mac: right-click the folder → Services →
New Terminal at Folder). Then:

```sh
cd apus-main    # only if you used the ZIP (folder name may vary)
```

From now on, run everything from inside this folder.

### Step 2 — install the tools

On **Mac** (installs the compiler if asked — say yes):

```sh
xcode-select --install 2>/dev/null; python3 -m venv .venv
.venv/bin/pip install numpy tokenizers huggingface_hub safetensors
```

On **Linux / WSL2**:

```sh
sudo apt update && sudo apt install -y build-essential python3-venv python3-pip
python3 -m venv .venv
.venv/bin/pip install numpy tokenizers huggingface_hub safetensors
```

On **Windows** (native): first get Python — open **PowerShell** and run
`winget install -e --id Python.Python.3.12` (or install it from
python.org and tick **"Add python.exe to PATH"**). Then in the **MSYS2
UCRT64** shell (Start menu → "MSYS2 UCRT64"):

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-gcc make
python -m venv .venv
.venv/Scripts/pip install numpy tokenizers huggingface_hub safetensors
```

This creates a small private Python environment called `.venv` inside the
folder — nothing is installed system-wide, and deleting the folder removes
everything.

> **Windows note:** a Windows venv puts its programs in `.venv/Scripts/`
> instead of `.venv/bin/`. In every command below that starts with
> `.venv/bin/`, use `.venv/Scripts/` instead (e.g.
> `.venv/Scripts/python tools/download.py ...`). Everything else —
> `make apus`, `./bin/apus run ...` — works as written in the UCRT64
> shell (it produces `bin/apus.exe` and runs it either way).

### Step 3 — download the model (the long part: ~160 GB)

```sh
.venv/bin/python tools/download.py \
    --repo deepseek-ai/DeepSeek-V4-Flash-0731 \
    --work weights/work --out weights/apus-0731
```

This downloads the model from DeepSeek's Hugging Face repo and repacks it
for streaming. Expect **1–6 hours depending on your internet**. It is safe
to interrupt (Ctrl-C, closing the lid, losing wifi): run the same command
again and it resumes exactly where it stopped. When it prints
`download+convert complete`, the model is in `weights/apus-0731/` — you do
not need to copy or move anything; the tools put every file where it
belongs.

### Step 4 — build the engine

```sh
make apus
```

Takes under a minute. You now have the engine at `bin/apus`.

### Step 5 — talk to it

```sh
.venv/bin/python tools/chat.py --model weights/apus-0731 --tiered
```

Wait ~10 seconds for it to load, then type a question at the `you>`
prompt. Answers appear slowly (about 1 token/second — this is normal for
a 284B model on a laptop). Type `/help` for commands, `/quit` to exit.

Prefer a one-off question instead of a chat?

```sh
./bin/apus run --model weights/apus-0731 --tiered \
    --prompt "The capital of France is" --max-tokens 100 --temp 0
```

Want an app-like UI (LM Studio, Open WebUI, your own scripts)? Run the
server and point any OpenAI-compatible client at
`http://localhost:8080/v1`:

```sh
.venv/bin/python tools/server.py --model weights/apus-0731 --tiered --port 8080
```

More options (speed knobs, troubleshooting, LM Studio detail):
`docs/USAGE.md`.


## How it works

DeepSeek-V4-Flash activates only 13B of its 284B parameters per token, but
naively you still need all 160 GB resident. apus instead keeps the dense
path (attention, shared expert, norms — FP8/BF16) in memory and stores the
11,264 routed experts as coalesced MXFP4 slabs on NVMe. A bounded RAM cache
pages in the 6 experts each token actually routes to, while the pilot —
a small predictor reading the router's own math one layer ahead — prefetches
likely experts so most demand loads hit warm RAM. All quantization happens
at exactly the points DeepSeek's reference kernels specify, so the streamed
model is numerically the same model, just slower when the cache is cold.
Full design: `docs/ARCHITECTURE.md`.

## Quality discipline

apus was built gate-first: every subsystem was verified against DeepSeek's
reference implementations before integration — MXFP4/FP8 kernels against
numpy ports of the reference TileLang kernels (max rel err ~1e-7), the
tokenizer against an exhaustive 1.2M-codepoint probe of the reference
(0 mismatches), the chat encoding against DeepSeek's conformance pairs
(byte-exact), and the forward pass against a layerwise numpy oracle. Every
memory/caching/prefetch configuration is required to produce bitwise
identical tokens — insufficient memory may cost speed, never output
quality. The external R1b validation against DeepSeek's hosted API is
written up in `docs/R1B-RESULTS.md`.

## Testing

Every milestone shipped with a hard-gate suite under `tests/`; the full
battery is what CI runs. Fixture/golden directories are gitignored and
regenerated deterministically by the make targets. On macOS the suites
use the local venv python; elsewhere pass `PY=python3`.

| Target | Suite |
|---|---|
| `test-m2` | Tokenizer + chat encoding, byte-exact vs DeepSeek's reference |
| `test-m3` | MXFP4 (E2M1 + UE8M0) kernel hard gate |
| `test-m4a` | FP8 (E4M3) dense kernel + mHC hard gate |
| `test-m4c` | C single-layer forward vs the M4b oracle goldens |
| `test-m5` | Full-model forward pass, end-to-end on a synthetic mini-model |
| `test-m6a` | Expert-store tiering (NVMe slabs, bounded RAM cache) |
| `test-m6b` | Router-lookahead prefetch ("pilot") + recall measurement |
| `test-m6c` | Decode performance pass (threading, routing, scratch, VM pressure) |
| `test-m7a` | OpenAI-compatible server end-to-end (scripted fixtures) |
| `test-m7b` | Metal GPU backend (macOS only) |
| `test-m8` | MTP speculative decoding (spec == non-spec, bitwise) |
| `test-m9a`–`test-m9e` | Kernel/dispatch rework gates (NEON ILP, BLAS dispatch, expert-I/O pipelining, prefill utilization, cache-bound dispatch) |
| `check-m11a` | DSpark oracle self-consistency (numpy) |
| `test-m11b` | DSpark speculative decoding in C (bitwise vs oracle) |
| `test-m12a2` | x86 kernel gates (AVX2 vs scalar anchor, bitwise) |
| `tests/m1` | Converter/downloader python suite: `python3 -m unittest discover -s tests/m1` |
| `tests/r1b` | API-comparison tooling unit tests: `python3 tests/r1b/test_compare.py`, `python3 tests/r1b/test_compare_api.py` |

On macOS/ARM, or in Docker for Linux/x86_64 (`tools/docker/test-linux.sh`
runs the whole portable battery in an ubuntu:24.04 container):

```sh
tools/docker/test-linux.sh                    # full portable battery
tools/docker/test-linux.sh test-m3 test-m4c   # selected targets
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the full battery on every
push to `main`: a `linux` job (ubuntu-latest, gcc/x86_64), a `macos` job
(macos-latest, clang/NEON + the Metal suite), and a `windows` job
(windows-latest, MinGW-w64 gcc via MSYS2 UCRT64 — same battery minus the
sanitizer twins and Metal, which is macOS-only).

## Repository layout

- `c/` — the engine: C11 header-only core + `apus.c` CLI/server driver +
  optional Metal backend (`backend_metal.mm`) + x86 kernels (`x86.h`)
- `tools/` — download/convert driver, chat client, OpenAI server,
  dockerized Linux test harness (`tools/docker/`)
- `tests/` — the milestone test battery (see Testing above)
- `reference/`, `reference-0731/` — DeepSeek's MIT-licensed reference
  implementations, tokenizer, and configs (both model revisions) that the
  gates verify against (see `LICENSE.deepseek` in each)
- `docs/` — architecture, usage, howto, R1/R1b validation writeups

## License and notices

apus is **source-available, not open-source**, under the
**PolyForm Noncommercial License 1.0.0** (see `LICENCE.PolyForm-Noncommercial`): you may use,
copy, modify, and distribute the apus source code for **noncommercial
purposes only**; any commercial use requires a separate license from the
authors.

Third-party components remain under their own licenses — see `NOTICE`:

- Design and adapted code from **colibri** (Apache License 2.0, text in
  `LICENSE.apache-2.0`)
- **DeepSeek-AI** model weights (downloaded separately by `tools/download.py`,
  not included in this repository) under the MIT License

apus is an independent project and is **not affiliated with, endorsed by,
or sponsored by DeepSeek-AI or the colibri project**.
