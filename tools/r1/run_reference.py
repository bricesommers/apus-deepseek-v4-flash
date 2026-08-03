#!/usr/bin/env python3
"""R1 golden-token generator — runs ON THE CUDA HOST.

Generates reference goldens for the R1 battery (tests/r1b/prompts.json) by
running DeepSeek's OFFICIAL reference inference (reference-0731/inference)
on DeepSeek-V4-Flash-0731. Output: r1_goldens.json, which is copied back to
the Mac and consumed by tools/r1/compare.py.

Expected layout (same relative layout as the apus repo — rsync the kit as
described in docs/R1.md):

    <root>/tools/r1/run_reference.py      (this file)
    <root>/reference-0731/inference/      (model.py kernel.py convert.py config.json)
    <root>/reference-0731/encoding/       (encoding_dsv4.py)
    <root>/tests/r1b/prompts.json         (the 8-case battery)

Key reference facts this script mirrors (DO NOT "fix" without re-reading
reference-0731/inference/{generate,model}.py):

  * 0731 moved sampling INTO Transformer.forward: it returns
    (output_ids, logits, main_hidden); output_ids = sample(logits,
    args.temperature) computed inside the forward (model.py:913-926).
  * sample() (model.py:939-946): temperature == 0 -> logits.argmax(dim=-1)
    (the greedy path). Otherwise Gumbel-max: softmax(logits/T) divided by an
    exponential_(1) draw, then argmax. THERE IS NO TOP-P in the reference.
  * generate.py's loop: tokens buffer prefilled with the prompt, one forward
    per position, `model.forward(tokens[:, prev:cur], prev)[0]`; the first
    forward is the full-prompt prefill, subsequent ones decode one token.
  * generate.py pins torch.manual_seed(33377335) at startup. We mirror that,
    and additionally re-pin per case when the case carries a "seed".
  * Checkpoint layout: <ckpt>/model{rank}-mp{world_size}.safetensors, produced
    by inference/convert.py from the HF shards (--convert below).
  * Prompt construction: raw = tokenizer.encode(prompt) verbatim (the
    tokenizer adds NO BOS by itself); chat/thinking =
    tokenizer.encode(encode_messages([{user}], thinking_mode=mode)) — the
    encoding adds the BOS string. This matches apus's serve "text" and
    "messages" paths respectively (c/encoding.h is byte-exact vs
    encoding_dsv4.py — M2/M10 gates).

Only rank 0 writes JSON; all ranks must execute the same decode loop
(the forwards are collective when world_size > 1).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEF_INFERENCE = ROOT / "reference-0731" / "inference"
DEF_ENCODING = ROOT / "reference-0731" / "encoding"
DEF_PROMPTS = ROOT / "tests" / "r1b" / "prompts.json"

# Base seed pinned at startup, exactly as reference-0731/inference/generate.py:79.
BASE_SEED = 33377335


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt-path",
                   help="Directory with model{rank}-mp{N}.safetensors "
                        "(output of the convert step). Required unless --convert-only.")
    p.add_argument("--hf-ckpt-path",
                   help="Directory with the original HF shards (input to --convert).")
    p.add_argument("--config", default=str(DEF_INFERENCE / "config.json"),
                   help="ModelArgs config JSON (default: reference-0731/inference/config.json).")
    p.add_argument("--prompts", default=str(DEF_PROMPTS))
    p.add_argument("--out", default="r1_goldens.json")
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")),
                   help="Model-parallel factor (default: $WORLD_SIZE or 1). "
                        "For N>1 launch with torchrun --nproc_per_node=N.")
    p.add_argument("--convert", action="store_true",
                   help="Run inference/convert.py (HF -> mp layout) before generating.")
    p.add_argument("--convert-only", action="store_true",
                   help="Run the convert step and exit.")
    p.add_argument("--cases", default="",
                   help="Comma-separated case ids to run (default: all).")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="Cap max_tokens for every case (0 = use the battery values). "
                        "Useful for a quick pipeline smoke (--max-tokens 4).")
    return p.parse_args()


def maybe_convert(args, cfg):
    """Wrap reference-0731/inference/convert.py: HF shards -> model{i}-mp{N}.safetensors."""
    if not args.convert and not args.convert_only:
        return
    if not args.hf_ckpt_path or not args.ckpt_path:
        sys.exit("--convert needs --hf-ckpt-path (HF shards) and --ckpt-path (output dir)")
    n_experts = cfg.get("n_routed_experts", 256)
    expert_dtype = cfg.get("expert_dtype") or "fp4"
    cmd = [sys.executable, str(DEF_INFERENCE / "convert.py"),
           "--hf-ckpt-path", args.hf_ckpt_path,
           "--save-path", args.ckpt_path,
           "--n-experts", str(n_experts),
           "--model-parallel", str(args.world_size),
           "--expert-dtype", expert_dtype]
    print(f"[convert] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[convert] done in {time.time() - t0:.0f}s -> {args.ckpt_path}", flush=True)
    if args.convert_only:
        sys.exit(0)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    maybe_convert(args, cfg)
    if not args.ckpt_path:
        sys.exit("--ckpt-path is required")

    # ---- host-only imports (keep this file py_compile-able on the Mac) ----
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from safetensors.torch import load_model

    sys.path.insert(0, str(DEF_INFERENCE))
    sys.path.insert(0, str(DEF_ENCODING))
    from model import Transformer, ModelArgs          # noqa: E402
    from encoding_dsv4 import encode_messages         # noqa: E402

    # ---- distributed + process setup: mirror generate.py main() exactly ----
    world_size = args.world_size
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(BASE_SEED)

    margs = ModelArgs(**cfg)
    margs.temperature = 0.0  # overridden per case via model.temperature
    print(f"[setup] ModelArgs: {margs}")
    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    shard = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    print(f"[setup] loading {shard}")
    t0 = time.time()
    load_model(model, shard, strict=False)
    torch.set_default_device("cuda")
    print(f"[setup] model loaded in {time.time() - t0:.0f}s")

    eos_id = tokenizer.eos_token_id

    with open(args.prompts) as f:
        battery = json.load(f)
    cases = battery["cases"]
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in want]

    goldens = {
        "format": "r1_goldens/v1",
        "meta": {
            "model": battery.get("model"),
            "ckpt_path": args.ckpt_path,
            "config": args.config,
            "world_size": world_size,
            "torch_version": torch.__version__,
            "base_seed": BASE_SEED,
            "rng_note": ("Greedy cases (temp 0) use the reference argmax path — no RNG. "
                         "Sampled cases re-pin torch.manual_seed(seed) immediately before "
                         "the case; sampling is the reference Gumbel-max trick "
                         "(one exponential_(1) draw over the vocab per step). The reference "
                         "has NO top-p; top_p in the battery applies to the apus side only."),
            "eos_token_id": eos_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "cases": {},
    }

    for ci, case in enumerate(cases):
        cid = case["id"]
        print(f"[case {ci + 1}/{len(cases)}] {cid} ...", flush=True)
        t_case = time.time()
        try:
            temp = float(case.get("temp", 0))
            max_new = int(case["max_tokens"])
            if args.max_tokens:
                max_new = min(max_new, args.max_tokens)
            mode = case["mode"]
            model.temperature = temp  # sampling happens inside model.forward
            seed = case.get("seed")
            if seed is not None:
                torch.manual_seed(int(seed))

            # Prompt ids: raw = verbatim tokenize (no auto-BOS); chat/thinking
            # = encode_messages (adds the BOS string) then tokenize.
            if mode == "raw":
                rendered = case["prompt"]
            else:
                rendered = encode_messages(
                    [{"role": "user", "content": case["prompt"]}],
                    thinking_mode=mode)
            prompt_ids = tokenizer.encode(rendered)

            # Decode loop: faithful batch-1 mirror of generate.py's generate().
            plen = len(prompt_ids)
            total_len = min(model.max_seq_len, max_new + plen)
            tokens = torch.full((1, total_len), -1, dtype=torch.long)
            tokens[0, :plen] = torch.tensor(prompt_ids, dtype=torch.long)
            steps = []
            prev = 0
            n_report = 0
            for cur in range(plen, total_len):
                out_ids, logits, _ = model.forward(tokens[:, prev:cur], prev)
                lf = logits[0].float()
                top2 = torch.topk(lf, 2)
                top1 = int(top2.indices[0])
                margin = float(top2.values[0] - top2.values[1])
                emitted = int(out_ids[0])
                if temp == 0 and emitted != top1:
                    # argmax path must make these identical; flag if not
                    print(f"  [warn] pos {cur - plen}: greedy emitted {emitted} != top1 {top1}")
                steps.append({"pos": cur - plen, "emitted": emitted,
                              "top1": top1, "margin": round(margin, 6)})
                tokens[0, cur] = emitted
                prev = cur
                n_report += 1
                if n_report % 16 == 0 or cur == total_len - 1:
                    print(f"  {cid}: {n_report}/{total_len - plen} tokens", flush=True)
                if emitted == eos_id:
                    break

            goldens["cases"][cid] = {
                "status": "ok",
                "mode": mode,
                "temp": temp,
                "top_p": case.get("top_p"),
                "seed": seed,
                "prompt": case["prompt"],
                "rendered_prompt": rendered,
                "prompt_ids": prompt_ids,
                "max_tokens": max_new,
                "stopped_at_eos": bool(steps) and steps[-1]["emitted"] == eos_id,
                "wall_s": round(time.time() - t_case, 1),
                "steps": steps,
            }
            print(f"[case {ci + 1}/{len(cases)}] {cid}: ok, {len(steps)} tokens, "
                  f"{time.time() - t_case:.0f}s", flush=True)
        except Exception:
            err = traceback.format_exc()
            goldens["cases"][cid] = {"status": "failed", "error": err,
                                     "mode": case.get("mode"), "prompt": case.get("prompt")}
            print(f"[case {ci + 1}/{len(cases)}] {cid}: FAILED (continuing)\n{err}",
                  flush=True)

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(goldens, f, ensure_ascii=False, indent=1)
        print(f"[done] wrote {args.out} "
              f"({sum(1 for c in goldens['cases'].values() if c['status'] == 'ok')}/{len(cases)} ok)")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
