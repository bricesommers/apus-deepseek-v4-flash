#!/usr/bin/env python3
"""M7a fixture generator — scripted "parrot" mini-models for server tests.

The server milestone needs a model whose generated token stream is a fixed,
known script (so tool-call DSML output, EOS handling, stop strings and usage
counts can be asserted end-to-end through the real C engine). Random weights
cannot do that. Instead:

  * All layer weights are ZERO (attn, MoE, mHC fn/base/scale): every
    sublayer output F(x) is 0, the mHC comb is the uniform 4x4 doubly
    stochastic matrix, and the 4x-replicated residual stream is preserved
    exactly. The final hidden state therefore depends ONLY on the last
    embedded token: hc_head (zero fn/base) averages the 4 copies, the final
    RMSNorm (weight 1) rescales. Logits = head @ f(embed[last]).
  * embed rows are random N(0,1) (pairwise near-orthogonal in 256 dims).
  * head is all zero except: for each scripted transition a -> b,
    head[b] = 8 * embed[a]. Greedy (and any sane temperature) decoding then
    follows the scripted Markov chain: self-dot ~8*256 vs cross-term noise
    sigma ~8*16.
  * Chain tokens with multi-character content are ADDED TOKENS (ids 300+),
    each used at exactly one chain position, so every transition source is
    unique. The canonical specials (BOS 256, EOS 257, user/assistant
    markers, <think>/</think>, DSML) are added tokens 256-262; repeated
    DSML markers in a chain use alias ids 400+ with identical content.
  * The byte-level vocab (ids 0-255, GPT-2 bytes_to_unicode alphabet, no
    merges) tokenizes arbitrary prompt text one byte per token.

Two variants are written:
  fixtures/model_chat/   thinking reply with a stop-string trigger + chat reply
  fixtures/model_tools/  thinking + DSML tool call (get_weather) + EOS

Regenerate: `make golden-m7a` (or run this file directly). Deterministic.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m3"))

import oracle  # noqa: E402
import stutil  # noqa: E402

SEED = 20260910

# Canonical added tokens (ids mirror the roles of the real model's specials).
BOS, EOS = 256, 257
SP_USER, SP_ASSISTANT = 258, 259
THINK_START, THINK_END = 260, 261
DSML = 262

SPECIALS = [
    (BOS, "<｜begin▁of▁sentence｜>"),
    (EOS, "<｜end▁of▁sentence｜>"),
    (SP_USER, "<｜User｜>"),
    (SP_ASSISTANT, "<｜Assistant｜>"),
    (THINK_START, "<think>"),
    (THINK_END, "</think>"),
    (DSML, "｜DSML｜"),
]

# ---- scripted chains -------------------------------------------------------
# chain: ordered token ids; CHUNKS: id -> decoded text for non-canonical ids.
# Transitions: last-prompt-token -> chain[0], then chain[i] -> chain[i+1].
# Entry points: thinking prompts end with THINK_START, chat prompts with
# THINK_END (which is mid-chain, so both modes share one continuation).

CHAT_CHUNKS = {
    300: "reasoning: thinking it over.",
    301: "The answer is ",
    302: "STOP",
    303: " right here.",
}
CHAT_CHAIN = [300, THINK_END, 301, 302, 303, EOS]
# full texts per entry mode (for tests/documentation):
#   thinking: "reasoning: thinking it over.</think>The answer is STOP right here." + EOS
#   chat:     "The answer is STOP right here." + EOS

TOOLS_CHUNKS = {
    300: "I should check the weather.",
    301: "\n\n<",
    302: "tool_calls>\n<",
    303: 'invoke name="get_weather">\n<',
    304: 'parameter name="location" string="true">Beijing</',
    305: "parameter>\n</",
    306: "invoke>\n</",
    307: "tool_calls>",
    400: "｜DSML｜",   # alias ids: same decode text, unique chain positions
    401: "｜DSML｜",
    402: "｜DSML｜",
    403: "｜DSML｜",
    404: "｜DSML｜",
}
TOOLS_CHAIN = [300, THINK_END, 301, DSML, 302, 400, 303, 401, 304,
               402, 305, 403, 306, 404, 307, EOS]
# thinking text: "I should check the weather.</think>\n\n<｜DSML｜tool_calls>\n"
#   "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" "
#   "string=\"true\">Beijing</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
#   "</｜DSML｜tool_calls>" + EOS
# chat entry starts at THINK_END -> 301 (the "\n\n<" chunk) directly.

VARIANTS = {
    "model_chat": (CHAT_CHUNKS, CHAT_CHAIN),
    "model_tools": (TOOLS_CHUNKS, TOOLS_CHAIN),
}


def bytes_to_unicode():
    """GPT-2 byte-level alphabet (identical rule to tok.h b2u_cp)."""
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD))
          + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()


def write_tokenizer(path, chunks):
    vocab = {B2U[b]: b for b in range(256)}
    added = [
        {"id": i, "content": c, "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in SPECIALS
    ]
    added += [
        {"id": i, "content": c, "special": False, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in sorted(chunks.items())
    ]
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=1)


def zeroed_layer_recs(cfg, layer, li):
    """gen_layer_tensors with all VALUES zeroed (scales kept valid from the
    quantization of the random source weights); tid2eid rows = [0..topk)."""
    lr = np.random.default_rng(SEED + 1000 * li)
    recs = []
    for name, dtype, shape, payload in oracle.gen_layer_tensors(cfg, layer, li, lr):
        if dtype in ("F8_E4M3", "I8", "BF16", "F32"):
            payload = b"\x00" * len(payload)
        elif dtype == "I64":  # hash-routing table: rows [0, 1, ..., topk)
            payload = np.broadcast_to(
                np.arange(shape[1], dtype=np.int64), tuple(shape)).copy().tobytes()
        recs.append((name, dtype, shape, payload))
    return recs


def build_variant(out_dir, chunks, chain):
    cfg = dict(oracle.FULL_CFG)
    cfg["bos_token_id"] = BOS
    cfg["eos_token_id"] = EOS
    # byte-level fixture tokenizer makes prompts long (the DSML tools
    # template alone is ~1 KB); give the position tables real headroom
    cfg["max_pos"] = 8192
    V, dim = cfg["vocab_size"], cfg["dim"]

    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir)

    rng = np.random.default_rng(SEED)
    embed = rng.standard_normal((V, dim)).astype(np.float32)

    # scripted transitions: entry THINK_START -> chain[0], then pairwise
    transitions = {THINK_START: chain[0]}
    for a, b in zip(chain, chain[1:]):
        assert a not in transitions, f"duplicate chain source {a}"
        transitions[a] = b
    head = np.zeros((V, dim), dtype=np.float32)
    for a, b in transitions.items():
        head[b] += np.float32(8.0) * embed[a]

    dense_recs = [
        ("embed.weight", "BF16", (V, dim), oracle.f32_to_bf16_bytes(embed)),
        ("norm.weight", "BF16", (dim,),
         oracle.f32_to_bf16_bytes(np.ones(dim, dtype=np.float32))),
        ("head.weight", "BF16", (V, dim), oracle.f32_to_bf16_bytes(head)),
        ("hc_head_fn", "F32", (cfg["hc_mult"], cfg["hc_mult"] * dim),
         np.zeros(cfg["hc_mult"] * cfg["hc_mult"] * dim, dtype=np.float32).tobytes()),
        ("hc_head_scale", "F32", (1,),
         np.ones(1, dtype=np.float32).tobytes()),
        ("hc_head_base", "F32", (cfg["hc_mult"],),
         np.zeros(cfg["hc_mult"], dtype=np.float32).tobytes()),
    ]
    expert_recs = []
    for li, layer in enumerate(cfg["layers"]):
        for rec in zeroed_layer_recs(cfg, layer, li):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)

    shards = [("apus-00001.safetensors", dense_recs),
              ("apus-00002.safetensors", expert_recs)]
    weight_map, total = {}, 0
    for fname, recs in shards:
        stutil.write_shard(os.path.join(weights_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(weights_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  f, indent=1)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    write_tokenizer(os.path.join(out_dir, "tokenizer.json"), chunks)

    # self-check the Markov margin with the stored (bf16-rounded) values
    from oracle import bf16_bytes_to_f32 as b2f
    e = b2f(oracle.f32_to_bf16_bytes(embed)).reshape(V, dim)
    h = b2f(oracle.f32_to_bf16_bytes(head)).reshape(V, dim)
    yn = e / np.sqrt((e * e).mean(axis=1, keepdims=True) + 1e-6)
    logits = yn @ h.T
    worst = 1e30
    for a, b in transitions.items():
        margin = logits[a, b] - np.max(np.delete(logits[a], b))
        worst = min(worst, margin)
    print(f"  {os.path.basename(out_dir)}: {len(transitions)} transitions, "
          f"min logit margin {worst:.1f}")
    assert worst > 100, "scripted chain margin too small"


def main():
    base = os.path.join(ROOT, "tests", "m7a", "fixtures")
    for name, (chunks, chain) in VARIANTS.items():
        build_variant(os.path.join(base, name), chunks, chain)
    print("m7a fixtures done")


if __name__ == "__main__":
    main()
