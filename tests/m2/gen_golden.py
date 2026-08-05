#!/usr/bin/env python3
"""Generate golden test data for M2 (tokenizer + message encoding).

Uses the HF `tokenizers` library on reference/tokenizer.json (byte-identical
to reference-0731/tokenizer.json — same sha256) and the normative
reference-0731/encoding/encoding_dsv4.py (0731 reasoning_effort semantics:
low/high/max). Outputs into tests/m2/golden/:

  tok_manifest.txt            one case name per line
  <name>.txt                  input text (raw UTF-8 bytes)
  <name>.ids                  [u32 n][n x u32 LE] token ids (encode, specials on)
  <name>.dec                  decode(ids, skip_special_tokens=False) bytes
  <name>.nosplit.ids          ids with special strings treated as plain text
  specials.bin                per-added-token records (covers every added id)
  enc_conf_<N>.ids            token ids of the N-th conformance prompt
  enc_extra_<N>.json          input spec for extra encoding case
  enc_extra_<N>.txt           expected prompt (from reference encoder)
  enc_extra_<N>.ids           its token ids
  codepoints.bin              (--exhaustive only) per-codepoint probe records
"""

import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True  # keep reference-0731/ pristine (no __pycache__)
sys.path.insert(0, os.path.join(ROOT, "reference-0731", "encoding"))

from tokenizers import Tokenizer  # noqa: E402
import encoding_dsv4  # noqa: E402

OUT = os.path.join(ROOT, "tests", "m2", "golden")
TOK_JSON = os.path.join(ROOT, "reference", "tokenizer.json")

EXHAUSTIVE = "--exhaustive" in sys.argv


def write_ids(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack(f"<{len(ids)}I", *ids) if ids else b"")


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer.from_file(TOK_JSON)

    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    def dec(ids):
        return tok.decode(ids, skip_special_tokens=False).encode("utf-8")

    paragraph = (
        "The quick brown fox jumps over the lazy dog. In 2024, large language "
        "models grew to 284B parameters — a 37.5% increase over expectations.\n\n"
        "深度求索发布了新一代模型，它在推理、编程和数学方面表现出色。"
        "日本語のテキストも正しくトークン化されるべきです。한국어도 마찬가지입니다.\n"
        "مرحبا بالعالم! هذا اختبار للنص العربي مع الأرقام ١٢٣٤٥.\n"
        "Code: `def f(x): return x**2 + 2*x - 1  # O(n) solution`\n"
        "URLs: https://example.com/path?q=1&r=2 — emails: user@example.com\n"
        "Emoji: 👋🌍🎉🚀 and math: ∑∫√π ≈ 3.14159, ±0.001, ①②③\n"
        "Combining: café (e + ́) vs café, Devanagari: नमस्ते, Thai: สวัสดี\n"
        "Quotes: “smart” ‘quotes’ — dashes… ellipsis! Punctuation: (a)[b]{c}<d>\n"
        "    indented line\n\t tabbed line\n\n\nmultiple blank lines above\n"
    )

    cases = {
        "ascii": "Hello, world! This is a test of the tokenizer: "
                 "numbers 123, 4567, 89012 and symbols!!! Don't stop—it's "
                 "fine. 3.14159 and 1e10; snake_case and CamelCase.",
        "utf8_multi": "中文测试：深度学习模型。日本語：こんにちは世界。"
                      "한국어: 안녕하세요. العربية: مرحبا بالعالم ١٢٣. "
                      "Emoji 👋🌍🎉🚀👨‍👩‍👧‍👦. Combining: áéó "
                      "नमस्ते दुनिया. สวัสดีชาวโลก. €£¥₿ ∑∫√π±≠≤≥.",
        "specials_text": "Use <｜User｜> and <think> like this: <｜Assistant｜>"
                         "<think>reasoning</think>answer<｜end▁of▁sentence｜>. "
                         "Also ｜DSML｜ markup <｜DSML｜invoke name=\"x\"> and "
                         "<｜begin▁of▁sentence｜> mid-text plus </think> and "
                         "<｜latest_reminder｜>2026-01-01 and <|EOT|> here.",
        "whitespace": "a\n\n  b\t c \n d   \n\n\ne\r\nf \t\ng \n \n h"
                      "\n \n\n \t\n\n\n   trailing   \n\n\n",
        "empty": "",
        "paragraph": paragraph,
        "long": paragraph * 40,
    }

    manifest = []
    for name, text in cases.items():
        ids = enc(text)
        with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
        write_ids(os.path.join(OUT, name + ".ids"), ids)
        with open(os.path.join(OUT, name + ".dec"), "wb") as f:
            f.write(dec(ids))
        # round-trip must be stable for valid UTF-8 input
        assert dec(ids).decode("utf-8") == text, f"round-trip unstable: {name}"
        manifest.append(name)

    # nosplit variants: special strings treated as plain text
    tj = json.load(open(TOK_JSON))
    tj_plain = dict(tj)
    tj_plain["added_tokens"] = []
    tmp_path = os.path.join(OUT, "_tokenizer_plain.json")
    with open(tmp_path, "w") as f:
        json.dump(tj_plain, f)
    tok_plain = Tokenizer.from_file(tmp_path)
    os.remove(tmp_path)
    for name in ("specials_text", "ascii"):
        ids = tok_plain.encode(cases[name], add_special_tokens=False).ids
        write_ids(os.path.join(OUT, name + ".nosplit.ids"), ids)

    with open(os.path.join(OUT, "tok_manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    # ---- specials coverage: every added token id at least once ----
    covered = set()
    with open(os.path.join(OUT, "specials.bin"), "wb") as f:
        for a in tj["added_tokens"]:
            content = a["content"]
            cid = a["id"]
            data = content.encode("utf-8")
            ids = enc(content)
            if ids != [cid]:
                raise RuntimeError(
                    f"added token {cid} {content!r} encodes to {ids}, expected [{cid}]")
            f.write(struct.pack("<III", cid, len(data), len(ids)))
            f.write(data)
            f.write(struct.pack(f"<{len(ids)}I", *ids))
            covered.add(cid)
    n_added = len(tj["added_tokens"])
    assert len(covered) == n_added, (len(covered), n_added)
    print(f"specials: all {n_added} added ids covered")

    # ---- conformance prompts: ids for the 4 reference pairs ----
    # (reference-0731/encoding/tests is byte-identical to reference/'s)
    tdir = os.path.join(ROOT, "reference-0731", "encoding", "tests")
    modes = {1: "thinking", 2: "thinking", 3: "thinking", 4: "chat"}
    for i in (1, 2, 3, 4):
        td = json.load(open(os.path.join(tdir, f"test_input_{i}.json")))
        if isinstance(td, dict):
            messages = td["messages"]
            messages[0]["tools"] = td["tools"]
        else:
            messages = td
        prompt = encoding_dsv4.encode_messages(messages, thinking_mode=modes[i])
        gold = open(os.path.join(tdir, f"test_output_{i}.txt")).read()
        assert prompt == gold, f"reference encoder disagrees on pair {i}"
        write_ids(os.path.join(OUT, f"enc_conf_{i}.ids"), enc(prompt))

    # ---- extra encoding cases (exercise paths the 4 pairs don't) ----
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
    calc_tool = {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Calculate",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}, "prec": {"type": "integer"}},
            },
        },
    }
    extras = [
        # 1: reasoning_effort="max" prefix + generation prompt
        #    (0731 semantics: "max" = the NEW "Beyond maximum..." prefix)
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement simply."},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": "max",
        },
        # 2: chat mode + tools + two tool results given in reverse call order
        {
            "messages": [
                {"role": "system", "content": "You are helpful.",
                 "tools": [weather_tool, calc_tool]},
                {"role": "user", "content": "Weather in Paris and 2+2?"},
                {"role": "assistant",
                 "reasoning_content": "Need two calls.",
                 "tool_calls": [
                     {"id": "call_a", "type": "function",
                      "function": {"name": "get_weather",
                                   "arguments": "{\"location\": \"Paris\"}"}},
                     {"id": "call_b", "type": "function",
                      "function": {"name": "calc",
                                   "arguments": "{\"expr\": \"2+2\", \"prec\": 0}"}},
                 ]},
                {"role": "tool", "tool_call_id": "call_b", "content": "4"},
                {"role": "tool", "tool_call_id": "call_a", "content": "sunny, 20°C"},
                {"role": "assistant", "reasoning_content": "Have both results.",
                 "content": "Paris is sunny, 20°C; 2+2 = 4."},
            ],
            "thinking_mode": "chat",
        },
        # 3: drop_thinking=false + tool-result list content + merged user text +
        #    response_format + wo_eos + query task
        {
            "messages": [
                {"role": "system", "content": "Reply in JSON.",
                 "response_format": {"type": "json_object"},
                 "tools": [calc_tool]},
                {"role": "user", "content": "What is 17 * 23?"},
                {"role": "assistant",
                 "reasoning_content": "Compute with the tool.",
                 "tool_calls": [
                     {"id": "c1", "type": "function",
                      "function": {"name": "calc",
                                   "arguments": "{\"expr\": \"17*23\"}"}},
                 ]},
                {"role": "tool", "tool_call_id": "c1",
                 "content": [{"type": "text", "text": "391"},
                             {"type": "image"}]},
                {"role": "user", "content": "Thanks, and 5! ?"},
                {"role": "assistant", "reasoning_content": "5! = 120.",
                 "content": "5! = 120", "wo_eos": True},
                {"role": "user", "content": "factorial intent", "task": "query"},
            ],
            "thinking_mode": "thinking",
            "drop_thinking": False,
        },
        # 4: developer role + title task + add_bos=false, chat mode
        {
            "messages": [
                {"role": "system", "content": "System note."},
                {"role": "developer", "content": "Developer instructions here."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Chat About Greetings",
                 "task": "title"},
            ],
            "thinking_mode": "chat",
            "add_default_bos_token": False,
        },
        # 5: DSML arguments with floats, bools, null, nested JSON, big ints,
        #    non-string values (string="false" + Python json.dumps rendering)
        {
            "messages": [
                {"role": "system", "content": "You are precise.",
                 "tools": [calc_tool]},
                {"role": "user", "content": "Run the numbers."},
                {"role": "assistant",
                 "tool_calls": [
                     {"id": "n1", "type": "function",
                      "function": {"name": "calc", "arguments":
                                   "{\"f1\": 1.5, \"f2\": -0.0, \"f3\": 1e3, "
                                   "\"f4\": 0.0001, \"f5\": 1e-5, "
                                   "\"f6\": 3.141592653589793, \"f7\": 1e16, "
                                   "\"f8\": 2.5e-8, "
                                   "\"big\": 123456789012345678901234567890, "
                                   "\"t\": true, \"z\": false, \"n\": null, "
                                   "\"arr\": [1, 2.25, \"x\"], "
                                   "\"obj\": {\"k\": \"v\", \"e\": {}}, "
                                   "\"s\": \"plain string\"}"}},
                 ]},
                {"role": "tool", "tool_call_id": "n1", "content": "ok"},
                {"role": "assistant", "content": "Done."},
            ],
            "thinking_mode": "thinking",
        },
        # 6: context parameter (BOS suppressed; drop_thinking context quirk)
        {
            "messages": [
                {"role": "user", "content": "second question"},
                {"role": "assistant", "reasoning_content": "fresh reasoning",
                 "content": "second answer"},
            ],
            "context": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "reasoning_content": "old reasoning",
                 "content": "first answer"},
            ],
            "thinking_mode": "thinking",
        },
        # 7: reasoning_effort="high" (0731: the OLD preview "max" prefix,
        #    "Absolute maximum...")
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement simply."},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": "high",
        },
        # 8: reasoning_effort="low" (0731 default: no prefix)
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement simply."},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": "low",
        },
        # 9: reasoning_effort="max" in CHAT mode — no-op (prefix is
        #    thinking-mode only)
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement simply."},
            ],
            "thinking_mode": "chat",
            "reasoning_effort": "max",
        },
        # 10: no reasoning_effort (None == "low": no prefix)
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement simply."},
            ],
            "thinking_mode": "thinking",
        },
    ]
    for i, spec in enumerate(extras, start=1):
        messages = spec["messages"]
        kwargs = {"thinking_mode": spec["thinking_mode"]}
        for k in ("drop_thinking", "add_default_bos_token", "reasoning_effort"):
            if k in spec:
                kwargs[k] = spec[k]
        if "context" in spec:
            kwargs["context"] = spec["context"]
        prompt = encoding_dsv4.encode_messages(messages, **kwargs)
        with open(os.path.join(OUT, f"enc_extra_{i}.json"), "w") as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)
        with open(os.path.join(OUT, f"enc_extra_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        write_ids(os.path.join(OUT, f"enc_extra_{i}.ids"), enc(prompt))
    print(f"encoding: 4 conformance pairs + {len(extras)} extra cases")

    # ---- exhaustive per-codepoint probes (optional) ----
    if EXHAUSTIVE:
        cps = list(range(0x00, 0x30000))
        cps += range(0x30000, 0x110000, 17)
        cps = [c for c in cps if not (0xD800 <= c <= 0xDFFF)]
        texts = []
        index = []  # (cp,) per group of 5
        for cp in cps:
            c = chr(cp)
            texts += [c, c * 2, "a" + c + "b", c + "\n ", c * 4]
            index.append(cp)
        print(f"encoding {len(texts)} probe strings...")
        results = tok.encode_batch(texts, add_special_tokens=False)
        path = os.path.join(OUT, "codepoints.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<II", 0xC0DE0001, len(index) * 5))
            k = 0
            for cp in index:
                for _ in range(5):
                    s = texts[k].encode("utf-8")
                    ids = results[k].ids
                    f.write(struct.pack("<II", len(s), len(ids)))
                    f.write(s)
                    if ids:
                        f.write(struct.pack(f"<{len(ids)}I", *ids))
                    k += 1
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")

    print("golden data written to", OUT)


if __name__ == "__main__":
    main()
