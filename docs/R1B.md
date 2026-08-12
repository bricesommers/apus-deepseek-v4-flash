# R1b — External verification: apus vs DeepSeek-V4-Flash-0731 via OpenRouter

R1b is the API-based replacement for the R1 GPU-rental plan
(`docs/R1.md`, `tools/r1/` — kept, untouched). It answers the same open
question from `docs/STATUS.md`: *does apus, running entirely on this Mac,
produce the same answers as DeepSeek-V4-Flash-0731 served by an independent
stack?* No GPU rental, no 167 GB second copy — just an OpenRouter API key
and the model you already have in `weights/apus-0731`.

No expertise assumed — follow the steps in order.

---

## 1. What R1b proves — and what it can't

**The idea in one paragraph:** we send an 8-prompt battery
(`tests/r1b/prompts.json`) to DeepSeek-V4-Flash-0731 through OpenRouter
(an API that routes to providers serving the same weights) and run the
same prompts through apus on this Mac. Then we compare the **answer
text**. Identical or near-identical answers with the same factual content
across greedy chat, thinking, code, CJK, and agentic prompts mean apus's
numerics, tokenizer, and chat encoding are right — a real serving stack
would have to be wrong in exactly the same way for that to be a
coincidence.

**Why byte-identity is NOT expected (and that's fine).** R1-with-a-GPU was
going to compare *token ids*, because both sides would run DeepSeek's own
kernels. OpenRouter's providers run their own serving stacks (different
GPUs, different kernels, different accumulation orders, possibly different
quantization). Floating-point addition is not associative, so two correct
implementations legitimately produce slightly different logits; at
"near-tie" positions (where the top-2 tokens are almost equally likely)
the streams can flip and then continue differently yet remain equally
correct. On top of that, OpenRouter gives us **text only** — no token ids,
no logprobs, no margins — so text comparison is the only gate available.
The verdicts are therefore:

| Verdict | Meaning | Counted as |
|---|---|---|
| `identical` | byte-equal answer content | pass |
| `normalized-identical` | equal after Unicode/whitespace normalization | pass |
| `diverges-at-N` | common prefix of N chars, then the streams part | pass **with WARN** — printed for human review |
| `format-different` | raw-mode case: apus continues a fragment, the API answers a chat message — different *formats* by construction; only content checks apply | pass with WARN — human review |
| `contradiction` | both sides name a capital city and they **differ** | **FAIL** (exit 1) |
| `api-error` / `apus-error` | API unreachable, engine crash, empty output | **FAIL** (exit 1) |

Divergences are **for your eyes**: the tool prints the common-prefix ratio
and the first 120 characters of each continuation. Read them. "Both are a
correct 2-day Paris plan with different phrasing" is a pass; "apus's
answer is garbled" is a bug — the tool deliberately does not try to make
that judgment for you.

**What R1b cannot prove:** exact-token equivalence (no token ids from the
API), RNG-stream equality (the fixed-seed sampled case runs on a different
sampler on a different stack — its divergence is expected and
informational), and provider fidelity (you are trusting OpenRouter's
provider to serve unmodified 0731 weights; pick the official DeepSeek
provider if the routing lets you). If you ever need bit-level proof, the
R1 GPU plan in `docs/R1.md` remains available.

## 2. Get an OpenRouter key (never committed)

1. Sign up at https://openrouter.ai and create a key at
   https://openrouter.ai/keys. Put a few dollars of credit on it — this
   battery costs **a few cents** (8 short greedy generations; see §5).
2. In the shell where you run the tool:

   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```

   The tool reads only this environment variable. It is sent as the
   `Authorization` header, scrubbed from every error message and results
   file, and never printed. Do not paste it into any file, command line,
   or chat.

## 3. Find the exact model id

OpenRouter model ids change as providers list new variants; the 0731
release may exist alongside the preview. List what your key can see:

```bash
.venv/bin/python tools/r1b/compare_api.py --list-models
```

This prints every id containing "deepseek". Pick the
**DeepSeek-V4-Flash-0731** entry and pass it as `--model` (the default,
`deepseek/deepseek-v4-flash`, is only a guess).

## 4. Run the comparison

Sanity first (no key needed, no API calls):

```bash
# what WOULD be sent, both sides, per case:
.venv/bin/python tools/r1b/compare_api.py --dry-run

# apus side only (smoke-test the engine path):
.venv/bin/python tools/r1b/compare_api.py --apus-only --cases france_greedy
```

Then the real thing:

```bash
export OPENROUTER_API_KEY="sk-or-..."
.venv/bin/python tools/r1b/compare_api.py --model deepseek/<the-0731-id>
```

Useful flags: `--cases france_greedy,spain_chat` (subset),
`--timeout 120` (per API call), `--retries 2` (with backoff, default),
`--apus-timeout 1800` (per read on the engine; decode is ~0.5 tok/s on
this Mac, so the thinking cases take minutes), `--no-tiered`,
`--verbose` (engine stderr + retry logging).

**Time and cost:** the apus side dominates wall clock (at ~0.3–0.9 tok/s,
the full 8-case battery is roughly 30–60 minutes; `--cases` for a quick
look). The API side is seconds per case and pennies in total. Requests
are strictly sequential — no rate-limit risk. A budget guard refuses to
run more than 8 cases in one go unless you pass `--yes-i-know`.

## 5. Reading the output

```
  [OK  ] spain_chat: identical
  [OK~ ] code_chat: normalized-identical
  [DIV ] math_thinking: diverges-at-141 (prefix ratio 0.42)
       api : …'so 347 × 89 = 30,883. Let me verify…'
       apus: …'so 347 multiplied by 89 gives 30883. Checking…'
  [FMT ] france_greedy: format-different (content check: match; raw vs chat — human review)
=== summary ===
  identical: 3
  diverges: 2
  format-different: 2
RESULT: PASS with WARN — 4 case(s) diverge or are format-different …
```

Exit code: **0** for pass (including PASS with WARN), **1** for hard
failure (API unreachable, apus crash, empty output, or a factual
contradiction), **2** for setup errors (missing key, bad battery, budget
guard). Full per-case detail — the exact request bodies, response
metadata, usage, reasoning presence, both full texts — lands in
`tests/r1b/results/r1b_<timestamp>.json` (gitignored).

## 6. Per-case notes baked into the tool

- **Thinking cases** (`paris_thinking`, `math_thinking`, `cjk_thinking`,
  `agentic_tool`): the model thinks by default on the API side; the tool
  captures `reasoning_content` (or OpenRouter's normalized `reasoning`)
  and `content` separately. The verdict compares **content**; the
  reasoning prefix ratio is reported as informational — different stacks
  think differently, and a thinking divergence does not imply a wrong
  answer. Note that API-side reasoning **consumes the same `max_tokens`
  budget** as the answer, so a case can come back with empty content and
  finish_reason `length`; that is reported as an api-error with a hint to
  re-run with a higher cap, not as an apus failure.
- **Chat cases** (`spain_chat`, `code_chat`): apus runs with thinking
  OFF, so the API is asked to do the same. OpenRouter has no single
  documented switch for this model, so the tool sends **both** dialects
  defensively: `chat_template_kwargs: {"thinking": false}` (pass-through
  to the chat template — the same dialect apus's own server honors) and
  `reasoning: {"enabled": false}` (OpenRouter's unified reasoning
  switch). Which one a given provider honors cannot be known without a
  live key; the results JSON records `reasoning_present` per case. **If
  a chat case comes back with reasoning present, its comparison is
  thinking-vs-non-thinking and divergence is expected** — re-check
  against the record before reading anything into it.
- **Raw cases** (`france_greedy`, `france_sampled`): apus continues the
  fragment verbatim (no chat template). The API has no raw-completion
  endpoint for chat models, so the prompt goes as a single user message —
  a different *format* by construction. The tool never byte-compares
  these; it checks normalized-substring containment and, for the
  capital-of-France prompt, extracts the named city on both sides
  (a narrow regex applied only when **both** sides parse — no match is
  "no opinion", never a failure). Verdict is always `format-different`;
  read both texts in the results JSON.
- **`france_sampled`** (temp 1.0, seed 42): the seed is sent to both
  sides, but two different samplers on two different stacks cannot be
  expected to draw the same stream. Any divergence here is informational,
  not a failure.
- **Temperature 0** is sent for greedy cases; some providers clamp or
  substitute a tiny epsilon. The actual request body is recorded per case
  in the results JSON — what was sent is what you see.
- **The fact check** (`contradiction`) currently only knows how to
  extract capital-city answers. It is a tripwire for "obviously wrong",
  not a semantic judge; everything else is left to human review by
  design.

## 7. If something fails

- `error: env OPENROUTER_API_KEY is not set` → §2.
- `HTTP 401` → wrong/expired key. `HTTP 402` → out of credit.
  `HTTP 404` → the model id is wrong; re-run `--list-models`.
- `api-error … empty content` on a thinking case → the API burned all
  `max_tokens` on reasoning; retry that case with a battery copy that
  raises its `max_tokens`.
- apus-side failures (engine died, timeout) are real bugs — keep the
  results JSON and report.
- Everything in `tests/r1b/results/` is scrubbed of the key and safe to
  share.

## 8. Tests

```bash
.venv/bin/python tests/r1b/test_compare_api.py
```

31 tests, no API key needed: classification logic, retry/backoff and
error paths against a local HTTP stub, end-to-end runs against the stub
plus the scripted m7a parrot model, exit codes, budget guard, dry-run /
apus-only, and a key-never-printed check on every output channel.
