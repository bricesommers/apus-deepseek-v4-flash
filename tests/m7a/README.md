# tests/m7a — OpenAI-compatible server (M7a)

Milestone M7a: an OpenAI-compatible local server for apus, following the
colibri split — a **Python stdlib-only HTTP gateway** (`tools/server.py`)
driving the **C engine** (`bin/apus serve`) as a persistent subprocess.
No new dependencies (gateway: `http.server`, `json`, `subprocess`,
`threading`; engine: libc/pthreads only). No numerics/router changes:
the serve path reuses `encoding.h` (rendering), `model.h` (forward) and
`sample.h` (sampling) exactly as the `run` CLI does.

Run:

```
make golden-m7a     # regenerate tests/m7a/fixtures (scripted parrot models)
make test-m7a       # 34 tests, exit 0 iff all pass
make ubsan-m7a      # same suite with bin/apus built -fsanitize=undefined
```

Manual use:

```
bin/apus serve --model DIR [--tiered]                       # engine, stdio
.venv/bin/python tools/server.py --model DIR --port 8000    # gateway, HTTP
```

## Architecture: why stdio NDJSON (not a socket)

The C engine speaks **one-JSON-object-per-line (NDJSON) on stdin/stdout**;
the gateway owns all networking. Rationale: the engine stays libc-only (no
socket code in C to audit/fuzz), the process model matches colibri's
proven gateway-drives-engine pattern, pipes make the protocol trivially
testable without a TCP stack, and a single engine process serves exactly
one gateway so a second process is the only concurrency story. The
`--port P` option mentioned in the milestone sketch lives on the gateway
(`tools/server.py --port`), not the engine.

### Engine protocol (request → `bin/apus serve` stdin)

```json
{"id": <any>, "cmd": "encode",
 "messages": [...], "tools": [...]|null,
 "thinking": true|false, "reasoning_effort": "low"|"high"|"max"|null}

{"id": <any>, "cmd": "generate",
 "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 "tools": [...]|null, "thinking": bool, "reasoning_effort": str|null,
 "max_tokens": int, "temperature": float, "top_p": float,
 "seed": uint, "stop": [str, ...]}
```

- `tools` (OpenAI format) is attached to the first `system`/`developer`
  message (the rule the M2 conformance tests use); with no such message an
  empty `system` carrier is synthesized. `encoding.h` renders tools only
  from those roles — same as the reference.
- `"text"` is tokenized verbatim (no chat template, no BOS) — the
  `/v1/completions` path. `"ids"` feeds raw ids (synthetic models without
  a tokenizer).
- Defaults: `max_tokens` 32, `temperature` 1.0, `top_p` 1.0, `seed` 0;
  `temperature <= 0` = greedy. Generation is clamped to the model's
  `max_pos`.

### Engine protocol (events → stdout)

```json
{"id","type":"encoded","text","ids"}                 encode reply
{"id","type":"prompt","prompt_tokens"}               generate, first
{"id","type":"token","token_id","text"}              per generated token
{"id","type":"done","finish_reason","prompt_tokens",
 "completion_tokens","text"}                         terminal
{"id","type":"error","message"}                      request failed
```

- The terminating EOS token is never emitted as a `token` event;
  `finish_reason` is `"stop"` (EOS), `"length"` (max_tokens/max_pos), or
  `"stop_string"`.
- Stop strings are matched against the assembled decoded text; on a match
  the text is truncated at the match start (a partial piece of the last
  token is emitted if it precedes the match) and generation ends
  `"stop_string"`. `text` fields require a tokenizer in the model dir.
- The process stays alive across requests (model loads once). Every
  request gets a **fresh KV state**; conversation state is the gateway's
  job and multi-turn context is re-prefilled. KV reuse across turns is a
  later optimization (needs prefix-aware state save/restore in `model.h`).

## Gateway endpoints (`tools/server.py`)

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status","model","engine"}`; no auth |
| `GET /v1/models` | single-model OpenAI list |
| `POST /v1/chat/completions` | full OpenAI shape; `stream:true` → SSE |
| `POST /v1/completions` | `prompt` (string only); stream supported |
| `POST /debug/encode` | NON-STANDARD test endpoint: rendered prompt text + token ids for a message list (conformance/usage verification) |

Honored chat fields: `messages`, `tools`, `temperature`, `top_p`,
`max_tokens`/`max_completion_tokens`, `seed`, `stream`,
`stream_options.include_usage`, `stop` (string or list), `model`,
`chat_template_kwargs.thinking`, `reasoning_effort` (top-level or in
`chat_template_kwargs`). Unknown fields are ignored.

**Thinking-mode decision.** The reference encoder's default is
`thinking_mode="thinking"` (`encoding_dsv4.py` / `DS_ENC_OPTS_DEFAULT`),
so the server defaults to thinking on; a request sets
`chat_template_kwargs: {"thinking": false}` for chat mode, and the
server default can be flipped with `--no-thinking` / `APUS_THINKING=0`.
Responses expose reasoning as `reasoning_content` (DeepSeek API style).

**SSE streaming.** Chunks are `chat.completion.chunk` with
`choices[0].delta`: first `{"role":"assistant"}`, then
`reasoning_content` deltas (buffered until `</think>` is unambiguous),
then `content` deltas, then a final chunk with empty delta and
`finish_reason`, an optional usage chunk (`stream_options.include_usage`,
`choices: []`), and `data: [DONE]`. Served with `Connection: close` (no
chunked encoding).

**DSML ↔ OpenAI tool calls.** Request side: `tools` render into the
system prompt as DSML instructions via `encoding.h` (unchanged, M2
conformance). Response side: the gateway ports
`encoding_dsv4.py parse_message_from_completion_text` /
`parse_tool_calls` / `decode_dsml_to_arguments` to Python
(`tools/server.py parse_completion`), splitting the raw completion into
`reasoning_content` / `content` / OpenAI `tool_calls` (`id` =
`call_<24 hex>`, `type: "function"`, `function.arguments` = JSON string;
`string="false"` DSML parameters keep their raw JSON value). The port is
*tolerant*: EOS not required (length/stop finishes), unterminated
thinking → all `reasoning_content`, malformed DSML → plain content.
`finish_reason`: `"length"` for length, `"tool_calls"` when tool calls
were parsed, else `"stop"`. Streaming limitation: deltas forward the raw
text (DSML markup included) — only the final `finish_reason` reflects the
parse; the parsed structure is available non-streaming.

**Errors.** `400` malformed JSON / bad fields, `404` unknown model or
path, `401` auth failure — all in the OpenAI shape
`{"error": {"message","type","param","code"}}` (`invalid_request_error` /
`not_found_error` / `authentication_error` / `engine_error`).

**Concurrency.** One engine ⇒ requests are serialized through a single
lock in the gateway: concurrent HTTP requests queue in lock-acquisition
order and run one at a time (no interleaving, no batching, predictable
FIFO-ish behavior). **Auth:** env `APUS_API_KEY` → `Authorization:
Bearer` required on `/v1/*` and `/debug/*` (off by default; `/health`
open).

## The scripted "parrot" fixtures (`tests/m7a/fixtures/`)

Random-weight models can't produce a *known* token stream, which the
server tests need (DSML tool-call output, EOS, stop strings, exact usage
counts). `gen_fixtures.py` builds two mini-models on the M5 dims/schedule
with **all layer weights zero** (sublayer outputs vanish; the 4× mHC
residual is preserved exactly), random embed rows, and a **scripted
head**: for each transition `a → b`, `head[b] = 8·embed[a]`, so decoding
follows a fixed Markov chain (min logit margin >1600, checked at gen
time; robust at any temperature). Chain tokens with multi-char content
are added tokens (ids 300+; repeated `｜DSML｜` markers get alias ids
400+ with identical decode text), plus a byte-level vocab (ids 0–255,
no merges) and the canonical specials (BOS 256, EOS 257, `<｜User｜>`
258, `<｜Assistant｜>` 259, `<think>` 260, `</think>` 261, `｜DSML｜`
262). `max_pos` is raised to 8192 because byte-level prompts are long
(the DSML tools template alone is ~1 KB).

- `model_chat/`: thinking reply `"reasoning: thinking it over.</think>The
  answer is STOP right here."` + EOS; chat-mode reply without the
  reasoning (both share one continuation after `</think>`).
- `model_tools/`: reasoning + a complete DSML `get_weather` tool call +
  EOS (thinking and chat entries).

## Test coverage (34 tests, `test_server.py`)

- **Pipe protocol** (7): encode text+ids; malformed line / unknown cmd /
  error recovery; generate event sequence and EOS non-emission; usage
  counts == `encode` id count (tok.h counts); stop-string truncation
  (streamed pieces never contain the stop string); max_tokens clamp;
  ids-only path on the tokenizer-less M5 fixture incl. clean error for
  message requests.
- **DSML parser** (10): differential vs `reference/encoding/
  encoding_dsv4.py` on well-formed completions (thinking/chat, single &
  multiple invokes, typed `string="false"` params, CJK values); OpenAI
  `tool_calls` shape; tolerant tails (no EOS, unterminated think,
  malformed DSML); `ThinkSplitter` char-by-char exactness.
- **HTTP chat** (13 on model_chat): health/models; thinking & chat
  modes; usage == `/debug/encode` ids; stop strings (list & bare string);
  seed determinism; SSE (chunk shape, event order, reasoning/content
  reassembly == non-stream, usage chunk, `[DONE]`); 6-way concurrency all
  correct; error shapes; `/v1/completions` (+SSE); gateway-vs-pipe encode
  equality + byte-for-byte vs the reference encoder for a multi-turn
  conversation with `reasoning_content` dropping; **the 4 reference
  encoding conformance pairs through the full server path**.
- **HTTP tools** (3 on model_tools): full tool-call round trip
  (`tool_calls` JSON, `finish_reason: "tool_calls"`, usage); streaming
  finish reason; role=`tool` follow-up rendering (`<tool_result>`),
  byte-for-byte vs the reference encoder.
- **Auth** (1): `APUS_API_KEY` 401/200, open `/health`.

## Known limitations

- No KV reuse across turns — every request re-prefills the full
  conversation (single-user local serving; revisit with prefix caching).
- Single engine: requests serialized (see Concurrency). Multi-user
  throughput is a non-goal (ARCHITECTURE §1).
- Streaming tool calls arrive as raw DSML text in `content` deltas;
  parse client-side or use non-streaming.
- `/v1/completions` accepts a single string prompt (no batching, no
  `echo`/`logprobs`/`suffix`).
- Engine stop strings are matched on decoded text only (needs a
  tokenizer); at most 16 per request.
- `max_tokens` is clamped to the model's `max_pos`.

## M7b (Metal) / M8 (MTP) touchpoints

- **M7b**: the serve path calls only `apus_model_forward` — a Metal
  backend activated at model load (`apus_model_load_ex`) is transparent
  to the protocol and gateway. Nothing in the server layer should need
  changes; re-run `test-m7a` on a Metal build to prove tokens unchanged.
- **M8 (MTP speculative decoding)**: output tokens must be identical to
  non-speculative per seed, so the protocol can stay unchanged if
  speculation stays inside the forward loop. If speculation wants
  per-request control, add a `"speculative": bool` request field and a
  `done` counter field (accepted/drafted tokens) for the acceptance-rate
  metric; the gateway passes unknown fields through only if whitelisted
  in `_sampling_params`. KV reuse across turns (deferred) also interacts
  with MTP draft state — design them together.
