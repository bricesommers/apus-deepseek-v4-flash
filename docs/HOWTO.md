# How to use apus — personal guide

DeepSeek-V4-Flash-0731 (284B params) running **fully locally** on this
MacBook Pro M1 (32 GB). No internet, no account, no API key needed at
runtime. Weights live in `weights/apus-0731/` (156 GB, do not delete).

Project root: `cd ~/Desktop/AI-PROJECTS/Apus`

---

## 1. Terminal — interactive chat (daily use)

```bash
cd ~/Desktop/AI-PROJECTS/Apus
.venv/bin/python tools/chat.py --model weights/apus-0731 --tiered
```

Type at the `you>` prompt. While it generates you'll see the thinking
(dimmed), then the answer, then a stats line `[N tok, Xs, Y tok/s, stop]`.

### In-chat commands

| command | what it does |
|---|---|
| `/quit` | exit (Ctrl-D or Ctrl-C also work) |
| `/reset` | clear the conversation (new topic — keeps it fast) |
| `/thinking off` | skip reasoning, answer directly (fast for simple questions) |
| `/thinking on` | reasoning mode back on (better for hard problems) |
| `/temp 0` | deterministic answers (same question → same reply) |
| `/temp 0.8` | more variety (default is 1.0) |
| `/max 4000` | max tokens per reply (default 2048; raise if you see `length`) |
| `/system <text>` | set a system message (e.g. "Answer in French") |
| `/raw <text>` | raw completion, no chat template, this turn only |
| `/help` | show all commands |

If a reply ends with **`length`** instead of `stop`, the model had more to
say — raise `/max` and ask it to continue.

## 2. Terminal — one-shot generation

```bash
cd ~/Desktop/AI-PROJECTS/Apus
./bin/apus run --model weights/apus-0731 --tiered \
    --prompt "Your prompt here" --max-tokens 200 --temp 0
```

`--temp 0` = greedy. Add `--seed 42` for reproducibility. `--spec` exists
but is intentionally off on this machine (DSpark is slower here — see
`tests/m11b/README.md`).

## 3. LM Studio (or any OpenAI-compatible app)

LM Studio cannot host apus's weights (it's not GGUF/MLX), but it can be
the **chat UI** in front of apus's server.

### Step 1 — start the apus server (leave it running)

```bash
cd ~/Desktop/AI-PROJECTS/Apus
.venv/bin/python tools/server.py --model weights/apus-0731 --tiered \
    --port 8080 --model-id deepseek-v4-flash-apus
```

Check it answers: `curl http://localhost:8080/health`

### Step 2 — connect LM Studio

1. In LM Studio, install the plugin **"OpenAI-Compatible Endpoint"**
   (`ankh/openai-compat-endpoint`).
2. Add a provider with:
   - **Base URL**: `http://localhost:8080/v1`
   - **API key**: `apus` (anything non-empty)
   - **Model**: `deepseek-v4-flash-apus`
3. Chat from the LM Studio UI. Thinking appears as reasoning content.

The same URL works for other apps (Open WebUI, Jan, Chatbox, your own
scripts with the OpenAI SDK pointed at `http://localhost:8080/v1`).

### Server notes

- One client at a time: requests queue (it's a single engine).
- Request options supported: `messages`, `temperature`, `top_p`,
  `max_tokens`, `seed`, `stream`, `stop`, `tools` (OpenAI tool calling),
  `chat_template_kwargs: {"thinking": false}`, `reasoning_effort`
  (`low`/`high`/`max`).
- Optional auth: `APUS_API_KEY=secret` before starting the server →
  clients must send `Authorization: Bearer secret`.

## 4. Speed — how to get the best tok/s

Measured on this machine: decode 0.3–1.07 tok/s, prefill ~9 tok/s.
The #1 factor is FREE RAM:

- **Close browsers/IDEs before a session** — worth up to 3×
  (0.27 → 0.93 tok/s measured).
- First tokens of a session are slow (cold cache); it speeds up as it
  warms. Long generations cruise ~1 tok/s.
- Each chat turn re-reads the conversation — `/reset` on new topics.
- Knobs (env vars before the command):
  `APUS_EXPERT_CACHE_MB=8192` (bigger expert cache if RAM is free),
  `APUS_THREADS=8`, `APUS_PILOT_K=8` (default, measured optimum).
- Expect ~10 s model load at start, then ~0.7–9 tok/s prefill (longer
  prompts are faster per token).

## 5. Troubleshooting

| symptom | what to do |
|---|---|
| very slow all of a sudden | another app is eating RAM → close it; or check `top`/`ps` for a second apus process |
| reply ends with `length` | raise `/max` (e.g. `/max 4000`) and say "continue" |
| weird pause on first token | normal — 10 s load + prefill of your message |
| server port busy | another server is running: `pkill -f server.py` |
| engine stuck (no output for minutes) | Ctrl-C the client; `pkill -f "apus serve"`; restart |
| want to reset learned expert pins | delete `apus.usage` files (regenerated automatically) |

## 6. What is where

| path | what |
|---|---|
| `weights/apus-0731/` | the model (156 GB) — never delete |
| `tools/chat.py` | terminal chat client |
| `tools/server.py` | OpenAI-compatible server |
| `tools/download-status.sh` | (only for downloads) live progress |
| `docs/USAGE.md` | the compact usage doc |
| `docs/STATUS.md` | project state, test evidence, decisions |
| `docs/R1B-RESULTS.md` | the external correctness proof |
| `HISTORY.md` | full project narrative |

If the machine reboots: nothing to reinstall — just start chat or the
server again. State is safe in git; the model is on disk.
