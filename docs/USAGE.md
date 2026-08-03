# apus — terminal usage

DeepSeek-V4-Flash, fully local on this Mac. Nothing leaves the machine.

## Interactive chat (like colibri's `coli chat`)

```bash
cd ~/Desktop/AI-PROJECTS/Apus
.venv/bin/python tools/chat.py --model weights/apus-0731 --tiered
```

In-chat commands:

| command | effect |
|---|---|
| `/quit` | exit (Ctrl-D also works) |
| `/reset` | clear conversation history |
| `/system <text>` | set a system message |
| `/thinking on\|off` | thinking mode (default on; off = direct answers) |
| `/temp 0` | greedy/deterministic (default 1.0) |
| `/max 2000` | max tokens per reply (default 2048) |
| `/raw <text>` | raw completion for this turn (no chat template) |
| `/help` | show all commands |

## One-shot generation

```bash
cd ~/Desktop/AI-PROJECTS/Apus
./bin/apus run --model weights/apus-0731 --tiered \
    --prompt "Your prompt here" --max-tokens 100 --temp 0
```

`--temp 0` = greedy. Add `--seed 42` for reproducibility. Add `--spec`
to try MTP speculative decoding (off by default; not always faster).

## OpenAI-compatible server (for other apps)

```bash
cd ~/Desktop/AI-PROJECTS/Apus
.venv/bin/python tools/server.py --model weights/apus-0731 --tiered --port 8080
```

Then point any OpenAI client at `http://localhost:8080/v1` (any non-empty
API key). Quick check:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "apus", "stream": true,
  "messages": [{"role": "user", "content": "Hello"}]}'
```

## What to expect

- First turn: ~10 s model load, then ~0.8 tok/s prefill, ~0.3 tok/s decode.
  A 100-token answer takes ~5 minutes. Normal for 160 GB on 32 GB RAM.
- Peak RAM ~14 GB — close heavy apps for best speed.
- Multi-turn chat re-reads the conversation each turn (no KV reuse yet);
  use `/reset` for new topics.
- Stop the server/chat with Ctrl-C; engine state (expert usage history) is
  saved on exit and makes later runs slightly faster.

## Useful knobs (environment variables)

| var | default | effect |
|---|---|---|
| `APUS_EXPERT_CACHE_MB` | 4096 | expert RAM cache; try 8192 with free RAM |
| `APUS_THREADS` | P-cores | compute threads |
| `APUS_PILOT_K` | 8 | prefetch depth (measured optimum) |
| `APUS_METAL=1` | off | GPU dense offload (needs `bin/apus_metal`) |

If numbers look off, compare with the baselines in `docs/STATUS.md`.
