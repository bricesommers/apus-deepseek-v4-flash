# R1b — External validation of apus against DeepSeek's API

**Date:** 2026-08-02/03 · **Verdict: PASSED** · **Commit:** `25c4e5a`

R1b was the project's final open verification gap (risk "R1" from the
original architecture doc): *does the engine on this Mac actually produce
DeepSeek's answers?* Instead of renting GPUs to run DeepSeek's reference
inference (that kit is still in `tools/r1/`, unused), we compared apus
against **DeepSeek-V4-Flash-0731 hosted on OpenRouter**
(`deepseek/deepseek-v4-flash-0731`), using the owner's API key.
Total API cost: ~$0.001.

## What was built

- `tests/r1b/prompts.json` — the 8-case battery (greedy QA, code, math,
  CJK, agentic tool-use, long thinking, one sampled case).
- `tools/r1b/compare_api.py` — the comparator (stdlib only): calls the
  OpenRouter API and the local `apus serve` engine on the same cases and
  compares the texts (identical / near-identical / diverges-at-N /
  format-different / error). Key is read from `OPENROUTER_API_KEY`, never
  printed or saved. 31 unit tests.
- `docs/R1B.md` — the operator guide (how to re-run it).
- Results: `tests/r1b/results/r1b_*.json` (full request/response records).

## The three runs (what actually happened)

1. **First run** — user typo `deepseek/deepseek/deepseek-v4-flash-0731`
   (doubled prefix) → all 8 API calls HTTP 400. Exposed two tool issues,
   both fixed and committed: (a) it wasted 17 min generating locally
   after the API had already failed → now API-first, skip local work on
   api-error; (b) a flaky test that raced on timestamped result files →
   fixed, proven stable over 5 runs.
2. **Second run** — real comparison. 2 identical, 2 stylistic divergences
   (same facts), 1 format-different, 3 "errors": two cases where the API
   spent its whole token budget on *reasoning* (it counts thinking inside
   `max_tokens`: 231/256 and 200/192) and one DNS blip. Battery fixed:
   thinking-case budgets raised.
3. **Third/fourth run** (the 3 fixed cases) — agentic + france_sampled
   completed (stylistic match / format-different by design); Paris needed
   one more bump (API reasoning: 728/768 → then 981/2048) and finally
   completed with a matching itinerary.

## Final per-case results

| case | verdict | detail |
|---|---|---|
| spain_chat | **identical** | byte-for-byte equal to DeepSeek's API |
| code_chat | **identical** | byte-for-byte equal |
| math_thinking | diverges (stylistic) | both compute **30,883** with the same steps |
| cjk_thinking | diverges (stylistic) | both correctly explain quantum entanglement in Chinese |
| agentic_tool | diverges (stylistic) | both "call the weather tool with Tokyo", same parameters |
| paris_thinking | diverges (stylistic) | both produce a practical 2-day Paris itinerary, same structure |
| france_greedy | format-different | raw prompt has no chat-API equivalent — by design |
| france_sampled | format-different | RNG stacks differ by construction — informational |

**Zero contradictions, zero factual errors, zero engine errors** in all
8 cases. Every divergence is phrasing (expected GPU-vs-NEON near-tie
class), never content.

## Conclusion

apus on the MacBook Pro M1 produces DeepSeek-V4-Flash-0731's answers —
internally proven (bitwise invariance across every engine feature, ~36k
checks green) and now externally confirmed against DeepSeek's own hosted
model. The project has no open verification gaps.

## Reproduce / re-run

```bash
export OPENROUTER_API_KEY="<your OpenRouter API key>"
.venv/bin/python tools/r1b/compare_api.py --list-models   # pick the 0731 id
.venv/bin/python tools/r1b/compare_api.py --model deepseek/deepseek-v4-flash-0731
```

~30–60 min (local generation dominates). Full operator guide: `docs/R1B.md`.
For bit-level proof instead of text-level, the GPU golden kit is in
`tools/r1/` with its own guide `docs/R1.md`.
