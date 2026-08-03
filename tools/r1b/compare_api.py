#!/usr/bin/env python3
"""R1b comparator — apus vs DeepSeek-V4-Flash-0731 via the OpenRouter API.

Stdlib only (urllib). Replaces the GPU-rental plan (tools/r1/, docs/R1.md)
with a text-level comparison against the same model served by OpenRouter
providers. See docs/R1B.md for the operator guide and the honest limits of
an API-based comparison.

Flow per case of tests/r1b/prompts.json:
  1. OpenRouter: POST /chat/completions with the case prompt as a single
     user message. Thinking cases rely on the model's default (it thinks);
     non-thinking cases ask for no thinking via BOTH dialects we know
     OpenRouter/DeepSeek honor — `chat_template_kwargs: {"thinking": false}`
     (pass-through to the chat template, the dialect apus itself honors)
     and `reasoning: {"enabled": false}` (OpenRouter's unified switch).
     Which one was honored cannot be known a priori; the report records
     `reasoning_present` per case so a thinking answer on a "chat" case is
     visible instead of silently miscompared.
  2. apus: `bin/apus serve` NDJSON (spawned once), same case — chat →
     messages with thinking:false, thinking → thinking:true, raw → the
     verbatim "text" path (no chat template).
  3. Compare TEXT (the API gives no token ids / logprobs — text is the
     gate, see docs/R1B.md §1).

Classification per case:
  identical              — byte-equal answer content.
  normalized-identical   — equal after Unicode NFKC + whitespace collapse.
  diverges-at-N          — common prefix of N chars; both continuations
                           (first 120 chars) are printed for HUMAN review.
                           Not a failure by itself (provider numerics
                           differ); counted, exit 0 with WARN.
  format-different       — raw-mode cases: apus continues the fragment,
                           the API answers a chat message — byte comparison
                           is meaningless by construction. Content is
                           checked sensibly instead: normalized-substring
                           containment plus, for "capital of" prompts, a
                           narrow extracted-answer check (below).
  contradiction          — HARD: both sides name a capital city (narrow
                           regex, only applied when BOTH extractions
                           succeed) and they differ. Exit 1.
  api-error / apus-error / error (empty output) — HARD. Exit 1.

Raw-case rationale: the only shared ground truth between a raw completion
("...is Paris, the...") and a chat answer ("The capital of France is
Paris.") is the factual content, not the surface form. We therefore (a)
never compare raw cases byte-wise, (b) check normalized containment in
both directions, and (c) extract the named capital when the prompt asks
for one. Anything else is reported for human review, never auto-failed.

Exit codes: 0 = pass (WARN printed when divergences/format-different cases
exist — they need a human eye); 1 = hard failure; 2 = setup/usage error
(missing API key, bad battery, budget guard).

The API key comes from env OPENROUTER_API_KEY, is sent only as the
Authorization header, and is scrubbed from every error message, printout,
and results file. It is NEVER printed.
"""

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request

API_BASE_DEFAULT = "https://openrouter.ai/api/v1"
MODEL_DEFAULT = "deepseek/deepseek-v4-flash"
KEY_ENV = "OPENROUTER_API_KEY"
MAX_CASES = 8          # budget guard: refuse more without --yes-i-know
RETRY_STATUSES = {429, 500, 502, 503, 504}
THINK_END = "</think>"


class ApiError(Exception):
    """Non-retryable API failure (4xx, malformed response)."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def scrub(text, key):
    """Remove any occurrence of the API key from a string."""
    if key and isinstance(text, str):
        return text.replace(key, "***")
    return text


# --------------------------------------------------------------------------
# OpenRouter client
# --------------------------------------------------------------------------

class OpenRouter:
    def __init__(self, key, base=API_BASE_DEFAULT, timeout=120, retries=2,
                 backoff=2.0, verbose=False):
        self.key = key
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.verbose = verbose

    def _request(self, method, path, body=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                url, data=data, method=method,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode(errors="replace")[:2000]
                except Exception:
                    detail = ""
                msg = scrub(f"HTTP {e.code} from {url}: {detail}", self.key)
                if e.code in RETRY_STATUSES and attempt < self.retries:
                    last = msg
                    if self.verbose:
                        print(f"    retry {attempt + 1}/{self.retries} "
                              f"after {msg.splitlines()[0][:120]}")
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise ApiError(msg, status=e.code)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                msg = scrub(f"network error from {url}: {e}", self.key)
                if attempt < self.retries:
                    last = msg
                    if self.verbose:
                        print(f"    retry {attempt + 1}/{self.retries} "
                              f"after {msg[:120]}")
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise ApiError(msg)
        raise ApiError(last or "unreachable")

    def chat(self, body):
        return self._request("POST", "/chat/completions", body)

    def models(self):
        return self._request("GET", "/models")


def build_api_body(case, model):
    """The exact request body we send for one battery case."""
    body = {"model": model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": float(case.get("temp", 0)),
            "top_p": float(case.get("top_p", 1.0)),
            "max_tokens": int(case["max_tokens"])}
    if "seed" in case:
        body["seed"] = int(case["seed"])
    if case["mode"] != "thinking":
        # Ask for no thinking in both known dialects; the report records
        # whether reasoning came back anyway (reasoning_present).
        body["chat_template_kwargs"] = {"thinking": False}
        body["reasoning"] = {"enabled": False}
    return body


def extract_api_text(resp):
    """(content, reasoning, meta) from an OpenRouter chat.completion."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    # OpenRouter normalizes to `reasoning`; some providers use
    # `reasoning_content` (DeepSeek API style). Accept both.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    meta = {"model": resp.get("model"), "id": resp.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage"),
            "reasoning_present": bool(reasoning)}
    return content, reasoning, meta


# --------------------------------------------------------------------------
# apus serve driver (same NDJSON protocol as tools/r1/compare.py)
# --------------------------------------------------------------------------

class ApusServe:
    def __init__(self, apus_bin, model_dir, tiered=True, verbose=False,
                 timeout=1800):
        cmd = [apus_bin, "serve", "--model", model_dir]
        if tiered:
            cmd.append("--tiered")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
            text=True, bufsize=1)
        self.timeout = timeout
        self._req = 0

    def _read_msg(self):
        fd = self.proc.stdout.fileno()
        r, _, _ = select.select([fd], [], [], self.timeout)
        if not r:
            raise TimeoutError(f"apus serve: no output for {self.timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("apus serve: EOF on stdout (engine died?)")
        return json.loads(line)

    def generate(self, case):
        """One battery case -> {"text", "finish_reason", "completion_tokens"}.
        Raises on engine error."""
        self._req += 1
        rid = f"r1b-{self._req}"
        fields = {"id": rid, "cmd": "generate",
                  "max_tokens": int(case["max_tokens"]),
                  "temperature": float(case.get("temp", 0)),
                  "top_p": float(case.get("top_p", 1.0)),
                  "seed": int(case.get("seed") or 0)}
        if case["mode"] == "raw":
            fields["text"] = case["prompt"]
        else:
            fields["messages"] = [{"role": "user", "content": case["prompt"]}]
            fields["thinking"] = case["mode"] == "thinking"
        self.proc.stdin.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            msg = self._read_msg()
            if msg.get("id") != rid:
                continue
            t = msg.get("type")
            if t == "token":
                continue
            if t == "done":
                return {"text": msg.get("text", ""),
                        "finish_reason": msg.get("finish_reason"),
                        "completion_tokens": msg.get("completion_tokens")}
            if t == "error":
                raise RuntimeError(f"apus serve error: {msg.get('message')}")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def split_apus_text(case, text):
    """apus's raw completion -> (content, reasoning), mirroring how
    tools/server.py splits at the first </think>."""
    if case["mode"] == "thinking" and THINK_END in text:
        reasoning, content = text.split(THINK_END, 1)
        return content, reasoning
    return text, ""


# --------------------------------------------------------------------------
# Text classification (pure, importable for tests)
# --------------------------------------------------------------------------

def normalize_text(s):
    """Unicode NFKC + whitespace collapse. Absorbs cosmetic differences
    (full-width chars, non-breaking spaces, newline style) without
    pretending different words are equal."""
    return " ".join(unicodedata.normalize("NFKC", s or "").split())


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# Narrow fact check for the capital-QA prompts. Only fires when BOTH sides
# yield a city; a missing pattern match is "no opinion", never a failure.
_CAPITAL_RE = re.compile(r"capital of [A-Za-zÀ-ÿ' -]+? is (?:the city of )?"
                         r"([A-Z][A-Za-zÀ-ÿ-]+)")


def extract_capital(text):
    m = _CAPITAL_RE.search(normalize_text(text))
    return m.group(1) if m else None


def classify_case(case, api_content=None, api_reasoning="", api_error=None,
                  apus_content=None, apus_reasoning="", apus_error=None):
    """Classify one case from the two answer texts. Returns a verdict dict.
    Compares CONTENT (the reasoning trace is informational only — providers
    think differently, and a thinking divergence does not imply a wrong
    final answer)."""
    cid = case["id"]
    mode = case["mode"]
    v = {"case": cid, "mode": mode, "sampled": float(case.get("temp") or 0) > 0}

    if api_error:
        v.update(verdict="api-error", error=api_error)
        return v
    if apus_error:
        v.update(verdict="apus-error", error=apus_error)
        return v
    if not (apus_content or "").strip():
        v.update(verdict="apus-error",
                 error="apus produced empty content")
        return v
    if not (api_content or "").strip():
        v.update(verdict="api-error",
                 error="API returned empty content "
                       "(reasoning may have consumed max_tokens)")
        return v

    # Reasoning similarity is informational only.
    if api_reasoning or apus_reasoning:
        na, np_ = normalize_text(api_reasoning), normalize_text(apus_reasoning)
        denom = max(len(na), len(np_), 1)
        v["reasoning_prefix_ratio"] = round(
            common_prefix_len(na, np_) / denom, 3)

    # Fact check (capital prompts): a proven contradiction is hard.
    cap_api, cap_apus = extract_capital(api_content), \
        extract_capital(apus_content)
    v["capital_api"], v["capital_apus"] = cap_api, cap_apus
    if cap_api and cap_apus and cap_api.lower() != cap_apus.lower():
        v.update(verdict="contradiction",
                 error=f"API says {cap_api}, apus says {cap_apus}")
        return v

    na, np_ = normalize_text(api_content), normalize_text(apus_content)

    if mode == "raw":
        # Raw completion vs chat answer: different FORMATS by construction
        # (see module docstring). Content-sensible checks only.
        if na == np_ or (na and np_ and (np_ in na or na in np_)):
            content_check = "match"
        elif cap_api and cap_apus and cap_api.lower() == cap_apus.lower():
            content_check = "match"
        elif cap_api or cap_apus:
            content_check = "unknown"  # one side didn't name it parseably
        else:
            content_check = "unknown"
        v.update(verdict="format-different", content_check=content_check,
                 api_text=api_content[:400], apus_text=apus_content[:400])
        return v

    if api_content == apus_content:
        v.update(verdict="identical", compared=len(apus_content))
        return v
    if na == np_:
        v.update(verdict="normalized-identical", compared=len(na))
        return v

    p = common_prefix_len(api_content, apus_content)
    denom = max(len(api_content), len(apus_content), 1)
    v.update(verdict=f"diverges-at-{p}", diverges_at=p,
             prefix_ratio=round(p / denom, 3),
             api_continuation=api_content[p:p + 120],
             apus_continuation=apus_content[p:p + 120])
    return v


HARD_VERDICTS = ("api-error", "apus-error", "contradiction")


def is_hard(v):
    return v["verdict"] in HARD_VERDICTS


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_verdict(v):
    tag = {"identical": "OK  ", "normalized-identical": "OK~ ",
           "format-different": "FMT ", "contradiction": "HARD",
           "api-error": "ERR ", "apus-error": "ERR "}.get(
               v["verdict"] if not v["verdict"].startswith("diverges")
               else "div", "DIV ")
    line = f"  [{tag}] {v['case']}: {v['verdict']}"
    if v["verdict"].startswith("diverges"):
        line += (f" (prefix ratio {v['prefix_ratio']})\n"
                 f"       api : …{v['api_continuation']!r}\n"
                 f"       apus: …{v['apus_continuation']!r}")
    elif v["verdict"] == "format-different":
        line += (f" (content check: {v['content_check']}; "
                 "raw vs chat — human review)")
    elif is_hard(v):
        line += f" — {v.get('error', '')[:200]}"
    if "reasoning_prefix_ratio" in v:
        line += f" [reasoning prefix {v['reasoning_prefix_ratio']}]"
    print(line)


# --------------------------------------------------------------------------

def load_battery(path):
    with open(path) as f:
        return json.load(f)["cases"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompts", default="tests/r1b/prompts.json")
    p.add_argument("--model", default=MODEL_DEFAULT,
                   help="OpenRouter model id (default: %(default)s — run "
                        "--list-models first and pick the 0731 entry).")
    p.add_argument("--api-base", default=API_BASE_DEFAULT)
    p.add_argument("--apus", default="bin/apus")
    p.add_argument("--apus-model", default="weights/apus-0731")
    p.add_argument("--cases", default="", help="Comma-separated case ids.")
    p.add_argument("--list-models", action="store_true",
                   help="List OpenRouter model ids matching 'deepseek' "
                        "and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what WOULD be sent; no API calls, no apus.")
    p.add_argument("--apus-only", action="store_true",
                   help="Run only the apus side (no API key needed).")
    p.add_argument("--yes-i-know", action="store_true",
                   help="Lift the 8-case budget guard.")
    p.add_argument("--out-dir", default="tests/r1b/results")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-API-call timeout (s).")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--apus-timeout", type=int, default=1800,
                   help="Per-read timeout (s) waiting on apus serve.")
    p.add_argument("--no-tiered", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    key = os.environ.get(KEY_ENV, "")
    try:
        cases = load_battery(args.prompts)
    except Exception as e:
        print(f"error: cannot load battery {args.prompts}: {e}")
        return 2
    want = set(args.cases.split(",")) if args.cases else None
    cases = [c for c in cases if not want or c["id"] in want]
    if not cases:
        print("error: no cases selected")
        return 2
    if len(cases) > MAX_CASES and not args.yes_i_know:
        print(f"error: {len(cases)} cases selected but the budget guard is "
              f"{MAX_CASES} (each case costs API tokens). Pass --yes-i-know "
              "to override.")
        return 2

    if args.list_models or not (args.dry_run or args.apus_only):
        if not key:
            print(f"error: env {KEY_ENV} is not set. Get a key at "
                  "https://openrouter.ai/keys and export it — never "
                  "paste it into files or the command line.")
            return 2
    client = OpenRouter(key, base=args.api_base, timeout=args.timeout,
                        retries=args.retries, verbose=args.verbose) \
        if key else None

    if args.list_models:
        try:
            data = client.models()
        except ApiError as e:
            print(f"error listing models: {e}")
            return 1
        ids = sorted(m.get("id", "") for m in data.get("data", []))
        hits = [i for i in ids if "deepseek" in i.lower()]
        print(f"{len(hits)} model id(s) matching 'deepseek':")
        for i in hits:
            print(f"  {i}")
        print("\nPick the DeepSeek-V4-Flash **0731** entry and pass it as "
              "--model.")
        return 0

    if args.dry_run:
        print("=== DRY RUN — nothing is sent ===")
        for c in cases:
            print(f"\n--- case {c['id']} ({c['mode']}) ---")
            print("API request body:")
            print(json.dumps(build_api_body(c, args.model),
                             ensure_ascii=False, indent=1))
            apus_req = ({"cmd": "generate", "text": c["prompt"]}
                        if c["mode"] == "raw" else
                        {"cmd": "generate",
                         "messages": [{"role": "user",
                                       "content": c["prompt"]}],
                         "thinking": c["mode"] == "thinking"})
            apus_req.update(max_tokens=c["max_tokens"],
                            temperature=float(c.get("temp", 0)),
                            top_p=float(c.get("top_p", 1.0)),
                            seed=int(c.get("seed") or 0))
            print("apus NDJSON request:")
            print(json.dumps(apus_req, ensure_ascii=False, indent=1))
        return 0

    # ---------------- run ----------------
    results = {"meta": {"tool": "r1b/compare_api.py",
                        "model": args.model, "api_base": args.api_base,
                        "apus_model": args.apus_model,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime())},
               "cases": {}}
    srv = None
    if not args.apus_only:
        print(f"Model (API): {args.model}")
    print(f"Starting {args.apus} serve --model {args.apus_model} ...",
          flush=True)
    t0 = time.time()
    verdicts = []
    srv = ApusServe(args.apus, args.apus_model, tiered=not args.no_tiered,
                    verbose=args.verbose, timeout=args.apus_timeout)
    try:
        for i, c in enumerate(cases):
            cid = c["id"]
            print(f"  [{i + 1}/{len(cases)}] {cid} ({c['mode']}) ...",
                  flush=True)
            rec = {"mode": c["mode"]}
            # API side FIRST: a hard API failure (e.g. bad model id) makes
            # the expensive apus generation pointless — skip it.
            if not args.apus_only:
                body = build_api_body(c, args.model)
                rec["api_request"] = body
                try:
                    resp = client.chat(body)
                    content, reasoning, meta = extract_api_text(resp)
                    rec["api"] = {"meta": meta, "content": content,
                                  "reasoning": reasoning}
                    print(f"    api: finish={meta['finish_reason']} "
                          f"reasoning_present={meta['reasoning_present']} "
                          f"usage={meta['usage']}", flush=True)
                except ApiError as e:
                    rec["api_error"] = str(e)
                    print(f"    api FAILED: {e}", flush=True)
            # apus side
            if args.apus_only or "api_error" not in rec:
                try:
                    t = time.time()
                    gen = srv.generate(c)
                    content, reasoning = split_apus_text(c, gen["text"])
                    rec["apus"] = {"finish_reason": gen["finish_reason"],
                                   "completion_tokens":
                                       gen["completion_tokens"],
                                   "content": content, "reasoning": reasoning,
                                   "seconds": round(time.time() - t, 1)}
                except Exception as e:
                    rec["apus_error"] = str(e)
                    print(f"    apus FAILED: {e}", flush=True)
            else:
                print("    apus: skipped (api-error)", flush=True)
            if args.apus_only:
                if "apus_error" in rec:
                    v = {"case": cid, "mode": c["mode"],
                         "verdict": "apus-error",
                         "error": rec["apus_error"]}
                elif not (rec["apus"]["content"] or "").strip():
                    v = {"case": cid, "mode": c["mode"],
                         "verdict": "apus-error",
                         "error": "apus produced empty content"}
                else:
                    v = {"case": cid, "mode": c["mode"],
                         "verdict": "skipped-api"}
            else:
                v = classify_case(
                    c,
                    api_content=(rec.get("api") or {}).get("content"),
                    api_reasoning=(rec.get("api") or {}).get("reasoning", ""),
                    api_error=rec.get("api_error"),
                    apus_content=(rec.get("apus") or {}).get("content"),
                    apus_reasoning=(rec.get("apus") or {}).get("reasoning",
                                                                ""),
                    apus_error=rec.get("apus_error"))
            rec["verdict"] = v
            verdicts.append(v)
            results["cases"][cid] = v | {"record": rec}
    finally:
        srv.close()
    print(f"runs finished in {time.time() - t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       time.strftime("r1b_%Y%m%d_%H%M%S.json", time.gmtime()))
    blob = scrub(json.dumps(results, ensure_ascii=False, indent=1), key)
    with open(out, "w") as f:
        f.write(blob)
    print(f"results saved to {out}")

    # ---------------- report ----------------
    print("\n=== R1b comparison report ===")
    for v in verdicts:
        print_verdict(v)
    counts = {}
    for v in verdicts:
        base = ("diverges" if v["verdict"].startswith("diverges")
                else v["verdict"])
        counts[base] = counts.get(base, 0) + 1
    print("\n=== summary ===")
    for k in ("identical", "normalized-identical", "diverges",
              "format-different", "skipped-api", "contradiction",
              "api-error", "apus-error"):
        if k in counts:
            print(f"  {k}: {counts[k]}")
    hard = [v for v in verdicts if is_hard(v)]
    if hard:
        print(f"\nRESULT: FAIL — {len(hard)} hard error(s)/contradiction(s). "
              "See the report above and the results JSON.")
        return 1
    soft = counts.get("diverges", 0) + counts.get("format-different", 0)
    if soft:
        print(f"\nRESULT: PASS with WARN — {soft} case(s) diverge or are "
              "format-different. Text comparison cannot prove equivalence "
              "there; review the printed continuations by eye "
              "(docs/R1B.md §4).")
    elif counts.get("skipped-api"):
        print("\nRESULT: PASS (apus-only) — apus side ran clean; the API "
              "side was skipped, so nothing was compared.")
    else:
        print("\nRESULT: PASS — all compared cases identical or "
              "normalized-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
