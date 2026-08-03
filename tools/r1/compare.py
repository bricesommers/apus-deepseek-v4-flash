#!/usr/bin/env python3
"""R1 comparator — runs ON THE MAC (stdlib only, no torch).

Compares apus's tokens against the reference goldens (r1_goldens.json,
produced by tools/r1/run_reference.py on the CUDA host).

What it does per case of tests/r1b/prompts.json:
  1. Drives `bin/apus serve` (NDJSON stdio protocol) with the GOLDEN PROMPT
     IDS ({"cmd":"generate","ids":[...]}) so prompt-encoding differences
     cannot mask model differences. Model comparison is token-for-token.
  2. Separately validates prompt encoding, cheaply (no extra forward):
     {"cmd":"encode"} for one chat and one thinking case must reproduce the
     golden prompt ids byte-exactly (this is the M2/M10 encoding gate
     re-checked against the real golden).
  3. One raw end-to-end case: {"cmd":"generate","text":...} (verbatim
     tokenization through apus's C tokenizer, no chat template) must produce
     the same tokens as the golden. Note: `apus run --prompt` always renders
     the thinking-mode chat template, so the serve "text" path is the raw
     end-to-end check.

Classification per case (see docs/R1.md):
  identical       — token-for-token equal (over the compared prefix).
  near_tie_flip   — first differing position has reference top1−top2 margin
                    < --near-tie (default 0.5 logit): acceptable accumulation-
                    order difference (GPU vs NEON), counted, not a failure.
  hard_mismatch   — first differing position has margin >= threshold: a real
                    bug. Exit code 1.
  rng_divergence  — sampled case (temp > 0): the reference samples via
                    Gumbel-max on torch's RNG, apus via its own splitmix64
                    sampler; the streams cannot match by construction, so the
                    first flip is expected and informational, NOT a failure.

Exit code: 0 iff no hard_mismatch, no failed apus run, and all encoding
checks pass.

Offline mode (--offline FILE) skips running apus and classifies from a
previously saved apus-output file — used for testing the classification
logic with fabricated goldens.
"""

import argparse
import json
import select
import subprocess
import sys
import time

NEAR_TIE_DEFAULT = 0.5


# --------------------------------------------------------------------------
# Classification (pure, importable for tests)
# --------------------------------------------------------------------------

def classify_case(case_id, golden_case, apus_ids, apus_finish, eos_id,
                  near_tie=NEAR_TIE_DEFAULT):
    """Classify one case. golden_case: a 'cases' entry from r1_goldens.json
    (status ok). apus_ids: token ids emitted by apus (WITHOUT eos — apus's
    serve stops at eos without emitting it). apus_finish: serve finish_reason.
    Returns a verdict dict."""
    steps = golden_case["steps"]
    ref_ids = [s["emitted"] for s in steps]
    sampled = float(golden_case.get("temp") or 0) > 0

    # Normalize eos: apus stops at eos silently; the reference records the eos
    # step. Append eos to apus's stream when it stopped there.
    ids = list(apus_ids)
    if apus_finish == "stop":
        ids.append(eos_id)

    n = min(len(ids), len(ref_ids))
    flip = next((i for i in range(n) if ids[i] != ref_ids[i]), None)

    v = {"case": case_id, "mode": golden_case.get("mode"),
         "sampled": sampled, "ref_tokens": len(ref_ids),
         "apus_tokens": len(ids),
         "flip_pos": None, "margin": None, "ref_top1": None,
         "apus_token": None, "ref_token": None}

    if flip is None:
        if len(ids) <= len(ref_ids):
            # All compared positions equal; apus shorter only when the run was
            # intentionally capped (--max-tokens). (A genuine early eos shows
            # up as an elementwise mismatch, not a length difference.)
            v["verdict"] = "identical"
            v["compared"] = len(ids)
            if len(ids) < len(ref_ids):
                v["capped"] = True
            return v
        # apus longer than the reference with an equal prefix — cannot happen
        # through the eos paths (ref's eos would force a mismatch or an apus
        # stop); treat defensively as a flip at the first uncompared position.
        flip = n
    v["compared"] = flip

    margin = steps[flip]["margin"] if flip < len(steps) else None
    v.update({
        "flip_pos": flip,
        "margin": margin,
        "ref_top1": steps[flip]["top1"] if flip < len(steps) else None,
        "ref_token": ref_ids[flip] if flip < len(ref_ids) else None,
        "apus_token": ids[flip] if flip < len(ids) else None,
        "prefix_match": flip,
    })
    if sampled:
        v["verdict"] = "rng_divergence"
    elif margin is not None and margin < near_tie:
        v["verdict"] = "near_tie_flip"
    else:
        v["verdict"] = "hard_mismatch"
    return v


def summarize(verdicts):
    counts = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    hard = [v for v in verdicts
            if v["verdict"] in ("hard_mismatch", "error")]
    return counts, hard


# --------------------------------------------------------------------------
# apus serve driver
# --------------------------------------------------------------------------

class ApusServe:
    """Drive `bin/apus serve` over its NDJSON stdio protocol."""

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

    def request(self, fields, on_token=None):
        """Send one request, collect messages until done/encoded/error."""
        self._req += 1
        rid = f"r1-{self._req}"
        fields = dict(fields, id=rid)
        self.proc.stdin.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        tokens, done = [], None
        n_tok = 0
        while True:
            msg = self._read_msg()
            if msg.get("id") != rid:
                continue
            t = msg.get("type")
            if t == "token":
                tokens.append(msg["token_id"])
                n_tok += 1
                if on_token and n_tok % 16 == 0:
                    on_token(n_tok)
            elif t == "done":
                done = msg
                break
            elif t == "encoded":
                return {"ids": msg.get("ids", []), "text": msg.get("text")}
            elif t == "error":
                raise RuntimeError(f"apus serve error: {msg.get('message')}")
        return {"ids": tokens, "finish_reason": done.get("finish_reason"),
                "text": done.get("text", "")}

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_verdict(v):
    tag = {"identical": "OK  ", "near_tie_flip": "NEAR",
           "rng_divergence": "RNG ", "hard_mismatch": "HARD",
           "error": "ERR "}.get(v["verdict"], "????")
    line = f"  [{tag}] {v['case']}: {v['verdict']}"
    if v["verdict"] == "identical":
        line += f" ({v.get('compared', 0)} tokens"
        if v.get("capped"):
            line += ", capped"
        line += ")"
    elif v["verdict"] == "error" or v.get("flip_pos") is None:
        line += f" — {v.get('error', '')[:200]}"
    else:
        line += (f" at pos {v['flip_pos']} (prefix {v['prefix_match']}): "
                 f"apus={v['apus_token']} ref={v['ref_token']} "
                 f"ref_top1={v['ref_top1']} margin={v['margin']}")
    print(line)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--goldens", default="r1_goldens.json")
    p.add_argument("--prompts", default="tests/r1b/prompts.json")
    p.add_argument("--model", default="weights/apus-0731")
    p.add_argument("--apus", default="bin/apus")
    p.add_argument("--cases", default="", help="Comma-separated case ids (default: all).")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="Cap apus generation per case (0 = golden length).")
    p.add_argument("--near-tie", type=float, default=NEAR_TIE_DEFAULT,
                   help="Margin threshold (logits) for near-tie classification "
                        f"(default: {NEAR_TIE_DEFAULT}).")
    p.add_argument("--out", default="r1_apus_out.json",
                   help="Where to save apus outputs (reuse with --offline).")
    p.add_argument("--offline", default="",
                   help="Skip running apus; classify from this apus-output file.")
    p.add_argument("--skip-e2e", action="store_true",
                   help="Skip the raw end-to-end (text) case.")
    p.add_argument("--skip-encode-checks", action="store_true")
    p.add_argument("--no-tiered", action="store_true")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-read timeout (s) waiting on apus serve.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    with open(args.goldens) as f:
        goldens = json.load(f)
    eos_id = goldens["meta"]["eos_token_id"]
    gcases = goldens["cases"]
    want = set(args.cases.split(",")) if args.cases else None
    case_ids = [c for c in gcases
                if gcases[c].get("status") == "ok" and (not want or c in want)]
    failed_golden = [c for c in gcases if gcases[c].get("status") != "ok"]
    if failed_golden:
        print(f"NOTE: golden cases failed on the host (skipped): {failed_golden}")

    if args.offline:
        with open(args.offline) as f:
            apus_out = json.load(f)
    else:
        with open(args.prompts) as f:
            battery = json.load(f)
        battery_by_id = {c["id"]: c for c in battery["cases"]}
        apus_out = {"cases": {}, "encode_checks": {}, "e2e": {}}
        print(f"Starting {args.apus} serve --model {args.model} "
              f"({'tiered' if not args.no_tiered else 'eager'}) ...", flush=True)
        t0 = time.time()
        srv = ApusServe(args.apus, args.model, tiered=not args.no_tiered,
                        verbose=args.verbose, timeout=args.timeout)
        try:
            # ---- cheap encoding checks (no forward pass) ----
            if not args.skip_encode_checks:
                for mode, thinking in (("chat", False), ("thinking", True)):
                    cid = next((c for c in case_ids
                                if gcases[c]["mode"] == mode), None)
                    if not cid:
                        continue
                    prompt = gcases[cid]["prompt"]
                    r = srv.request({"cmd": "encode",
                                     "messages": [{"role": "user",
                                                   "content": prompt}],
                                     "thinking": thinking})
                    ok = r["ids"] == gcases[cid]["prompt_ids"]
                    apus_out["encode_checks"][cid] = {
                        "mode": mode, "ok": ok, "apus_ids": r["ids"]}
                    print(f"  encode check {cid} ({mode}): "
                          f"{'byte-exact' if ok else 'MISMATCH'}", flush=True)

            # ---- per-case model comparison via golden prompt ids ----
            for i, cid in enumerate(case_ids):
                g = gcases[cid]
                max_new = len(g["steps"])
                if args.max_tokens:
                    max_new = min(max_new, args.max_tokens)
                bcase = battery_by_id.get(cid, {})
                print(f"  [{i + 1}/{len(case_ids)}] {cid}: {max_new} tokens ...",
                      flush=True)
                t = time.time()
                try:
                    r = srv.request(
                        {"cmd": "generate", "ids": g["prompt_ids"],
                         "max_tokens": max_new,
                         "temperature": float(bcase.get("temp", g.get("temp") or 0)),
                         "top_p": float(bcase.get("top_p") or 1.0),
                         "seed": int(bcase.get("seed") or 0)},
                        on_token=lambda n: print(f"    {n} tokens", flush=True))
                    apus_out["cases"][cid] = r
                    print(f"    done: {len(r['ids'])} tokens, "
                          f"{r['finish_reason']}, {time.time() - t:.0f}s", flush=True)
                except Exception as e:
                    apus_out["cases"][cid] = {"error": str(e)}
                    print(f"    FAILED: {e}", flush=True)

            # ---- raw end-to-end case (verbatim tokenizer path) ----
            if not args.skip_e2e:
                cid = next((c for c in case_ids if gcases[c]["mode"] == "raw"
                            and float(gcases[c].get("temp") or 0) == 0), None)
                if cid:
                    max_new = len(gcases[cid]["steps"])
                    if args.max_tokens:
                        max_new = min(max_new, args.max_tokens)
                    print(f"  [e2e] {cid} via raw text path: {max_new} tokens ...",
                          flush=True)
                    try:
                        r = srv.request(
                            {"cmd": "generate", "text": gcases[cid]["prompt"],
                             "max_tokens": max_new, "temperature": 0.0,
                             "top_p": 1.0, "seed": 0},
                            on_token=lambda n: print(f"    {n} tokens", flush=True))
                        apus_out["e2e"][cid] = r
                        print(f"    done: {len(r['ids'])} tokens, "
                              f"{r['finish_reason']}", flush=True)
                    except Exception as e:
                        apus_out["e2e"][cid] = {"error": str(e)}
                        print(f"    FAILED: {e}", flush=True)
        finally:
            srv.close()
        print(f"apus runs finished in {time.time() - t0:.0f}s")
        with open(args.out, "w") as f:
            json.dump(apus_out, f, ensure_ascii=False, indent=1)
        print(f"apus outputs saved to {args.out}")

    # ---------------- classification ----------------
    print("\n=== R1 comparison report ===")
    verdicts = []

    for cid, chk in (apus_out.get("encode_checks") or {}).items():
        v = {"case": f"{cid} [prompt-encoding]",
             "verdict": "identical" if chk["ok"] else "hard_mismatch",
             "compared": len(chk.get("apus_ids", []))}
        if not chk["ok"]:
            v["error"] = "apus prompt encoding != golden prompt ids"
            v["verdict"] = "hard_mismatch"
        verdicts.append(v)
        print_verdict(v)

    for cid in case_ids:
        r = (apus_out.get("cases") or {}).get(cid)
        if r is None:
            continue
        if "error" in r:
            v = {"case": cid, "verdict": "error", "error": r["error"]}
        else:
            v = classify_case(cid, gcases[cid], r["ids"],
                              r.get("finish_reason"), eos_id, args.near_tie)
        verdicts.append(v)
        print_verdict(v)

    for cid, r in (apus_out.get("e2e") or {}).items():
        if "error" in r:
            v = {"case": f"{cid} [e2e raw]", "verdict": "error",
                 "error": r["error"]}
        else:
            v = classify_case(f"{cid} [e2e raw]", gcases[cid], r["ids"],
                              r.get("finish_reason"), eos_id, args.near_tie)
        verdicts.append(v)
        print_verdict(v)

    counts, hard = summarize(verdicts)
    print("\n=== summary ===")
    for k in ("identical", "near_tie_flip", "rng_divergence",
              "hard_mismatch", "error"):
        if k in counts:
            print(f"  {k}: {counts[k]}")
    if hard:
        print(f"\nRESULT: FAIL — {len(hard)} hard mismatch(es)/error(s). "
              "This is a real bug; stop and report (see docs/R1.md).")
        return 1
    print("\nRESULT: PASS — no hard mismatches "
          f"({counts.get('near_tie_flip', 0)} near-tie flip(s), "
          f"{counts.get('rng_divergence', 0)} rng divergence(s) are acceptable; "
          "see docs/R1.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
