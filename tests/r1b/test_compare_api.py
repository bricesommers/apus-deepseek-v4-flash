#!/usr/bin/env python3
"""tests/r1b/test_compare_api.py — R1b comparator tests (stdlib only).

No real API key, no network beyond loopback:
  - OpenRouter is mocked by a local http.server stub in a thread.
  - The apus side runs against the scripted m7a parrot fixture
    (tests/m7a/fixtures/model_chat), whose replies are known exactly.

Covers: text classification (identical / normalized / diverges / raw
format-different / capital contradiction), retry behavior, error paths,
exit codes, --dry-run / --apus-only / budget guard / --list-models, and
that the API key appears in NO output or results file.

Run from the repo root: `.venv/bin/python tests/r1b/test_compare_api.py [-v]`
Env APUS_BIN selects the engine binary (like tests/m7a).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "r1b"))
import compare_api as ca  # noqa: E402

APUS = os.path.abspath(os.environ.get("APUS_BIN",
                                      str(ROOT / "bin" / "apus")))
FIXTURE = ROOT / "tests" / "m7a" / "fixtures" / "model_chat"
TOOL = ROOT / "tools" / "r1b" / "compare_api.py"
KEY = "test-key-r1b"

# The parrot model's scripted replies (tests/m7a/README.md).
PARROT_CHAT = "The answer is STOP right here."
PARROT_REASONING = "reasoning: thinking it over."

CHAT_Q = "chat question"
THINK_Q = "think question"
# The parrot's chain is only scripted from <think> (id 260), so the raw
# prompt must end with it for the apus side to be deterministic.
RAW_P = "raw fragment<think>"

BATTERY = {"cases": [
    {"id": "c_chat", "prompt": CHAT_Q, "max_tokens": 32, "temp": 0,
     "mode": "chat"},
    {"id": "c_think", "prompt": THINK_Q, "max_tokens": 64, "temp": 0,
     "mode": "thinking"},
    {"id": "c_raw", "prompt": RAW_P, "max_tokens": 16, "temp": 0,
     "mode": "raw"},
]}


def ensure_fixture():
    if (FIXTURE / "config.json").is_file():
        return
    subprocess.run([sys.executable,
                    str(ROOT / "tests" / "m7a" / "gen_fixtures.py")],
                   check=True, cwd=ROOT)


# ---------------------------------------------------------------------------
# OpenRouter stub
# ---------------------------------------------------------------------------

class StubState:
    def __init__(self):
        self.requests = []          # (path, body dict or None)
        self.fail_first_n = 0       # respond 500 this many times (POST)
        self.unauthorized = False   # respond 401 always
        self.echo_key_in_error = False  # malicious: echo the key in a 400


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        blob = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _auth_ok(self):
        return self.headers.get("Authorization") == f"Bearer {KEY}"

    def do_GET(self):
        st = self.server.state
        st.requests.append((self.path, None))
        if not self._auth_ok():
            self._send(401, {"error": {"message": "bad key"}})
            return
        if self.path == "/api/v1/models":
            self._send(200, {"data": [
                {"id": "deepseek/deepseek-v4-flash-0731"},
                {"id": "deepseek/deepseek-v4-flash"},
                {"id": "other/gpt-thing"}]})
        else:
            self._send(404, {"error": {"message": "nope"}})

    def do_POST(self):
        st = self.server.state
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        st.requests.append((self.path, body))
        if st.unauthorized:
            self._send(401, {"error": {"message": "invalid key"}})
            return
        if not self._auth_ok():
            self._send(401, {"error": {"message": "bad key"}})
            return
        if st.echo_key_in_error:
            self._send(400, {"error": {"message": f"bad request from {KEY}"}})
            return
        n_posts = sum(1 for p, _ in st.requests if "chat" in p)
        if n_posts <= st.fail_first_n:
            self._send(500, {"error": {"message": "provider exploded"}})
            return
        prompt = body["messages"][0]["content"]
        thinking = body.get("chat_template_kwargs", {}).get("thinking", True)
        msg = {"role": "assistant", "content": PARROT_CHAT}
        if THINK_Q in prompt:
            msg["reasoning_content"] = PARROT_REASONING
        elif RAW_P in prompt:
            msg["content"] = "The capital of France is Paris."
        self._send(200, {
            "id": "gen-stub", "model": body.get("model"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6}})


class Stub:
    def __init__(self):
        self.state = StubState()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        self.httpd.state = self.state
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 1. pure classification
# ---------------------------------------------------------------------------

class TestClassify(unittest.TestCase):
    def case(self, mode="chat", temp=0):
        return {"id": "x", "mode": mode, "temp": temp, "max_tokens": 8}

    def test_identical(self):
        v = ca.classify_case(self.case(), api_content="Paris.",
                             apus_content="Paris.")
        self.assertEqual(v["verdict"], "identical")

    def test_normalized_identical(self):
        v = ca.classify_case(self.case(),
                             api_content="The  answer\nis\u00a0here.",
                             apus_content="The answer is  here.")
        self.assertEqual(v["verdict"], "normalized-identical")

    def test_diverges(self):
        v = ca.classify_case(self.case(),
                             api_content="abcdeZZZ" + "a" * 200,
                             apus_content="abcdeQQQ" + "b" * 200)
        self.assertEqual(v["verdict"], "diverges-at-5")
        self.assertEqual(v["diverges_at"], 5)
        self.assertEqual(len(v["api_continuation"]), 120)
        self.assertFalse(ca.is_hard(v))

    def test_raw_format_different_content_match(self):
        v = ca.classify_case(self.case(mode="raw"),
                             api_content="The capital of France is Paris.",
                             apus_content="The capital of France is Paris, "
                                          "the City of Light")
        self.assertEqual(v["verdict"], "format-different")
        self.assertEqual(v["content_check"], "match")
        self.assertFalse(ca.is_hard(v))

    def test_raw_format_different_unknown_is_not_hard(self):
        v = ca.classify_case(self.case(mode="raw"),
                             api_content="It is the city of lights, Paris.",
                             apus_content=" Paris, obviously.")
        self.assertEqual(v["verdict"], "format-different")
        self.assertEqual(v["content_check"], "unknown")
        self.assertFalse(ca.is_hard(v))

    def test_capital_contradiction_is_hard(self):
        v = ca.classify_case(self.case(),
                             api_content="The capital of Spain is Madrid.",
                             apus_content="The capital of Spain is "
                                          "Barcelona.")
        self.assertEqual(v["verdict"], "contradiction")
        self.assertTrue(ca.is_hard(v))

    def test_capital_agreement(self):
        # Same fact, different surface form: diverges, but NOT hard.
        v = ca.classify_case(self.case(),
                             api_content="The capital of Spain is Madrid.",
                             apus_content="The capital of Spain is Madrid, "
                                          "of course.")
        self.assertTrue(v["verdict"].startswith("diverges-at-"))
        self.assertFalse(ca.is_hard(v))
        self.assertEqual(v["capital_api"], "Madrid")
        self.assertEqual(v["capital_apus"], "Madrid")

    def test_api_error_hard(self):
        v = ca.classify_case(self.case(), api_error="HTTP 401 ...")
        self.assertEqual(v["verdict"], "api-error")
        self.assertTrue(ca.is_hard(v))

    def test_apus_error_hard(self):
        v = ca.classify_case(self.case(), api_content="x",
                             apus_error="engine died")
        self.assertEqual(v["verdict"], "apus-error")
        self.assertTrue(ca.is_hard(v))

    def test_empty_content_hard(self):
        v = ca.classify_case(self.case(), api_content="ok", apus_content="  ")
        self.assertEqual(v["verdict"], "apus-error")
        self.assertTrue(ca.is_hard(v))
        v = ca.classify_case(self.case(mode="thinking"), api_content="",
                             api_reasoning="lots of thinking",
                             apus_content="ok")
        self.assertEqual(v["verdict"], "api-error")

    def test_reasoning_ratio_informational(self):
        v = ca.classify_case(self.case(mode="thinking"),
                             api_content="A", api_reasoning="same start div",
                             apus_content="A", apus_reasoning="same start x")
        self.assertEqual(v["verdict"], "identical")
        self.assertIn("reasoning_prefix_ratio", v)
        self.assertGreater(v["reasoning_prefix_ratio"], 0.5)


class TestSplitAndNormalize(unittest.TestCase):
    def test_split_thinking(self):
        c = {"id": "t", "mode": "thinking"}
        content, reasoning = ca.split_apus_text(
            c, "reasoning: thinking it over.</think>The answer.")
        self.assertEqual(content, "The answer.")
        self.assertEqual(reasoning, "reasoning: thinking it over.")

    def test_split_chat(self):
        c = {"id": "c", "mode": "chat"}
        self.assertEqual(ca.split_apus_text(c, "hello"), ("hello", ""))

    def test_unterminated_think_is_content(self):
        c = {"id": "t", "mode": "thinking"}
        self.assertEqual(ca.split_apus_text(c, "still thinking"),
                         ("still thinking", ""))

    def test_normalize_nfkc(self):
        self.assertEqual(ca.normalize_text("ｆｕｌｌ\u00a0width"),
                         "full width")

    def test_scrub(self):
        self.assertEqual(ca.scrub(f"key {KEY} leaked", KEY),
                         "key *** leaked")
        self.assertEqual(ca.scrub("nothing", KEY), "nothing")


# ---------------------------------------------------------------------------
# 2. API client against the stub
# ---------------------------------------------------------------------------

class TestClient(unittest.TestCase):
    def setUp(self):
        self.stub = Stub()

    def tearDown(self):
        self.stub.close()

    def client(self):
        return ca.OpenRouter(KEY, base=self.stub.base, timeout=10,
                             retries=2, backoff=0.01)

    def test_chat_ok(self):
        body = ca.build_api_body(BATTERY["cases"][0], "m")
        resp = self.client().chat(body)
        content, reasoning, meta = ca.extract_api_text(resp)
        self.assertEqual(content, PARROT_CHAT)
        self.assertFalse(meta["reasoning_present"])

    def test_reasoning_both_field_names(self):
        resp = {"choices": [{"message": {"content": "c", "reasoning": "r1"},
                             "finish_reason": "stop"}]}
        self.assertEqual(ca.extract_api_text(resp)[1], "r1")
        resp = {"choices": [{"message": {"content": "c",
                                         "reasoning_content": "r2"},
                             "finish_reason": "stop"}]}
        self.assertEqual(ca.extract_api_text(resp)[1], "r2")

    def test_retry_on_500_then_success(self):
        self.stub.state.fail_first_n = 1
        body = ca.build_api_body(BATTERY["cases"][0], "m")
        resp = self.client().chat(body)
        self.assertEqual(resp["id"], "gen-stub")
        posts = [r for r in self.stub.state.requests if "chat" in r[0]]
        self.assertEqual(len(posts), 2)

    def test_retry_exhausted_raises(self):
        self.stub.state.fail_first_n = 99
        with self.assertRaises(ca.ApiError) as cm:
            self.client().chat(ca.build_api_body(BATTERY["cases"][0], "m"))
        self.assertEqual(cm.exception.status, 500)
        posts = [r for r in self.stub.state.requests if "chat" in r[0]]
        self.assertEqual(len(posts), 3)  # 1 + 2 retries

    def test_401_not_retried(self):
        self.stub.state.unauthorized = True
        with self.assertRaises(ca.ApiError) as cm:
            self.client().chat(ca.build_api_body(BATTERY["cases"][0], "m"))
        self.assertEqual(cm.exception.status, 401)
        posts = [r for r in self.stub.state.requests if "chat" in r[0]]
        self.assertEqual(len(posts), 1)

    def test_key_scrubbed_from_error(self):
        self.stub.state.echo_key_in_error = True
        with self.assertRaises(ca.ApiError) as cm:
            self.client().chat(ca.build_api_body(BATTERY["cases"][0], "m"))
        self.assertNotIn(KEY, str(cm.exception))
        self.assertIn("***", str(cm.exception))

    def test_build_body_thinking_vs_chat(self):
        chat = ca.build_api_body({"id": "c", "prompt": "p", "max_tokens": 8,
                                  "temp": 0, "mode": "chat"}, "m")
        self.assertEqual(chat["chat_template_kwargs"], {"thinking": False})
        self.assertEqual(chat["reasoning"], {"enabled": False})
        think = ca.build_api_body({"id": "t", "prompt": "p", "max_tokens": 8,
                                   "temp": 0, "mode": "thinking"}, "m")
        self.assertNotIn("chat_template_kwargs", think)
        self.assertNotIn("reasoning", think)
        sampled = ca.build_api_body(
            {"id": "s", "prompt": "p", "max_tokens": 8, "temp": 1.0,
             "seed": 42, "mode": "raw"}, "m")
        self.assertEqual(sampled["seed"], 42)
        self.assertEqual(sampled["temperature"], 1.0)


# ---------------------------------------------------------------------------
# 3. CLI end-to-end against the stub + parrot model
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fixture()
        cls.stub = Stub()
        cls.td = tempfile.TemporaryDirectory()
        bp = Path(cls.td.name) / "battery.json"
        bp.write_text(json.dumps(BATTERY))
        cls.battery = str(bp)
        cls.out_dir = str(Path(cls.td.name) / "results")

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()
        cls.td.cleanup()

    def run_cli(self, extra, key=KEY):
        env = dict(os.environ)
        if key is None:
            env.pop("OPENROUTER_API_KEY", None)
        else:
            env["OPENROUTER_API_KEY"] = key
        return subprocess.run(
            [sys.executable, str(TOOL), "--prompts", self.battery,
             "--api-base", self.stub.base, "--apus", APUS,
             "--apus-model", str(FIXTURE), "--no-tiered",
             "--out-dir", self.out_dir, "--retries", "1"] + extra,
            capture_output=True, text=True, env=env, timeout=300)

    def test_full_run_pass_and_key_never_printed(self):
        r = self.run_cli([])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("identical", out)          # chat + thinking match
        self.assertIn("format-different", out)   # raw case
        self.assertIn("PASS with WARN", out)
        self.assertNotIn(KEY, out)
        # results JSON exists, well-formed, key-free — take the exact path
        # this run printed (globbing r1b_*.json races with other tests'
        # files when several runs land in the same second).
        m = re.search(r"results saved to (\S+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        blob = Path(m.group(1)).read_text()
        self.assertNotIn(KEY, blob)
        data = json.loads(blob)
        v = data["cases"]["c_think"]
        self.assertEqual(v["verdict"], "identical")
        # the chat request asked for no thinking (both dialects)
        reqs = [b for p, b in self.stub.state.requests if "chat" in p]
        chat_req = next(b for b in reqs
                        if b["messages"][0]["content"] == CHAT_Q)
        self.assertEqual(chat_req["chat_template_kwargs"],
                         {"thinking": False})
        self.assertEqual(chat_req["reasoning"], {"enabled": False})

    def test_cases_filter(self):
        r = self.run_cli(["--cases", "c_chat"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("c_chat", r.stdout)
        self.assertNotIn("c_think (thinking)", r.stdout)

    def test_dry_run_no_http_no_key_needed(self):
        before = len(self.stub.state.requests)
        r = self.run_cli(["--dry-run"], key=None)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DRY RUN", r.stdout)
        self.assertEqual(len(self.stub.state.requests), before)

    def test_apus_only_no_http(self):
        before = len(self.stub.state.requests)
        r = self.run_cli(["--apus-only"], key=None)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("skipped-api", r.stdout)
        self.assertEqual(len(self.stub.state.requests), before)

    def test_missing_key_exit2(self):
        r = self.run_cli([], key=None)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("OPENROUTER_API_KEY", r.stdout + r.stderr)
        self.assertNotIn(KEY, r.stdout + r.stderr)

    def test_api_failure_is_hard_exit1(self):
        self.stub.state.unauthorized = True
        try:
            r = self.run_cli(["--cases", "c_chat"])
        finally:
            self.stub.state.unauthorized = False
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("api-error", out)
        self.assertNotIn(KEY, out)

    def test_budget_guard(self):
        big = {"cases": [dict(c, id=f"c{i}") for i, c in
                         enumerate(BATTERY["cases"] * 3)]}
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as f:
            json.dump(big, f)
            path = f.name
        try:
            r = self.run_cli(["--prompts", path, "--dry-run"])
        finally:
            os.unlink(path)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--yes-i-know", r.stdout + r.stderr)

    def test_list_models(self):
        r = self.run_cli(["--list-models"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("deepseek/deepseek-v4-flash-0731", r.stdout)
        self.assertNotIn("other/gpt-thing", r.stdout)
        self.assertNotIn(KEY, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
