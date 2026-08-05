#!/usr/bin/env python3
"""tests/m7a/test_server.py — M7a server test suite (stdlib only).

Layers:
  1. Pipe protocol tests against `bin/apus serve` directly (NDJSON).
  2. DSML parser unit tests, differential vs reference/encoding/
     encoding_dsv4.py.
  3. HTTP gateway tests (tools/server.py) on the scripted m7a fixtures:
     chat (thinking/chat modes, stop strings, max_tokens, seed
     determinism, usage), SSE streaming, tool-call round trip, the 4
     reference encoding conformance pairs through /debug/encode, error
     paths, API-key auth, concurrent-request serialization.

Run from the repo root: `.venv/bin/python tests/m7a/test_server.py [-v]`.
Env APUS_BIN selects the engine binary (ubsan-m7a uses bin/apus_ubsan).
Fixtures are (re)generated with `make golden-m7a` if missing.
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APUS = os.path.abspath(os.environ.get("APUS_BIN", os.path.join(ROOT, "bin", "apus")))
SERVER = os.path.join(ROOT, "tools", "server.py")
FIX = os.path.join(ROOT, "tests", "m7a", "fixtures")
M5_FIX = os.path.join(ROOT, "tests", "m5", "fixtures")
ENC_TESTS = os.path.join(ROOT, "reference", "encoding", "tests")

sys.path.insert(0, os.path.join(ROOT, "reference", "encoding"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import encoding_dsv4 as ref_enc  # noqa: E402
import server as gw  # noqa: E402

THINKING_REPLY = "reasoning: thinking it over."
CHAT_REPLY = "The answer is STOP right here."
TOOL_DSML = ("\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
             "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing"
             "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>")


def ensure_fixtures():
    if os.path.isfile(os.path.join(FIX, "model_chat", "config.json")):
        return
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "tests", "m7a", "gen_fixtures.py")],
                   check=True, cwd=ROOT)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# 1. engine pipe protocol
# ---------------------------------------------------------------------------


class Pipe:
    def __init__(self, model_dir):
        self.proc = subprocess.Popen(
            [APUS, "serve", "--model", model_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=ROOT)

    def rpc(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        events = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("engine died")
            ev = json.loads(line)
            events.append(ev)
            if ev["type"] in ("done", "error", "encoded"):
                return events

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        self.proc.stdout.close()


class PipeProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.pipe = Pipe(os.path.join(FIX, "model_chat"))

    @classmethod
    def tearDownClass(cls):
        cls.pipe.close()

    def req(self, payload, i=1):
        return self.pipe.rpc(dict(payload, id=i))

    def test_encode_basic(self):
        ev = self.req({"cmd": "encode",
                       "messages": [{"role": "user", "content": "hi"}]})[0]
        self.assertEqual(ev["type"], "encoded")
        self.assertEqual(
            ev["text"],
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜><think>")
        self.assertEqual(ev["ids"], [256, 258, 104, 105, 259, 260])

    def test_error_recovery(self):
        # malformed line, then unknown cmd, then a good request — the loop
        # must survive all of it
        self.pipe.proc.stdin.write("this is not json\n")
        self.pipe.proc.stdin.flush()
        ev = json.loads(self.pipe.proc.stdout.readline())
        self.assertEqual(ev["type"], "error")
        self.assertIsNone(ev["id"])
        ev = self.req({"cmd": "bogus"})[0]
        self.assertEqual(ev["type"], "error")
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0})
        self.assertEqual(ev[-1]["type"], "done")

    def test_generate_events_and_eos(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 64, "temperature": 0})
        self.assertEqual(ev[0]["type"], "prompt")
        toks = [e for e in ev if e["type"] == "token"]
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "stop")   # EOS
        self.assertNotIn(257, [t["token_id"] for t in toks])  # EOS not emitted
        self.assertEqual(done["completion_tokens"], len(toks))
        self.assertEqual("".join(t["text"] for t in toks), done["text"])
        self.assertEqual(done["text"],
                         THINKING_REPLY + "</think>" + CHAT_REPLY)

    def test_usage_matches_encode_ids(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "count my tokens"}]
        enc = self.req({"cmd": "encode", "messages": msgs}, i=2)[0]
        gen = self.req({"cmd": "generate", "messages": msgs,
                        "max_tokens": 1, "temperature": 0}, i=3)
        self.assertEqual(gen[0]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(gen[-1]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(gen[-1]["completion_tokens"], 1)
        self.assertEqual(gen[-1]["finish_reason"], "length")

    def test_stop_string(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 64, "temperature": 0, "stop": ["STOP"]})
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "stop_string")
        self.assertEqual(done["text"],
                         THINKING_REPLY + "</think>" + "The answer is ")
        # the streamed pieces never contain the stop string
        toks = [e for e in ev if e["type"] == "token"]
        self.assertNotIn("STOP", "".join(t["text"] for t in toks))

    def test_max_tokens_clamp(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0})
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "length")
        self.assertEqual(done["completion_tokens"], 2)

    def test_ids_path_synthetic_m5(self):
        # the m5 fixture has no tokenizer: ids in, ids out, no text fields
        p = Pipe(M5_FIX)
        try:
            ev = p.rpc({"id": 9, "cmd": "generate", "ids": [3, 1, 4, 1, 5],
                        "max_tokens": 4, "temperature": 0})
            self.assertEqual(ev[0]["prompt_tokens"], 5)
            toks = [e for e in ev if e["type"] == "token"]
            self.assertEqual(len(toks), 4)
            self.assertNotIn("text", toks[0])
            # messages need a tokenizer -> clean protocol error, loop alive
            err = p.rpc({"id": 10, "cmd": "generate",
                         "messages": [{"role": "user", "content": "x"}]})[0]
            self.assertEqual(err["type"], "error")
            ev = p.rpc({"id": 11, "cmd": "generate", "ids": [7],
                        "max_tokens": 1, "temperature": 0})
            self.assertEqual(ev[-1]["type"], "done")
        finally:
            p.close()


# ---------------------------------------------------------------------------
# 2. DSML parser unit tests (differential vs the reference)
# ---------------------------------------------------------------------------


def ref_parse(text, thinking_mode):
    m = ref_enc.parse_message_from_completion_text(text, thinking_mode)
    return (m["reasoning_content"], m["content"],
            [(tc["function"]["name"], tc["function"]["arguments"])
             for tc in m["tool_calls"]])


def gw_parse(text, thinking):
    r, c, tcs = gw.parse_completion(text, thinking)
    return r, c, [(tc["function"]["name"], tc["function"]["arguments"])
                  for tc in tcs]


class ParserUnit(unittest.TestCase):
    EOS = "<｜end▁of▁sentence｜>"

    def _both(self, body, thinking=True):
        """Compare gateway port vs reference on a well-formed completion."""
        mode = "thinking" if thinking else "chat"
        text = body + self.EOS
        self.assertEqual(gw_parse(text, thinking), ref_parse(text, mode))

    def test_plain_thinking(self):
        self._both("some reasoning</think>the answer", True)

    def test_plain_chat(self):
        self._both("just the answer", False)

    def test_empty_reasoning(self):
        self._both("</think>content only", True)

    def test_single_tool_call(self):
        self._both("let me check</think>" + TOOL_DSML, True)

    def test_tool_call_chat_mode(self):
        self._both(TOOL_DSML, False)

    def test_tool_call_with_content(self):
        self._both("r</think>checking the weather first" + TOOL_DSML, True)

    def test_multiple_invokes_and_typed_params(self):
        dsml = ("\n\n<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"search\">\n"
                "<｜DSML｜parameter name=\"query\" string=\"true\">apus m7a"
                "</｜DSML｜parameter>\n"
                "<｜DSML｜parameter name=\"num_results\" string=\"false\">5"
                "</｜DSML｜parameter>\n"
                "<｜DSML｜parameter name=\"verbose\" string=\"false\">false"
                "</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "<｜DSML｜invoke name=\"get_weather\">\n"
                "<｜DSML｜parameter name=\"location\" string=\"true\">上海"
                "</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>")
        self._both("two calls</think>" + dsml, True)
        # and check the typed values survive the JSON round trip
        _, _, tcs = gw_parse("two calls</think>" + dsml + self.EOS, True)
        args = json.loads(tcs[0][1])
        self.assertEqual(args, {"query": "apus m7a", "num_results": 5,
                                "verbose": False})
        self.assertEqual(json.loads(tcs[1][1]), {"location": "上海"})

    def test_tool_calls_openai_shape(self):
        _, content, tcs = gw.parse_completion(TOOL_DSML, False)
        self.assertEqual(content, "")
        self.assertEqual(len(tcs), 1)
        tc = tcs[0]
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]),
                         {"location": "Beijing"})

    def test_tolerant_tails(self):
        # no EOS (finish by length): reference would assert; the port copes
        r, c, tcs = gw.parse_completion("reason</think>partial content", True)
        self.assertEqual((r, c, tcs), ("reason", "partial content", []))
        # unterminated thinking: everything is reasoning
        r, c, tcs = gw.parse_completion("never closed", True)
        self.assertEqual((r, c, tcs), ("never closed", "", []))
        # malformed DSML falls back to plain content (no exception)
        bad = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke oops"
        r, c, tcs = gw.parse_completion("x</think>" + bad, True)
        self.assertEqual(c, bad)
        self.assertEqual(tcs, [])

    def test_think_splitter(self):
        text = THINKING_REPLY + "</think>" + CHAT_REPLY
        # feed character by character: the split must be exact
        sp = gw.ThinkSplitter(thinking=True)
        parts = []
        for ch in text:
            parts.extend(sp.feed(ch))
        parts.extend(sp.flush())
        reasoning = "".join(t for k, t in parts if k == "reasoning_content")
        content = "".join(t for k, t in parts if k == "content")
        self.assertEqual(reasoning, THINKING_REPLY)
        self.assertEqual(content, CHAT_REPLY)
        # chat mode: all content
        sp = gw.ThinkSplitter(thinking=False)
        parts = sp.feed(text) + sp.flush()
        self.assertEqual("".join(t for k, t in parts if k == "content"), text)


# ---------------------------------------------------------------------------
# 3. HTTP gateway tests
# ---------------------------------------------------------------------------


class GatewayCase(unittest.TestCase):
    """Base: spawns tools/server.py on a fixture model."""
    MODEL_DIR = None
    MODEL_ID = "test-model"
    EXTRA_ARGS = ()
    EXTRA_ENV = ()

    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.port = free_port()
        env = dict(os.environ)
        env.pop("APUS_API_KEY", None)
        env.update(dict(cls.EXTRA_ENV))
        # Capture gateway stderr so a CI startup failure is diagnosable
        # (was DEVNULL: the failure mode is invisible otherwise).
        import tempfile
        cls._errlog = tempfile.NamedTemporaryFile(
            mode="rb", prefix="apus-gw-", suffix=".log", delete=False)
        cls.proc = subprocess.Popen(
            [sys.executable, SERVER, "--model", cls.MODEL_DIR,
             "--apus", APUS, "--port", str(cls.port),
             "--model-id", cls.MODEL_ID, *cls.EXTRA_ARGS],
            cwd=ROOT, env=env, stderr=cls._errlog)
        deadline = 120  # shared CI runners can be very slow to boot Metal
        import time
        t0 = time.time()
        while time.time() - t0 < deadline:
            try:
                st, body = cls.get("/health", auth=False)
                if st == 200 and body.get("engine") == "up":
                    return
            except OSError:
                pass
            time.sleep(0.2)
        cls.proc.kill()
        cls._errlog.close()
        with open(cls._errlog.name, "rb") as f:
            tail = f.read()[-2000:].decode("utf-8", "replace")
        raise AssertionError(
            f"gateway did not come up within {deadline}s\n"
            f"--- gateway stderr tail ---\n{tail}")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=30)
        cls.proc.stderr.close() if cls.proc.stderr else None

    @classmethod
    def _conn(cls):
        return http.client.HTTPConnection("127.0.0.1", cls.port, timeout=120)

    @classmethod
    def get(cls, path, auth=True):
        headers = {}
        if auth and os.environ.get("APUS_API_KEY_TEST"):
            headers["Authorization"] = ("Bearer "
                                        + os.environ["APUS_API_KEY_TEST"])
        c = cls._conn()
        c.request("GET", path, headers=headers)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body)

    @classmethod
    def post(cls, path, payload=None, raw=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth and os.environ.get("APUS_API_KEY_TEST"):
            headers["Authorization"] = ("Bearer "
                                        + os.environ["APUS_API_KEY_TEST"])
        c = cls._conn()
        c.request("POST", path,
                  body=raw if raw is not None else json.dumps(payload),
                  headers=headers)
        r = c.getresponse()
        body = r.read()
        status = r.status
        c.close()
        try:
            return status, json.loads(body)
        except ValueError:
            return status, body

    @classmethod
    def post_sse(cls, path, payload):
        """Returns (status, [parsed data: payloads]); "[DONE]" kept as str."""
        c = cls._conn()
        c.request("POST", path, body=json.dumps(payload),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read().decode("utf-8")
        status = r.status
        c.close()
        events = []
        for block in raw.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    data = line[6:]
                    events.append(data if data == "[DONE]"
                                  else json.loads(data))
        return status, events

    def chat(self, messages, **kw):
        payload = {"model": self.MODEL_ID, "messages": messages,
                   "temperature": 0}
        payload.update(kw)
        return self.post("/v1/chat/completions", payload)


class GatewayChat(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_chat")
    MODEL_ID = "test-chat"

    def test_health_and_models(self):
        st, body = self.get("/health")
        self.assertEqual(st, 200)
        self.assertEqual(body["status"], "ok")
        st, body = self.get("/v1/models")
        self.assertEqual(st, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], self.MODEL_ID)

    def test_chat_thinking(self):
        st, r = self.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(st, 200)
        self.assertEqual(r["object"], "chat.completion")
        self.assertTrue(r["id"].startswith("chatcmpl-"))
        self.assertEqual(r["model"], self.MODEL_ID)
        ch = r["choices"][0]
        self.assertEqual(ch["index"], 0)
        self.assertEqual(ch["finish_reason"], "stop")
        msg = ch["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["reasoning_content"], THINKING_REPLY)
        self.assertEqual(msg["content"], CHAT_REPLY)
        self.assertNotIn("tool_calls", msg)
        u = r["usage"]
        self.assertEqual(u["completion_tokens"], 5)
        self.assertEqual(u["total_tokens"],
                         u["prompt_tokens"] + u["completion_tokens"])

    def test_chat_mode(self):
        st, r = self.chat([{"role": "user", "content": "hi"}],
                          chat_template_kwargs={"thinking": False})
        self.assertEqual(st, 200)
        msg = r["choices"][0]["message"]
        self.assertNotIn("reasoning_content", msg)
        self.assertEqual(msg["content"], CHAT_REPLY)
        self.assertEqual(r["usage"]["completion_tokens"], 3)

    def test_usage_prompt_tokens_match_encoding(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "count my tokens"}]
        st, enc = self.post("/debug/encode", {"messages": msgs})
        self.assertEqual(st, 200)
        st, r = self.chat(msgs, max_tokens=1)
        self.assertEqual(r["usage"]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(r["usage"]["completion_tokens"], 1)
        self.assertEqual(r["choices"][0]["finish_reason"], "length")

    def test_stop_strings(self):
        st, r = self.chat([{"role": "user", "content": "hi"}],
                          stop=["STOP"])
        self.assertEqual(st, 200)
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "stop")
        self.assertEqual(ch["message"]["content"], "The answer is ")
        # stop as a bare string is accepted too
        st, r = self.chat([{"role": "user", "content": "hi"}], stop="STOP")
        self.assertEqual(ch["message"]["content"], "The answer is ")
        self.assertEqual(st, 200)

    def test_seed_determinism(self):
        msgs = [{"role": "user", "content": "hi"}]
        st, a = self.chat(msgs, temperature=0.7, top_p=0.9, seed=1234)
        st, b = self.chat(msgs, temperature=0.7, top_p=0.9, seed=1234)
        self.assertEqual(a["choices"][0]["message"],
                         b["choices"][0]["message"])

    def test_stream_sse(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        for c in chunks:
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertTrue(c["id"].startswith("chatcmpl-"))
            self.assertEqual(c["model"], self.MODEL_ID)
        # first chunk: role delta; then deltas; final: empty delta + reason
        self.assertEqual(chunks[0]["choices"][0]["delta"],
                         {"role": "assistant"})
        self.assertEqual(chunks[0]["choices"][0]["finish_reason"], None)
        final = chunks[-2] if chunks[-1].get("usage") else chunks[-1]
        # locate the finish chunk: empty delta with finish_reason
        finish_chunks = [c for c in chunks
                         if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish_chunks), 1)
        self.assertEqual(finish_chunks[0]["choices"][0]["finish_reason"],
                         "stop")
        self.assertEqual(finish_chunks[0]["choices"][0]["delta"], {})
        # reassembled stream == non-stream message
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "")
                            for c in chunks if c["choices"])
        content = "".join(c["choices"][0]["delta"].get("content", "")
                          for c in chunks if c["choices"])
        self.assertEqual(reasoning, THINKING_REPLY)
        self.assertEqual(content, CHAT_REPLY)
        # usage chunk present (include_usage) with empty choices
        usage_chunks = [c for c in chunks if c.get("usage")]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["choices"], [])
        self.assertEqual(usage_chunks[0]["usage"]["completion_tokens"], 5)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"],
                         usage_chunks[0]["usage"]["prompt_tokens"] + 5)

    def test_stream_order_before_done(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "stream": True})
        kinds = []
        for e in events:
            if e == "[DONE]":
                kinds.append("done")
            elif e["choices"] and e["choices"][0]["finish_reason"]:
                kinds.append("finish")
            else:
                kinds.append("delta")
        self.assertEqual(kinds[0], "delta")
        self.assertEqual(kinds[-2:], ["finish", "done"])
        self.assertNotIn("finish", kinds[:-2])

    def test_concurrent_requests_serialize(self):
        # 6 concurrent requests: single engine -> serialized, all correct
        results = [None] * 6

        def work(i):
            results[i] = self.chat([{"role": "user",
                                     "content": f"request {i}"}])
        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for st, r in results:
            self.assertEqual(st, 200)
            self.assertEqual(r["choices"][0]["message"]["content"], CHAT_REPLY)

    def test_errors(self):
        # malformed JSON -> 400 with the OpenAI error shape
        st, r = self.post("/v1/chat/completions", raw="{not json")
        self.assertEqual(st, 400)
        self.assertIn("error", r)
        self.assertEqual(r["error"]["type"], "invalid_request_error")
        self.assertIn("message", r["error"])
        # unknown model -> 404
        st, r = self.chat([{"role": "user", "content": "hi"}],
                          model="no-such-model")
        self.assertEqual(st, 404)
        self.assertEqual(r["error"]["type"], "not_found_error")
        # unknown path -> 404
        st, r = self.post("/v1/engines", {})
        self.assertEqual(st, 404)
        st, r = self.get("/v1/unknown")
        self.assertEqual(st, 404)
        # bad messages -> 400
        st, r = self.post("/v1/chat/completions",
                          {"model": self.MODEL_ID, "messages": "hi"})
        self.assertEqual(st, 400)

    def test_completions(self):
        st, r = self.post("/v1/completions", {
            "model": self.MODEL_ID, "prompt": "once upon a ",
            "max_tokens": 8, "temperature": 0})
        self.assertEqual(st, 200)
        self.assertEqual(r["object"], "text_completion")
        ch = r["choices"][0]
        self.assertEqual(ch["index"], 0)
        self.assertIsInstance(ch["text"], str)
        self.assertIn(ch["finish_reason"], ("stop", "length"))
        u = r["usage"]
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertLessEqual(u["completion_tokens"], 8)
        # streaming variant
        st, events = self.post_sse("/v1/completions", {
            "model": self.MODEL_ID, "prompt": "once upon a ",
            "max_tokens": 4, "temperature": 0, "stream": True})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        streamed = "".join(c["choices"][0]["text"] for c in chunks)
        finish = [c for c in chunks if c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish), 1)
        self.assertGreater(len(streamed), 0)

    def test_gateway_encode_matches_engine_pipe(self):
        # the gateway must not alter the messages: text+ids through
        # /debug/encode == direct pipe encode (same encoding.h path)
        msgs = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer",
                 "reasoning_content": "secret thoughts"},
                {"role": "user", "content": "second question"}]
        st, via_http = self.post("/debug/encode", {"messages": msgs})
        self.assertEqual(st, 200)
        pipe = Pipe(self.MODEL_DIR)
        try:
            direct = pipe.rpc({"id": 1, "cmd": "encode", "messages": msgs})[0]
        finally:
            pipe.close()
        self.assertEqual(via_http["text"], direct["text"])
        self.assertEqual(via_http["ids"], direct["ids"])
        # and against the Python reference (encoding.h is verified
        # byte-for-byte against it in tests/m2): multi-turn rendering,
        # including drop_thinking of the earlier reasoning_content
        ref = ref_enc.encode_messages(json.loads(json.dumps(msgs)),
                                      thinking_mode="thinking")
        self.assertEqual(via_http["text"], ref)
        self.assertNotIn("secret thoughts", via_http["text"])

    def test_conformance_pairs_through_server(self):
        # the 4 reference encoding conformance pairs exercised through the
        # full server path (HTTP -> gateway -> engine -> encoding.h)
        modes = {1: True, 2: True, 3: True, 4: False}
        for i in (1, 2, 3, 4):
            with open(os.path.join(ENC_TESTS, f"test_input_{i}.json"),
                      encoding="utf-8") as f:
                spec = json.load(f)
            payload = {"messages": (spec["messages"] if isinstance(spec, dict)
                                    else spec),
                       "chat_template_kwargs": {"thinking": modes[i]}}
            if isinstance(spec, dict) and "tools" in spec:
                payload["tools"] = spec["tools"]
            st, r = self.post("/debug/encode", payload)
            self.assertEqual(st, 200, f"pair {i}: {r}")
            with open(os.path.join(ENC_TESTS, f"test_output_{i}.txt"),
                      encoding="utf-8") as f:
                expected = f.read()
            self.assertEqual(r["text"], expected, f"conformance pair {i}")


class GatewayTools(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_tools")
    MODEL_ID = "test-tools"

    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]

    def test_tool_call_roundtrip(self):
        st, r = self.chat(
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "Weather in Beijing?"}],
            tools=self.TOOLS)
        self.assertEqual(st, 200)
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        msg = ch["message"]
        self.assertEqual(msg["reasoning_content"],
                         "I should check the weather.")
        self.assertIsNone(msg["content"])
        tcs = msg["tool_calls"]
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["type"], "function")
        self.assertTrue(tcs[0]["id"].startswith("call_"))
        self.assertEqual(tcs[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]),
                         {"location": "Beijing"})
        self.assertEqual(r["usage"]["completion_tokens"], 15)

    def test_tool_call_streaming_finish_reason(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": self.TOOLS, "temperature": 0, "stream": True})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        finish = [e for e in events if isinstance(e, dict) and e["choices"]
                  and e["choices"][0]["finish_reason"]]
        self.assertEqual(finish[0]["choices"][0]["finish_reason"],
                         "tool_calls")

    def test_tool_result_followup_rendering(self):
        # follow-up request with the assistant's tool_calls and a role=tool
        # message must render the DSML tool-result form, byte-for-byte vs
        # the reference encoder
        tool_call_id = "call_abc123"
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Beijing?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": tool_call_id, "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"location": "Beijing"}'}}]},
            {"role": "tool", "tool_call_id": tool_call_id,
             "content": "18°C and sunny"},
            {"role": "user", "content": "thanks, and tomorrow?"},
        ]
        st, r = self.post("/debug/encode",
                          {"messages": msgs, "tools": self.TOOLS})
        self.assertEqual(st, 200)
        text = r["text"]
        self.assertIn("<｜DSML｜invoke name=\"get_weather\">", text)
        self.assertIn("<tool_result>18°C and sunny</tool_result>", text)
        self.assertTrue(text.endswith("<｜Assistant｜><think>"))
        # byte-for-byte vs the reference encoder (same attach-tools rule as
        # tests/m2: tools ride on the first system message)
        ref_msgs = json.loads(json.dumps(msgs))
        ref_msgs[0]["tools"] = self.TOOLS
        ref = ref_enc.encode_messages(ref_msgs, thinking_mode="thinking")
        self.assertEqual(text, ref)


class GatewayAuth(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_chat")
    MODEL_ID = "test-auth"
    EXTRA_ENV = (("APUS_API_KEY", "m7a-secret"),)

    @classmethod
    def setUpClass(cls):
        os.environ["APUS_API_KEY_TEST"] = "m7a-secret"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("APUS_API_KEY_TEST", None)

    def test_api_key_required(self):
        st, r = self.get("/v1/models", auth=False)
        self.assertEqual(st, 401)
        self.assertEqual(r["error"]["type"], "authentication_error")
        st, r = self.chat([{"role": "user", "content": "hi"}], )
        self.assertEqual(st, 200)
        # health stays open
        st, r = self.get("/health", auth=False)
        self.assertEqual(st, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
