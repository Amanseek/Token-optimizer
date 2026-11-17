import json

import httpx

from tokenwise.pricing import Usage, cost

from .conftest import sse

HEADERS = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01", "x-tokenwise-user": "aman"}


def msg(text="look at main.py", stream=False, model="claude-opus-5-5"):
    return {"model": model, "max_tokens": 100, "stream": stream,
            "tools": [{"name": "Bash", "input_schema": {}}],
            "messages": [{"role": "user", "content": text}]}


def rows(store):
    return store.query("SELECT * FROM requests ORDER BY id")


def test_non_stream_forwards_and_logs(make_client, upstream):
    client, store = make_client()
    r = client.post("/v1/messages", json=msg(), headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["content"][0]["text"] == "hello"

    fwd = upstream.requests[0]
    assert fwd.headers["x-api-key"] == "sk-test"
    assert "x-tokenwise-user" not in fwd.headers  # our header is stripped

    [row] = rows(store)
    assert row["engineer"] == "aman"
    assert row["input_tokens"] == 1000 and row["cache_read_tokens"] == 5000
    expected = cost("claude-opus-5-5", Usage(1000, 200, 0, 5000))
    assert abs(row["cost_usd"] - expected) < 1e-9


def test_stream_bytes_pass_through_unchanged(make_client, upstream):
    client, store = make_client()
    r = client.post("/v1/messages", json=msg(stream=True), headers=HEADERS)
    assert r.status_code == 200
    usage = {"input_tokens": 1000, "output_tokens": 1, "cache_read_input_tokens": 5000, "cache_creation_input_tokens": 0}
    assert r.content == sse("claude-opus-5-5", usage, 200)
    [row] = rows(store)
    assert row["stream"] == 1
    assert row["output_tokens"] == 200 and row["input_tokens"] == 1000


def test_engineer_from_key_hash_when_no_header(make_client):
    client, store = make_client()
    client.post("/v1/messages", json=msg(), headers={"x-api-key": "sk-abc"})
    assert rows(store)[0]["engineer"].startswith("key-")


def test_routing_on_sends_cheap_model(make_client, upstream):
    client, store = make_client(routing_mode="on")
    r = client.post("/v1/messages", json=msg("explain what this does"), headers=HEADERS)
    assert r.status_code == 200
    assert upstream.bodies[0]["model"] == "claude-haiku-5-5"
    assert r.headers["x-tokenwise-model"] == "claude-haiku-5-5"
    row = rows(store)[0]
    assert row["route_decision"] == "routed"
    assert row["baseline_cost_usd"] > row["cost_usd"]


def test_shadow_mode_logs_but_keeps_model(make_client, upstream):
    client, store = make_client(routing_mode="shadow")
    client.post("/v1/messages", json=msg("explain what this does"), headers=HEADERS)
    assert upstream.bodies[0]["model"] == "claude-opus-5-5"
    row = rows(store)[0]
    assert row["route_decision"] == "shadow"
    assert row["shadow_model"] == "claude-haiku-5-5"
    assert row["shadow_cost_usd"] < row["cost_usd"]


def test_fallback_when_cheap_model_rejects(make_client, upstream):
    client, store = make_client(routing_mode="on")
    default = upstream.default

    def handler(request, body):
        if body["model"] == "claude-haiku-5-5":
            return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}})
        return default(request, body)

    upstream.handler = handler
    for stream in (False, True):
        r = client.post("/v1/messages", json=msg(f"explain this {stream}", stream=stream), headers=HEADERS)
        assert r.status_code == 200
    assert [b["model"] for b in upstream.bodies] == ["claude-haiku-5-5", "claude-opus-5-5"] * 2
    assert {r["route_decision"] for r in rows(store)} == {"fallback"}


def test_cascade_on_empty_answer(make_client, upstream):
    client, store = make_client(routing_mode="on")
    default = upstream.default

    def handler(request, body):
        if body["model"] == "claude-haiku-5-5":
            return httpx.Response(200, json={"content": [], "stop_reason": "end_turn",
                                             "usage": {"input_tokens": 10, "output_tokens": 0}})
        return default(request, body)

    upstream.handler = handler
    r = client.post("/v1/messages", json=msg("explain this"), headers=HEADERS)
    assert r.json()["content"][0]["text"] == "hello"
    assert [x["route_decision"] for x in rows(store)] == ["cascade-attempt", "cascaded"]
    # Conversation is now pinned to the original model.
    client.post("/v1/messages", json=msg("explain this"), headers=HEADERS)
    assert upstream.bodies[-1]["model"] == "claude-opus-5-5"


def test_upstream_error_passed_through(make_client, upstream):
    client, store = make_client()
    upstream.handler = lambda req, body: httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})
    r = client.post("/v1/messages", json=msg(stream=True), headers=HEADERS)
    assert r.status_code == 429
    assert rows(store)[0]["status"] == 429


def test_trimming_applied_on_the_wire(make_client, upstream):
    client, store = make_client()
    big = "\n".join(f"line {i}" for i in range(3000))
    body = msg()
    body["messages"] += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]},
    ]
    client.post("/v1/messages", json=body, headers=HEADERS)
    sent = upstream.bodies[0]["messages"][2]["content"][0]["content"]
    assert "lines trimmed" in sent
    assert rows(store)[0]["trimmed_chars"] > 10000


def test_passthrough_other_endpoints(make_client, upstream):
    client, _ = make_client()
    upstream.handler = lambda req, body: httpx.Response(200, json={"input_tokens": 42, "path": req.url.path})
    r = client.post("/v1/messages/count_tokens", json=msg(), headers=HEADERS)
    assert r.json() == {"input_tokens": 42, "path": "/v1/messages/count_tokens"}


def test_dashboard(make_client):
    client, _ = make_client(routing_mode="shadow")
    client.post("/v1/messages", json=msg("explain this"), headers=HEADERS)
    page = client.get("/dashboard")
    assert page.status_code == 200 and "aman" in page.text
    data = client.get("/api/summary").json()
    assert data["totals"]["requests"] == 1
    assert json.dumps(data)
