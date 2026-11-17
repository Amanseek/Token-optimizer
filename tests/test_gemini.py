import asyncio
import json
import socket
import threading

import httpx
import pytest

from tokenwise.gemini_pricing import gemini_cost

GH = {"x-goog-api-key": "AIza-test", "content-type": "application/json",
      "x-tokenwise-user": "prankundli", "x-tokenwise-feature": "tarot"}
GEN = "/v1beta/models/gemini-3.1-flash-lite:generateContent"
BODY = {"contents": [{"role": "user", "parts": [{"text": "Read my cards"}]}]}


def rows(store):
    return store.query("SELECT * FROM requests ORDER BY id")


# ------------------------------------------------------------------ pricing

def test_text_cost():
    c, t = gemini_cost("gemini-3.1-flash-lite", {"promptTokenCount": 1_000_000, "candidatesTokenCount": 1_000_000})
    assert c == pytest.approx(0.25 + 1.50)
    assert t.input_tokens == 1_000_000


def test_thinking_tokens_billed_as_output():
    c, t = gemini_cost("gemini-3.1-flash-lite", {"candidatesTokenCount": 0, "thoughtsTokenCount": 1_000_000})
    assert c == pytest.approx(1.50) and t.output_tokens == 1_000_000


def test_cached_tokens_discounted():
    full, _ = gemini_cost("gemini-3.1-flash-lite", {"promptTokenCount": 1_000_000})
    cached, t = gemini_cost("gemini-3.1-flash-lite", {"promptTokenCount": 1_000_000, "cachedContentTokenCount": 1_000_000})
    assert cached == pytest.approx(full * 0.1)
    assert t.cached_tokens == 1_000_000


def test_live_audio_priced_by_modality():
    um = {"promptTokenCount": 1_000_000, "responseTokenCount": 1_000_000,
          "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 1_000_000}],
          "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 1_000_000}]}
    c, t = gemini_cost("gemini-3.1-flash-live-preview", um)
    assert c == pytest.approx(3.00 + 12.00)
    assert t.audio_tokens == 2_000_000


def test_models_prefix_and_unknown_model():
    assert gemini_cost("models/gemini-3.1-flash-lite", {"promptTokenCount": 10})[0] > 0
    assert gemini_cost("some-other-model", {"promptTokenCount": 10})[0] == 0


# ------------------------------------------------------------------ REST proxy

def test_generate_content_forwarded_and_logged(make_client, gemini_upstream, upstream):
    client, store = make_client()
    r = client.post(GEN, json=BODY, headers=GH)
    assert r.status_code == 200
    assert r.json()["candidates"][0]["content"]["parts"][0]["text"] == "Your reading"
    sent = gemini_upstream.requests[0]
    assert sent.url.path == GEN and sent.headers["x-goog-api-key"] == "AIza-test"
    assert "x-tokenwise-feature" not in sent.headers
    assert upstream.requests == []  # nothing went to Anthropic
    [row] = rows(store)
    assert (row["provider"], row["feature"], row["engineer"]) == ("gemini", "tarot", "prankundli")
    assert row["input_tokens"] == 2000 and row["output_tokens"] == 500
    assert row["cost_usd"] == pytest.approx((2000 * 0.25 + 500 * 1.50) / 1e6)


def test_stream_generate_content(make_client, gemini_upstream):
    client, store = make_client()
    path = "/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent"
    r = client.post(path + "?alt=sse", json=BODY, headers=GH)
    assert r.status_code == 200 and "reading" in r.text
    [row] = rows(store)
    assert row["stream"] == 1 and row["output_tokens"] == 500  # totals from the last chunk


def test_gemini_error_passed_through(make_client, gemini_upstream):
    client, store = make_client()
    gemini_upstream.handler = lambda req, body: httpx.Response(429, json={"error": {"code": 429}})
    r = client.post(GEN, json=BODY, headers=GH)
    assert r.status_code == 429 and rows(store)[0]["status"] == 429


def test_other_gemini_endpoints_pass_through(make_client, gemini_upstream):
    client, _ = make_client()
    r = client.post("/v1beta/models/gemini-3.1-flash-lite:countTokens", json=BODY, headers=GH)
    assert r.json()["path"].endswith(":countTokens")
    r = client.get("/v1beta/models", headers=GH)
    assert r.json()["path"] == "/v1beta/models"


# ------------------------------------------------------------------ response cache

def test_response_cache_serves_exact_repeats(make_client, gemini_upstream):
    client, store = make_client(response_cache_enabled=True)
    first = client.post(GEN, json=BODY, headers=GH)
    second = client.post(GEN, json=BODY, headers=GH)
    assert second.headers.get("x-tokenwise-cache") == "hit"
    assert second.json() == first.json()
    assert len(gemini_upstream.requests) == 1
    r1, r2 = rows(store)
    assert r2["route_decision"] == "cache-hit" and r2["cost_usd"] == 0
    assert r2["baseline_cost_usd"] == pytest.approx(r1["cost_usd"])


def test_response_cache_misses_on_any_change(make_client, gemini_upstream):
    client, _ = make_client(response_cache_enabled=True)
    client.post(GEN, json=BODY, headers=GH)
    other = {"contents": [{"role": "user", "parts": [{"text": "Read my cards!"}]}]}
    client.post(GEN, json=other, headers=GH)
    client.post(GEN, json=BODY, headers={**GH, "x-goog-api-key": "another-customer"})
    assert len(gemini_upstream.requests) == 3


def test_response_cache_opt_out_and_image_models(make_client, gemini_upstream):
    client, _ = make_client(response_cache_enabled=True)
    for _ in range(2):
        client.post(GEN, json=BODY, headers={**GH, "x-tokenwise-no-cache": "1"})
    img = "/v1beta/models/gemini-2.5-flash-image:generateContent"
    for _ in range(2):
        client.post(img, json=BODY, headers=GH)
    assert len(gemini_upstream.requests) == 4


def test_response_cache_off_by_default(make_client, gemini_upstream):
    client, _ = make_client()
    client.post(GEN, json=BODY, headers=GH)
    client.post(GEN, json=BODY, headers=GH)
    assert len(gemini_upstream.requests) == 2


def test_anthropic_response_cache(make_client, upstream):
    client, _ = make_client(response_cache_enabled=True)
    msg = {"model": "claude-opus-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    h = {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    client.post("/v1/messages", json=msg, headers=h)
    r = client.post("/v1/messages", json=msg, headers=h)
    assert r.headers["x-tokenwise-cache"] == "hit" and len(upstream.requests) == 1


def test_dashboard_shows_provider_and_feature(make_client):
    client, _ = make_client(response_cache_enabled=True)
    client.post(GEN, json=BODY, headers=GH)
    client.post(GEN, json=BODY, headers=GH)
    data = client.get("/api/summary").json()
    assert data["by_provider"][0]["provider"] == "gemini"
    assert data["by_feature"][0]["feature"] == "tarot"
    assert data["response_cache"]["hits"] == 1
    assert "By feature" in client.get("/dashboard").text


# ------------------------------------------------------------------ Live API (WebSocket)

def _free_port():
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


@pytest.fixture
def live_upstream():
    """A real WebSocket server pretending to be Gemini Live."""
    import websockets

    port = _free_port()
    seen = {"headers": None, "path": None, "messages": []}
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def handler(conn):
        seen["headers"] = dict(conn.request.headers)
        seen["path"] = conn.request.path
        setup = json.loads(await conn.recv())
        seen["messages"].append(setup)
        await conn.send(json.dumps({"setupComplete": {}}))
        async for raw in conn:
            seen["messages"].append(json.loads(raw))
            usage = {"promptTokenCount": 1000, "responseTokenCount": 500,
                     "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 1000}],
                     "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 500}]}
            await conn.send(json.dumps({"serverContent": {"modelTurn": {"parts": [{"text": "namaste"}]},
                                                          "turnComplete": True}}).encode())
            await conn.send(json.dumps({"usageMetadata": usage}))

    stop = None

    async def main():
        nonlocal stop
        stop = asyncio.Event()
        async with websockets.serve(handler, "127.0.0.1", port):
            ready.set()
            await stop.wait()

    t = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    t.start()
    ready.wait(5)
    yield port, seen
    loop.call_soon_threadsafe(stop.set)
    t.join(5)


def test_live_websocket_relay_and_usage(make_client, live_upstream):
    port, seen = live_upstream
    client, store = make_client(gemini_upstream_url=f"http://127.0.0.1:{port}")
    path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    with client.websocket_connect(path, headers={"x-goog-api-key": "AIza-live", "x-tokenwise-feature": "voice"}) as ws:
        ws.send_text(json.dumps({"setup": {"model": "models/gemini-3.1-flash-live-preview"}}))
        assert json.loads(ws.receive_text()) == {"setupComplete": {}}
        for turn in range(2):
            ws.send_text(json.dumps({"clientContent": {"turns": [{"parts": [{"text": f"q{turn}"}]}], "turnComplete": True}}))
            assert b"namaste" in ws.receive_bytes()  # binary frames stay binary
            assert "usageMetadata" in ws.receive_text()
    assert seen["path"] == path
    assert seen["headers"]["x-goog-api-key"] == "AIza-live"
    assert "x-tokenwise-feature" not in seen["headers"]
    assert len(seen["messages"]) == 3

    for _ in range(50):  # the row is written as the session closes
        if rows(store):
            break
        import time
        time.sleep(0.05)
    [row] = rows(store)
    assert (row["provider"], row["route_decision"], row["feature"]) == ("gemini", "live", "voice")
    assert row["model_used"] == "gemini-3.1-flash-live-preview"
    assert row["input_tokens"] == 2000 and row["output_tokens"] == 1000 and row["audio_tokens"] == 3000
    assert row["cost_usd"] == pytest.approx((2000 * 3.0 + 1000 * 12.0) / 1e6)
