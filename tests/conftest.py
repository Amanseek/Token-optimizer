import json

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenwise.app import create_app
from tokenwise.config import Settings
from tokenwise.db import Store


def sse(model: str, usage_start: dict, output_tokens: int, text: str = "hi") -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "m1", "model": model, "usage": usage_start}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


class Upstream:
    """Fake Anthropic API. Set .handler to customize per test."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.handler = self.default

    def default(self, request: httpx.Request, body: dict) -> httpx.Response:
        usage = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 5000,
                 "cache_creation_input_tokens": 0}
        if body.get("stream"):
            return httpx.Response(200, content=sse(body["model"], {**usage, "output_tokens": 1}, 200),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "m1", "type": "message", "model": body["model"],
                                         "content": [{"type": "text", "text": "hello"}],
                                         "stop_reason": "end_turn", "usage": usage})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else {}
        self.bodies.append(body)
        resp = self.handler(request, body)
        # Return an unread stream, like a real network response.
        return httpx.Response(resp.status_code, headers=resp.headers, stream=httpx.ByteStream(resp.content))


@pytest.fixture
def upstream():
    return Upstream()


GEMINI_USAGE = {"promptTokenCount": 2000, "candidatesTokenCount": 400, "thoughtsTokenCount": 100,
                "totalTokenCount": 2500}


def gemini_reply(text="Your reading", usage=None):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage or GEMINI_USAGE, "modelVersion": "gemini-3.1-flash-lite"}


class GeminiUpstream(Upstream):
    def default(self, request, body):
        path = request.url.path
        if path.endswith(":streamGenerateContent"):
            chunks = [gemini_reply("Your ", {"promptTokenCount": 2000}), gemini_reply("reading")]
            data = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
            return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})
        if path.endswith(":generateContent"):
            return httpx.Response(200, json=gemini_reply())
        return httpx.Response(200, json={"path": path})


@pytest.fixture
def gemini_upstream():
    return GeminiUpstream()


@pytest.fixture
def make_client(tmp_path, upstream, gemini_upstream):
    clients = []

    def _make(**overrides):
        opts = {"upstream_url": "https://upstream.test", "gemini_upstream_url": "https://gemini.test", **overrides}
        s = Settings(db_path=str(tmp_path / "t.db"), **opts)
        store = Store(s.db_path)
        app = create_app(s, store, transport=httpx.MockTransport(upstream),
                         gemini_transport=httpx.MockTransport(gemini_upstream))
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c, store

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
