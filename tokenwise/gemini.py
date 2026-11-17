"""Google Gemini (AI Studio API) support.

Point the google-genai SDK at Tokenwise with no code changes:

    GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787

Handled here:
  POST /{version}/models/{model}:generateContent        logged, optional response cache
  POST /{version}/models/{model}:streamGenerateContent  logged, streamed through
  WS   /ws/...BidiGenerateContent                        Live API (voice), relayed and logged
Everything else (countTokens, embedContent, files, ...) is passed through.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from .common import cache_key, cacheable, forward_headers, identity, response_headers
from .config import Settings
from .db import Store
from .gemini_pricing import GeminiTokens, gemini_cost

log = logging.getLogger("tokenwise.gemini")

GEMINI_VERSIONS = {"v1", "v1beta", "v1alpha"}
WS_SKIP_HEADERS = {
    "host", "upgrade", "connection", "sec-websocket-key", "sec-websocket-version",
    "sec-websocket-extensions", "sec-websocket-protocol", "content-length", "accept-encoding",
}


def is_gemini_request(request: Request, path: str) -> bool:
    if request.headers.get("x-goog-api-key") or request.query_params.get("key"):
        return True
    return path.split("/", 1)[0] in {"v1beta", "v1alpha", "upload", "download"}


def _clean_model(model: str | None) -> str:
    return (model or "").split("/")[-1]


class GeminiStreamUsage:
    """Tracks usageMetadata in a streamGenerateContent response.

    With alt=sse each event is one JSON chunk; otherwise the body is one JSON
    array. Each chunk repeats the running totals, so the last one wins.
    """

    def __init__(self, sse: bool):
        self.sse = sse
        self.buf = ""
        self.usage: dict | None = None
        self.model_version: str | None = None

    def _take(self, data) -> None:
        items = data if isinstance(data, list) else [data]
        for d in items:
            if isinstance(d, dict):
                if d.get("usageMetadata"):
                    self.usage = d["usageMetadata"]
                self.model_version = d.get("modelVersion") or self.model_version

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk.decode("utf-8", errors="replace")
        if not self.sse:
            return
        self.buf = self.buf.replace("\r\n", "\n")
        while "\n\n" in self.buf:
            event, self.buf = self.buf.split("\n\n", 1)
            for line in event.splitlines():
                if line.startswith("data:"):
                    try:
                        self._take(json.loads(line[5:].strip()))
                    except ValueError:
                        pass

    def finish(self) -> None:
        if self.sse:
            self.feed(b"\n\n")
        else:
            try:
                self._take(json.loads(self.buf))
            except ValueError:
                pass


def register(app: FastAPI, s: Settings, store: Store) -> None:

    def row(who, model, status, started, stream, tokens: GeminiTokens, cost_usd, label="none", baseline=None):
        engineer, team, feature = who
        return {
            "provider": "gemini",
            "engineer": engineer,
            "team": team,
            "feature": feature,
            "model_requested": model,
            "model_used": model,
            "route_decision": label,
            "stream": int(stream),
            "status": status,
            "latency_ms": int((time.time() - started) * 1000),
            "input_tokens": tokens.input_tokens,
            "output_tokens": tokens.output_tokens,
            "cache_read_tokens": tokens.cached_tokens,
            "audio_tokens": tokens.audio_tokens,
            "cost_usd": cost_usd,
            "baseline_cost_usd": cost_usd if baseline is None else baseline,
        }

    @app.post("/{version}/models/{model_action}")
    async def gemini_models(version: str, model_action: str, request: Request):
        started = time.time()
        client: httpx.AsyncClient = app.state.gemini_client
        log_row = app.state.log_row
        path = f"/{version}/models/{model_action}"
        raw = await request.body()
        headers = forward_headers(request, s)
        params = dict(request.query_params)
        model, _, method = model_action.partition(":")

        if version not in GEMINI_VERSIONS or method not in {"generateContent", "streamGenerateContent"}:
            resp = await client.post(path, content=raw, headers=headers, params=params)
            return Response(content=resp.content, status_code=resp.status_code, headers=response_headers(resp))

        who = identity(request, s)
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = None

        # ---------------- generateContent
        if method == "generateContent":
            rkey = None
            if body is not None and cacheable(request, s, model):
                rkey = cache_key(request, "gemini", path, body)
                hit = await asyncio.to_thread(store.cache_get, rkey, s.response_cache_ttl)
                if hit:
                    await log_row(row(who, model, 200, started, False, GeminiTokens(), 0.0, "cache-hit", hit["cost_usd"]))
                    return Response(content=hit["body"], status_code=200, media_type=hit["content_type"],
                                    headers={"x-tokenwise-cache": "hit"})

            resp = await client.post(path, content=raw, headers=headers, params=params)
            tokens, cost_usd = GeminiTokens(), 0.0
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    cost_usd, tokens = gemini_cost(model, data.get("usageMetadata"))
                except ValueError:
                    pass
            await log_row(row(who, model, resp.status_code, started, False, tokens, cost_usd))
            if rkey and resp.status_code == 200:
                await asyncio.to_thread(store.cache_put, rkey, resp.content,
                                        resp.headers.get("content-type", "application/json"), model, cost_usd)
            return Response(content=resp.content, status_code=resp.status_code, headers=response_headers(resp))

        # ---------------- streamGenerateContent
        req = client.build_request("POST", path, content=raw, headers=headers, params=params)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            await log_row(row(who, model, resp.status_code, started, True, GeminiTokens(), 0.0))
            return Response(content=content, status_code=resp.status_code, headers=response_headers(resp))

        tracker = GeminiStreamUsage(sse=params.get("alt") == "sse")

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    tracker.feed(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                tracker.finish()
                cost_usd, tokens = gemini_cost(model, tracker.usage)
                await log_row(row(who, model, 200, started, True, tokens, cost_usd))

        return StreamingResponse(relay(), status_code=200, headers=response_headers(resp),
                                 media_type=resp.headers.get("content-type"))

    # ---------------- Live API (voice) over WebSocket
    @app.websocket("/ws/{path:path}")
    async def gemini_live(ws: WebSocket, path: str):
        import websockets  # imported here so the HTTP proxy works without it

        started = time.time()
        who = identity(ws, s)
        skip = WS_SKIP_HEADERS | {s.user_header, s.team_header, s.feature_header, s.nocache_header}
        headers = {k: v for k, v in ws.headers.items() if k.lower() not in skip}
        base = s.gemini_upstream_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
        query = ws.url.query
        uri = f"{base}/ws/{path}" + (f"?{query}" if query else "")

        await ws.accept()
        state = {"model": "", "turns": 0}
        tokens = GeminiTokens()
        cost_total = 0.0

        def note_client(msg: str | bytes) -> None:
            if state["model"]:
                return
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                return
            setup = data.get("setup") if isinstance(data, dict) else None
            if isinstance(setup, dict):
                state["model"] = _clean_model(setup.get("model"))

        def note_server(msg: str | bytes) -> None:
            nonlocal cost_total
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                return
            um = data.get("usageMetadata") if isinstance(data, dict) else None
            if um:
                # Live bills each turn on the whole session context; summing every
                # usage message matches the invoice.
                c, t = gemini_cost(state["model"], um)
                cost_total += c
                tokens.add(t)
                state["turns"] += 1

        try:
            extra = {}
            if uri.startswith("wss://") and os.environ.get("SSL_CERT_FILE"):
                extra["ssl"] = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
            upstream = await websockets.connect(uri, additional_headers=headers, max_size=None, **extra)
        except Exception as e:  # upstream refused (bad key, bad model, ...)
            log.warning("live upstream connect failed: %s", e)
            await ws.close(code=1011, reason="upstream connection failed")
            return

        async def client_to_upstream():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                data = msg.get("text") if msg.get("text") is not None else msg.get("bytes")
                if data is None:
                    continue
                note_client(data)
                await upstream.send(data)

        async def upstream_to_client():
            async for data in upstream:
                note_server(data)
                if isinstance(data, bytes):
                    await ws.send_bytes(data)
                else:
                    await ws.send_text(data)

        tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
        close_code, close_reason = 1000, ""
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed)):
                    log.warning("live relay error: %r", exc)
            if upstream.close_code is not None:
                close_code = upstream.close_code if upstream.close_code not in (1005, 1006) else 1000
                close_reason = upstream.close_reason or ""
        finally:
            # Log first and synchronously: the server may cancel this task while
            # the closes below are awaited, and the session's cost must not be lost.
            try:
                store.log_request(row(who, state["model"], 200, started, True, tokens, cost_total, "live"))
            except Exception:
                log.exception("failed to log live session")
            for t in tasks:
                t.cancel()
            await upstream.close()
            try:
                await ws.close(code=close_code, reason=close_reason)
            except RuntimeError:
                pass  # client already gone
