"""OpenAI-compatible APIs: OpenAI, OpenRouter, Together, Groq, DeepSeek, Fireworks, vLLM, Ollama...

Point the app's OpenAI SDK at Tokenwise with no code changes:

    OPENAI_BASE_URL=http://<tokenwise-host>:8787/openai/v1
    OPENAI_API_KEY=tw-...          # client key; or the provider's own key in pass-through mode

POST /openai/v1/chat/completions is logged, priced and (optionally) traced, streamed or not.
Everything else under /openai/v1/ (models, embeddings, ...) is passed through.

Prices: model names vary by host, so set them yourself (USD per million tokens):
    TOKENWISE_OPENAI_PRICES='{"gpt-5-mini": [0.25, 2.0], "deepseek-chat": [0.27, 1.1]}'
Unknown models are logged with tokens and a cost of 0.

Self-hosted models (vLLM, hf_server): set TOKENWISE_OPENAI_COMPARE_TO so "Without Tokenwise" shows
what the hosted API they replace would have charged, and TOKENWISE_GPU_COST_PER_HOUR for the GPU
(see selfhost.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import gateway, policies, selfhost, traces
from .common import response_headers
from .config import Settings
from .db import Store

PREFIX = "/openai/v1"
log = logging.getLogger("tokenwise.openai")


def _prices() -> dict[str, tuple[float, float]]:
    try:
        raw = json.loads(os.getenv("TOKENWISE_OPENAI_PRICES", "{}"))
        return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    except (ValueError, TypeError, IndexError):
        return {}


def price(model: str, usage: dict | None) -> tuple[float, int, int, int]:
    """(cost_usd, input_tokens, output_tokens, cached_tokens) from an OpenAI usage object."""
    u = usage or {}
    inp, out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    table = _prices()
    p = table.get(model)
    if p is None:  # longest matching prefix, e.g. "gpt-5-mini" for "gpt-5-mini-2026-08-01"
        matches = [k for k in table if model.startswith(k)]
        p = table[max(matches, key=len)] if matches else None
    if p is None:
        return 0.0, inp, out, cached
    return (inp * p[0] + out * p[1]) / 1_000_000, inp, out, cached


def register(app: FastAPI, s: Settings, store: Store) -> None:

    def log_row(ctx: gateway.Ctx, model: str, status: int, started: float, stream: bool, usage: dict | None,
                decision: str = "none", requested: str | None = None):
        engineer, team, feature = ctx.who
        c, inp, out, cached = price(model, usage)
        compare = selfhost.compare_to(model)  # self-hosted model: baseline is the hosted API it replaces
        baseline = price(compare, usage)[0] if compare else c
        return {
            "provider": "openai", "engineer": engineer, "team": team, "feature": feature,
            "model_requested": requested or model, "model_used": model, "route_decision": decision,
            "stream": int(stream),
            "status": status, "latency_ms": int((time.time() - started) * 1000),
            "input_tokens": inp, "output_tokens": out, "cache_read_tokens": cached,
            "cost_usd": c, "baseline_cost_usd": baseline,
        }, c, inp, out

    @app.post(PREFIX + "/chat/completions")
    async def chat_completions(request: Request):
        started = time.time()
        ctx = gateway.prepare(request, "openai", s, store)
        if isinstance(ctx, Response):
            return ctx
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, 400)
        model = body.get("model") or ""
        stream = bool(body.get("stream"))
        if stream and s_stream_usage():
            # Ask for a final usage chunk so streamed calls can be priced.
            body = {**body, "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
            raw = json.dumps(body).encode()
        client: httpx.AsyncClient = app.state.openai_client
        url = ctx.upstream_url + "/chat/completions"
        ctx.headers["content-type"] = "application/json"
        collector = traces.OutputCollector("openai")

        pol = policies.get(ctx.client)
        fb = pol.get("fallback")
        requested = model

        if not stream:
            try:
                resp = await client.post(url, content=raw, headers=ctx.headers, params=ctx.params)
                status, content, out_headers = resp.status_code, resp.content, response_headers(resp)
            except httpx.HTTPError:
                if not fb:
                    raise
                status, content, out_headers = None, b"", {}
            data = _json(content) if status == 200 else None
            reason = policies.failure(status, data, bool(fb.get("require_json"))) if fb else None
            extra_cost = 0.0
            if reason:  # safety net: same request to the backup model
                if data:
                    extra_cost = price(model, data.get("usage"))[0]  # the failed answer was still billed
                fb_url, fb_model, fb_headers = policies.target(fb)
                try:
                    r2 = await client.post(fb_url, json={**body, "model": fb_model}, headers=fb_headers)
                    status, content, out_headers = r2.status_code, r2.content, response_headers(r2)
                except httpx.HTTPError as e:
                    status, content, out_headers = 502, json.dumps({"error": {
                        "message": f"main and backup model both failed ({reason}; backup: {type(e).__name__})",
                        "type": "upstream_error"}}).encode(), {}
                data = _json(content) if status == 200 else None
                model = fb_model
                log.warning("fallback for %s: %s -> %s", ctx.client.name if ctx.client else "-", reason, fb_model)
            usage = None
            if data:
                usage = data.get("usage")
                model = data.get("model") or model
                collector.add(data)
            decision = f"fallback:{reason}" if reason else "none"
            row, c, inp, out = log_row(ctx, model, status, started, False, usage, decision, requested)
            row["cost_usd"] += extra_cost
            await app.state.log_row(row)
            await gateway.record(ctx, store, s, model=model, status=status, stream=False, body=body,
                                 collector=collector, input_tokens=inp, output_tokens=out, cost_usd=c, started=started)
            sh = pol.get("shadow")
            if sh and not reason and data and random.random() < float(sh.get("rate", 0)):
                task = asyncio.create_task(_shadow(app, store, ctx, sh, body, data, model, c))
                app.state.shadow_tasks.add(task)
                task.add_done_callback(app.state.shadow_tasks.discard)
            extra = {"x-tokenwise-fallback": reason} if reason else {}
            return Response(content=content, status_code=status,
                            headers={**out_headers, **gateway.trace_headers(ctx), **extra})

        req = client.build_request("POST", url, content=raw, headers=ctx.headers, params=ctx.params)
        decision = "none"
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError:
            if not fb:
                raise
            resp = None
        reason = policies.failure(resp.status_code if resp else None, None, False) if fb else None
        if reason:  # safety net for streams: only when the main model fails before answering
            if resp is not None:
                await resp.aclose()
            fb_url, fb_model, fb_headers = policies.target(fb)
            req = client.build_request("POST", fb_url, json={**body, "model": fb_model}, headers=fb_headers)
            resp = await client.send(req, stream=True)
            model, decision = fb_model, f"fallback:{reason}"
        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            row, *_ = log_row(ctx, model, resp.status_code, started, True, None, decision, requested)
            await app.state.log_row(row)
            await gateway.record(ctx, store, s, model=model, status=resp.status_code, stream=True, body=body,
                                 collector=collector, started=started)
            return Response(content=content, status_code=resp.status_code, headers=response_headers(resp))

        state = {"buf": "", "usage": None, "model": model}

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    state["buf"] += chunk.decode("utf-8", errors="replace")
                    events, state["buf"] = traces.sse_events(state["buf"])
                    for ev in events:
                        collector.add(ev)
                        state["usage"] = ev.get("usage") or state["usage"]
                        state["model"] = ev.get("model") or state["model"]
                    yield chunk
            finally:
                await resp.aclose()
                events, _ = traces.sse_events(state["buf"] + "\n\n")
                for ev in events:
                    collector.add(ev)
                    state["usage"] = ev.get("usage") or state["usage"]
                row, c, inp, out = log_row(ctx, state["model"], 200, started, True, state["usage"], decision,
                                           requested)
                await app.state.log_row(row)
                await gateway.record(ctx, store, s, model=state["model"], status=200, stream=True, body=body,
                                     collector=collector, input_tokens=inp, output_tokens=out, cost_usd=c,
                                     started=started)

        return StreamingResponse(relay(), status_code=200,
                                 headers={**response_headers(resp), **gateway.trace_headers(ctx),
                                          **({"x-tokenwise-fallback": reason} if reason else {})},
                                 media_type=resp.headers.get("content-type", "text/event-stream"))

    @app.api_route(PREFIX + "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def openai_passthrough(path: str, request: Request):
        ctx = gateway.prepare(request, "openai", s, store)
        if isinstance(ctx, Response):
            return ctx
        client: httpx.AsyncClient = app.state.openai_client
        req = client.build_request(request.method, f"{ctx.upstream_url}/{path}", content=await request.body(),
                                   headers=ctx.headers, params=ctx.params)
        resp = await client.send(req, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(relay(), status_code=resp.status_code, headers=response_headers(resp))


def s_stream_usage() -> bool:
    return os.getenv("TOKENWISE_OPENAI_STREAM_USAGE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _json(content: bytes) -> dict | None:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


async def _shadow(app: FastAPI, store: Store, ctx: gateway.Ctx, cfg: dict, body: dict, primary: dict,
                  primary_model: str, primary_cost: float) -> None:
    """Send a copy to the shadow model and store how far the answers agree (scores only, no text)."""
    try:
        url, model, headers = policies.target(cfg)
        sbody = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        started = time.time()
        try:
            r = await app.state.openai_client.post(url, json={**sbody, "model": model}, headers=headers)
            status, data = r.status_code, (_json(r.content) if r.status_code == 200 else None)
        except httpx.HTTPError:
            status, data = 0, None
        ms = int((time.time() - started) * 1000)
        ptext = policies.answer_text(primary)
        stext = policies.answer_text(data) if data else ""
        score, exact = policies.agreement(ptext, stext) if data else (None, False)
        pusage = primary.get("usage") or {}
        susage = (data or {}).get("usage") or {}
        await asyncio.to_thread(policies.record_shadow, store, {
            "client_id": ctx.client_id, "feature": ctx.who[2], "primary_model": primary_model, "shadow_model": model,
            "status": status, "agreement": score, "exact": int(exact),
            "primary_json": int(policies.as_json(ptext) is not None),
            "shadow_json": int(policies.as_json(stext) is not None) if data else 0,
            "shadow_ms": ms, "input_tokens": int(pusage.get("prompt_tokens") or 0),
            "shadow_output_tokens": int(susage.get("completion_tokens") or 0),
            "primary_cost_usd": primary_cost, "shadow_cost_usd": price(model, susage)[0] if data else 0.0,
        })
    except Exception:
        log.exception("shadow test failed")
