"""A stand-in for the Gemini API, for testing without an API key.

Serves generateContent, streamGenerateContent and the Live WebSocket. Prompt
tokens are estimated as characters / 4; every reply is a fixed ~450-token
reading. It counts what it receives, so a benchmark can compare the
"provider bill" with and without Tokenwise. GET /stats, POST /reset.
"""

import json
import math

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

REPLY = ("The Tower in the past position shows a sudden upheaval you have already lived through. "
         "The Star in the present brings healing and quiet hope; trust it. ") * 12  # ~1,850 chars

app = FastAPI()
STATS = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "live_sessions": 0, "live_turns": 0}


def _text(body: dict) -> str:
    out = []
    for c in body.get("contents") or []:
        for p in (c.get("parts") if isinstance(c, dict) else []) or []:
            out.append(p.get("text", ""))
    si = body.get("systemInstruction") or body.get("system_instruction") or {}
    for p in si.get("parts", []) if isinstance(si, dict) else []:
        out.append(p.get("text", ""))
    return "\n".join(out)


def _usage(prompt_chars: int) -> dict:
    p = math.ceil(prompt_chars / 4)
    o = math.ceil(len(REPLY) / 4)
    return {"promptTokenCount": p, "candidatesTokenCount": o, "totalTokenCount": p + o}


def _reply(usage: dict, text: str = REPLY) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage, "modelVersion": "fake"}


@app.post("/{version}/models/{model_action}")
async def models(version: str, model_action: str, request: Request):
    body = await request.json()
    usage = _usage(len(_text(body)))
    STATS["calls"] += 1
    STATS["prompt_tokens"] += usage["promptTokenCount"]
    STATS["output_tokens"] += usage["candidatesTokenCount"]
    if model_action.endswith(":streamGenerateContent"):
        half = len(REPLY) // 2
        chunks = [_reply({"promptTokenCount": usage["promptTokenCount"]}, REPLY[:half]), _reply(usage, REPLY[half:])]
        data = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
        return StreamingResponse(iter([data.encode()]), media_type="text/event-stream")
    return JSONResponse(_reply(usage))


@app.websocket("/ws/{path:path}")
async def live(ws: WebSocket, path: str):
    await ws.accept()
    STATS["live_sessions"] += 1
    await ws.receive_text()  # setup
    await ws.send_text(json.dumps({"setupComplete": {}}))
    context = 0
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        context += 300  # each turn adds ~12 s of audio
        STATS["live_turns"] += 1
        usage = {"promptTokenCount": context, "responseTokenCount": 250,
                 "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": context}],
                 "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 250}]}
        await ws.send_text(json.dumps({"serverContent": {"modelTurn": {"parts": [{"text": "Namaste"}]},
                                                         "turnComplete": True}, "usageMetadata": usage}))


@app.get("/stats")
async def stats():
    return STATS


@app.post("/reset")
async def reset():
    for k in STATS:
        STATS[k] = 0
    return STATS
