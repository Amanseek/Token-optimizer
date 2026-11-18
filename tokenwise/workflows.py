"""Feedback API and the per-workflow view of captured traces.

POST /tokenwise/v1/feedback   (auth: the same tw- client key the app uses)
    {"trace_id": "...", "rating": "good" | "bad", "note": "...", "corrected_output": "..."}
    trace_id is the x-tokenwise-trace-id header Tokenwise returned with the answer.
    corrected_output: a better answer (from a reviewer or the user's edit); it is used
    for training instead of the model's answer.

GET /api/workflows?days=30    per-workflow volume, success, feedback, cost, tool sequences
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import clients, traces
from .config import Settings
from .db import Store
from .masking import mask_text


def summary(store: Store, days: int = 30, client_id: str | None = None) -> list[dict]:
    since = time.time() - days * 86400
    where, params = "ts >= ?", [since]
    if client_id:
        where += " AND client_id = ?"
        params.append(client_id)
    rows = store.query(
        f"""SELECT t.client_id, COALESCE(c.name, '(no client)') AS client, t.workflow_key, t.workflow_label,
                   COUNT(*) AS requests, SUM(t.success) AS ok,
                   SUM(CASE WHEN t.feedback = 1 THEN 1 ELSE 0 END) AS good,
                   SUM(CASE WHEN t.feedback = 0 THEN 1 ELSE 0 END) AS bad,
                   SUM(CASE WHEN t.corrected_output IS NOT NULL THEN 1 ELSE 0 END) AS corrected,
                   SUM(CASE WHEN t.reason = 'retried' THEN 1 ELSE 0 END) AS retried,
                   AVG(t.input_tokens) AS avg_in, AVG(t.output_tokens) AS avg_out,
                   COALESCE(SUM(t.cost_usd), 0) AS cost, GROUP_CONCAT(DISTINCT t.model) AS models,
                   GROUP_CONCAT(DISTINCT t.provider) AS providers
            FROM traces t LEFT JOIN clients c ON c.id = t.client_id
            WHERE {where.replace('ts', 't.ts').replace('client_id', 't.client_id')}
            GROUP BY t.client_id, t.workflow_key ORDER BY requests DESC""",
        tuple(params),
    )
    for r in rows:
        r["success_pct"] = round(100 * (r["ok"] or 0) / r["requests"], 1) if r["requests"] else 0.0
        # Usable for training: succeeded and not rated bad, or corrected by a person.
        r["trainable"] = store.query(
            "SELECT COUNT(*) AS n FROM traces WHERE workflow_key = ? AND COALESCE(client_id,'') = COALESCE(?,'') "
            "AND ts >= ? AND ((success = 1 AND COALESCE(feedback, 1) = 1) OR corrected_output IS NOT NULL)",
            (r["workflow_key"], r["client_id"], since))[0]["n"]
        seqs = store.query(
            "SELECT tool_calls FROM traces WHERE workflow_key = ? AND COALESCE(client_id,'') = COALESCE(?,'') "
            "AND ts >= ? ORDER BY ts DESC LIMIT 500", (r["workflow_key"], r["client_id"], since))
        top = Counter(traces.tool_sequence(traces.unseal(x["tool_calls"])) for x in seqs).most_common(3)
        r["top_tool_sequences"] = [{"sequence": k, "count": v} for k, v in top]
    return rows


def set_feedback(store: Store, trace_id: str, client_id: str | None, rating: str | None, note: str | None,
                 corrected: str | None, mask: bool) -> bool:
    rows = store.query("SELECT client_id FROM traces WHERE id = ?", (trace_id,))
    if not rows or (rows[0]["client_id"] or None) != client_id:
        return False  # unknown trace, or another client's trace
    fb = {"good": 1, "bad": 0}.get(rating or "")
    if mask:
        note = mask_text(note) if note else note
        corrected = mask_text(corrected) if corrected else corrected
    store.execute(
        "UPDATE traces SET feedback = COALESCE(?, feedback), feedback_note = COALESCE(?, feedback_note), "
        "corrected_output = COALESCE(?, corrected_output) WHERE id = ?",
        (fb, traces.seal(note), traces.seal(corrected), trace_id),
    )
    return True


def register(app: FastAPI, s: Settings, store: Store) -> None:

    @app.post("/tokenwise/v1/feedback")
    async def feedback(request: Request):
        client, sent_key = clients.resolve(request, store)
        if sent_key and client is None:
            return JSONResponse({"error": "unknown or disabled Tokenwise key"}, status_code=401)
        if client is None and s.require_client_key:
            return JSONResponse({"error": "Tokenwise client key required"}, status_code=401)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        trace_id = str(body.get("trace_id") or "")
        rating = body.get("rating")
        if rating not in (None, "good", "bad"):
            return JSONResponse({"error": "rating must be 'good' or 'bad'"}, status_code=400)
        if not trace_id or (rating is None and not body.get("corrected_output") and not body.get("note")):
            return JSONResponse({"error": "send trace_id and a rating, note or corrected_output"}, status_code=400)
        mask = client.mask if client else s.traces_mask
        ok = await asyncio.to_thread(set_feedback, store, trace_id, client.id if client else None, rating,
                                     body.get("note"), body.get("corrected_output"), mask)
        if not ok:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return {"ok": True}

    @app.post("/tokenwise/v1/delete")
    async def delete(request: Request):
        """Delete stored conversations: for one end user (their x-tokenwise-user value), by trace ids, or all."""
        client, sent_key = clients.resolve(request, store)
        if sent_key and client is None:
            return JSONResponse({"error": "unknown or disabled Tokenwise key"}, status_code=401)
        if client is None and s.require_client_key:
            return JSONResponse({"error": "Tokenwise client key required"}, status_code=401)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        user, ids, everything = body.get("user"), body.get("trace_ids") or [], bool(body.get("all"))
        if not (user or ids or everything) or not isinstance(ids, list):
            return JSONResponse({"error": "send 'user', 'trace_ids' (list) or 'all': true"}, status_code=400)
        cid = client.id if client else None
        n = await asyncio.to_thread(traces.delete_for, store, cid, user=user, trace_ids=[str(i) for i in ids],
                                    everything=everything)
        await asyncio.to_thread(store.audit, f"client:{client.name}" if client else "anonymous", "delete",
                                "all" if everything else (f"user:{user}" if user else f"{len(ids)} trace ids"),
                                f"{n} traces deleted")
        return {"ok": True, "deleted": n}

    @app.get("/api/workflows")
    async def api_workflows(request: Request, days: int = 30):
        if s.admin_token and s.admin_token not in (request.query_params.get("token"),
                                                   request.headers.get("x-tokenwise-admin")):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        await asyncio.to_thread(store.audit, "admin", "view_workflows", request.client.host if request.client else "")
        return JSONResponse(await asyncio.to_thread(summary, store, days))
