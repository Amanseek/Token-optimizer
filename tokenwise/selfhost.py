"""Own-GPU economics: what self-hosted models cost versus the hosted API they replace.

    TOKENWISE_OPENAI_COMPARE_TO   the hosted model each self-hosted model replaces:
        gemini-2.5-flash                      every OpenAI-format model is compared with it
        {"demo-scam": "gemini-2.5-flash"}     per model ("*" = all others)
      The compare model's price must be in TOKENWISE_OPENAI_PRICES.
    TOKENWISE_GPU_COST_PER_HOUR   USD per hour of the GPU machine, tax included (e.g. 1.34 for ₹118/hr).
      Counted for the hours this Tokenwise process runs, so run it on the GPU machine
      (start_e2e.sh does); if Tokenwise runs elsewhere the GPU hours won't match the bill.
    TOKENWISE_INR_PER_USD         optional; also show rupees on the dashboard.

A GPU costs the same per hour however busy it is, so it only beats the API above a certain
number of requests per hour. The dashboard shows that break-even next to the busiest hour seen.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from .db import Store

UPTIME_SQL = "CREATE TABLE IF NOT EXISTS uptime (id TEXT PRIMARY KEY, started REAL, last_seen REAL)"


def _float_env(name: str) -> float:
    try:
        return max(float(os.getenv(name, "") or 0), 0.0)
    except ValueError:
        return 0.0


def gpu_cost_per_hour() -> float:
    return _float_env("TOKENWISE_GPU_COST_PER_HOUR")


def inr_per_usd() -> float:
    return _float_env("TOKENWISE_INR_PER_USD")


def _compare_map() -> dict[str, str]:
    raw = (os.getenv("TOKENWISE_OPENAI_COMPARE_TO") or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return {str(k): str(v) for k, v in json.loads(raw).items() if v}
        except ValueError:
            return {}
    return {"*": raw}


def compare_to(model: str) -> str | None:
    """The hosted model a self-hosted model is compared with, or None if it isn't self-hosted."""
    m = _compare_map()
    return m.get(model) or m.get("*")


# ---------------------------------------------------------------- GPU hours
def start_session(store: Store) -> str:
    store.execute(UPTIME_SQL)
    sid, now = uuid.uuid4().hex, time.time()
    store.execute("INSERT INTO uptime (id, started, last_seen) VALUES (?, ?, ?)", (sid, now, now))
    return sid


def beat(store: Store, sid: str) -> None:
    store.execute("UPDATE uptime SET last_seen = ? WHERE id = ?", (time.time(), sid))


def gpu_hours(store: Store, since: float) -> float:
    store.execute(UPTIME_SQL)
    rows = store.query("SELECT started, last_seen FROM uptime WHERE last_seen >= ?", (since,))
    return sum(max(r["last_seen"] - max(r["started"], since), 0) for r in rows) / 3600


# ---------------------------------------------------------------- report
def report(store: Store, since: float) -> dict | None:
    """Self-hosted vs hosted API for the period, or None when nothing is configured."""
    if not _compare_map():
        return None
    rows = store.query(
        """SELECT model_used AS model, COUNT(*) AS requests, SUM(cost_usd) AS cost,
                  SUM(baseline_cost_usd) AS api_cost
           FROM requests WHERE ts >= ? AND provider = 'openai' GROUP BY model_used""", (since,))
    rows = [r for r in rows if compare_to(r["model"])]
    models = [r["model"] for r in rows]
    requests = sum(r["requests"] for r in rows)
    api_cost = sum(r["api_cost"] or 0 for r in rows)
    token_cost = sum(r["cost"] or 0 for r in rows)
    rate = gpu_cost_per_hour()
    hours = gpu_hours(store, since) if rate else 0.0
    gpu_cost = hours * rate
    own_cost = token_cost + gpu_cost
    busiest = 0
    if models:
        marks = ",".join("?" for _ in models)
        top = store.query(
            f"""SELECT COUNT(*) AS n FROM requests WHERE ts >= ? AND provider = 'openai'
                AND model_used IN ({marks}) GROUP BY CAST(ts / 3600 AS INTEGER) ORDER BY n DESC LIMIT 1""",
            (since, *models))
        busiest = top[0]["n"] if top else 0
    per_request = api_cost / requests if requests else 0.0
    return {
        "compare_to": sorted({compare_to(m) for m in models}) or sorted(set(_compare_map().values())),
        "models": models,
        "requests": requests,
        "api_cost": api_cost,           # what the hosted API would have charged for the same tokens
        "token_cost": token_cost,       # per-token price set for the own models (usually 0)
        "gpu_hours": hours,
        "gpu_cost_per_hour": rate,
        "gpu_cost": gpu_cost,
        "own_cost": own_cost,
        "saved": api_cost - own_cost,   # negative = the API would have been cheaper
        "api_cost_per_request": per_request,
        # requests per hour at which the GPU costs the same as the API (None until there's data)
        "breakeven_per_hour": (rate / per_request) if (rate and per_request) else None,
        "busiest_hour": busiest,
    }
