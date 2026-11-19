"""Per-client safety net (fallback) and shadow testing for OpenAI-format traffic.

Safety net: when the client's main model fails, the same request is sent to a backup model:
    the main model is unreachable or errors (5xx, 404, 429), the answer is empty, cut off at the
    length limit, or (with require_json) not valid JSON. Streams fall back only if the main model
    fails before the first byte. The app gets the backup's answer, labelled x-tokenwise-fallback.

Shadow test: a share of successful, non-streamed requests is also sent to a second model in the
    background. Only scores are stored (how far the answers agree, sizes, speed), never the text.
    Use it to show a client, on their own traffic, how often their own model would agree with the
    model they pay for today, before anything is switched.

    python -m tokenwise.admin fallback acme --base-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --model gemini-2.5-flash --key-env ACME_GEMINI_KEY --require-json
    python -m tokenwise.admin shadow acme --base-url https://gpu.example.in/v1 --model acme-support \\
        --key-env MODEL_API_KEY --rate 0.2
    python -m tokenwise.admin policy acme              # show;  ... fallback acme --off  /  shadow acme --off

Shadow copies send the client's prompts to the shadow host: only turn it on with their agreement.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from . import clients
from .db import Store

FALLBACK_STATUSES = {404, 408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530}

SHADOW_SQL = """CREATE TABLE IF NOT EXISTS shadow_results (
    id TEXT PRIMARY KEY, ts REAL, client_id TEXT, feature TEXT,
    primary_model TEXT, shadow_model TEXT, status INTEGER,
    agreement REAL, exact INTEGER, primary_json INTEGER, shadow_json INTEGER,
    shadow_ms INTEGER, input_tokens INTEGER, shadow_output_tokens INTEGER,
    primary_cost_usd REAL, shadow_cost_usd REAL
)"""


# ---------------------------------------------------------------- settings
def get(client: clients.Client | None) -> dict:
    if client is None:
        return {}
    try:
        return json.loads(client._row.get("policy") or "{}")
    except ValueError:
        return {}


def _save(store: Store, name: str, policy: dict) -> None:
    if not store.execute("UPDATE clients SET policy = ? WHERE name = ?", (json.dumps(policy), name)):
        raise ValueError(f"no client named {name}")


def _load(store: Store, name: str) -> dict:
    rows = store.query("SELECT policy FROM clients WHERE name = ?", (name,))
    if not rows:
        raise ValueError(f"no client named {name}")
    return json.loads(rows[0]["policy"] or "{}")


def set_target(store: Store, name: str, kind: str, base_url: str, model: str, key: str, **extra) -> None:
    """kind: 'fallback' or 'shadow'. The key is stored encrypted."""
    if kind not in ("fallback", "shadow"):
        raise ValueError(kind)
    if not (base_url.startswith("https://") or base_url.startswith("http://")):
        raise ValueError("base URL must start with https:// (or http:// for a server on the same machine)")
    policy = _load(store, name)
    policy[kind] = {"base_url": base_url.rstrip("/"), "model": model, "key_enc": clients.encrypt(key), **extra}
    _save(store, name, policy)


def clear(store: Store, name: str, kind: str) -> None:
    policy = _load(store, name)
    policy.pop(kind, None)
    _save(store, name, policy)


def describe(store: Store, name: str) -> dict:
    """The policy without keys, for display."""
    return {k: {kk: vv for kk, vv in v.items() if kk != "key_enc"} for k, v in _load(store, name).items()}


def target(cfg: dict) -> tuple[str, str, dict[str, str]]:
    """(url, model, headers) for a fallback/shadow entry."""
    return (cfg["base_url"] + "/chat/completions", cfg["model"],
            {"authorization": f"Bearer {clients.decrypt(cfg['key_enc'])}", "content-type": "application/json"})


# ---------------------------------------------------------------- checks
def answer_text(data: dict) -> str:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def as_json(text: str):
    try:
        return json.loads(_strip_fences(text))
    except (ValueError, TypeError):
        return None


def failure(status: int | None, data: dict | None, require_json: bool) -> str | None:
    """Why an answer should go to the backup model, or None if it's fine."""
    if status is None:
        return "unreachable"
    if status in FALLBACK_STATUSES or status >= 500:
        return f"http_{status}"
    if status != 200 or data is None:
        return None  # other client errors (400, 401...) are the app's to fix; a backup won't help
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "no_answer"
    if (choice.get("message") or {}).get("tool_calls"):
        return None
    text = answer_text(data)
    if not text.strip():
        return "empty"
    if choice.get("finish_reason") == "length":
        return "cut_off"
    if require_json and as_json(text) is None:
        return "bad_json"
    return None


# ---------------------------------------------------------------- comparing answers
def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _f1(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return float(wa == wb)
    common, rest = 0, list(wb)
    for w in wa:
        if w in rest:
            rest.remove(w)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(wb), common / len(wa)
    return 2 * p * r / (p + r)


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return sorted(_norm(x) for x in v) if all(not isinstance(x, (dict, list)) for x in v) else [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def agreement(primary: str, shadow: str) -> tuple[float, bool]:
    """(score 0-1, exact). JSON answers: share of the main answer's fields the shadow got the same.
    Text answers: word overlap (F1)."""
    pj, sj = as_json(primary), as_json(shadow)
    if isinstance(pj, dict) and pj:
        if not isinstance(sj, dict):
            return 0.0, False
        same = sum(1 for k, v in pj.items() if k in sj and _norm(sj[k]) == _norm(v))
        return same / len(pj), _norm(pj) == _norm(sj)
    if pj is not None and sj is not None:
        return float(_norm(pj) == _norm(sj)), _norm(pj) == _norm(sj)
    exact = primary.strip() == shadow.strip()
    return (1.0 if exact else _f1(primary, shadow)), exact


def record_shadow(store: Store, row: dict) -> None:
    store.execute(SHADOW_SQL)
    store.insert("shadow_results", {"id": uuid.uuid4().hex, "ts": time.time(), **row})


def shadow_summary(store: Store, since: float) -> list[dict]:
    store.execute(SHADOW_SQL)
    return store.query(
        """SELECT COALESCE(c.name, '-') AS client, COALESCE(r.feature, '(unlabeled)') AS feature,
                  r.primary_model, r.shadow_model, COUNT(*) AS samples,
                  SUM(CASE WHEN r.status = 200 THEN 1 ELSE 0 END) AS answered,
                  AVG(CASE WHEN r.status = 200 THEN r.agreement END) AS agreement,
                  AVG(CASE WHEN r.status = 200 THEN r.exact END) AS exact,
                  AVG(CASE WHEN r.status = 200 AND r.primary_json = 1 THEN r.shadow_json END) AS shadow_json,
                  AVG(CASE WHEN r.status = 200 THEN r.shadow_ms END) AS shadow_ms,
                  SUM(r.primary_cost_usd) AS primary_cost, SUM(r.shadow_cost_usd) AS shadow_cost
           FROM shadow_results r LEFT JOIN clients c ON c.id = r.client_id
           WHERE r.ts >= ? GROUP BY client, feature, r.primary_model, r.shadow_model
           ORDER BY samples DESC""", (since,))


def fallback_summary(store: Store, since: float) -> list[dict]:
    return store.query(
        """SELECT engineer AS client, route_decision AS decision, COUNT(*) AS requests
           FROM requests WHERE ts >= ? AND route_decision LIKE 'fallback:%'
           GROUP BY engineer, route_decision ORDER BY requests DESC""", (since,))
