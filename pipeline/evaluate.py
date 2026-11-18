"""Compare models on a held-out test set and decide whether the new one can take traffic.

    python pipeline/evaluate.py --test data/test.jsonl \\
        --candidate http://gpu-host:8000/v1 acme-support \\
        --current  https://openrouter.ai/api/v1 deepseek/deepseek-chat --current-key-env OPENROUTER_KEY \\
        --out reports/acme-support.json

Both sides are OpenAI-compatible endpoints (vLLM, Tokenwise, OpenAI, OpenRouter, ...).
Each test example's last assistant message is the reference answer.

Scores per model:
  exact        normalised exact match with the reference
  f1           word-overlap F1 with the reference (rough similarity for free text)
  json_valid   share of answers that parse as JSON (only when references are JSON)
  json_fields  share of reference JSON fields matched exactly (only when references are JSON)
  latency_ms   median per request

Decision: promote = candidate's main score >= current's main score - --margin
(main score: json_fields when references are JSON, else f1). With no --current,
the candidate is compared against --min-score instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def f1(pred: str, ref: str) -> float:
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def as_json(s: str):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    m = re.search(r"\{.*\}|\[.*\]", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def field_score(pred, ref) -> float:
    if not isinstance(ref, dict) or not ref:
        return float(pred == ref)
    if not isinstance(pred, dict):
        return 0.0
    return sum(pred.get(k) == v for k, v in ref.items()) / len(ref)


def load_test(path: str, limit: int | None) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            msgs = json.loads(line)["messages"]
            if msgs and msgs[-1]["role"] == "assistant":
                rows.append({"prompt": msgs[:-1], "reference": msgs[-1]["content"]})
    return rows[:limit] if limit else rows


async def run_model(url: str, model: str, key: str | None, rows: list[dict], concurrency: int,
                    max_tokens: int, headers: dict) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    h = {"content-type": "application/json", **headers}
    if key:
        h["authorization"] = f"Bearer {key}"
    async with httpx.AsyncClient(timeout=120) as client:
        async def one(r):
            async with sem:
                t0 = time.time()
                try:
                    resp = await client.post(url.rstrip("/") + "/chat/completions", headers=h, json={
                        "model": model, "messages": r["prompt"], "temperature": 0, "max_tokens": max_tokens})
                    text = resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else ""
                    err = None if resp.status_code == 200 else f"http {resp.status_code}: {resp.text[:200]}"
                except Exception as e:  # network errors count as failures, not crashes
                    text, err = "", repr(e)[:200]
                return {"answer": text or "", "error": err, "ms": int((time.time() - t0) * 1000)}
        return await asyncio.gather(*(one(r) for r in rows))


def score(rows: list[dict], outs: list[dict]) -> dict:
    json_refs = [as_json(r["reference"]) for r in rows]
    is_json = sum(j is not None for j in json_refs) >= 0.8 * len(rows)
    res = {
        "n": len(rows),
        "errors": sum(o["error"] is not None for o in outs),
        "exact": round(sum(norm(o["answer"]) == norm(r["reference"]) for o, r in zip(outs, rows)) / len(rows), 4),
        "f1": round(sum(f1(o["answer"], r["reference"]) for o, r in zip(outs, rows)) / len(rows), 4),
        "latency_ms": int(statistics.median(o["ms"] for o in outs)),
    }
    if is_json:
        preds = [as_json(o["answer"]) for o in outs]
        res["json_valid"] = round(sum(p is not None for p in preds) / len(rows), 4)
        res["json_fields"] = round(sum(field_score(p, j) for p, j in zip(preds, json_refs)) / len(rows), 4)
    res["main"] = res["json_fields"] if is_json else res["f1"]
    res["main_metric"] = "json_fields" if is_json else "f1"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--candidate", nargs=2, metavar=("BASE_URL", "MODEL"), required=True)
    ap.add_argument("--candidate-key-env")
    ap.add_argument("--current", nargs=2, metavar=("BASE_URL", "MODEL"))
    ap.add_argument("--current-key-env")
    ap.add_argument("--header", action="append", default=[], help="extra header 'name: value' (both sides)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    rows = load_test(a.test, a.limit)
    if not rows:
        raise SystemExit("no test examples")
    headers = dict(h.split(":", 1) for h in a.header)
    headers = {k.strip(): v.strip() for k, v in headers.items()}
    env = lambda n: os.getenv(n) if n else None  # noqa: E731

    report = {"test": a.test, "examples": len(rows)}
    cand_out = asyncio.run(run_model(*a.candidate, env(a.candidate_key_env), rows, a.concurrency, a.max_tokens, headers))
    report["candidate"] = {"model": a.candidate[1], **score(rows, cand_out)}
    if a.current:
        cur_out = asyncio.run(run_model(*a.current, env(a.current_key_env), rows, a.concurrency, a.max_tokens, headers))
        report["current"] = {"model": a.current[1], **score(rows, cur_out)}
        report["promote"] = report["candidate"]["main"] >= report["current"]["main"] - a.margin
        report["rule"] = f"candidate {report['candidate']['main_metric']} >= current - {a.margin}"
    else:
        report["promote"] = report["candidate"]["main"] >= a.min_score
        report["rule"] = f"candidate {report['candidate']['main_metric']} >= {a.min_score}"
    report["samples"] = [{"prompt": r["prompt"][-1]["content"][:300], "reference": r["reference"][:300],
                          "candidate": c["answer"][:300]} for r, c in list(zip(rows, cand_out))[:10]]

    for side in ("candidate", "current"):
        if side in report:
            s = report[side]
            extra = f"  json_valid {s['json_valid']:.1%}  json_fields {s['json_fields']:.1%}" if "json_fields" in s else ""
            print(f"{side:<9} {s['model']:<40} exact {s['exact']:.1%}  f1 {s['f1']:.1%}{extra}  "
                  f"errors {s['errors']}  p50 {s['latency_ms']} ms")
    print("PROMOTE" if report["promote"] else "KEEP CURRENT", "-", report["rule"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
