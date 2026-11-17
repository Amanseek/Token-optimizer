"""Compare cost and quality with and without Tokenwise.

Each task is sent twice: straight to Anthropic, and through a running Tokenwise
proxy. For each run we record cost (from the usage block) and whether the
answer passes the task's check (a regex).

Usage:
    export ANTHROPIC_API_KEY=...
    # in another terminal: TOKENWISE_ROUTING=on python -m tokenwise.main
    python evals/run_eval.py evals/tasks.jsonl --proxy http://127.0.0.1:8787

This is a single-turn harness to get started. The stronger test is replaying
real agent sessions on your own repos and checking that the test suite still
passes; see README.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenwise.pricing import Usage, cost  # noqa: E402

DIRECT = "https://api.anthropic.com"


def run(base_url: str, task: dict, key: str, label: str) -> dict:
    body = {
        "model": task.get("model", "claude-opus-5-5"),
        "max_tokens": task.get("max_tokens", 1024),
        "messages": [{"role": "user", "content": task["prompt"]}],
    }
    if task.get("system"):
        body["system"] = task["system"]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "x-tokenwise-user": "eval"}
    r = httpx.post(f"{base_url}/v1/messages", json=body, headers=headers, timeout=300)
    r.raise_for_status()
    data = r.json()
    model_used = r.headers.get("x-tokenwise-model", data.get("model", body["model"]))
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    passed = bool(re.search(task["check"], text, re.IGNORECASE | re.DOTALL))
    return {
        "task": task["id"], "run": label, "model": model_used,
        "cost_usd": round(cost(model_used, Usage.from_api(data.get("usage"))), 6),
        "passed": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--proxy", default="http://127.0.0.1:8787")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY first.")

    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    results = []
    for t in tasks:
        for label, url in (("direct", DIRECT), ("tokenwise", args.proxy)):
            res = run(url, t, key, label)
            results.append(res)
            print(f"{res['task']:<24} {label:<10} {res['model']:<20} ${res['cost_usd']:.5f}  {'PASS' if res['passed'] else 'FAIL'}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    for label in ("direct", "tokenwise"):
        rs = [r for r in results if r["run"] == label]
        total = sum(r["cost_usd"] for r in rs)
        passed = sum(r["passed"] for r in rs)
        print(f"\n{label}: cost ${total:.4f}, passed {passed}/{len(rs)}")
    d = sum(r["cost_usd"] for r in results if r["run"] == "direct")
    t = sum(r["cost_usd"] for r in results if r["run"] == "tokenwise")
    if d:
        print(f"Saved {100 * (d - t) / d:.1f}% (results in {args.out})")


if __name__ == "__main__":
    main()
