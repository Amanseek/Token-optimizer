"""Export captured traces as a training set (chat-format JSONL) plus a held-out test set.

    python -m tokenwise.export --list                        # workflows and how many usable traces each has
    python -m tokenwise.export --client acme --workflow f:support-reply --out data/

What's exported for each trace: system prompt, the conversation, and the target answer.
Target answer, in order of preference:
  1. corrected_output (a person's better answer)  - always allowed
  2. the model's own answer, if the trace succeeded and wasn't rated bad

Provider terms: OpenAI, Google (Gemini) and Anthropic restrict using their outputs to
train models that compete with them. For traces answered by those hosts, model answers
are only exported with --allow-provider-outputs (you confirm your use is permitted, e.g.
personal or non-competing). Corrected answers and open-model answers export by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .db import Store
from .traces import is_proprietary


def usable_rows(store: Store, client: str | None, workflow: str | None, days: int) -> list[dict]:
    sql = ("SELECT t.*, c.name AS client_name FROM traces t LEFT JOIN clients c ON c.id = t.client_id "
           "WHERE t.ts >= ? AND ((t.success = 1 AND COALESCE(t.feedback, 1) = 1) OR t.corrected_output IS NOT NULL)")
    params: list = [time.time() - days * 86400]
    if client:
        sql += " AND c.name = ?"
        params.append(client)
    if workflow:
        sql += " AND t.workflow_key = ?"
        params.append(workflow)
    return store.query(sql + " ORDER BY t.ts", tuple(params))


def to_example(row: dict, allow_provider_outputs: bool) -> tuple[dict | None, str]:
    """(chat example or None, why skipped)."""
    if row.get("corrected_output"):
        target, source = row["corrected_output"], "corrected"
    else:
        if is_proprietary(row.get("upstream_host") or "") and not allow_provider_outputs:
            return None, "provider output (needs --allow-provider-outputs)"
        if not (row.get("output_text") or "").strip():
            return None, "no text answer (tool calls only)"
        target, source = row["output_text"], "model"
    try:
        convo = json.loads(row.get("messages") or "[]")
    except ValueError:
        return None, "unreadable conversation"
    if not isinstance(convo, list) or (convo and isinstance(convo[0], dict) and convo[0].get("truncated")):
        return None, "conversation was truncated"
    messages = []
    if (row.get("system") or "").strip():
        messages.append({"role": "system", "content": row["system"]})
    for m in convo:
        role = m.get("role")
        if role in ("user", "assistant") and (m.get("content") or "").strip():
            messages.append({"role": role, "content": m["content"]})
    if not messages or messages[-1]["role"] != "user":
        return None, "conversation doesn't end with a user turn"
    messages.append({"role": "assistant", "content": target})
    return {"messages": messages, "meta": {"trace_id": row["id"], "workflow": row["workflow_key"],
                                           "source": source, "model": row.get("model")}}, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="tokenwise.db")
    ap.add_argument("--client", help="client name")
    ap.add_argument("--workflow", help="workflow key, e.g. f:tarot or s:1a2b3c4d5e6f (see --list)")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default="export")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--allow-provider-outputs", action="store_true",
                    help="include answers from OpenAI/Gemini/Claude hosts; only if your use is permitted")
    ap.add_argument("--list", action="store_true", help="list workflows and exit")
    a = ap.parse_args(argv)
    store = Store(a.db)

    if a.list:
        from .workflows import summary
        for r in summary(store, a.days):
            print(f"{r['client']:<16} {r['workflow_key']:<16} {r['requests']:>6} traces  "
                  f"{r['trainable']:>6} usable  {r['workflow_label']}")
        return 0

    rows = usable_rows(store, a.client, a.workflow, a.days)
    examples, skipped = [], {}
    for r in rows:
        ex, why = to_example(r, a.allow_provider_outputs)
        if ex:
            examples.append(ex)
        else:
            skipped[why] = skipped.get(why, 0) + 1
    if not examples:
        print("No usable traces." + (f" Skipped: {skipped}" if skipped else ""), file=sys.stderr)
        return 1
    random.Random(a.seed).shuffle(examples)
    n_test = max(1, int(len(examples) * a.test_frac)) if len(examples) >= 10 else 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", examples[n_test:]), ("test.jsonl", examples[:n_test])):
        with (out / name).open("w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples) - n_test} train + {n_test} test examples to {out}/"
          + (f"; skipped {skipped}" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
