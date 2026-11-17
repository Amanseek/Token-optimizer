# Tokenwise

A proxy that sits between AI coding agents (like Claude Code) and the Anthropic API.
It **tracks AI spend per engineer** and **cuts token costs** without changing your workflow.

```
Claude Code ──► Tokenwise proxy ──► api.anthropic.com
                   │
                   ├─ logs usage & cost per engineer (SQLite)
                   ├─ trims noisy tool output
                   ├─ removes duplicate tool output
                   ├─ adds prompt-cache breakpoints
                   ├─ routes simple conversations to a cheaper model
                   └─ dashboard at /dashboard
```

## What it does

| Layer | What happens | Risk |
|---|---|---|
| **Logging** | Every request's tokens and cost, per engineer and team | None |
| **Caching** | Adds prompt-cache breakpoints where the client set none (max 4). Optional 1-hour TTL | Very low |
| **Trimming** | Long command output (e.g. 5,000 lines of test logs) keeps the head, the tail and any error/warning lines | Low. Only applies to tools in `TOKENWISE_TRIM_TOOLS` (default `Bash`, `BashOutput`). File reads are never trimmed, because agents need exact contents to edit files |
| **Dedupe** | If a tool returns exactly the same large output twice, the second copy becomes a short pointer to the first | Low |
| **Routing** | Simple conversations ("explain this", "rename", "commit message") go to a cheap model. Hard ones ("debug", "refactor", "migrate") stay | Medium. Starts in **shadow mode** (logs only) |
| **Fallback / cascade** | If the cheap model rejects a request, it is retried on the original model. For non-streaming requests, an empty or cut-off cheap answer is also retried | — |

Design rules that keep it safe:
- **Deterministic transforms.** The same history always produces the same output, so Anthropic's prompt cache (which matches exact prefixes) keeps working.
- **Assistant messages are never modified.** They can contain signed thinking blocks.
- **Sticky routing.** A conversation's model is chosen once, on its first request. Switching models mid-conversation would lose the cache.

## Quick start

```bash
pip install -r requirements.txt
python -m tokenwise.main          # listens on http://127.0.0.1:8787
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_CUSTOM_HEADERS="x-tokenwise-user: aman"   # optional; otherwise a hash of the API key is used
claude
```

Open **http://127.0.0.1:8787/dashboard** to see spend, savings, cache hit rate and per-engineer numbers. JSON is at `/api/summary`.

With Docker:

```bash
docker build -t tokenwise .
docker run -p 8787:8787 -v $PWD/data:/data tokenwise
```

## Recommended rollout

1. **Week 1: shadow mode (default).** Logging, caching, trimming and dedupe are on. Routing only logs what it *would* do. The dashboard shows potential routing savings.
2. Check the routing table: are the "would route" conversations really simple?
3. **Turn routing on** with `TOKENWISE_ROUTING=on`.
4. Run the eval (below) to confirm quality holds.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `TOKENWISE_UPSTREAM` | `https://api.anthropic.com` | Where requests go |
| `TOKENWISE_DB` | `tokenwise.db` | SQLite file |
| `TOKENWISE_HOST` / `TOKENWISE_PORT` | `127.0.0.1` / `8787` | Listen address |
| `TOKENWISE_CACHE` | `true` | Add cache breakpoints |
| `TOKENWISE_CACHE_TTL` | `5m` | `1h` costs more per write but survives long pauses |
| `TOKENWISE_TRIM` | `true` | Trim long tool output |
| `TOKENWISE_TRIM_TOOLS` | `Bash,BashOutput` | Tools whose output may be trimmed |
| `TOKENWISE_TRIM_MAX_CHARS` | `6000` | Output shorter than this is left alone |
| `TOKENWISE_TRIM_HEAD` / `_TAIL` / `_ERRORS` | `40` / `60` / `60` | Lines kept |
| `TOKENWISE_DEDUPE` | `true` | Replace repeated identical tool output |
| `TOKENWISE_ROUTING` | `shadow` | `off`, `shadow` or `on` |
| `TOKENWISE_CHEAP_MODEL` | `claude-haiku-5-5` | Target for simple conversations |
| `TOKENWISE_ROUTABLE` | `claude-opus,claude-sonnet,claude-fable` | Model prefixes that may be routed down |
| `TOKENWISE_ROUTE_MAX_TOKENS` | `30000` | Larger contexts are never routed |
| `TOKENWISE_CASCADE` | `true` | Retry failed cheap answers on the original model |
| `TOKENWISE_PRICES_JSON` | — | Override prices, e.g. `{"claude-opus-5-5": {"input": 4, "output": 20, "cache_write_5m": 5, "cache_write_1h": 8, "cache_read": 0.2}}` |

## How savings are calculated

The dashboard shows **actual cost** (from the API's usage block and the price table) and an **estimated cost without Tokenwise**:

- Routing: the same token counts priced at the originally requested model.
- Trimming and dedupe: removed characters ÷ 4, priced at the **cache-read** rate. This is deliberately conservative; much of the removed text would otherwise have been a cache write or full-price input.
- Shadow mode: potential routing savings, priced on the same token counts (the cheap model's real output length will differ).

These are estimates. For real proof, use the eval.

## Evaluating quality vs. cost

```bash
export ANTHROPIC_API_KEY=...
TOKENWISE_ROUTING=on python -m tokenwise.main &
python evals/run_eval.py evals/tasks.jsonl
```

Each task runs once directly and once through the proxy. You get cost and pass/fail for both, plus a CSV. Add your own tasks to `tasks.jsonl` (each has a `prompt` and a `check` regex).

The stronger test, for selling to teams: take 50–100 real tasks from your own repos, run the agent on each with and without Tokenwise, and compare **cost per task where the tests still pass**.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Known limits

- **Anthropic API only** for now (Claude Code). OpenAI/Codex is the next step.
- **Prices** in `tokenwise/pricing.py` come from public sources (Nov 2026). Check them against Anthropic's pricing page.
- **Cascade** only works for non-streaming requests. Claude Code streams, so for it only the HTTP-error fallback applies.
- **Routing rules are keyword-based.** The logs (`route_decisions` and `requests` tables) are the training data for a learned router later.
- **The cheap model may not support every feature** the client sends. Rejected requests fall back automatically, and that conversation stops being routed.
- **No authentication on the dashboard.** Run it on localhost or behind your own auth. Don't expose it to the internet.
- **Subscription (OAuth) users** don't pay per token. The dashboard then shows the API-equivalent cost.

## Roadmap

- Learned router trained on logged outcomes
- OpenAI / Codex support
- Code retrieval: send only relevant functions instead of whole files
- Summarizing old turns in long sessions
- Team login, budgets and alerts; Postgres for multi-user deployments
