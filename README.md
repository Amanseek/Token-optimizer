# Tokenwise

A gateway that sits between your apps or AI coding agents and the model APIs (**OpenAI-compatible hosts**, **Google Gemini** and **Anthropic**).
It **tracks AI spend per client, engineer, app feature and model**, **cuts token costs**, and **captures traces of real work** (prompts, answers, tool calls, success and feedback) so you can train smaller custom models on them. No app code changes: apps only switch their base URL and API key.

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

## Multi-client gateway and trace capture

Each client gets a Tokenwise key (`tw-...`). Their app sends it instead of the provider key; Tokenwise swaps in the client's real provider key (stored encrypted on the server), so the app can't bypass the gateway.

```bash
python -m tokenwise.admin new-secret                  # once; export the output as TOKENWISE_SECRET
export ACME_OPENAI_KEY=sk-...                         # the client's provider key (never typed as an argument)
python -m tokenwise.admin create acme --openai-key-env ACME_OPENAI_KEY \
    --openai-base-url https://openrouter.ai/api/v1    # prints acme's tw- key (shown once)
python -m tokenwise.admin set-key acme gemini --key-env ACME_GEMINI_KEY
python -m tokenwise.admin list | rotate acme | disable acme | capture acme off | mask acme off
```

The client's app changes two settings:

| SDK | Base URL | API key |
|---|---|---|
| OpenAI (and any OpenAI-compatible host) | `OPENAI_BASE_URL=http://<host>:8787/openai/v1` | `tw-...` |
| google-genai | `GOOGLE_GEMINI_BASE_URL=http://<host>:8787` | `tw-...` |
| Anthropic | `ANTHROPIC_BASE_URL=http://<host>:8787` | `tw-...` |

Optional header `x-tokenwise-feature: <name>` labels the app feature (e.g. `support-reply`). Without it, traces are grouped by system prompt.

**What a trace holds:** system prompt, the conversation (normalised across providers), declared tools, the answer, tool calls made, finish reason, tokens, cost and latency.

**Personal data is masked before storing** (emails, phone numbers, UPI IDs, PAN, Aadhaar, card numbers, long digit runs). The system prompt is the client's own and is kept. Masking can be turned off per client.

**Success signals:**
- automatic: HTTP error, blocked, cut off (length) or empty means not good; the same user input again within 10 minutes marks the earlier answer not good (a retry);
- from the app: every response carries `x-tokenwise-trace-id`; the app can send
  `POST /tokenwise/v1/feedback {"trace_id": ..., "rating": "good"|"bad", "note": ..., "corrected_output": ...}`
  with its `tw-` key. A corrected answer is the best training data.

**Workflows:** `/dashboard` and `/api/workflows` show, per client and workflow: volume, success rate, 👍/👎, corrected, retried, how many traces are usable for training, spend, and the most common tool sequences.

**Client keeps its own provider key ("bring your own key"):** create the client without keys
(`python -m tokenwise.admin create acme`); their app keeps sending its own OpenAI/Gemini key as usual and adds
the header `x-tokenwise-key: tw-...`. Tokenwise uses their key for that request only and never stores it.

**Safety net and shadow tests (per client, OpenAI format):**
```bash
# backup model when the main one is offline, errors, gets cut off or (with --require-json) returns broken JSON
python -m tokenwise.admin fallback acme --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --model gemini-2.5-flash --key-env ACME_GEMINI_KEY --require-json
# copy 20% of successful requests to a second model and score how often it agrees (scores only, no text stored)
python -m tokenwise.admin shadow acme --base-url https://gpu.example.in/v1 --model acme-support \
    --key-env MODEL_API_KEY --rate 0.2
```
Answers from the backup carry `x-tokenwise-fallback: <reason>`. The dashboard shows agreement per feature and
fallbacks per reason. Shadow copies send the client's prompts to the shadow host: turn it on with their agreement.

**Export a training set:**
```bash
python -m tokenwise.export --list
python -m tokenwise.export --client acme --workflow f:support-reply --out data/   # train.jsonl + test.jsonl
```
Corrected answers always export. Answers from OpenAI, Google and Anthropic hosts export only with `--allow-provider-outputs`, because their terms restrict training models that compete with them; answers from open models (e.g. via OpenRouter, Together, vLLM) export by default. Check each host's terms.

### Data protection

| Control | How |
|---|---|
| Encryption in transit | Serve over HTTPS: `TOKENWISE_SSL_CERTFILE` / `TOKENWISE_SSL_KEYFILE`, or a host/proxy that terminates TLS |
| Encryption at rest | With `TOKENWISE_SECRET` set, stored prompts, answers, tool calls, notes and corrections are encrypted (upstream keys always are). The workflow label (first 60 characters of the client's own system prompt) stays readable for the dashboard |
| Masking | Personal data masked before storing (per-client switch) |
| Store less | `admin sample acme 0.1` stores 10% of conversations; `admin capture acme off` stores none (cost is still logged) |
| Retention | Traces auto-delete after `retention_days` (default 90; `admin retention acme 30`); checked hourly |
| Delete on request | `POST /tokenwise/v1/delete` with the client key: `{"user": "<x-tokenwise-user value>"}`, `{"trace_ids": [...]}` or `{"all": true}`; or `admin purge acme --user U123` |
| Isolation | A client key only reaches that client's traces; exports are per client |
| Access log | Dashboard/API views, exports, deletions and admin changes are recorded: `admin audit` |

For per-user deletion, the client's app should send its own (pseudonymous) user id in `x-tokenwise-user`. The cost log (`requests` table) holds no conversation content.

Traffic without a `tw-` key still passes through with the caller's own key (set `TOKENWISE_REQUIRE_CLIENT_KEY=1` to refuse it). Capture for that traffic is off unless `TOKENWISE_TRACES=1`.

## Google Gemini

Any app using the official `google-genai` SDK can go through Tokenwise with **one environment variable** and no code changes:

```bash
export GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787
```

| Gemini endpoint | What Tokenwise does |
|---|---|
| `models/*:generateContent` | Logs tokens and cost; optional response cache |
| `models/*:streamGenerateContent` | Streams through; logs final usage |
| Live API (`/ws/...BidiGenerateContent`, voice) | Relays the WebSocket in both directions; logs per-session cost |
| Everything else (`countTokens`, `embedContent`, files) | Passed through |

Costs are calculated per **modality**, because audio is several times more expensive than text. Thinking tokens are billed as output, and cached tokens get the cache discount. Live sessions are billed every turn on the whole session context, so Tokenwise adds up every usage message in the session.

**Voice needs TLS.** The SDK always opens Live sessions over `wss://`, even when the base URL is `http://`. So if your app uses Live, run Tokenwise behind HTTPS: deploy it on a host that provides TLS (Railway, Render, Cloud Run), or set `TOKENWISE_SSL_CERTFILE` and `TOKENWISE_SSL_KEYFILE`.

### Feature labels (optional)

Send `x-tokenwise-feature: tarot` (or `voice`, `rag`, ...) to see spend per app feature on the dashboard. With google-genai: `genai.Client(api_key=..., http_options=types.HttpOptions(headers={"x-tokenwise-feature": "tarot"}))`. The header is stripped before forwarding.

## Response cache (optional)

`TOKENWISE_RESPONSE_CACHE=true` answers **exact repeats** of a non-streaming request (same model, same prompt, same API key) from cache, without calling the model. This helps with page refreshes, retries and reopened readings, and it also keeps answers consistent for identical questions.

- Off by default. TTL `TOKENWISE_RESPONSE_CACHE_TTL` (seconds, default 7 days).
- Models containing `image` or `tts` are never cached (`TOKENWISE_RESPONSE_CACHE_EXCLUDE`).
- Skip it for one request with the header `x-tokenwise-no-cache: 1`.
- Works for Anthropic and Gemini. Cache hits show on the dashboard with the money saved.

## Recommended rollout

1. **Week 1: shadow mode (default).** Logging, caching, trimming and dedupe are on. Routing only logs what it *would* do. The dashboard shows potential routing savings.
2. Check the routing table: are the "would route" conversations really simple?
3. **Turn routing on** with `TOKENWISE_ROUTING=on`.
4. Run the eval (below) to confirm quality holds.

## Configuration

New gateway settings: `TOKENWISE_SECRET` (encrypts upstream keys), `TOKENWISE_OPENAI_UPSTREAM` (default `https://api.openai.com/v1`, for traffic without a client key), `TOKENWISE_OPENAI_PRICES` (JSON, USD per million tokens, e.g. `{"deepseek-chat": [0.27, 1.1]}`), `TOKENWISE_REQUIRE_CLIENT_KEY`, `TOKENWISE_ADMIN_TOKEN` (protects `/dashboard` and `/api/*`), `TOKENWISE_TRACES`, `TOKENWISE_TRACES_MASK`, `TOKENWISE_RETRY_WINDOW` (seconds, default 600), `TOKENWISE_TRACE_RETENTION_DAYS` (traffic without a client key, default 90).


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
| `TOKENWISE_GEMINI_UPSTREAM` | `https://generativelanguage.googleapis.com` | Gemini API |
| `TOKENWISE_GEMINI_PRICES_JSON` | — | Override Gemini prices, e.g. `{"gemini-3.1-flash-lite": {"input": {"TEXT": 0.25, "AUDIO": 0.5}, "output": {"TEXT": 1.5}}}` |
| `TOKENWISE_RESPONSE_CACHE` | `false` | Answer exact repeats from cache |
| `TOKENWISE_RESPONSE_CACHE_TTL` | `604800` | Seconds |
| `TOKENWISE_RESPONSE_CACHE_EXCLUDE` | `image,tts` | Never cache models containing these |
| `TOKENWISE_SSL_CERTFILE` / `_KEYFILE` | — | Serve HTTPS directly (needed for Gemini Live without a TLS host) |

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

For a Gemini app, `evals/prankundli_bench.py` runs real app code (PranKundali's tarot module and the SDK's Live client) twice, directly and through Tokenwise, and compares what reaches the API. By default it uses `evals/fake_gemini.py`, so it needs no key; pass `--real` with `GEMINI_API_KEY` for real numbers.

The stronger test, for selling to teams: take 50–100 real tasks from your own repos, run the agent on each with and without Tokenwise, and compare **cost per task where the tests still pass**.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Known limits

- **Anthropic and Gemini (AI Studio API)** only. Vertex AI and OpenAI/Codex are not supported yet.
- **Prices** in `tokenwise/pricing.py` and `tokenwise/gemini_pricing.py` come from public sources (Nov 2026), and some Gemini entries are approximate. Check them against the providers' pricing pages.
- **Routing, trimming and dedupe apply to Anthropic only.** Gemini requests are logged and optionally cached, never modified.
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
