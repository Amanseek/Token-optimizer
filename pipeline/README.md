# Pipeline: train, serve and evaluate client models

Everything needed to go from a client's captured traffic to a small model serving their requests.

```
client app ──tw- key──► Tokenwise gateway ──► vLLM on one GPU: base model + one LoRA adapter per client task
                            │                        ▲
                            └─ traces ─► export ─► train.py ─► adapter ─► evaluate.py ─► register_adapter.py
```

| File | What it does |
|---|---|
| `train.py` | Tokenwise export (chat JSONL) → LoRA adapter. Plain transformers + PEFT + TRL; loss only on the answer; `--qlora` for small GPUs |
| `evaluate.py` | Runs a test set against any two OpenAI-compatible endpoints (new adapter vs the client's current model) and prints PROMOTE / KEEP CURRENT |
| `serve/start_vllm.sh` | Serves one base model plus every adapter in `ADAPTERS_DIR` (OpenAI-compatible, API key required) |
| `serve/register_adapter.py` | Adds or removes an adapter on a running server, no restart |
| `serve/Dockerfile` | The serving server as a container |
| `kaggle_e2e.ipynb` | The whole flow on one free Kaggle GPU (train → serve → call through Tokenwise → evaluate) |

## First client: step by step

**0. Before any client (one time)**
- Accounts: a GPU cloud in India (E2E Networks, or the IndiaAI compute portal with your Udyam certificate), optionally a serverless host.
- Tokenwise running on a small CPU server with HTTPS, `TOKENWISE_SECRET` and `TOKENWISE_ADMIN_TOKEN` set.

**1. Connect the client (day 1)**
```bash
python -m tokenwise.admin create acme --openai-key-env ACME_CURRENT_KEY --openai-base-url <their current provider>
```
Their app switches its base URL and key to Tokenwise. Traffic is logged and traced (masked, encrypted). Use `x-tokenwise-feature` per task if they can.

**2. Collect (1–4 weeks)**
Watch `/dashboard` → Workflows. Pick a task with high volume, a high success rate and enough trainable traces (about 1,000+).

**3. Export**
```bash
python -m tokenwise.export --client acme --workflow f:support-reply --out data/acme-support
```
If the answers came from OpenAI, Google or Anthropic, only corrected answers export unless the client's use is permitted (see the main README).

**4. Train (rent a GPU for 1–2 hours)**
```bash
# A100 / H100 / L40S
python pipeline/train.py --train data/acme-support/train.jsonl --out adapters/acme-support
# T4 / L4 (24 GB or less)
python pipeline/train.py --train data/acme-support/train.jsonl --out adapters/acme-support --qlora
```

**5. Serve**
```bash
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 ADAPTERS_DIR=/data/adapters VLLM_API_KEY=... pipeline/serve/start_vllm.sh
# or: docker run --gpus all ... tokenwise-serve (see serve/Dockerfile)
```
Later adapters: copy the folder to `/data/adapters/<name>` and `python pipeline/serve/register_adapter.py <name> /data/adapters/<name>`.

**6. Evaluate before switching**
```bash
python pipeline/evaluate.py --test data/acme-support/test.jsonl \
  --candidate http://gpu-host:8000/v1 acme-support --candidate-key-env VLLM_API_KEY \
  --current <their provider URL> <their model> --current-key-env ACME_CURRENT_KEY --out reports/acme.json
```

**7. Switch**
Only on PROMOTE: point the client in Tokenwise at the GPU server
(`admin set-key acme openai --key-env VLLM_API_KEY` and its base URL) and have the app use model name `acme-support`.
Gradual traffic shifting with automatic rollback is the next milestone.

## Sizing and cost (Nov 2026 list prices, before GST; check before booking)

| Need | GPU | Approx. |
|---|---|---|
| Training one adapter (4B base, few thousand examples) | 1× A100 for 1–2 h | ₹200–500 per run |
| Pilot serving, low traffic | serverless GPU, scales to zero | pay per use |
| Steady serving, several clients | 1× L4 24 GB (4B–8B base, many adapters) | ~₹36,000/month |
| Larger base models (30B+) | L40S / A100 | ₹100–220/hour |

Use the smallest base model that passes the evaluation; classification and extraction tasks often work with 1.7B–4B.
