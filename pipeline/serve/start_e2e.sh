#!/usr/bin/env bash
# Start the client API on an E2E TIR GPU container: model server + Tokenwise gateway + public HTTPS link.
#
#   cd /home/jovyan && (git clone -q https://github.com/Amanseek/Token-optimizer tokenwise 2>/dev/null || git -C tokenwise pull -q) \
#     && bash tokenwise/pipeline/serve/start_e2e.sh
#
#   bash tokenwise/pipeline/serve/start_e2e.sh stop     # stop all three (then Stop the container in TIR)
#
# Safe to run again. Keys and the Tokenwise database live in /home/jovyan/tw (kept between Stop/Start).
# The public link (https://....trycloudflare.com) changes on every start; good for demos, not for clients.
set -euo pipefail

WS=${WS:-/home/jovyan}         # the only folder E2E keeps between Stop and Start
TW=$WS/tokenwise                      # this repo
STATE=$WS/tw                          # keys + database (private)
ADAPTERS=$WS/adapters                 # one folder per adapter: adapters/<model name>/
KAGGLE_RUN=${KAGGLE_RUN:-amangupta786/notebook62ba0b2e52}
CLIENT=${CLIENT:-demo-client}
export HF_HOME=$WS/hf                 # downloaded base model stays in the workspace

stop_all() { pkill -f "hf_server.py|tokenwise.main:app|cloudflared tunnel" 2>/dev/null || true; }
if [ "${1:-}" = "stop" ]; then stop_all; echo "stopped. Now Stop the container in TIR to stop GPU billing."; exit 0; fi

quiet() { grep -v -i -E "warning|dependency|requires|incompatible|already satisfied" || true; }
wait_for() {  # wait_for <url> <log> <seconds>
  for _ in $(seq "$3"); do curl -s -o /dev/null "$1" && return 0; sleep 1; done
  echo "Did not start. Last lines of $2:"; tail -25 "$2"; exit 1
}

echo "== 1/6 tools"
if [ "${LOCAL_TEST:-}" != 1 ]; then
  pip install -q -U transformers peft bitsandbytes accelerate fastapi uvicorn httpx kaggle 2>&1 | quiet
  # the image's cryptography is a Debian package pip can't upgrade; install ours alongside it
  python -c "import cryptography, sys; sys.exit(int(cryptography.__version__.split('.')[0]) < 42)" 2>/dev/null \
    || pip install -q --ignore-installed "cryptography>=42" 2>&1 | quiet
fi

echo "== 2/6 GPU"
if [ "${LOCAL_TEST:-}" = 1 ]; then echo "(local test: GPU check skipped)"; else
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! python - <<'EOF'
import sys, torch
major, minor = torch.cuda.get_device_capability(0)
ok = f"sm_{major}{minor}" in torch.cuda.get_arch_list()
print(f"torch {torch.__version__}: GPU sm_{major}{minor}", "supported" if ok else "NOT supported")
sys.exit(0 if ok else 1)
EOF
then
  # E2E's PyTorch image is NVIDIA's build without older GPUs (V100). Install the standard one.
  echo "installing standard PyTorch for this GPU (2-4 min)..."
  PIP_CONSTRAINT="" pip install -q "torch==2.8.0+cu126" --extra-index-url https://download.pytorch.org/whl/cu126 2>&1 | quiet
  PIP_CONSTRAINT="" pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
  python -c "import torch; torch.randn(2, device='cuda'); print('GPU works with torch', torch.__version__)"
fi
fi

echo "== 3/6 adapters"
mkdir -p "$ADAPTERS"
if [ ! -f "$ADAPTERS/demo-scam/adapter_config.json" ]; then
  if [ ! -f "$WS/run/adapters/demo-scam/adapter_config.json" ]; then
    echo "downloading the trained adapter from Kaggle ($KAGGLE_RUN)..."
    kaggle kernels output "$KAGGLE_RUN" -p "$WS/run" > /dev/null
  fi
  cp -r "$WS/run/adapters/demo-scam" "$ADAPTERS/demo-scam"
fi
ls "$ADAPTERS"

echo "== 4/6 keys"
mkdir -p "$STATE" && chmod 700 "$STATE"
if [ ! -f "$STATE/secrets.env" ]; then  # created once, reused after every restart
  umask 077
  {
    echo "export MODEL_API_KEY=$(python -c 'import secrets; print("mk-" + secrets.token_urlsafe(24))')"
    echo "export TOKENWISE_ADMIN_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(18))')"
    echo "export TOKENWISE_SECRET=$(cd "$TW" && python -m tokenwise.admin new-secret 2>/dev/null)"
  } > "$STATE/secrets.env"
fi
# shellcheck disable=SC1091
source "$STATE/secrets.env"
export TOKENWISE_DB=$STATE/tokenwise.db
if [ ! -f "$STATE/client_key.txt" ]; then
  (cd "$TW" && python -m tokenwise.admin create "$CLIENT" --openai-key-env MODEL_API_KEY \
      --openai-base-url http://127.0.0.1:8000/v1 2>/dev/null) | grep -o 'tw-[A-Za-z0-9_-]*' | head -1 > "$STATE/client_key.txt"
  chmod 600 "$STATE/client_key.txt"
fi
CLIENT_KEY=$(cat "$STATE/client_key.txt")
[ -n "$CLIENT_KEY" ] || { echo "could not create the client key"; exit 1; }
echo "client '$CLIENT' ready (key in $STATE/client_key.txt)"

echo "== 5/6 start"
stop_all; sleep 2
nohup python -u "$TW/pipeline/serve/hf_server.py" --adapters-dir "$ADAPTERS" --port 8000 > "$WS/model_server.log" 2>&1 &
echo "model server loading (1-4 min the first time)..."
wait_for http://127.0.0.1:8000/health "$WS/model_server.log" 600
(cd "$TW" && nohup python -m uvicorn tokenwise.main:app --host 127.0.0.1 --port 8787 > "$WS/tokenwise.log" 2>&1 &)
wait_for http://127.0.0.1:8787/ "$WS/tokenwise.log" 60
mkdir -p "$WS/bin"
if [ ! -x "$WS/bin/cloudflared" ]; then
  curl -sL -o "$WS/bin/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$WS/bin/cloudflared"
fi
nohup "$WS/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8787 > "$WS/tunnel.log" 2>&1 &
URL=""
for _ in $(seq 60); do URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" "$WS/tunnel.log" | grep -v "://api\." | head -1) && [ -n "$URL" ] && break; sleep 1; done
[ -n "$URL" ] || { echo "No public link. Last lines of the tunnel log:"; tail -15 "$WS/tunnel.log"; exit 1; }

echo "== 6/6 test through the public link"
sleep 5  # the new link needs a few seconds to work
# the instruction the scam adapter was trained with (from the scam-checker repo)
[ -d "$WS/scam-checker" ] || git clone -q --depth 1 https://github.com/Amanseek/scam-checker "$WS/scam-checker"
SYSTEM=$(cd "$WS/scam-checker" && python -c "from schema.labels import INSTRUCTION; print(INSTRUCTION)")
python - "$URL" "$CLIENT_KEY" "$SYSTEM" <<'EOF'
import sys, time, httpx
url, key, system = sys.argv[1:]
msg = "Dear customer your SBI account will be blocked today, update KYC now http://sbi-kyc-update.xyz"
for attempt in range(6):
    try:
        t = time.time()
        r = httpx.post(f"{url}/openai/v1/chat/completions", timeout=120,
                       headers={"authorization": f"Bearer {key}", "x-tokenwise-feature": "scam-check"},
                       json={"model": "demo-scam", "max_tokens": 80, "temperature": 0,
                             "messages": [{"role": "system", "content": system}, {"role": "user", "content": msg}]})
        r.raise_for_status()
        print(f"OK in {time.time() - t:.1f}s:", r.json()["choices"][0]["message"]["content"].strip())
        break
    except Exception as e:
        print("retrying:", str(e)[:120]); time.sleep(5)
else:
    sys.exit("The API did not answer through the public link. Check model_server.log and tokenwise.log in /home/jovyan.")
EOF

cat <<EOF

READY.
  API for apps (OpenAI format):  $URL/openai/v1
  Model name:                    demo-scam
  Client key:                    cat $STATE/client_key.txt   (private: don't share or screenshot)
  Dashboard (private link):      $URL/dashboard?token=\$TOKENWISE_ADMIN_TOKEN
                                 see it with: source $STATE/secrets.env && echo "$URL/dashboard?token=\$TOKENWISE_ADMIN_TOKEN"
  Stop:                          bash $TW/pipeline/serve/start_e2e.sh stop   (then Stop the container in TIR)
EOF
