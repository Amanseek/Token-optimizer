"""Builds pipeline/kaggle_e2e.ipynb: the whole first-client flow on one Kaggle GPU.

    train adapter -> serve with vLLM (multi-LoRA) -> client calls go through Tokenwise
    -> evaluate the new model against the base model -> show captured workflows

Demo data: the scam-checker messages (public, synthetic), standing in for a client's export.

    python pipeline/build_kaggle_e2e.py
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Tokenwise pipeline: end-to-end test on one GPU

Stands in for a rented GPU server. Steps: **train** a LoRA adapter (plain TRL + PEFT) →
**serve** base model + adapter with **vLLM** → the "client app" calls it **through Tokenwise**
with a `tw-` key → **evaluate** new vs base model → check the **captured workflow**.

Settings: GPU T4 x2, Internet on. Run all (about 40–60 minutes)."""),

    (CODE, """%%capture
!pip install -q -U trl peft bitsandbytes datasets
!pip uninstall -y -q torchao
!pip install -q uv"""),

    (CODE, """import json, os, subprocess, sys, time, random
W = "/kaggle/working"
!rm -rf {W}/tokenwise {W}/scam-checker
!git clone -q --depth 1 https://github.com/Amanseek/Token-optimizer {W}/tokenwise
!git clone -q --depth 1 https://github.com/Amanseek/scam-checker {W}/scam-checker
BASE = "Qwen/Qwen3-4B-Instruct-2507"
N_TRAIN, N_TEST = 1500, 150

# Stand-in for a client's Tokenwise export: chat JSONL (system, user, assistant).
sys.path.insert(0, f"{W}/scam-checker")
from datagen.generate import to_chat
def rows(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
random.seed(0)
train = random.sample(rows(f"{W}/scam-checker/data/train.jsonl"), N_TRAIN)
test = random.sample(rows(f"{W}/scam-checker/data/test.jsonl"), N_TEST)
os.makedirs(f"{W}/data", exist_ok=True)
for name, part in (("train", train), ("test", test)):
    with open(f"{W}/data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\\n")
print(open(f"{W}/data/test.jsonl").readline()[:400])"""),

    (MD, "## 1. Train the client's adapter (QLoRA on T4)"),
    (CODE, """t0 = time.time()
!python {W}/tokenwise/pipeline/train.py --train {W}/data/train.jsonl --out {W}/adapters/demo-scam --base {BASE} --qlora --epochs 1 --max-len 1024
print(f"train step took {(time.time()-t0)/60:.1f} min"); print(open(f"{W}/adapters/demo-scam/training.json").read())"""),

    (MD, "## 2. Serve base + adapter with vLLM (separate environment, so it can't clash with training libraries)"),
    (CODE, """%%capture
!uv venv -q {W}/vllm-env --python 3.12 || uv venv -q {W}/vllm-env
!VIRTUAL_ENV={W}/vllm-env uv pip install -q vllm httpx"""),

    (CODE, """VLLM_KEY = "vllm-" + os.urandom(8).hex()
env = {**os.environ, "PATH": f"{W}/vllm-env/bin:" + os.environ["PATH"], "VLLM_API_KEY": VLLM_KEY,
       "BASE_MODEL": BASE, "ADAPTERS_DIR": f"{W}/adapters", "CUDA_VISIBLE_DEVICES": "0",
       "MAX_MODEL_LEN": "2048", "GPU_MEM": "0.85", "VIRTUAL_ENV": f"{W}/vllm-env"}
vllm_log = open(f"{W}/vllm.log", "w")
vllm = subprocess.Popen(["bash", f"{W}/tokenwise/pipeline/serve/start_vllm.sh"], env=env, stdout=vllm_log, stderr=subprocess.STDOUT)
import httpx
for i in range(120):
    try:
        if httpx.get("http://127.0.0.1:8000/v1/models", headers={"authorization": f"Bearer {VLLM_KEY}"}, timeout=5).status_code == 200:
            break
    except Exception:
        pass
    if vllm.poll() is not None:
        break
    time.sleep(10)
print(open(f"{W}/vllm.log").read()[-3000:])
print(httpx.get("http://127.0.0.1:8000/v1/models", headers={"authorization": f"Bearer {VLLM_KEY}"}).json())"""),

    (MD, "## 3. Tokenwise gateway in front of it, with a client account"),
    (CODE, """%%capture
!pip install -q fastapi uvicorn httpx cryptography"""),
    (CODE, """os.chdir(f"{W}/tokenwise")
tw_env = {**os.environ, "TOKENWISE_DB": f"{W}/tw.db", "UPSTREAM_KEY": VLLM_KEY}
tw_env["TOKENWISE_SECRET"] = subprocess.check_output([sys.executable, "-m", "tokenwise.admin", "new-secret"], env=tw_env, stderr=subprocess.DEVNULL).decode().strip()
CLIENT_KEY = subprocess.check_output([sys.executable, "-m", "tokenwise.admin", "--db", f"{W}/tw.db", "create", "demo-client",
    "--openai-key-env", "UPSTREAM_KEY", "--openai-base-url", "http://127.0.0.1:8000/v1"], env=tw_env, stderr=subprocess.DEVNULL).decode().strip()
tw = subprocess.Popen([sys.executable, "-m", "uvicorn", "tokenwise.main:app", "--port", "8787"], env=tw_env,
                      stdout=open(f"{W}/tw.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
r = httpx.post("http://127.0.0.1:8787/openai/v1/chat/completions",
               headers={"authorization": f"Bearer {CLIENT_KEY}", "x-tokenwise-feature": "scam-check"},
               json={"model": "demo-scam", "messages": json.loads(open(f"{W}/data/test.jsonl").readline())["messages"][:-1],
                     "temperature": 0, "max_tokens": 80}, timeout=120)
print(r.status_code, r.json()["choices"][0]["message"]["content"], "| trace", r.headers.get("x-tokenwise-trace-id"))"""),

    (MD, "## 4. Evaluate: new adapter (through Tokenwise) vs base model"),
    (CODE, """os.environ["CLIENT_KEY"], os.environ["VLLM_KEY"] = CLIENT_KEY, VLLM_KEY
!python {W}/tokenwise/pipeline/evaluate.py --test {W}/data/test.jsonl --candidate http://127.0.0.1:8787/openai/v1 demo-scam --candidate-key-env CLIENT_KEY --current http://127.0.0.1:8000/v1 {BASE} --current-key-env VLLM_KEY --header "x-tokenwise-feature: scam-check" --max-tokens 80 --out {W}/eval_report.json
rep = json.load(open(f"{W}/eval_report.json"))
for s in rep["samples"][:3]:
    print("REF ", s["reference"]); print("NEW ", s["candidate"]); print()"""),

    (MD, "## 5. What Tokenwise captured"),
    (CODE, """for w in httpx.get("http://127.0.0.1:8787/api/workflows").json():
    print(w["client"], "|", w["workflow_label"], "| requests", w["requests"], "| success", w["success_pct"], "% | trainable", w["trainable"])
summary = {"training": json.load(open(f"{W}/adapters/demo-scam/training.json")), "eval": {k: rep[k] for k in ("candidate", "current", "promote", "rule")}}
json.dump(summary, open(f"{W}/e2e_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
tw.terminate(); vllm.terminate()"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    p = Path(__file__).with_name("kaggle_e2e.ipynb")
    p.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print("wrote", p)
