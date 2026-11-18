"""Load (or unload) a client's adapter into a running vLLM server without restarting it.

    python register_adapter.py acme-faq /data/adapters/acme-faq            # load
    python register_adapter.py acme-faq --unload                           # unload
    VLLM_URL=http://127.0.0.1:8000 VLLM_API_KEY=... (defaults shown)
"""

import argparse
import os
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--unload", action="store_true")
    a = ap.parse_args()
    url = os.getenv("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
    headers = {"authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    if a.unload:
        r = httpx.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": a.name}, headers=headers, timeout=60)
    else:
        if not a.path:
            ap.error("path is required to load an adapter")
        r = httpx.post(f"{url}/v1/load_lora_adapter", json={"lora_name": a.name, "lora_path": a.path},
                       headers=headers, timeout=300)
    print(r.status_code, r.text[:300])
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
