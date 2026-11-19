"""Small OpenAI-compatible server: one base model + many LoRA adapters, plain transformers + peft.

For GPUs vLLM doesn't support (V100, T4) and for quick demos. On an L4 or better, prefer
start_vllm.sh: same API, much faster with many users.

    MODEL_API_KEY=... python pipeline/serve/hf_server.py --adapters-dir /data/adapters --port 8000

Every sub-folder of --adapters-dir with an adapter_config.json is served under the folder's name
(e.g. /data/adapters/demo-scam -> model "demo-scam"). The base model comes from the adapters'
own config unless --base is given; requesting the base model's name answers without any adapter.

Endpoints (Bearer MODEL_API_KEY required): GET /v1/models, POST /v1/chat/completions, GET /health.
Requests are answered one at a time on the GPU (fine for demos and light traffic).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_MAX_TOKENS = 256
MAX_TOKENS_CAP = 2048


def find_adapters(adapters_dir: str | Path) -> dict[str, Path]:
    """name -> folder, for every sub-folder holding an adapter_config.json (sorted by name)."""
    root = Path(adapters_dir)
    if not root.is_dir():
        return {}
    return {d.name: d for d in sorted(root.iterdir()) if (d / "adapter_config.json").is_file()}


def base_of(adapters: dict[str, Path]) -> str:
    """The base model all adapters were trained on (they must agree)."""
    bases = {json.loads((p / "adapter_config.json").read_text())["base_model_name_or_path"] for p in adapters.values()}
    if len(bases) != 1:
        raise SystemExit(f"adapters use different base models {sorted(bases)}; serve them on separate servers or pass --base")
    return bases.pop()


class Engine:
    """Loads the model once; generate() switches adapter per request."""

    def __init__(self, base: str, adapters: dict[str, Path], four_bit: bool | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.base = base
        self.lock = threading.Lock()
        cuda = torch.cuda.is_available()
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        kwargs: dict = {"dtype": dtype}
        if cuda:
            kwargs["device_map"] = {"": 0}
            if four_bit is None or four_bit:  # same 4-bit setup the adapters were trained with
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        first = next(iter(adapters.values()), None)
        self.tok = AutoTokenizer.from_pretrained(first or base)
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        self.adapters = list(adapters)
        if adapters:
            names = iter(adapters.items())
            name, path = next(names)
            model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
            for name, path in names:
                model.load_adapter(str(path), adapter_name=name)
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

    def models(self) -> list[str]:
        return [self.base, *self.adapters]

    def generate(self, model_name: str, messages: list[dict], max_tokens: int, temperature: float,
                 top_p: float, stop: list[str]) -> tuple[str, int, int, str]:
        """Returns (text, prompt_tokens, completion_tokens, finish_reason)."""
        torch = self.torch
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        gen = {"max_new_tokens": max_tokens, "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen.update(do_sample=False)
        with self.lock, torch.inference_mode():
            if model_name in self.adapters:
                self.model.set_adapter(model_name)
                out = self.model.generate(**enc, **gen)
            elif self.adapters:  # base model requested: switch adapters off for this call
                with self.model.disable_adapter():
                    out = self.model.generate(**enc, **gen)
            else:
                out = self.model.generate(**enc, **gen)
        new = out[0][n_in:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return text, n_in, len(new), "length" if len(new) >= max_tokens else "stop"


def create_app(engine, api_key: str) -> FastAPI:
    """engine: anything with .models() and .generate(...) (tests use a fake)."""
    if not api_key:
        raise SystemExit("set MODEL_API_KEY: the server refuses to run without a key")
    app = FastAPI(title="Tokenwise model server")

    def check(authorization: str | None) -> None:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")

    @app.get("/health")
    def health():
        return {"ok": True, "models": engine.models()}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(None)):
        check(authorization)
        now = int(time.time())
        return {"object": "list", "data": [{"id": m, "object": "model", "created": now, "owned_by": "tokenwise"}
                                           for m in engine.models()]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str | None = Header(None)):
        check(authorization)
        body = await request.json()
        name = body.get("model") or ""
        if name not in engine.models():
            raise HTTPException(404, f"model '{name}' not found; available: {engine.models()}")
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages is required")
        messages = [{"role": m.get("role", "user"), "content": _text(m.get("content"))} for m in messages]
        max_tokens = min(int(body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS),
                         MAX_TOKENS_CAP)
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        temperature = float(body.get("temperature", 0) or 0)
        top_p = float(body.get("top_p", 1) or 1)

        import anyio
        text, n_in, n_out, finish = await anyio.to_thread.run_sync(
            lambda: engine.generate(name, messages, max_tokens, temperature, top_p, stop))
        for s in stop:  # cut at the first stop string, as OpenAI does
            if s and s in text:
                text, finish = text[: text.index(s)], "stop"
        cid, now = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}
        if body.get("stream"):  # answer is computed in one go; sent as a single chunk for streaming clients
            def events():
                head = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": name}
                yield _sse({**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                                 "finish_reason": None}]})
                last = {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                if (body.get("stream_options") or {}).get("include_usage"):
                    last["usage"] = usage
                yield _sse(last)
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": now, "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": usage,
        })

    return app


def _text(content) -> str:
    """OpenAI content can be a string or a list of parts; keep the text parts."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible server for a base model + LoRA adapters")
    ap.add_argument("--adapters-dir", default=os.getenv("ADAPTERS_DIR", "/data/adapters"))
    ap.add_argument("--base", help="base model (default: read from the adapters)")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 when Tokenwise runs on the same machine")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-4bit", action="store_true", help="load the base model in 16-bit (needs more GPU memory)")
    args = ap.parse_args(argv)

    adapters = find_adapters(args.adapters_dir)
    base = args.base or (base_of(adapters) if adapters else None)
    if not base:
        raise SystemExit(f"no adapters in {args.adapters_dir}; pass --base to serve a plain model")
    key = os.getenv("MODEL_API_KEY", "")
    print(f"base: {base} | adapters: {', '.join(adapters) or '(none)'}", flush=True)
    engine = Engine(base, adapters, four_bit=False if args.no_4bit else None)
    app = create_app(engine, key)
    print("ready", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
