# GPU server for client models: vLLM + LoRA adapters mounted at /data/adapters.
#   docker build -t tokenwise-serve pipeline/serve
#   docker run --gpus all -p 8000:8000 -v /data/adapters:/data/adapters -v /data/hf:/root/.cache/huggingface \
#     -e VLLM_API_KEY=... -e BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 tokenwise-serve
FROM vllm/vllm-openai:latest
RUN pip install --no-cache-dir httpx
COPY start_vllm.sh register_adapter.py /app/
WORKDIR /app
ENTRYPOINT ["/app/start_vllm.sh"]
