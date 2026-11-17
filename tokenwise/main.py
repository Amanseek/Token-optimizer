"""Entry point: `python -m tokenwise.main` or `uvicorn tokenwise.main:app`.

Set TOKENWISE_SSL_CERTFILE and TOKENWISE_SSL_KEYFILE to serve HTTPS directly.
The google-genai SDK always opens Live (voice) sessions over wss://, so
Gemini Live traffic needs Tokenwise behind TLS: either these settings, or a
host that terminates TLS for you (Railway, Render, Cloud Run, nginx...).
"""

import os

import uvicorn

from .app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("TOKENWISE_HOST", "127.0.0.1"),
        port=int(os.getenv("TOKENWISE_PORT", "8787")),
        ssl_certfile=os.getenv("TOKENWISE_SSL_CERTFILE") or None,
        ssl_keyfile=os.getenv("TOKENWISE_SSL_KEYFILE") or None,
    )
