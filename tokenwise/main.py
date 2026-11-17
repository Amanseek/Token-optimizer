"""Entry point: `python -m tokenwise.main` or `uvicorn tokenwise.main:app`."""

import os

import uvicorn

from .app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("TOKENWISE_HOST", "127.0.0.1"),
        port=int(os.getenv("TOKENWISE_PORT", "8787")),
    )
