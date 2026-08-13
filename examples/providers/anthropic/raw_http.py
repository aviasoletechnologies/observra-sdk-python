"""Anthropic Messages API through raw HTTP.

Set GATEWAY_KEY and ANTHROPIC_API_KEY before running.
"""

import os
from pathlib import Path

import httpx
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

response = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
    },
    json={
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["content"][0]["text"])
