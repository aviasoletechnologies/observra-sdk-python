"""Cerebras OpenAI-compatible API through raw HTTP.

Set GATEWAY_KEY and CEREBRAS_API_KEY before running.
"""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

import observra

load_dotenv()

gateway_key = os.getenv("GATEWAY_KEY")
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not gateway_key or not cerebras_api_key:
    raise RuntimeError("Set GATEWAY_KEY and CEREBRAS_API_KEY in the repository root .env")

observra.configure(gateway_key=gateway_key)
observra.instrument()

response = httpx.post(
    "https://api.cerebras.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {cerebras_api_key}"},
    json={
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
