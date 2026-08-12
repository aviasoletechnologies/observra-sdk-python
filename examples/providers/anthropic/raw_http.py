"""Anthropic Messages API through raw HTTP.

Set OBSERVRA_GATEWAY_KEY and ANTHROPIC_API_KEY before running.
"""

import os

import httpx
import observra

observra.configure(gateway_key="")
observra.instrument()

response = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": "",
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
