"""Ollama Cloud chat API through raw HTTP.

Set OBSERVRA_GATEWAY_KEY and OLLAMA_API_KEY before running.
For local Ollama, use http://localhost:11434/api/chat instead.
"""

import os

import httpx
import observra

observra.configure(gateway_key="")
observra.instrument()

response = httpx.post(
    "https://ollama.com/api/chat",
    headers={"Authorization": f"Bearer {""}"},
    json={
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
        "stream": False,
    },
    timeout=60.0,
)
response.raise_for_status()
print(response.json()["message"]["content"])
