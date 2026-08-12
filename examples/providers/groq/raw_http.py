"""Groq OpenAI-compatible API through raw HTTP.

Set OBSERVRA_GATEWAY_KEY and GROQ_API_KEY before running.
"""

import os

import httpx
import observra

observra.configure(gateway_key="")
observra.instrument()

response = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {""}"},
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
