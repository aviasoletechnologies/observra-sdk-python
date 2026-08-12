"""OpenAI Chat Completions API through raw HTTP.

Set OBSERVRA_GATEWAY_KEY and OPENAI_API_KEY before running.
"""

import os

import httpx
import observra

observra.configure(gateway_key="")
observra.instrument()

response = httpx.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {""}"},
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
