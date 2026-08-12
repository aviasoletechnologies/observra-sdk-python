"""Ollama Cloud native SDK with transparent Observra routing.

Install: pip install ollama
Set OBSERVRA_GATEWAY_KEY and OLLAMA_API_KEY before running.
"""

import os

import observra
from ollama import Client

observra.configure(gateway_key="")
observra.instrument()

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {""}"},
)
response = client.chat(
    model="gpt-oss:120b-cloud",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.message.content)
