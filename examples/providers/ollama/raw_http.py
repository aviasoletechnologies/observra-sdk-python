"""Ollama Cloud chat API through raw HTTP.

Set GATEWAY_KEY and OLLAMA_API_KEY before running.
For local Ollama, use http://localhost:11434/api/chat instead.
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
    "https://ollama.com/api/chat",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
    json={
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
        "stream": False,
    },
    timeout=60.0,
)
response.raise_for_status()
print(response.json()["message"]["content"])
