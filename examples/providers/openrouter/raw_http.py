"""OpenRouter OpenAI-compatible API through raw HTTP.

Set GATEWAY_KEY and OPENROUTER_API_KEY before running.
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
    "https://api.openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
