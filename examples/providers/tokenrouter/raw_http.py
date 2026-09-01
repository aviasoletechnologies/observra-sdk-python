"""TokenRouter OpenAI-compatible API through raw HTTPX.

Set GATEWAY_KEY and TOKENROUTER_API_KEY before running.
"""

import os

import httpx
from dotenv import load_dotenv

import observra

load_dotenv()

gateway_key = os.getenv("GATEWAY_KEY")
tokenrouter_api_key = os.getenv("TOKENROUTER_API_KEY")
if not gateway_key or not tokenrouter_api_key:
    raise RuntimeError("Set GATEWAY_KEY and TOKENROUTER_API_KEY in the repository root .env")

observra.configure(gateway_key=gateway_key)
observra.instrument()

response = httpx.post(
    "https://api.tokenrouter.com/v1",
    headers={"Authorization": f"Bearer {tokenrouter_api_key}"},
    json={
        "model": os.getenv("TOKENROUTER_MODEL", "z-ai/glm-5.3-free"),
        "messages": [{"role": "user", "content": "Explain observability in one sentence."}],
    },
    timeout=30.0,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
