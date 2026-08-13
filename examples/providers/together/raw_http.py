"""Together raw HTTP with transparent Observra routing. Requires httpx and python-dotenv."""

import os
from pathlib import Path

import httpx
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


response = httpx.post("https://api.together.xyz/v1/chat/completions", headers={"Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"}, json={"model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"), "messages": [{"role": "user", "content": "Explain observability in one sentence."}]})
print(response.json())
