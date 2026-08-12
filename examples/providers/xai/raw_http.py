"""xAI raw HTTP with transparent Observra routing. Requires httpx and python-dotenv."""

import os
from pathlib import Path

import httpx
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


response = httpx.post("https://api.x.ai/v1/chat/completions", headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"}, json={"model": os.getenv("XAI_MODEL", "grok-3-mini"), "messages": [{"role": "user", "content": "Explain observability in one sentence."}]})
print(response.json())
