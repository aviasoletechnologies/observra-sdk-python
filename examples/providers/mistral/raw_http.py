"""Mistral raw HTTP with transparent Observra routing. Requires httpx and python-dotenv."""

import os
from pathlib import Path

import httpx
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


response = httpx.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"}, json={"model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"), "messages": [{"role": "user", "content": "Explain observability in one sentence."}]})
print(response.json())
