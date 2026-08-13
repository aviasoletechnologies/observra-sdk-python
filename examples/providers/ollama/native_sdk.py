"""Ollama Cloud native SDK with transparent Observra routing.

Install: pip install ollama
Set GATEWAY_KEY and OLLAMA_API_KEY before running.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from ollama import Client

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
)
response = client.chat(
    model="gpt-oss:120b-cloud",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.message.content)
