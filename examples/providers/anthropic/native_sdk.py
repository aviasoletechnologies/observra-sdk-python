"""Anthropic native SDK with transparent Observra routing.

Install: pip install anthropic
Set GATEWAY_KEY and ANTHROPIC_API_KEY before running.
"""

import os
from pathlib import Path

import observra
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=128,
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(message.content[0].text)
