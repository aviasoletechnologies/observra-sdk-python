"""Fireworks native SDK tracing. Install: pip install fireworks-ai python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from fireworks.client import Fireworks

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
response = client.chat.completions.create(model=os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"), messages=[{"role": "user", "content": "Explain observability in one sentence."}])
print(response.choices[0].message.content)
