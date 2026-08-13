"""OpenRouter through OpenAI native SDK with transparent Observra routing.

Install: pip install openai
Set GATEWAY_KEY and OPENROUTER_API_KEY before running.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://api.openrouter.ai/api/v1",
)
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
