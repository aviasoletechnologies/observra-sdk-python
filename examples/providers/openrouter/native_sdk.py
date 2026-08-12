"""OpenRouter through OpenAI native SDK with transparent Observra routing.

Install: pip install openai
Set OBSERVRA_GATEWAY_KEY and OPENROUTER_API_KEY before running.
"""

import os

import observra
from openai import OpenAI

observra.configure(gateway_key="")
observra.instrument()

client = OpenAI(
    api_key="",
    base_url="https://api.openrouter.ai/api/v1",
)
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
