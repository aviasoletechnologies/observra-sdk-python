"""OpenAI native SDK with transparent Observra routing.

Install: pip install openai
Set OBSERVRA_GATEWAY_KEY and OPENAI_API_KEY before running.
"""

import os

import observra
from openai import OpenAI

observra.configure(gateway_key="")
observra.instrument()

client = OpenAI(api_key="")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
