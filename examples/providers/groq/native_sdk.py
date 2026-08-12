"""Groq native SDK with transparent Observra routing.

Install: pip install groq
Set OBSERVRA_GATEWAY_KEY and GROQ_API_KEY before running.
"""

import os

import observra
from groq import Groq

observra.configure(gateway_key="")
observra.instrument()

client = Groq(api_key="")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
