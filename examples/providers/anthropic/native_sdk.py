"""Anthropic native SDK with transparent Observra routing.

Install: pip install anthropic
Set OBSERVRA_GATEWAY_KEY and ANTHROPIC_API_KEY before running.
"""

import os

import observra
from anthropic import Anthropic

observra.configure(gateway_key="")
observra.instrument()

client = Anthropic(api_key="")
message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=128,
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(message.content[0].text)
