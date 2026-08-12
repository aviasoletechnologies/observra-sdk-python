"""Direct Gemini usage — Observra configures routing without a wrapper client.

Set OBSERVRA_GATEWAY_KEY and GEMINI_API_KEY before running this example.
OBSERVRA_GATEWAY_URL is optional; it defaults to https://gateway.observra.in.
"""

import os

import observra
from google import genai

# Configures gateway routing, tracing, and guardrails for supported provider traffic.
observra.configure(gateway_key="")
observra.instrument()

# Native Google GenAI client; Observra patches its underlying transport automatically.
client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Summarize this support ticket: the customer can't reset their password.",
)

print(response.text)
