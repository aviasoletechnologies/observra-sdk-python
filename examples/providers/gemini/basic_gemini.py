"""Direct Gemini usage — Observra configures routing without a wrapper client.

Set GATEWAY_KEY and GEMINI_API_KEY before running this example.
OBSERVRA_GATEWAY_URL is optional; it defaults to https://gateway.observra.in.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Configures gateway routing, tracing, and guardrails for supported provider traffic.
observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

# Native Google GenAI client; Observra patches its underlying transport automatically.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Summarize this support ticket: the customer can't reset their password.",
)

print(response.text)
