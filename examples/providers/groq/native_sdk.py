"""Groq native SDK with transparent Observra routing.

Install: pip install groq
Set GATEWAY_KEY and GROQ_API_KEY before running.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
