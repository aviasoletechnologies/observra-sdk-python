"""Cohere native SDK tracing. Install: pip install cohere python-dotenv."""

import os
from pathlib import Path

import cohere
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(
    gateway_key=os.getenv("GATEWAY_KEY"),
)
observra.instrument()

client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
response = client.chat(
    model=os.getenv("COHERE_MODEL", "command-a-03-2025"),
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)
print(response.message.content[0].text)
