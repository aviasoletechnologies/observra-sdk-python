"""Mistral native SDK tracing. Install: pip install mistralai python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
response = client.chat.complete(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)
print(response.choices[0].message.content)
