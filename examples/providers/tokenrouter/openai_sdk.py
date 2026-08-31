"""TokenRouter through OpenAI SDK with transparent Observra routing.

Install: pip install 'observra-sdk-python[openai,examples]'
Set GATEWAY_KEY and TOKENROUTER_API_KEY before running.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import observra

load_dotenv()

gateway_key = os.getenv("GATEWAY_KEY")
tokenrouter_api_key = os.getenv("TOKENROUTER_API_KEY")
if not gateway_key or not tokenrouter_api_key:
    raise RuntimeError("Set GATEWAY_KEY and TOKENROUTER_API_KEY in the repository root .env")

observra.configure(gateway_key=gateway_key)
observra.instrument()

client = OpenAI(
    api_key=tokenrouter_api_key,
    base_url="https://api.tokenrouter.com/v1",
)
response = client.chat.completions.create(
    model=os.getenv("TOKENROUTER_MODEL", "openai/gpt-5"),
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
