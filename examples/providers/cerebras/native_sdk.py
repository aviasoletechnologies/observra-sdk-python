"""Cerebras native SDK with transparent Observra routing.

Install: pip install 'observra-sdk-python[cerebras,examples]'
Set GATEWAY_KEY and CEREBRAS_API_KEY before running.
"""

import os
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

import observra

load_dotenv()

gateway_key = os.getenv("GATEWAY_KEY")
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not gateway_key or not cerebras_api_key:
    raise RuntimeError("Set GATEWAY_KEY and CEREBRAS_API_KEY in the repository root .env")

# Configure before constructing Cerebras so its HTTPX client receives gateway mounts.
observra.configure(gateway_key=gateway_key)
observra.instrument()

client = Cerebras(api_key=cerebras_api_key)
response = client.chat.completions.create(
    model="gemma-4-31b",
    messages=[{"role": "user", "content": "Explain observability in one sentence."}],
)

print(response.choices[0].message.content)
