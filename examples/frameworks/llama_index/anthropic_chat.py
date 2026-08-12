"""LlamaIndex + Anthropic with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-anthropic.
Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = Anthropic(
    model="claude-3-5-haiku-latest",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=128,
)
response = llm.complete("Explain observability in one sentence.")
print(response.text)
