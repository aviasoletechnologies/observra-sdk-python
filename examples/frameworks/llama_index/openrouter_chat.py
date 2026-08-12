"""LlamaIndex + OpenRouter with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-openrouter.
Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = OpenRouter(
    model=os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=128,
    context_window=4096,
)
response = llm.complete("Explain observability in one sentence.")
print(response.text)
