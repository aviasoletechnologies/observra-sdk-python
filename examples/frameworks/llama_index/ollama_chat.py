"""LlamaIndex + Ollama Cloud with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-ollama.
Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = Ollama(
    model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
    base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
    api_key=os.getenv("OLLAMA_API_KEY"),
)
response = llm.complete("Explain observability in one sentence.")
print(response.text)
