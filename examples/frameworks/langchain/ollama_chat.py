"""LangChain + Ollama Cloud with Observra tracing.

Requires python-dotenv and langchain-ollama. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
    base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
    client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
)
response = llm.invoke("Explain observability in one sentence.")
print(response.content)
