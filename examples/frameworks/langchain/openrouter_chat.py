"""LangChain + OpenRouter with Observra tracing.

Requires python-dotenv and langchain-openai. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://api.openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/observra/observra-python",
        "X-Title": "Observra LangChain example",
    },
)
response = llm.invoke("Explain observability in one sentence.")
print(response.content)
