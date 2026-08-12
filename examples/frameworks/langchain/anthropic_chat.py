"""LangChain + Anthropic with Observra tracing.

Requires python-dotenv and langchain-anthropic. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=128,
)
response = llm.invoke("Explain observability in one sentence.")
print(response.content)
