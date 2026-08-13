"""LangChain + xAI tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("XAI_MODEL", "grok-3-mini"), api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
print(llm.invoke("Explain observability in one sentence.").content)
